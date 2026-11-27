"""Embodiogenesis, step one: the fly brain discovers the dog's body by babbling. No goals, no rewards, no prediction.

Like a newborn on its back, the dog is held in the air and its brain moves at random. Spontaneous bursts of activity
in the motor neurons twist the joints; the joints move until they hit their limits. The brain feels all of it through
its joints' stretch and pressure sense (angle, speed, effort, and the push-back at a limit), carried by anonymous
sensory nerves into its own sensory neurons. Nobody tells it which neuron moves which leg, or which nerve comes from
which joint.

Plastic synapses from each motor neuron to each sensory neuron strengthen when that motor neuron's activity is
followed by a change in that sensory neuron (Hebbian, local, no backprop). After babbling, those synapses are its
body map. Then we ask the brain to use it: to drive one joint at a time to each end of its range.

Measured (truth known only to the test, never to the brain):
  coverage  how much of each joint's real range the babbling reached, and whether the limits were felt
  map       for each motor neuron, whether its strongest learned link is to the joint it really moves, in the right direction
  control   asked to move one joint to one end, how far toward that real limit it gets, and how still the other joints stay

Kaggle CPU:  scripts/kaggle_job.py push babble --machine CPU --dataset bio-bot-cores
"""
JOB_ARGS: dict = {}   # filled in by scripts/kaggle_job.py

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_COMMIT = "367e3d9884401dcf6f9c27fa69f118992539039f"
GENOME_G1 = 20261127     # the G1's wiring: which copied neurons it reaches (grown when the humanoid arrives, 27 Nov)
COPY_NOISE = 0.1         # a copied neuron's synapses: the template's, each x (1 + 0.1 N(0,1)), clipped to keep its sign
SLOT_KEYS = ("sense", "motor", "drives", "push", "n_motors", "fibre_neuron", "fibre_gain", "fibre_col", "intention",
             "mean", "var", "seen", "map", "Q", "Qc", "rest_sense", "rest_mean", "rest_baseline", "motor_baseline",
             "reflex_scale", "turn_line", "walk_line", "cpg_f", "R", "reflex_cols")
GENOME = 20261117        # the body's fixed wiring to the brain (which neuron drives which motor, which nerve goes where)
BRAIN_DT = 0.02          # s: the brain updates 50 times a second
ALPHA = 0.5              # how far a neuron moves toward its input each step (a 40 ms membrane time constant)
GAIN = 1.5               # connectome strength
BABBLE_TAU = 0.3         # s: how long a spontaneous burst lasts
BABBLE_SIZE = 1.2        # strength of spontaneous motor activity
NERVE_FIBRES = 16        # sensory neurons per nerve (half excited by an increase, half by a decrease)
TRACE = 0.7              # motor activity trace kept from step to step (60 ms): what the body felt shortly after
HEBB_RATE = 1e-3         # how fast a motor-to-sensory synapse strengthens
HELD_AT = 0.6            # m: the trunk is held in the air at this height, as a parent holds a baby
REPOSITION_S = 15.0      # s: on the ground, a parent sets the dog down somewhere new this often (as babies get moved)
POSES = ("stand", "belly", "left_side", "right_side")
FOOT_LOADED_N = 5.0      # N: a foot counts as bearing weight above this (the dog weighs ~150 N)
G1_HELP_START = 0.5      # balance practice starts with the parent carrying half (it always stays up at 0.5, §6.62)
G1_BABBLE_HELP = 1.0     # the G1 babbles in a parent's hands that carry its weight (it falls in ~2 s alone, §6.58)
G1_UP_HEIGHT = 0.60      # m: the G1's pelvis above this counts as up (it stands at 0.79)
STANDING_HEIGHT = 0.16   # m: trunk above this counts as up on its legs (measured: lying 0.07-0.14; with tone it stands at 0.19)
# --tone: muscle tone. Each joint springs toward a set point (the Go2's standard stiffness and damping); the motor
# neurons move the set point; when the dog is set down, the set points take the posture it is in.
TONE_KP = 20.0           # N m / rad
TONE_KD = 0.5            # N m s / rad
SET_SPEED = 4.0          # rad/s: how fast a fully driven motor pool moves its joint's set point
MOTOR_TAU = 3.0          # s: the motor pool adapts away its own steady resting drive, as the senses do
TONE_LEAK_S = 0.5        # --leak: the set point relaxes back to the posture it was set down in with this time constant,
                         # so a steady push gives a bounded offset (up to SET_SPEED x TONE_LEAK_S = 2 rad) instead of
                         # endless drift. Measured drift without it: 1.45 rad/s while babbling (§6.18), one-way
                         # collapses of the load reflexes (§6.19), and legs folding under practice noise (§6.20+).
# --reflex: the brain's own feedback. While babbling, synapses from sensory back to motor neurons are tuned by the
# twitches themselves (as in newborn rats' spinal reflexes, Petersson et al. 2003): a link grows when the motor
# neuron's activity is followed by a FALL in that sensory neuron, so the finished reflex opposes change. They stay
# silent while babbling (as reflexes are tuned during sleep twitches) and act afterwards. They are the body map's
# reverse direction with the opposite sign: the same co-activity, learned once.
REFLEX_GAINS = (0.0, 0.25, 0.5, 1.0, 2.0)   # reflex strength, in units of the map's own scale (0 = switched off)
TILT_XY_TOO = False      # --tilt_xy: also ask each tilt naming only the two directional gravity lines (diagnosis)
SETTLE_S = 30.0          # s: before tests, set down standing at rest while senses and motor pools settle (~3 time constants)
# --practice: standing practice after babbling. Its reflex synapses keep learning while it stands, gated by a
# dopamine-like signal it makes from its own senses: positive when what it feels returns toward how standing still
# feels, negative when it moves away. Each synapse's eligibility is its recent pre (sensory change since rest) x post
# (motor neuron above its usual rate) activity. The body map itself is not changed.
PRACTICE_RATE = 1e-5     # reflex synapse change per step per unit of (standardised dopamine x eligibility). At 1e-4,
                         # random drift was 68% of the reflexes in 2 min (practice1) and grows as sqrt(time); 1e-5
                         # keeps 30 min of drift near 26% instead of overwriting the reflexes.
PRACTICE_TRACE = 0.9     # eligibility kept from step to step (~0.2 s)
PRACTICE_NOISE = 0.5     # share of babbling's spontaneous motor activity kept while practising (to try variations)
EPISODE_S = 5.0          # s: each time it is set down standing
# --fair_dopamine: dopamine relative to what is usual at the current level of upset. After any jolt, what it feels
# drifts back on its own; without this, whatever pushed hardest while it was upset gets the credit, grows, upsets it
# more, and runs away (leak-practice1: up 58% at 8 min -> 0% at 18 min, reflexes changing 2% -> 12% per 2 min).
# And synaptic scaling: the reflex synapses' total strength stays at its start, so they can turn but not run away.
SUPPORT_FADE_SHARE = 2 / 3   # --support: a parent's hands hold the trunk (springs toward standing height and level)
                             # during practice; the help fades linearly to zero over this share of practice, after
                             # which it practises alone. Physical help only: the brain is never told about it.
SUPPORT_SAG = 0.05       # m: at full support the hands carry its whole weight with this much sag
# --contexts: posture neurons, grown as needed (append-only). Each is tuned to a posture: a pattern in its slowly
# smoothed senses (all nerves, in their adapted units). A new one grows when no existing one matches; it starts as a
# copy of the nearest (cell type copied, then diverges). Each has its own reflex synapses, tuned by the same twitch rule
# but only while it is active, and scaled to their own size at test.
CTX_MAX = 24             # most posture neurons it can grow
CTX_GROW = 1.0           # grows one when the posture is this far (RMS, adapted units per nerve) from every existing one
CTX_SIGMA = 0.5          # how broadly a posture neuron is tuned (same units)
CTX_TAU = 0.5            # s: the senses are smoothed over this long to tell posture
CTX_LEARN = 0.01         # how fast a posture neuron's preferred posture follows what it sees while active
CTX_ACTIVE = 0.05        # a posture neuron learns and acts only above this share of the activity
CTX_WINNER = False       # --ctx_winner: only the best-matching posture neuron moves its preferred posture (lateral
                         # inhibition); without it they all drift to the average posture (ctx1: 12 grown, top one 10%)
CTX_SLOW = False         # --ctx_slow: read posture mostly from the senses that change slowly. While babbling, each
                         # nerve's slow variance (of its smoothed value) and fast variance (step to step) are tracked;
                         # its weight is their ratio (capped at 10x the median). Posture is what stays steady.
CTX_STATS_RATE = 1e-4    # how fast those variances are estimated (~200 s)
CTX_START = int(30.0 / BRAIN_DT)   # no posture neurons in the first 30 s of life, while its senses calibrate
# --skills: reward-based practice of whole-body skills (as songbirds learn song: variability plus reward). Plastic
# skill synapses from sensory to motor neurons start at zero. Each 2 s trial from standing asks for a felt change
# (a tilt, more weight on a foot, a joint move to rehearse, or staying as it is) by shifting that sense's felt
# reference. Motor neurons get small random variation. Dopamine: getting closer to the asked feeling this step,
# relative to what is usual for that kind of trial. A synapse grows when its sensory input together with its motor
# neuron's own variation preceded dopamine (node perturbation). The twitch-tuned reflexes stay on, weak.
SKILL_REFLEX = 0.125     # twitch-tuned reflexes during skill practice (stands well at this strength, §6.26)
SKILL_NOISE = 0.3        # motor variation, as a share of babbling's
SKILL_RATE = 1e-3        # skill synapse change per step per unit of (standardised dopamine x eligibility)
SKILL_TRACE = 0.9        # eligibility kept per step (~0.2 s)
SKILL_TRIAL_S = 2.0
# --cpg: a small pattern generator, grown append-only. Four neurons inhibit each other and tire (adaptation), so the
# active one hands over to the next; each is helped a little by its predecessor, which sets an order. It runs only
# while the brain asks to turn (its descending command), and its synapses onto the motor neurons start at zero and
# are learned by the same reward practice. Nothing about legs, phases or gaits is set.
CPG_N = 4
CPG_TAU = 0.05           # s: membrane
CPG_ADAPT_TAU = 0.15     # s: tiring and recovery
CPG_ADAPT_G = 1.5        # how strongly a firing neuron tires
CPG_INH = 2.0            # mutual inhibition
CPG_SEQ = 0.5            # each neuron is inhibited less by its predecessor
SKILL_VARY = False       # --skill_vary: practise under the test's conditions (varied set-down, +-0.1 wobble), §6.45
CPG_WP = False           # --cpg_wp: learn the rhythm synapses by reward-gated synaptic noise (weight perturbation)
CPG_WP_SIZE = 1.0        # each turn trial explores Qc + e, |e| = this x the map (random draws this size turn up to 13%, all stay up)
CPG_WP_MAX = 2.0         # cap on |Qc| (x the map): random draws turn up to 37% at 2x, but topple mostly above it (§6.43)
CPG_WP_RATE = 0.03       # Qc += rate * dz * e per turn trial (random walk ~0.8 x the map over ~750 trials per direction)
CPG_SUBSTEPS = 10        # integrate it in 2 ms steps: in 20 ms steps it chattered (cpg1: 41.6 hand-overs/s, silent 40%)
FOOT_FLOOR_N = 20.0      # N: floor on a foot's starting load in weight/lift requests (half its usual ~40 N); with a
                         # varied set-down a foot can read ~0 N at the start, which blew the scores up (vary1: +198716%)
STILL_TRIAL = False      # --still_trial: on still trials the skill synapses learn once per trial, from the felt
                         # turning averaged over the trial plus level (the spin is 0.115 sd, invisible moment by moment)
WALK_V = 0.3             # m/s: a walk request asks to move forward/back at this speed (felt through the accelerometer)
WALK_ACC = 0.5           # m/s^2: the walk request as a shift of the forward-acceleration nerve's felt reference
GROW_NERVES = False      # --accel: a saved brain may continue with new nerves appended after its own (prefix-equal)
NEW_NERVE_VAR = 1.0      # an appended nerve's starting spread: 1 (m/s^2) for the accelerometer
ACTIVE_REST = False      # --active_rest: record the at-rest senses while it stands by itself (brain on). With the
                         # output cut, active standing differs a little, and the skill synapses answer that constant
                         # difference with a constant push: the spin with nothing asked (§6.52)
STILL_TURN = False       # --still_turn: "still" also means not turning (body z); the promoted dog spins ~4 deg/s when
                         # nothing is asked and nothing penalised it (§6.49-6.50)
SKILL_NO_Q_TURN = False  # --no_q_in_turn: the skill synapses do not learn on turn trials (their per-step dopamine
                         # tracks the true turn at r 0.2-0.6; suspected of teaching the crouch in wpmean1, §6.47)
CPG_WP_MEAN = False      # --cpg_wp_mean: a turn trial's dopamine is the felt turning added up over the trial (tracks the
                         # true turn at r 0.98-0.99, vs 0.19-0.59 for moment-by-moment closeness, vary1), plus level
TURN_RATE = 0.5          # rad/s: a turn request is feeling itself turn at this rate (the turning nerve, body z)
SKILL_PER_TASK = False   # --skill_per_task: dopamine relative to what is usual for that exact request (each foot,
                         # each tilt direction), not for the kind of task. Passive drift (weight sliding back, the nose
                         # sagging up) otherwise makes some requests free and others look like failures (skills6).
SKILL_LEVEL = False      # --skill_level: a request means "feel this, and otherwise feel level as when standing". The
                         # distance also counts how far the inner ear's gravity is from standing (unless tilt is what is
                         # asked). Without it, falling over took weight off a paw and was rewarded (skills7b: still up
                         # 96% -> 29%, "lift a paw" +58% from falls).
SKILL_LEVEL_JOINTS = True  # --no_level_joints: leave joint-move trials out of the stay-level term (moving one joint
                           # far tilts the trunk; penalising that may teach it to resist joint moves: joints 33 -> 20%)
SKILL_CURRICULUM = False  # --skill_curriculum: weight and lift requests start small and grow per foot as it succeeds
CURRICULUM = {"weight": (0.1, 0.5), "lift": (0.2, 1.0)}   # (first step, full request as in the tests)
SKILL_MIX = ()           # --skill_mix: the practice task mix, e.g. tilt,weight,weight,lift,lift,joint,still
SKILL_LIFT = False       # --skill_lift: also practise lifting a paw (take the weight off one foot); equal task mix
SKILL_MAX = 0.25         # synaptic scaling: skill synapses never exceed this share of the twitch map's size
                         # (skills1 at rate 1e-3: 16x the map's size within 5 min, knocking it over every trial)
HELD_BABBLE_SHARE = 0.5  # --held_babble: share of babbling set-downs where a parent holds it standing (like holding a
HELD_BABBLE_HELP = 0.5   # baby upright on its feet), carrying this share of its weight; it babbled standing only
                         # 0.6% of the time otherwise (§6.25). Never during tests.
PRACTICE_GAIN0 = 0.25    # reflexes start weak in practice (they knock it over within ~1 s at x0.5 and above) ...
PRACTICE_GAIN_STEP = 1.25  # ... and the trainer strengthens them by this factor after each 2 min with it still up in
                           # at least half the set-downs, up to x1. The brain is never told whether it is up.


def ensure_packages():
    try:
        import mujoco  # noqa: F401
        import scipy  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "mujoco==3.14.0", "scipy"])


def fetch_model(root: Path, folder: str) -> Path:
    d = root / "mujoco_menagerie"
    if not (d / folder / "scene.xml").exists():
        run = lambda *a: subprocess.check_call(list(a))
        if not (d / ".git").exists():
            run("git", "clone", "-q", "--filter=blob:none", "--no-checkout", MENAGERIE_URL, str(d))
            run("git", "-C", str(d), "sparse-checkout", "set", folder)
            run("git", "-C", str(d), "checkout", "-q", MENAGERIE_COMMIT)
        else:
            run("git", "-C", str(d), "sparse-checkout", "add", folder)
    return d / folder


def fetch_go2(root: Path) -> Path:
    return fetch_model(root, "unitree_go2")


BRAIN_KEYS = ("map", "Q", "mean", "var", "seen", "motor_baseline", "rest_baseline", "rest_mean", "rest_sense",
              "reflex_scale", "drives", "push", "fibre_neuron", "fibre_gain", "Qc")


def save_brain(brain, path):
    """Everything it learned and its fixed wiring, so practice can continue on this exact brain. A grown body's parts
    are saved too, under its name (g1_...); the dog's stay under the plain names."""
    active = brain.body_name
    if brain.slots:
        brain.use("go2")
    data = {k: np.asarray(getattr(brain, k)) for k in BRAIN_KEYS if getattr(brain, k, None) is not None}
    for name, slot in (brain.slots or {}).items():
        if name == "go2":
            continue
        data.update({f"{name}_{k}": np.asarray(slot[k]) for k in BRAIN_KEYS if slot.get(k) is not None})
    if brain.slots:
        brain.use(active)
    np.savez_compressed(path, **data)


def load_body_parts(brain, path, name):
    """After growing a body's periphery from its genome: check its wiring equals the saved one, then restore what it
    learned. A brain without saved parts for that body starts them fresh (it has not met that body yet)."""
    with np.load(path) as z:
        keys = [k for k in z.files if k.startswith(name + "_")]
        if not keys:
            return False
        slot = brain.slots[name]
        for k in ("drives", "push", "fibre_neuron", "fibre_gain"):
            if not np.array_equal(z[f"{name}_{k}"], slot[k]):
                raise SystemExit(f"{path}: the {name}'s wiring '{k}' differs from this growth; refusing to continue a different brain")
        for k in keys:
            v = z[k]
            slot[k[len(name) + 1:]] = v.item() if v.shape == () else v.copy()
        return True


def load_brain(brain, path):
    """Continue an existing brain. Its fixed wiring must match exactly; learned parts must all be present."""
    with np.load(path) as z:
        n_old = len(z["fibre_neuron"])
        for k in ("drives", "push", "fibre_neuron", "fibre_gain"):
            cur = getattr(brain, k)
            same = np.array_equal(z[k], cur)
            grown = (GROW_NERVES and k in ("fibre_neuron", "fibre_gain") and len(z[k]) < len(cur)
                     and np.array_equal(z[k], cur[:len(z[k])]))                # new nerves appended after the old ones
            if not (same or grown):
                raise SystemExit(f"{path}: its wiring '{k}' differs from this genome; refusing to continue a different brain")
        for k in ("map", "mean", "var", "seen"):
            if k not in z.files:
                raise SystemExit(f"{path}: missing '{k}'; refusing to continue with part of a brain")
        for k in z.files:
            if k in ("drives", "push", "fibre_neuron", "fibre_gain"):
                continue
            v = z[k]
            setattr(brain, k, v.item() if v.shape == () else v.copy())
        if brain.map.shape != (len(brain.motor), len(brain.sense)):
            raise SystemExit(f"{path}: map shape {brain.map.shape} does not fit this brain")
        n_new = len(brain.fibre_neuron)
        if n_old < n_new:                                                 # appended nerves: at rest they report zero,
            for k, fill in (("mean", 0.0), ("var", NEW_NERVE_VAR), ("rest_mean", 0.0)):   # with a spread of 1 (m/s^2)
                v = getattr(brain, k)
                if v is not None and len(v) == n_old:
                    setattr(brain, k, np.concatenate([v, np.full(n_new - n_old, fill)]))
            print(f"{path}: {n_new - n_old} new nerves appended after its {n_old}; existing wiring unchanged", flush=True)


def load_core():
    found = sorted(Path("/kaggle/input").glob("**/core_v2.npz")) if Path("/kaggle/input").exists() else []
    path = found[0] if found else Path(__file__).resolve().parent.parent / "data" / "malecns" / "core_v2.npz"
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


class Body:
    """The Go2: held in the air by its trunk, or free on the ground. Torque motors; every joint has a hard limit.
    On the ground it also has touch in each foot and an inner ear (which way gravity pulls, and how it is turning)."""

    def __init__(self, go2: Path, held: bool = True, tone: bool = False, leak: bool = False, accel: bool = False):
        import mujoco
        self.accel = accel                         # --accel: the IMU's accelerometer (trunk frame), appended nerves
        self._acc, self._v_prev = np.zeros(3), None
        self.mj = mujoco
        self.kind = "g1" if (go2 / "g1.xml").exists() else "go2"         # --body g1: the humanoid
        self.m = mujoco.MjModel.from_xml_path(str(go2 / "scene.xml"))
        self.d = mujoco.MjData(self.m)
        self.subs = max(1, round(BRAIN_DT / self.m.opt.timestep))
        self.joint = [int(self.m.actuator_trnid[a, 0]) for a in range(self.m.nu)]       # the joint each motor turns
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in self.joint])
        self.vadr = np.array([self.m.jnt_dofadr[j] for j in self.joint])
        self.lo, self.hi = self.m.jnt_range[self.joint, 0].copy(), self.m.jnt_range[self.joint, 1].copy()
        self.tmax = self.m.actuator_ctrlrange[:, 1].copy()
        self.names = [self.m.joint(j).name.replace("_joint", "") for j in self.joint]
        self.held = held
        self.tone = tone
        self.leak = leak
        self.leak_s = TONE_LEAK_S
        self.set_speed = SET_SPEED
        if self.kind == "g1":                       # position servos (kp 500): the servo is the muscle, ctrl its set point
            self.base = self.m.body("pelvis").id
            feet = [self.m.body(f).id for f in ("left_ankle_roll_link", "right_ankle_roll_link")]
            self.feet = [{g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == b and self.m.geom_contype[g]} for b in feet]
            self.key, self.stand_height, self.up_height, self.min_feet = "stand", 0.79, G1_UP_HEIGHT, 1
        else:
            self.base = self.m.body("base").id
            self.feet = [{self.m.geom(f).id} for f in ("FL", "FR", "RL", "RR")]
            self.key, self.stand_height, self.up_height, self.min_feet = "home", 0.27, STANDING_HEIGHT, 3
        self._force = np.zeros(6)
        self.support = 0.0                         # a parent's hands on the trunk: 0 (none) .. 1 (carries its weight)
        self.mass = float(self.m.body_subtreemass[self.base])
        self.reset()

    def reset(self, pose: str = "stand", rng=None):
        """Held: back in the parent's hands. On the ground: set down standing, on its belly, or on one side."""
        self.mj.mj_resetDataKeyframe(self.m, self.d, self.m.keyframe(self.key).id)
        self.d.xfrc_applied[:] = 0.0                                     # no hands left on it from before
        self._acc, self._v_prev = np.zeros(3), None
        if self.held:
            self.hold()
            self.mj.mj_forward(self.m, self.d)
            self.target = self.angles()
            self.rest_q = self.target.copy()
            return
        q = self.d.qpos
        if pose != "stand" and self.kind != "go2":
            raise SystemExit(f"reset('{pose}'): only 'stand' is defined for the {self.kind}")
        if pose != "stand":
            q[self.qadr] = np.tile([0.0, 1.3, -2.6], 4)                 # legs folded
            q[2] = 0.14
            if pose != "belly":
                a = (1.0 if pose == "left_side" else -1.0) * np.pi / 2
                q[2], q[3:7] = 0.2, [np.cos(a / 2), np.sin(a / 2), 0.0, 0.0]
        if rng is not None:
            q[self.qadr] += rng.uniform(-0.2, 0.2, len(self.qadr))
        self.d.qvel[:] = 0.0
        self.d.ctrl[:] = 0.0
        if pose == "stand":                                              # set on its feet; holding itself up is its job
            self.mj.mj_forward(self.m, self.d)
            self.target = self.angles()
            self.rest_q = self.target.copy()
            return
        for _ in range(int(0.5 / self.m.opt.timestep)):                 # let it settle onto the floor, limp
            self.mj.mj_step(self.m, self.d)
        self.target = self.angles()                                      # the muscles take the posture it was set in
        self.rest_q = self.target.copy()

    def hold(self):
        self.d.qpos[0:3] = [0.0, 0.0, HELD_AT]
        self.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.d.qvel[0:6] = 0.0

    def step(self, out):
        """out: each motor pool's output in [-1, 1]. Without tone it is a torque (share of the motor's maximum); with
        tone it moves the joint's set point, and the muscle pulls toward it."""
        if self.accel and self._v_prev is None:
            self._v_prev = self.trunk_velocity()
        if self.tone:
            relax = (self.target - self.rest_q) * BRAIN_DT / self.leak_s if self.leak else 0.0
            self.target = np.clip(self.target + self.set_speed * BRAIN_DT * out - relax, self.lo - 0.1, self.hi + 0.1)
        else:
            self.d.ctrl[:] = np.clip(self.tmax * out, -self.tmax, self.tmax)
        for _ in range(self.subs):
            self.d.xfrc_applied[self.base] = self.hands() if self.support > 0 else 0.0
            if self.tone and self.kind == "g1":
                self.d.ctrl[:] = self.target                             # the position servo pulls toward it
            elif self.tone:
                pull = TONE_KP * (self.target - self.d.qpos[self.qadr]) - TONE_KD * self.d.qvel[self.vadr]
                self.d.ctrl[:] = np.clip(pull, -self.tmax, self.tmax)
            self.mj.mj_step(self.m, self.d)
            if self.held:
                self.hold()
        if self.accel:                                                    # how fast the trunk's own velocity changed
            v = self.trunk_velocity()
            self._acc = (v - self._v_prev) / BRAIN_DT
            self._v_prev = v

    def trunk_velocity(self):
        R = self.d.xmat[self.base].reshape(3, 3)
        return R.T @ self.d.qvel[0:3]                                     # free joint: world-frame linear velocity

    def hands(self):
        """The parent's hands: a spring-damper up toward standing height (0.27 m) and a spring-damper turning the
        trunk back to level (yaw free), scaled by how much help is left."""
        mg = self.mass * 9.81
        kz = mg / SUPPORT_SAG
        cz = np.sqrt(kz * self.mass)
        fz = np.clip(kz * (self.stand_height - self.d.qpos[2]) - cz * self.d.qvel[2], -mg, mg)
        R = self.d.xmat[self.base].reshape(3, 3)
        kr = 2.0 * mg * self.stand_height                      # twice the tipping torque per radian of tilt
        cr = 2.0 * np.sqrt(kr * 0.1)
        w = R @ self.d.qvel[3:6]                               # trunk turning, world frame
        tq = kr * np.cross(R[:, 2], [0.0, 0.0, 1.0]) - cr * np.array([w[0], w[1], 0.0])
        return self.support * np.array([0.0, 0.0, fz, tq[0], tq[1], tq[2]])

    def angles(self):
        return self.d.qpos[self.qadr].copy()

    def feel(self):
        """What the joints' own sense organs report: angle, speed, effort, and the push-back at a limit. On the
        ground also: the pressure under each foot, and the inner ear."""
        joints = [self.d.qpos[self.qadr], self.d.qvel[self.vadr], self.d.actuator_force, self.d.qfrc_constraint[self.vadr]]
        if self.held:
            return np.concatenate(joints)
        lines = [*joints, self.touch(), self.gravity(), self.d.qvel[3:6]]
        if self.accel:                                                    # appended last: earlier nerves keep their wiring
            lines.append(self._acc)
        return np.concatenate(lines)

    def touch(self):
        f = np.zeros(len(self.feet))
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            for k, g in enumerate(self.feet):
                if c.geom1 in g or c.geom2 in g:
                    self.mj.mj_contactForce(self.m, self.d, i, self._force)
                    f[k] += abs(self._force[0])
        return f

    def gravity(self):
        """Which way gravity pulls, in the trunk's own frame: (0, 0, -1) when upright."""
        return self.d.xmat[self.base].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])

    def yaw(self):
        """True heading of the trunk (rad), for the tests only."""
        R = self.d.xmat[self.base].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def tilt(self):
        """True roll and pitch of the trunk (rad), for the tests only."""
        g = self.gravity()
        return float(np.arctan2(-g[1], -g[2])), float(np.arcsin(np.clip(g[0], -1.0, 1.0)))


class Brain:
    """The fly connectome as a rate model, wired to the body by a fixed random genome it is never told about."""

    def __init__(self, core, n_lines: int, n_motors: int, rng):
        from scipy import sparse
        role = core["role"]
        self.N = len(role)
        pre, post = core["pre"].astype(np.int64), core["post"].astype(np.int64)
        syn = core["synapses"].astype(np.float64)
        z = np.bincount(post, weights=syn, minlength=self.N) + 1.0      # each neuron: synapse-weighted mean of its inputs
        w = core["sign"].astype(np.float64)[pre] * syn / z[post]
        self.W = sparse.csr_matrix((w * GAIN, (post, pre)), shape=(self.N, self.N))
        self.sense = np.where(role == 0)[0]
        self.motor = np.where(role == 1)[0]
        # genome, part 1: each motor neuron drives one motor, pushing one way or the other
        self.drives = rng.integers(0, n_motors, len(self.motor))
        self.push = rng.choice([-1.0, 1.0], len(self.motor))
        self.n_motors = n_motors
        # genome, part 2: each nerve reaches NERVE_FIBRES sensory neurons; half fire when it rises, half when it falls
        pick = rng.permutation(len(self.sense))[: n_lines * NERVE_FIBRES].reshape(n_lines, NERVE_FIBRES)
        self.fibre_neuron = self.sense[pick]                                          # [lines, fibres]
        self.fibre_gain = rng.uniform(0.5, 1.5, pick.shape) * np.where(np.arange(NERVE_FIBRES) % 2 == 0, 1.0, -1.0)
        self.v = np.zeros(self.N)
        self.intention = np.zeros(n_lines)      # a shift of each nerve's felt reference (0: feel things as they are)
        # sensory adaptation: each nerve settles to what it usually reports
        self.mean, self.var, self.seen = np.zeros(n_lines), np.ones(n_lines), 0
        # the plastic motor-to-sensory synapses: its body map, all zero at birth
        self.map = np.zeros((len(self.motor), len(self.sense)))
        self.fibre_col = pick                                                         # [lines, fibres] -> column in map

    def rates(self):
        return np.tanh(np.maximum(self.v, 0.0))

    adapting = True
    adapt_var = True
    reflex = 0.0             # strength of its own sensory-to-motor reflexes (0 while babbling)
    reflex_scale = 1.0       # the map's own scale (99th percentile of its links per nerve)
    rest_sense = None        # how its sensory neurons fire when it stands still at rest
    rest_mean = None         # what its senses report at rest, standing
    reflex_cols = None       # diagnosis only: use the reflexes from these sensory neurons alone (None: all)
    R = None                 # its reflex synapses once they learn on their own (standing practice); None: the twitch-tuned ones
    Q = None                 # skill synapses, sensory to motor, learned from reward (--skills)
    use_ctx = False          # posture neurons (--contexts)
    cpg = None               # the grown pattern generator (--cpg)
    walk_line = None         # --walk: the forward-acceleration nerve; its request drives the generator too
    Qc = None
    turn_line = None
    adapt_g = 0.0            # neuron adaptation: a neuron that keeps firing tires (subtracts adapt_g x its recent rate)
    adapt_tau = 0.3          # s: how fast it tires and recovers
    adapt = None

    def cpg_init(self, rng):
        self.cpg = {"u": rng.uniform(0.0, 0.1, CPG_N), "a": np.zeros(CPG_N)}
        cols = 2 * CPG_N * (2 if self.walk_line is not None else 1)
        if getattr(self, "Qc", None) is None:
            self.Qc = np.zeros((len(self.motor), cols))
        elif self.Qc.shape == (len(self.motor), 2 * CPG_N) and cols == 4 * CPG_N:   # --walk: new sets appended after
            self.Qc = np.concatenate([self.Qc, np.zeros((len(self.motor), 2 * CPG_N))], 1)  # the learned turning ones
            print(f"--walk: rhythm synapses for walking forward/back appended after the {2 * CPG_N} turning ones", flush=True)
        elif self.Qc.shape != (len(self.motor), cols):
            raise SystemExit(f"--cpg: the saved rhythm synapses have shape {self.Qc.shape}, expected "
                             f"{(len(self.motor), cols)} (one set per turning/walking direction); not re-initialising them")
        self.cpg_f = np.zeros(cols)

    body_name = "go2"
    slots = None

    def grow_body(self, name, n_lines, n_motors, rng):
        """A new body arrives: grow its own sensory and motor neurons by copying the fly's (Embodiogenesis §2.4),
        appended after every existing neuron. Sensory copies take only their template's outgoing synapses (the new
        nerves are their input); motor copies take only their template's incoming synapses (they drive the new body's
        motors). Without that body plugged in, the sensory copies are silent and the motor copies drive nothing, so
        nothing the brain already does changes. The new body gets its own wiring and its own learned parts."""
        from scipy import sparse
        if self.slots is None:
            self.slots = {}
        n_s, n_m = n_lines * NERVE_FIBRES, len(self.motor)
        t_s = rng.choice(self.sense, n_s)                                  # templates: the fly's sensory neurons
        t_m = self.motor.copy()                                            # and each motor neuron once
        N0, N1 = self.N, self.N + n_s + n_m
        new_s, new_m = np.arange(N0, N0 + n_s), np.arange(N0 + n_s, N1)
        W = self.W.tocsc()
        rows, cols, vals = [], [], []
        for k, t in enumerate(t_s):                                       # outgoing synapses of each sensory template
            a, b = W.indptr[t], W.indptr[t + 1]
            rows.append(W.indices[a:b]); cols.append(np.full(b - a, new_s[k])); vals.append(W.data[a:b])
        Wr = self.W.tocsr()
        for k, t in enumerate(t_m):                                       # incoming synapses of each motor template
            a, b = Wr.indptr[t], Wr.indptr[t + 1]
            rows.append(np.full(b - a, new_m[k])); cols.append(Wr.indices[a:b]); vals.append(Wr.data[a:b])
        rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        vals = vals * np.clip(1.0 + COPY_NOISE * rng.standard_normal(len(vals)), 0.5, 1.5)
        old = self.W.tocoo()
        self.W = sparse.csr_matrix((np.concatenate([old.data, vals]), (np.concatenate([old.row, rows]),
                                    np.concatenate([old.col, cols]))), shape=(N1, N1))
        self.N = N1
        self.v = np.concatenate([self.v, np.zeros(n_s + n_m)])
        if self.adapt is not None:
            self.adapt = np.concatenate([self.adapt, np.zeros(n_s + n_m)])
        pick = rng.permutation(n_s).reshape(n_lines, NERVE_FIBRES)
        slot = {k: None for k in SLOT_KEYS}
        slot.update(sense=new_s, motor=new_m, drives=rng.integers(0, n_motors, n_m), push=rng.choice([-1.0, 1.0], n_m),
                    n_motors=n_motors, fibre_neuron=new_s[pick], fibre_col=pick,
                    fibre_gain=rng.uniform(0.5, 1.5, pick.shape) * np.where(np.arange(NERVE_FIBRES) % 2 == 0, 1.0, -1.0),
                    intention=np.zeros(n_lines), mean=np.zeros(n_lines), var=np.ones(n_lines), seen=0,
                    map=np.zeros((n_m, n_s)), reflex_scale=1.0,
                    motor_baseline=None if self.motor_baseline is None else np.zeros(n_motors))
        self.slots[name] = slot
        self.grown = getattr(self, "grown", []) + [(name, n_s, n_m, int(len(vals)))]
        return n_s, n_m, int(len(vals))

    def use(self, name):
        """Plug the brain into this body: its own neurons, wiring and learned parts become the active ones."""
        if name == self.body_name:
            return
        if self.slots is None or name not in self.slots:
            raise SystemExit(f"use('{name}'): this brain has not grown a periphery for that body")
        self.slots[self.body_name] = {k: getattr(self, k, None) for k in SLOT_KEYS}
        for k, v in self.slots[name].items():
            setattr(self, k, v)
        self.body_name = name
        self.v[:] = 0.0

    def cpg_step(self, drive):
        u, a = self.cpg["u"], self.cpg["a"]
        dt = BRAIN_DT / CPG_SUBSTEPS                                       # its own time scale, not the brain step's
        for _ in range(CPG_SUBSTEPS):
            r = np.maximum(u, 0.0)
            inh = CPG_INH * (r.sum() - r) - CPG_SEQ * np.roll(r, 1)
            u += dt / CPG_TAU * (-u + drive - inh - CPG_ADAPT_G * a)
            r = np.maximum(u, 0.0)
            a += dt / CPG_ADAPT_TAU * (r - a)
        return r

    def ctx_init(self):
        self.use_ctx = True
        n = len(self.mean)
        self.slow = np.zeros(n)
        self.ctx_p, self.ctx_map, self.ctx_scale = [], [], []
        self.slow_m, self.slow_v, self.fast_v = np.zeros(n), np.zeros(n), np.full(n, 1e-3)
        self.prev_z = None
        self.ctx_w = np.ones(n)
        self.ctx_skip = 0                        # steps right after being set down, not counted (being moved is not its own change)

    def ctx_stats(self, z):
        """While babbling: how slowly each sense changes, and the weights posture is read with."""
        if self.ctx_skip > 0:
            self.ctx_skip -= 1
            return
        r = CTX_STATS_RATE
        self.slow_m += r * (self.slow - self.slow_m)
        self.slow_v += r * ((self.slow - self.slow_m) ** 2 - self.slow_v)
        if self.prev_z is not None:
            self.fast_v += r * ((z - self.prev_z) ** 2 - self.fast_v)
        if CTX_SLOW:
            ratio = self.slow_v / np.maximum(self.fast_v, 1e-6)
            ratio = np.minimum(ratio, 10.0 * max(float(np.median(ratio)), 1e-12))
            self.ctx_w = ratio / max(float(ratio.mean()), 1e-12)

    def ctx_activity(self):
        """Each posture neuron's share of the activity, and how far the current posture is from each."""
        if not self.ctx_p:
            return np.zeros(0), np.zeros(0)
        d = np.sqrt((self.ctx_w * (np.array(self.ctx_p) - self.slow) ** 2).mean(1))
        a = np.exp(-d ** 2 / (2 * CTX_SIGMA ** 2)) + 1e-12
        return a / a.sum(), d

    def ctx_learn(self, update):
        """Grow a posture neuron if none fits; the active ones learn the twitch-tuned update and drift to this posture."""
        if self.seen < CTX_START:
            return
        a, d = self.ctx_activity()
        if len(a) == 0 or (d.min() > CTX_GROW and len(self.ctx_p) < CTX_MAX):
            near = int(np.argmin(d)) if len(d) else None
            self.ctx_p.append(self.slow.copy())
            self.ctx_map.append(self.ctx_map[near].copy() if near is not None else np.zeros(self.map.shape, np.float32))
            a, d = self.ctx_activity()
        for k in np.where(a > CTX_ACTIVE)[0]:
            self.ctx_map[k] += (a[k] * update).astype(np.float32)
            if not CTX_WINNER:
                self.ctx_p[k] += CTX_LEARN * a[k] * (self.slow - self.ctx_p[k])
        if CTX_WINNER:
            w = int(np.argmax(a))
            self.ctx_p[w] += CTX_LEARN * (self.slow - self.ctx_p[w])

    def zscore(self, felt):
        return np.clip((felt - self.mean) / np.sqrt(self.var + 1e-6), -4.0, 4.0)

    def step(self, felt, motor_drive):
        if self.adapting:
            self.seen += 1
            a = max(1.0 / self.seen, 0.002)                              # adapts within ~10 s
            self.mean += a * (felt - self.mean)
            if self.adapt_var:
                self.var += a * ((felt - self.mean) ** 2 - self.var)
        if self.use_ctx:
            z = self.zscore(felt)
            self.slow += BRAIN_DT / CTX_TAU * (z - self.slow)
            if self.adapting:
                self.ctx_stats(z)
            self.prev_z = z
        zf = self.zscore(felt) + self.intention
        I = np.zeros(self.N)
        np.add.at(I, self.fibre_neuron.ravel(), (self.fibre_gain * zf[:, None]).ravel())
        I[self.motor] += motor_drive
        if self.cpg is not None and self.turn_line is not None:          # driven by its own request to turn
            want = -self.intention[self.turn_line]
            wantw = -self.intention[self.walk_line] if self.walk_line is not None else 0.0
            rc = self.cpg_step(min(1.0, max(abs(want), abs(wantw)) / 0.25))
            # each turning direction has its own synapses from the rhythm (a sign flip of one set is the same gait
            # half a cycle later, so one set cannot turn both ways); so does each walking direction (--walk)
            parts = [rc * (want > 0), rc * (want < 0)]
            if self.walk_line is not None:
                parts += [rc * (wantw > 0), rc * (wantw < 0)]
            self.cpg_f = np.concatenate(parts)
            I[self.motor] += self.Qc @ self.cpg_f
        if self.Q is not None and self.rest_sense is not None:  # learned skills: respond to what it feels vs rest
            I[self.motor] += self.Q @ (self.rates()[self.sense] - self.rest_sense)
        if self.reflex and self.rest_sense is not None:      # its own reflexes: oppose what changed since rest
            change = self.rates()[self.sense] - self.rest_sense
            Wr = self.map if self.R is None else self.R
            if self.use_ctx and self.ctx_map and self.ctx_scale:           # the active posture neurons' own reflexes
                a, _ = self.ctx_activity()
                for k in np.where(a > CTX_ACTIVE)[0]:
                    I[self.motor] -= self.reflex * a[k] / self.ctx_scale[k] * (self.ctx_map[k] @ change)
            elif self.reflex_cols is None:
                I[self.motor] -= self.reflex / self.reflex_scale * (Wr @ change)
            else:
                c = self.reflex_cols
                I[self.motor] -= self.reflex / self.reflex_scale * (Wr[:, c] @ change[c])
        if self.adapt_g:
            if self.adapt is None:
                self.adapt = np.zeros(self.N)
            I -= self.adapt_g * self.adapt
        self.v += ALPHA * (-self.v + self.W @ self.rates() + I)
        r = self.rates()
        if self.adapt_g:
            self.adapt += BRAIN_DT / self.adapt_tau * (r - self.adapt)
        return r

    motor_baseline = None    # with tone: each motor pool's steady resting drive, adapted away
    rest_baseline = None     # that baseline after settling at rest; every test trial starts from it

    def fresh(self):
        """A new trial from rest: neurons quiet, and the motor pools' adaptation as it is after resting."""
        self.v[:] = 0.0
        self.intention[:] = 0.0
        if self.rest_baseline is not None:
            self.motor_baseline[:] = self.rest_baseline
        if self.rest_mean is not None:
            self.mean[:] = self.rest_mean
        if self.cpg is not None:
            self.cpg["u"][:] = 0.0
            self.cpg["u"][0] = 0.1
            self.cpg["a"][:] = 0.0

    def command(self, r):
        net = np.bincount(self.drives, weights=self.push * r[self.motor], minlength=self.n_motors)
        x = 1.5 * net / np.sqrt(np.bincount(self.drives, minlength=self.n_motors) + 1.0)
        self.last_raw = x
        if self.motor_baseline is not None:
            self.motor_baseline += BRAIN_DT / MOTOR_TAU * (x - self.motor_baseline)
            x = x - self.motor_baseline
        return np.tanh(x)


def babble(body, brain, minutes, rng, log, label, diag=None, held=False):
    """Spontaneous movement; the body map grows by Hebbian learning. Returns what was explored and felt."""
    steps = int(minutes * 60 / BRAIN_DT)
    lines = len(body.feel())
    J = len(body.joint)
    burst = np.zeros(len(brain.motor))
    r = brain.rates()
    s_prev = r[brain.sense].copy()
    trace = np.zeros(len(brain.motor))
    m_mean = np.zeros(len(brain.motor))
    seen_lo, seen_hi = body.angles(), body.angles()
    limit_felt = np.zeros((J, 2), int)                                    # steps pressed against each end
    t0 = time.time()
    every = int(REPOSITION_S / BRAIN_DT)
    for t in range(steps):
        if not body.held and t % every == 0:
            pose = POSES[rng.integers(len(POSES))] if body.kind == "go2" else "stand"
            body.support = 0.0 if body.kind == "go2" else G1_BABBLE_HELP   # the G1 cannot stand yet: always held up
            if held and rng.random() < HELD_BABBLE_SHARE:
                pose = "stand"
                body.support = HELD_BABBLE_HELP
            body.reset(pose, rng)
            trace[:] = 0.0                                                # being moved is not something it did
            if brain.use_ctx:
                brain.ctx_skip = int(0.5 / BRAIN_DT)
            if diag is not None:
                if diag["up_since"] is not None:
                    diag["stand_lasted_s"].append((t - diag["up_since"]) * BRAIN_DT)
                diag["up_since"] = t if pose == "stand" else None
        burst += BRAIN_DT / BABBLE_TAU * -burst + np.sqrt(2 * BRAIN_DT / BABBLE_TAU) * rng.standard_normal(len(burst))
        felt = body.feel()
        r = brain.step(felt, BABBLE_SIZE * burst)
        out_cmd = brain.command(r)
        body.step(out_cmd)
        m = r[brain.motor]
        m_mean += 0.01 * (m - m_mean)
        s = r[brain.sense]
        update = HEBB_RATE * np.outer(trace, s - s_prev)                # activity, then what was felt shortly after
        brain.map += update
        if brain.use_ctx:
            brain.ctx_learn(update)
        if diag is not None:                                              # measurement only; changes nothing
            loaded = int((felt[4 * J:4 * J + 4] > FOOT_LOADED_N).sum())
            standing = int(loaded >= 3 and -felt[4 * J + 6] > 0.8 and body.d.qpos[2] > STANDING_HEIGHT)
            diag["feet_loaded"][loaded] += 1
            diag["standing"] += standing
            diag["steps"] += 1
            diag["part"][standing] += HEBB_RATE * np.outer(trace, (s - s_prev)[diag["cols"]])
            diag["abs_command"] += np.abs(out_cmd)
            if diag["up_since"] is not None and not standing and t > diag["up_since"] + 1:
                diag["stand_lasted_s"].append((t - diag["up_since"]) * BRAIN_DT)
                diag["up_since"] = None
        trace = TRACE * trace + (1 - TRACE) * (m - m_mean)
        s_prev = s
        q = body.angles()
        seen_lo, seen_hi = np.minimum(seen_lo, q), np.maximum(seen_hi, q)
        push = body.d.qfrc_constraint[body.vadr]
        limit_felt[:, 0] += (q < body.lo + 0.02) & (np.abs(push) > 1.0)
        limit_felt[:, 1] += (q > body.hi - 0.02) & (np.abs(push) > 1.0)
        if (t + 1) % int(5 * 60 / BRAIN_DT) == 0 or t == steps - 1:
            acc = map_accuracy(brain, body)
            cov = (seen_hi - seen_lo) / (body.hi - body.lo)
            log(f"[{label}] babbled {(t + 1) * BRAIN_DT / 60:4.1f} min ({time.time() - t0:5.0f} s): range explored "
                f"{cov.mean():.0%} (least {cov.min():.0%}); limits felt {int((limit_felt > 0).sum())}/{2 * J}; "
                f"body map: right joint {acc['joint']:.0%}, right joint and direction {acc['both']:.0%} (chance {1 / J:.0%}, "
                f"{1 / (2 * J):.0%})")
    body.support = 0.0
    return {"coverage": ((seen_hi - seen_lo) / (body.hi - body.lo)).tolist(), "seen_lo": seen_lo.tolist(),
            "seen_hi": seen_hi.tolist(), "limit_felt_steps": limit_felt.tolist()}


def joint_scores(brain, body):
    """[motor neurons, joints]: each motor neuron's learned link to each joint's angle nerve (sign: which way)."""
    J = len(body.joint)
    cols, gains = brain.fibre_col[:J], brain.fibre_gain[:J]                # lines 0..J-1 are the angle nerves
    return np.stack([(brain.map[:, cols[j]] * gains[j]).sum(1) for j in range(J)], 1)


def map_accuracy(brain, body):
    sc = joint_scores(brain, body)
    best = np.abs(sc).argmax(1)
    right = best == brain.drives
    direction = np.sign(sc[np.arange(len(best)), brain.drives]) == brain.push
    return {"joint": float(right.mean()), "both": float((right & direction).mean())}


def settle_at_rest(body, brain, seconds=15.0):
    """Before a battery of tests: set down standing, the brain's output cut, until its motor pools' adaptation matches
    their resting drive (it otherwise remembers the last few seconds of babbling or of the previous trial)."""
    if brain.motor_baseline is None:
        return
    body.reset("stand")
    brain.v[:] = 0.0
    for _ in range(int(seconds / BRAIN_DT)):
        brain.step(body.feel(), np.zeros(len(brain.motor)))
        brain.command(brain.rates())
        body.step(np.zeros(len(body.joint)))
    brain.rest_baseline = brain.motor_baseline.copy()


def control_test(body, brain, scores, log, label, seconds=1.5):
    """Ask the brain to drive each joint to each end, using only its own body map."""
    settle_at_rest(body, brain)
    J = len(body.joint)
    best = np.abs(scores).argmax(1)
    rows = []
    for j in range(J):
        for d in (1.0, -1.0):
            body.reset()
            brain.fresh()
            mine = best == j
            drive = np.where(mine, np.maximum(d * scores[:, j], 0.0), 0.0)
            drive = 3.0 * drive / (drive.max() + 1e-12)
            q0 = body.angles()
            for _ in range(int(seconds / BRAIN_DT)):
                r = brain.step(body.feel(), drive)
                body.step(brain.command(r))
            q1 = body.angles()
            end = body.hi[j] if d > 0 else body.lo[j]
            reach = (q1[j] - q0[j]) / (end - q0[j]) if abs(end - q0[j]) > 1e-6 else 0.0
            others = float(np.delete(np.abs(q1 - q0), j).mean())
            rows.append({"joint": body.names[j], "to": "high" if d > 0 else "low", "reach": float(reach),
                         "others_moved_rad": others, "neurons_used": int(mine.sum())})
    reach = np.array([r_["reach"] for r_ in rows])
    log(f"[{label}] control: asked to move one joint to one end, it gets {np.mean(np.clip(reach, -1, 1)):.0%} of the way "
        f"on average (median {np.median(reach):.0%}; {np.mean(reach > 0.8):.0%} of 24 moves reach 80% of the way); "
        f"other joints move {np.mean([r_['others_moved_rad'] for r_ in rows]):.3f} rad on average")
    return rows


def line_scores(brain, M=None):
    """[motor neurons, nerves]: each motor neuron's learned link to each nerve (sign: which way it moves it)."""
    M = brain.map if M is None else M
    return np.einsum("mlf,lf->ml", M[:, brain.fibre_col], brain.fibre_gain)


def ctx_scales(brain):
    if brain.use_ctx:
        brain.ctx_scale = [max(float(np.percentile(np.abs(line_scores(brain, M)), 99)), 1e-12) for M in brain.ctx_map]


def steer(body, brain, S, targets, seconds, flail=None):
    """Use only its own body map to make chosen senses go where asked. targets: {nerve: value in its own units}.
    Each step: how far each targeted sense is from where it should be (in its adapted units), times each motor
    neuron's learned link to that sense, drives that neuron. flail: babbling instead (a baseline), no map."""
    lines = np.array(list(targets))
    goal = np.array([targets[l] for l in lines])
    scale = max(float(np.percentile(np.abs(S[:, lines]), 99)), 1e-12)
    burst = np.zeros(len(brain.motor))
    traj = []
    for _ in range(int(seconds / BRAIN_DT)):
        felt = body.feel()
        err = np.zeros(len(lines))
        if flail is not None:
            burst += BRAIN_DT / BABBLE_TAU * -burst + np.sqrt(2 * BRAIN_DT / BABBLE_TAU) * flail.standard_normal(len(burst))
            drive = BABBLE_SIZE * burst
        else:
            z_now, z_goal = brain.zscore(felt)[lines], (goal - brain.mean[lines]) / np.sqrt(brain.var[lines] + 1e-6)
            err = np.clip(z_goal - z_now, -3.0, 3.0)
            drive = np.clip(np.maximum(S[:, lines] @ err, 0.0) / scale, 0.0, 3.0)
        r = brain.step(felt, drive)
        out_cmd = brain.command(r)
        body.step(out_cmd)
        traj.append((float(body.d.qpos[2]), float(-body.gravity()[2]), body.touch(), out_cmd, brain.last_raw.copy(),
                     err.copy(), body.tilt(), float(np.abs(drive).sum())))
    return traj


def story(traj, progress):
    """Step by step: progress toward the asked change (1 = reached), pools' mean |output|, height; when it first
    reached the target, and when it fell (trunk below standing height)."""
    h = np.array([t[0] for t in traj])
    cmd = np.array([np.abs(t[3]).mean() for t in traj])
    fell = int(np.argmax(h < STANDING_HEIGHT)) if (h < STANDING_HEIGHT).any() else None
    reached = int(np.argmax(progress >= 1.0)) if (progress >= 1.0).any() else None
    return {"progress": progress.round(3).tolist(), "mean_abs_output": cmd.round(3).tolist(), "height": h.round(3).tolist(),
            "drive_sum": [round(t[7], 2) for t in traj], "reached_s": None if reached is None else (reached + 1) * BRAIN_DT,
            "fell_s": None if fell is None else (fell + 1) * BRAIN_DT,
            "output_when_reached": None if reached is None else float(cmd[reached]),
            "output_when_fell": None if fell is None else float(cmd[fell]),
            "progress_when_fell": None if fell is None else float(progress[fell]),
            "peak_output": float(cmd.max()), "peak_output_s": (int(cmd.argmax()) + 1) * BRAIN_DT}


def tell(log, label, name, st):
    log(f"[{label}]   DIAG {name}: reached the target at {st['reached_s']} s (output then {st['output_when_reached']}), "
        f"fell at {st['fell_s']} s (progress then {st['progress_when_fell']}, output then {st['output_when_fell']}); "
        f"pools' mean |output| peaked at {st['peak_output']:.2f} at {st['peak_output_s']:.2f} s; progress at 0.2/0.5/1/2 s: "
        + "/".join(f"{st['progress'][min(int(t / BRAIN_DT) - 1, len(st['progress']) - 1)]:+.2f}" for t in (0.2, 0.5, 1.0, 2.0)))


def end_state(traj):
    h = np.array([t[0] for t in traj])
    f = traj[-1][2]
    return {"min_height_m": float(h.min()), "end_height_m": float(h[-1]), "end_upright": float(traj[-1][1]),
            "end_touch_N": f.tolist(), "end_feet_loaded": int((f > FOOT_LOADED_N).sum())}


def ground_tests(body, brain, log, label, rng):
    """On the ground, using only its own map: tilt its trunk, put weight on one foot, get up from its side."""
    J = len(body.joint)
    S = line_scores(brain)
    nerve = {"touch": 4 * J + np.arange(4), "gravity": 4 * J + 4 + np.arange(3)}
    brain.adapting = False
    out = {}
    # tilt: ask its inner ear to feel the trunk rolled or pitched by 0.25 rad
    tilts = {"roll left": (0.25, 0.0), "roll right": (-0.25, 0.0), "nose down": (0.0, 0.25), "nose up": (0.0, -0.25)}
    got = []
    for name, (roll, pitch) in tilts.items():
        body.reset("stand")
        brain.fresh()
        g = np.array([np.sin(pitch), -np.sin(roll), -np.cos(roll) * np.cos(pitch)])
        r0, p0 = body.tilt()
        traj = steer(body, brain, S, dict(zip(nerve["gravity"].tolist(), g.tolist())), 2.0)
        r1, p1 = body.tilt()
        want = np.array([roll, pitch])
        moved = np.array([r1 - r0, p1 - p0])
        got.append(float(moved @ want / (want @ want)))                  # share of the asked tilt achieved
        prog = np.array([(np.array(t[6]) - np.array([r0, p0])) @ want / (want @ want) for t in traj])
        out[f"tilt {name}"] = {"asked_rad": want.tolist(), "moved_rad": moved.tolist(), "share": got[-1],
                               "end": end_state(traj), "story": story(traj, prog)}
        tell(log, label, f"tilt {name}", out[f"tilt {name}"]["story"])
    log(f"[{label}] tilt its trunk 0.25 rad as asked (roll left/right, nose down/up): achieved "
        + ", ".join(f"{v:.0%}" for v in got) + " of the asked tilt")
    # weight: ask for more pressure under one foot (one and a half times what it feels standing)
    shares = []
    for k, foot in enumerate(("FL", "FR", "RL", "RR")):
        body.reset("stand")
        brain.fresh()
        base = body.touch()
        traj = steer(body, brain, S, {int(nerve["touch"][k]): float(1.5 * base[k] + 1.0)}, 2.0)
        now = body.touch()
        s0, s1 = base[k] / max(base.sum(), 1e-6), now[k] / max(now.sum(), 1e-6)
        shares.append((s0, s1))
        want_share = 1.5 * base[k] / max(base.sum(), 1e-6)
        prog = np.array([(t[2][k] / max(t[2].sum(), 1e-6) - s0) / max(want_share - s0, 1e-6) for t in traj])
        out[f"weight on {foot}"] = {"share_before": float(s0), "share_after": float(s1), "end": end_state(traj),
                                    "story": story(traj, prog)}
        tell(log, label, f"weight on {foot}", out[f"weight on {foot}"]["story"])
    log(f"[{label}] shift its weight onto one foot as asked: that foot's share of the weight "
        + ", ".join(f"{f} {a:.0%}->{b:.0%}" for f, (a, b) in zip(("FL", "FR", "RL", "RR"), shares)))
    for key in ("weight, map switched off", "map off, brain's output cut", "map off, babbling on"):
        body.reset("stand")
        brain.fresh()
        start = body.target.copy() if body.tone else body.angles()
        baseline = None if brain.motor_baseline is None else brain.motor_baseline.copy()
        if key == "map off, brain's output cut":
            traj = []
            for _ in range(int(2.0 / BRAIN_DT)):
                brain.step(body.feel(), np.zeros(len(brain.motor)))
                brain.command(brain.rates())
                body.step(np.zeros(len(body.joint)))
                traj.append((float(body.d.qpos[2]), float(-body.gravity()[2]), body.touch(), np.zeros(len(body.joint)),
                             brain.last_raw.copy(), np.zeros(1), body.tilt(), 0.0))
        else:
            traj = steer(body, brain, np.zeros_like(S), {int(nerve["touch"][0]): 0.0}, 2.0,
                         flail=np.random.default_rng(77) if key == "map off, babbling on" else None)
        cmd = np.array([t_[3] for t_ in traj])
        raw = np.array([t_[4] for t_ in traj])
        moved = (body.target if body.tone else body.angles()) - start
        if baseline is not None:
            brain.motor_baseline[:] = baseline                            # each trial starts from the same brain
        out[key] = {"end": end_state(traj), "mean_output": cmd.mean(0).tolist(), "mean_raw_drive": raw.mean(0).tolist(),
                    "baseline_at_start": None if baseline is None else baseline.tolist(), "set_point_moved_rad": moved.tolist()}
        e = out[key]["end"]
        log(f"[{label}] DIAG set down standing, {key}: after 2 s trunk {e['end_height_m']:.2f} m, feet bearing weight "
            f"{e['end_feet_loaded']}/4; each pool's mean output " + " ".join(f"{x:+.2f}" for x in cmd.mean(0))
            + "; raw resting drive " + " ".join(f"{x:+.2f}" for x in raw.mean(0))
            + ("" if baseline is None else "; what it learned to cancel " + " ".join(f"{x:+.2f}" for x in baseline))
            + f"; {'set points' if body.tone else 'joints'} moved " + " ".join(f"{x:+.2f}" for x in moved) + " rad")
    for key in [f"weight on {f}" for f in ("FL", "FR", "RL", "RR")] + ["weight, map switched off"]:
        e = out[key]["end"]
        log(f"[{label}]   {key}: after 2 s, feet bearing weight {e['end_feet_loaded']}/4 (forces "
            + "/".join(f"{x:.0f}" for x in e["end_touch_N"]) + f" N), trunk height {e['end_height_m']:.2f} m "
            f"(lowest {e['min_height_m']:.2f}; set down standing at 0.27), upright {e['end_upright']:.2f}")
    # getting up: lying on its side, ask its inner ear to feel upright; against flailing and doing nothing
    trials = 8
    up = {"its map": 0, "flailing": 0, "nothing": 0}
    for t in range(trials):
        side = "left_side" if t % 2 == 0 else "right_side"
        for how in up:
            body.reset(side, np.random.default_rng(500 + t))
            brain.fresh()
            target = dict(zip(nerve["gravity"].tolist(), [0.0, 0.0, -1.0]))
            if how == "its map":
                steer(body, brain, S, target, 6.0)
            elif how == "flailing":
                steer(body, brain, S, target, 6.0, flail=np.random.default_rng(900 + t))
            else:
                e = end_state(steer(body, brain, np.zeros_like(S), target, 6.0))
                out.setdefault("get_up_nothing_end", []).append(e)
            up[how] += int(-body.gravity()[2] > 0.8 and body.d.qpos[2] > STANDING_HEIGHT
                           and (body.touch() > FOOT_LOADED_N).sum() >= 3)
    out["get_up_from_side"] = {k: v / trials for k, v in up.items()}
    log(f"[{label}] lying on its side, get up (after 6 s: trunk upright, above {STANDING_HEIGHT:g} m, 3-4 feet bearing weight), {trials} tries: "
        + ", ".join(f"{k} {v}/{trials}" for k, v in up.items()))
    log(f"[{label}]   doing nothing from its side, where it ends: trunk height "
        + ", ".join(f"{e['end_height_m']:.2f}" for e in out["get_up_nothing_end"]) + " m; feet bearing weight "
        + ", ".join(str(e["end_feet_loaded"]) for e in out["get_up_nothing_end"]))
    brain.adapting = True
    return out


GROUND_LINES = ("FL pressure", "FR pressure", "RL pressure", "RR pressure", "gravity x (pitch)", "gravity y (roll)",
                "gravity z (upright)")


def true_effects(body, pose, steps=3):
    """[motors, ground lines]: what each motor really does, in this pose, to foot pressures and the inner ear's gravity
    within 60 ms (3 brain steps, the span its Hebbian trace covers). Half the difference between pushing each way."""
    J = len(body.joint)
    lines = 4 * J + np.arange(7)
    eff = np.zeros((J, 7))
    for a in range(J):
        d = []
        for sign in (1.0, -1.0):
            body.reset(pose)
            f0 = body.feel()[lines]
            out = np.zeros(J)
            out[a] = sign * 0.5                                          # half output (half torque, or half set-point speed)
            for _ in range(steps):
                body.step(out)
            d.append(body.feel()[lines] - f0)
        eff[a] = (d[0] - d[1]) / 2
    return eff


def per_motor(brain, S):
    """[motors, lines]: what the brain's map says pushing each motor does (its neurons' links, signed by their push)."""
    return np.stack([(brain.push[brain.drives == a, None] * S[brain.drives == a]).mean(0) for a in range(brain.n_motors)])


def agreement(learned, true):
    """Per line: share of the true effect (weighted by its size) whose sign the map has right; and the correlation."""
    out = []
    for l in range(true.shape[1]):
        w = np.abs(true[:, l])
        agree = float((w * (np.sign(learned[:, l]) == np.sign(true[:, l]))).sum() / max(w.sum(), 1e-12))
        ok = learned[:, l].std() > 0 and true[:, l].std() > 0
        out.append({"sign_right": agree, "corr": float(np.corrcoef(learned[:, l], true[:, l])[0, 1]) if ok else None})
    return out


def diagnose(body, brain, diag, log, label, out_dir, dog):
    J = len(body.joint)
    n = max(diag["steps"], 1)
    log(f"[{label}] DIAG babbling time with 0/1/2/3/4 feet bearing weight (> {FOOT_LOADED_N:g} N): "
        + "/".join(f"{c / n:.0%}" for c in diag["feet_loaded"]) + f"; standing (upright, trunk above {STANDING_HEIGHT:g} m, on 3-4 feet): {diag['standing'] / n:.1%}")
    lasted = np.array(diag["stand_lasted_s"]) if diag["stand_lasted_s"] else np.zeros(1)
    log(f"[{label}] DIAG set down standing {len(diag['stand_lasted_s'])} times while babbling; it stayed up for "
        f"median {np.median(lasted):.2f} s (10th-90th percentile {np.percentile(lasted, 10):.2f}-{np.percentile(lasted, 90):.2f} s, "
        f"set down for 15 s each time); motor pools' mean |output| while babbling "
        + " ".join(f"{a / n:.2f}" for a in diag["abs_command"])
        + (f" (with tone: set points move {body.set_speed * diag['abs_command'].mean() / n:.2f} rad/s on average)" if body.tone else ""))
    gains = brain.fibre_gain[4 * J:4 * J + 7]
    part = [np.einsum("mlf,lf->ml", p.reshape(len(brain.motor), 7, NERVE_FIBRES), gains) for p in diag["part"]]
    S = line_scores(brain)[:, 4 * J:4 * J + 7]
    share_standing = float(np.abs(part[1]).sum() / max(np.abs(part[0]).sum() + np.abs(part[1]).sum(), 1e-12))
    log(f"[{label}] DIAG share of the learned foot/inner-ear links that came from standing: {share_standing:.1%}")
    c = np.corrcoef(S[:, :4].T)
    log(f"[{label}] DIAG how alike the map's links to the four feet are (correlation across its 370 motor neurons): "
        f"mean {c[np.triu_indices(4, 1)].mean():+.2f} (FL-FR {c[0, 1]:+.2f}, FL-RL {c[0, 2]:+.2f}, FL-RR {c[0, 3]:+.2f}, "
        f"FR-RL {c[1, 2]:+.2f}, FR-RR {c[1, 3]:+.2f}, RL-RR {c[2, 3]:+.2f})")
    learned = {"whole map": per_motor(brain, S), "learned standing": per_motor(brain, part[1]),
               "learned lying/other": per_motor(brain, part[0])}
    truth = {pose: true_effects(body, pose) for pose in POSES}
    report = {"feet_loaded_share": [c_ / n for c_ in diag["feet_loaded"]], "standing_share": diag["standing"] / n,
              "share_of_links_from_standing": share_standing, "foot_link_corr": c.tolist(), "compare": {}}
    for pose, tr in truth.items():
        for what, le in learned.items():
            ag = agreement(le, tr)
            report["compare"][f"{what} vs truth {pose}"] = ag
            log(f"[{label}] DIAG {what:20s} vs what the motors really do {pose:10s}: sign right "
                + ", ".join(f"{nm.split()[0] if 'pressure' in nm else nm.split()[2]} {a['sign_right']:.0%}"
                            for nm, a in zip(GROUND_LINES, ag)))
    if body.tone:                                                     # the muscle needs time to move the joint
        for steps in (10, 25):
            for pose in POSES:
                tr = true_effects(body, pose, steps)
                truth[f"{pose}_{steps * BRAIN_DT:g}s"] = tr
                ag = agreement(learned["whole map"], tr)
                report["compare"][f"whole map vs truth {pose} over {steps * BRAIN_DT:g} s"] = ag
                log(f"[{label}] DIAG whole map vs what the motors really do {pose:10s} over {steps * BRAIN_DT:g} s "
                    f"(size of effect {np.abs(tr).mean():.1e}): sign right "
                    + ", ".join(f"{nm.split()[0] if 'pressure' in nm else nm.split()[2]} {a['sign_right']:.0%}"
                                for nm, a in zip(GROUND_LINES, ag)))
    np.savez_compressed(out_dir / f"babble_diag_dog{dog}.npz", map=brain.map, line_scores=line_scores(brain),
                        drives=brain.drives, push=brain.push, part_other=part[0], part_standing=part[1],
                        **{f"truth_{p}": t for p, t in truth.items()})
    return report


def settle_reflex(body, brain, seconds=SETTLE_S):
    """Set down standing with the brain's output cut, until its senses' reference and its motor pools' resting drive
    match standing still (the senses keep the gains babbling gave them); note how its sensory neurons fire at rest.
    Tests then start from this state, with adaptation paused for the trial."""
    body.reset("stand")
    brain.v[:] = 0.0
    brain.intention[:] = 0.0
    reflex, brain.reflex = brain.reflex, 0.0
    brain.adapting, brain.adapt_var = True, False
    n = int(seconds / BRAIN_DT)
    rest = []
    for i in range(n):
        r = brain.step(body.feel(), np.zeros(len(brain.motor)))
        brain.command(r)
        body.step(np.zeros(len(body.joint)))
        if i >= n - int(2.0 / BRAIN_DT):
            rest.append(r[brain.sense].copy())
    brain.rest_sense = np.mean(rest, 0)
    brain.rest_mean = brain.mean.copy()
    brain.rest_baseline = None if brain.motor_baseline is None else brain.motor_baseline.copy()
    brain.adapting, brain.adapt_var = False, True
    brain.reflex = reflex
    if ACTIVE_REST and reflex > 0:                                        # --active_rest: rest is how it feels standing
        for _ in range(2):                                                # by itself (brain on), not with its output cut
            rest = []
            for i in range(int(3.0 / BRAIN_DT)):
                r = brain.step(body.feel(), np.zeros(len(brain.motor)))
                body.step(brain.command(r))
                if i >= int(1.0 / BRAIN_DT):
                    rest.append(r[brain.sense].copy())
            brain.rest_sense = np.mean(rest, 0)


def intend(body, brain, targets, seconds):
    """The test only names what the dog should come to feel: {nerve: value, in that nerve's own units}. It enters once,
    at the start, as a shift of that nerve's felt reference (it feels as if it were on the other side of the target).
    The brain's own reflexes move the body until the shift is cancelled. The test drives no motor neuron."""
    for l, g in targets.items():
        brain.intention[l] = -(g - brain.mean[l]) / np.sqrt(brain.var[l] + 1e-6)
    if brain.use_ctx:                                                     # a fresh trial: posture as it is now
        brain.slow = brain.zscore(body.feel())
    traj = []
    for _ in range(int(seconds / BRAIN_DT)):
        r = brain.step(body.feel(), np.zeros(len(brain.motor)))
        out_cmd = brain.command(r)
        body.step(out_cmd)
        traj.append((float(body.d.qpos[2]), float(-body.gravity()[2]), body.touch(), out_cmd, brain.last_raw.copy(),
                     brain.intention.copy(), body.tilt(), float(np.abs(out_cmd).sum())))
    brain.intention[:] = 0.0
    return traj


def is_up(body):
    return bool(-body.gravity()[2] > 0.8 and body.d.qpos[2] > body.up_height and (body.touch() > FOOT_LOADED_N).sum() >= body.min_feet)


def reflex_tests(body, brain, log, label):
    """Everything by its own reflexes, at each strength (x0 = reflexes switched off, the baseline)."""
    J = len(body.joint)
    brain.reflex_scale = max(float(np.percentile(np.abs(line_scores(brain)), 99)), 1e-12)
    ctx_scales(brain)
    touch, grav = 4 * J + np.arange(4), 4 * J + 4 + np.arange(3)
    results = {}
    for gain in REFLEX_GAINS:
        brain.reflex = gain
        settle_reflex(body, brain)
        tag = f"{label}, own reflexes x{gain:g}"
        res = {}
        # stand still for 5 s, nothing asked
        body.reset("stand")
        brain.fresh()
        res["stand_still_5s"] = end_state(intend(body, brain, {}, 5.0))
        e = res["stand_still_5s"]
        log(f"[{tag}] set down standing, nothing asked, after 5 s: trunk {e['end_height_m']:.2f} m, feet bearing weight "
            f"{e['end_feet_loaded']}/4, upright {e['end_upright']:.2f}")
        # one joint to one end
        rows = []
        for j in range(J):
            for d in (1.0, -1.0):
                body.reset("stand")
                brain.fresh()
                q0 = body.angles()
                end = float(body.hi[j] if d > 0 else body.lo[j])
                intend(body, brain, {j: end}, 1.5)
                q1 = body.angles()
                rows.append({"joint": body.names[j], "to": "high" if d > 0 else "low",
                             "reach": float((q1[j] - q0[j]) / (end - q0[j])),
                             "others_moved_rad": float(np.delete(np.abs(q1 - q0), j).mean()), "up_after": is_up(body)})
        res["joints"] = rows
        reach = np.array([r_["reach"] for r_ in rows])
        log(f"[{tag}] move one joint to one end: {np.mean(np.clip(reach, -1, 1)):.0%} of the way on average "
            f"({np.mean(reach > 0.8):.0%} of 24 reach 80%); other joints move "
            f"{np.mean([r_['others_moved_rad'] for r_ in rows]):.3f} rad; still standing after {sum(r_['up_after'] for r_ in rows)}/24")
        # tilt
        got = []
        for name, (roll, pitch) in {"roll left": (0.25, 0.0), "roll right": (-0.25, 0.0), "nose down": (0.0, 0.25),
                                    "nose up": (0.0, -0.25)}.items():
            body.reset("stand")
            brain.fresh()
            g = np.array([np.sin(pitch), -np.sin(roll), -np.cos(roll) * np.cos(pitch)])
            r0, p0 = body.tilt()
            want = np.array([roll, pitch])
            traj = intend(body, brain, dict(zip(grav.tolist(), g.tolist())), 2.0)
            prog = np.array([(np.array(t_[6]) - np.array([r0, p0])) @ want / (want @ want) for t_ in traj])
            st = story(traj, prog)
            got.append((float(prog[-1]), is_up(body)))
            res[f"tilt {name}"] = {"share": got[-1][0], "up_after": got[-1][1], "end": end_state(traj), "story": st}
            if TILT_XY_TOO:                                               # diagnosis: name only the two directional lines
                body.reset("stand")
                brain.fresh()
                traj = intend(body, brain, dict(zip(grav[:2].tolist(), g[:2].tolist())), 2.0)
                prog = np.array([(np.array(t_[6]) - np.array([r0, p0])) @ want / (want @ want) for t_ in traj])
                res[f"tilt {name}, directional lines only"] = {"share": float(prog[-1]), "up_after": is_up(body)}
        log(f"[{tag}] tilt 0.25 rad as asked (roll left/right, nose down/up): "
            + ", ".join(f"{a:+.0%}{'' if u else ' (fell)'}" for a, u in got) + " of the asked tilt")
        if TILT_XY_TOO:
            log(f"[{tag}]   DIAG same tilts, naming only the two directional gravity lines (not 'how straight down'): "
                + ", ".join(f"{res[f'tilt {n}, directional lines only']['share']:+.0%}"
                            f"{'' if res[f'tilt {n}, directional lines only']['up_after'] else ' (fell)'}"
                            for n in ("roll left", "roll right", "nose down", "nose up")))
        # weight
        shares = []
        for k, foot in enumerate(("FL", "FR", "RL", "RR")):
            body.reset("stand")
            brain.fresh()
            base = body.touch()
            traj = intend(body, brain, {int(touch[k]): float(1.5 * base[k])}, 2.0)
            now = body.touch()
            s0, s1 = base[k] / max(base.sum(), 1e-6), now[k] / max(now.sum(), 1e-6)
            shares.append((foot, s0, s1, is_up(body)))
            res[f"weight on {foot}"] = {"share_before": float(s0), "share_after": float(s1), "up_after": shares[-1][3],
                                        "end": end_state(traj)}
        log(f"[{tag}] more weight on one foot as asked (its share, before -> after 2 s): "
            + ", ".join(f"{f} {a:.0%}->{b:.0%}{'' if u else ' (fell)'}" for f, a, b, u in shares))
        lifts = []
        for k, foot in enumerate(("FL", "FR", "RL", "RR")):
            body.reset("stand")
            brain.fresh()
            f0 = float(body.touch()[k])
            intend(body, brain, {int(touch[k]): 0.0}, 2.0)
            f1 = float(body.touch()[k])
            lifts.append((foot, f0, f1, is_up(body)))
            res[f"lift {foot}"] = {"force_before_N": f0, "force_after_N": f1, "up_after": lifts[-1][3]}
        log(f"[{tag}] lift one paw as asked (its force, before -> after 2 s): "
            + ", ".join(f"{f} {a:.0f}->{b:.0f} N{'' if u else ' (fell)'}" for f, a, b, u in lifts))
        turns = []
        for name, rate in (("left", TURN_RATE), ("right", -TURN_RATE)):
            body.reset("stand")
            brain.fresh()
            y0 = body.yaw()
            intend(body, brain, {4 * J + 9: rate}, 3.0)
            dy = (body.yaw() - y0 + np.pi) % (2 * np.pi) - np.pi
            turns.append((name, float(dy / (rate * 3.0)), is_up(body)))
            res[f"turn {name}"] = {"turned_rad": float(dy), "share": turns[-1][1], "up_after": turns[-1][2]}
        log(f"[{tag}] turn as asked ({TURN_RATE:g} rad/s for 3 s): "
            + ", ".join(f"{n} {a:+.0%}{'' if u else ' (fell)'}" for n, a, u in turns) + " of the asked turn")
        # lying on its side, nothing asked: its reflexes' reference is standing still
        ups = 0
        ends = []
        for t in range(8):
            body.reset("left_side" if t % 2 == 0 else "right_side", np.random.default_rng(500 + t))
            brain.fresh()
            ends.append(end_state(intend(body, brain, {}, 6.0)))
            ups += int(is_up(body))
        res["get_up_from_side"] = {"up": ups, "of": 8, "ends": ends}
        log(f"[{tag}] lying on its side, nothing asked, after 6 s up on its legs (upright, above {STANDING_HEIGHT:g} m, "
            f"3-4 feet): {ups}/8; trunk heights " + ", ".join(f"{e['end_height_m']:.2f}" for e in ends))
        results[f"x{gain:g}"] = res
    brain.reflex = 0.0
    return results


def reflex_diag(body, brain, log, label, gain=1.0):
    """Which of its reflexes knock it over, and how: set down standing, nothing asked, 5 s, with one kind of reflex
    at a time. Drift: pools push one way steadily (|mean| / mean |output| near 1). Wobble: they flip back and forth."""
    J = len(body.joint)
    groups = {"joint angle": range(0, J), "joint speed": range(J, 2 * J), "effort": range(2 * J, 3 * J),
              "limit push-back": range(3 * J, 4 * J), "foot pressure": range(4 * J, 4 * J + 4),
              "gravity": range(4 * J + 4, 4 * J + 7), "turning": range(4 * J + 7, 4 * J + 10)}
    cols = {k: brain.fibre_col[list(v)].ravel() for k, v in groups.items()}
    reached = np.zeros(len(brain.sense), bool)
    reached[brain.fibre_col.ravel()] = True
    cols["no nerve (connectome only)"] = np.where(~reached)[0]
    cols["none"] = np.zeros(0, int)
    cols["all"] = np.arange(len(brain.sense))
    brain.reflex_scale = max(float(np.percentile(np.abs(line_scores(brain)), 99)), 1e-12)
    ctx_scales(brain)
    brain.reflex = gain
    settle_reflex(body, brain)
    out = {}
    for name, c in cols.items():
        brain.reflex_cols = c
        body.reset("stand")
        brain.fresh()
        start = body.target.copy() if body.tone else body.angles()
        traj = intend(body, brain, {}, 5.0)
        h = np.array([t_[0] for t_ in traj])
        cmd = np.array([t_[3] for t_ in traj])
        fell = float((np.argmax(h < STANDING_HEIGHT) + 1) * BRAIN_DT) if (h < STANDING_HEIGHT).any() else None
        drift = float(np.mean(np.abs(cmd.mean(0)) / (np.abs(cmd).mean(0) + 1e-9)))
        flips = float(np.mean((np.diff(np.sign(cmd), axis=0) != 0).sum(0)) / 5.0)
        moved = (body.target if body.tone else body.angles()) - start
        e = end_state(traj)
        out[name] = {"fell_s": fell, "end": e, "drift": drift, "flips_per_s": flips, "mean_abs_output": float(np.abs(cmd).mean()),
                     "set_point_moved_rad": moved.tolist(), "height": h[::5].round(3).tolist(),
                     "reflex_neurons": int(len(c))}
        log(f"[{label}] DIAG reflexes from {name:26s} ({len(c):4d} sensory neurons), x{gain:g}, standing, nothing asked: "
            f"fell at {fell} s; after 5 s trunk {e['end_height_m']:.2f} m, {e['end_feet_loaded']}/4 feet; pools' mean |output| "
            f"{np.abs(cmd).mean():.2f}, one-way {drift:.2f}, flips {flips:.1f}/s; set points moved mean {np.abs(moved).mean():.2f} rad; "
            f"height at 0.5/1/2/5 s " + "/".join(f"{h[int(t / BRAIN_DT) - 1]:.2f}" for t in (0.5, 1.0, 2.0, 5.0)))
    brain.reflex_cols = None
    brain.reflex = 0.0
    return out


def practice(body, brain, minutes, rng, log, label, gain=PRACTICE_GAIN0, support=False, fair=False):
    """Standing practice: set down standing again and again; its reflex synapses learn from its own dopamine-like
    signal. Height, feet and 'still up' are measured for the log only; the brain never receives them."""
    J = len(body.joint)
    groups = {"joint angle": range(0, J), "joint speed": range(J, 2 * J), "effort": range(2 * J, 3 * J),
              "limit push-back": range(3 * J, 4 * J), "foot pressure": range(4 * J, 4 * J + 4),
              "gravity": range(4 * J + 4, 4 * J + 7), "turning": range(4 * J + 7, 4 * J + 10)}
    cols = {k: brain.fibre_col[list(v)].ravel() for k, v in groups.items()}
    brain.reflex_scale = max(float(np.percentile(np.abs(line_scores(brain)), 99)), 1e-12)
    ctx_scales(brain)
    brain.reflex = gain
    settle_reflex(body, brain)
    brain.R = brain.map.copy()
    R0 = brain.R.copy()
    steps, ep, report = int(minutes * 60 / BRAIN_DT), int(EPISODE_S / BRAIN_DT), int(2 * 60 / BRAIN_DT)
    e = np.zeros_like(brain.R)
    burst = np.zeros(len(brain.motor))
    m_mean = np.zeros(len(brain.motor))
    d_mean, d_var = 0.0, 1e-4
    dev_mean, dev_var, cov = 1.0, 1e-2, 0.0                               # for the usual settling at this level of upset
    R0_norm = float(np.linalg.norm(R0))
    prev_dev = None
    ups, heights, devs, history = [], [], [], []
    first = int(1.0 / BRAIN_DT)                                           # premise test: the first second of each set-down
    if support:
        first = ep                                                        # with help it does not fall: use every step
    push_by_group, dopamine = [], []
    fade = SUPPORT_FADE_SHARE * steps
    t0 = time.time()
    for t in range(steps):
        if t % ep == 0:
            if t:
                ups.append(is_up(body))
                heights.append(float(body.d.qpos[2]))
            body.reset("stand")
            brain.fresh()
            e[:] = 0.0
            prev_dev = None
        body.support = max(0.0, 1.0 - t / fade) if support else 0.0
        burst += BRAIN_DT / BABBLE_TAU * -burst + np.sqrt(2 * BRAIN_DT / BABBLE_TAU) * rng.standard_normal(len(burst))
        felt = body.feel()
        dev = float(np.sqrt(np.mean(brain.zscore(felt) ** 2)))           # how far what it feels is from standing still
        pre = brain.rates()[brain.sense] - brain.rest_sense
        if t % ep < first:                                                # measurement only
            push_by_group.append([float(np.abs(brain.R[:, c] @ pre[c]).sum()) for c in cols.values()])
        r = brain.step(felt, PRACTICE_NOISE * BABBLE_SIZE * burst)
        body.step(brain.command(r))
        m = r[brain.motor]
        m_mean += 0.01 * (m - m_mean)
        e *= PRACTICE_TRACE
        e += np.outer(m - m_mean, pre)
        if prev_dev is not None:
            D = prev_dev - dev                                            # dopamine: back toward standing still feels good
            if fair:                                                      # minus what usually happens at this level of upset
                dev_mean += 0.001 * (prev_dev - dev_mean)
                dev_var += 0.001 * ((prev_dev - dev_mean) ** 2 - dev_var)
                cov += 0.001 * ((prev_dev - dev_mean) * (D - d_mean) - cov)
                D = D - cov / dev_var * (prev_dev - dev_mean)
            d_mean += 0.001 * (D - d_mean)
            d_var += 0.001 * ((D - d_mean) ** 2 - d_var)
            dz = float(np.clip((D - d_mean) / np.sqrt(d_var + 1e-12), -3.0, 3.0))
            brain.R -= PRACTICE_RATE * dz * e                             # reflex input is -R: strengthen what preceded good
            if fair:
                brain.R *= R0_norm / float(np.linalg.norm(brain.R))       # synaptic scaling
        else:
            D = 0.0
        if t % ep < first:
            dopamine.append(D - d_mean if fair else D)
        prev_dev = dev
        devs.append(dev)
        if (t + 1) % report == 0 or t == steps - 1:
            k = max(1, int(report / ep))
            change = float(np.linalg.norm(brain.R - R0) / np.linalg.norm(R0))
            same = {g: float(np.corrcoef(brain.R[:, c].ravel(), R0[:, c].ravel())[0, 1]) for g, c in cols.items()}
            if ups and np.mean(ups[-k:]) >= 0.5 and brain.reflex < 1.0:
                brain.reflex = min(1.0, brain.reflex * PRACTICE_GAIN_STEP)
            credit = None
            if len(dopamine) > 3 * first:
                Pw, Dw = np.array(push_by_group[-report:]), np.array(dopamine[-report:])
                credit = {g: float(np.corrcoef(Pw[:-3, i], Dw[3:])[0, 1]) for i, g in enumerate(cols)}
            history.append({"minute": (t + 1) * BRAIN_DT / 60, "gain": brain.reflex, "support": body.support,
                            "credit_last_2min": credit,
                            "up_share": float(np.mean(ups[-k:])) if ups else None,
                            "mean_end_height": float(np.mean(heights[-k:])) if heights else None,
                            "mean_felt_deviation": float(np.mean(devs[-report:])), "reflex_change": change,
                            "corr_with_start": same})
            log(f"[{label}] practice {(t + 1) * BRAIN_DT / 60:4.1f} min ({time.time() - t0:4.0f} s), reflexes now x{brain.reflex:.2f}, "
                f"help {body.support:.0%}: still up after 5 s in "
                f"{np.mean(ups[-k:]) if ups else float('nan'):.0%} of the last {min(k, len(ups))} set-downs (end height "
                f"{np.mean(heights[-k:]) if heights else float('nan'):.2f} m); felt deviation from standing still "
                f"{np.mean(devs[-report:]):.2f}; reflexes changed {change:.0%}; still like the twitch-tuned ones (corr): "
                + ", ".join(f"{g} {v:+.2f}" for g, v in same.items())
                + ("" if credit is None else "; credit (push vs dopamine 60 ms later): "
                   + ", ".join(f"{g} {v:+.3f}" for g, v in credit.items())))
    body.support = 0.0
    # premise: in the first second of a set-down, does each kind of reflex's push line up with the dopamine 60 ms later?
    P, Dd = np.array(push_by_group), np.array(dopamine)
    lag = 3
    ok = np.array([i % first < first - lag for i in range(len(Dd))])
    idx = np.where(ok)[0]
    credit = {g: float(np.corrcoef(P[idx, i], Dd[idx + lag])[0, 1]) for i, g in enumerate(cols)}
    log(f"[{label}] DIAG premise: in the first second of {len(Dd) // first} set-downs, how each kind of reflex's push "
        f"lines up with its dopamine 60 ms later (corr; negative = that reflex is followed by things getting worse): "
        + ", ".join(f"{g} {v:+.3f}" for g, v in credit.items()))
    history.append({"credit_corr": credit})
    return history


def ctx_diag(body, brain, log, label):
    """How many posture neurons it grew, which one is active when it stands still or lies on its side, and whether
    that one's reflex synapses know what its motors really do in that posture (against the single global table)."""
    J = len(body.joint)
    adapting, brain.adapting = brain.adapting, False
    out = {"posture_neurons": len(brain.ctx_p)}
    for pose in ("stand", "belly", "left_side"):
        body.reset(pose)
        brain.v[:] = 0.0
        for _ in range(int(1.0 / BRAIN_DT)):                              # let the slow senses settle on this posture
            brain.step(body.feel(), np.zeros(len(brain.motor)))
            body.step(np.zeros(J))
        a, d = brain.ctx_activity()
        k = int(np.argmax(a))
        truth = true_effects(body, pose)
        local = agreement(per_motor(brain, line_scores(brain, brain.ctx_map[k].astype(float))[:, 4 * J:4 * J + 7]), truth)
        glob = agreement(per_motor(brain, line_scores(brain)[:, 4 * J:4 * J + 7]), truth)
        out[pose] = {"neuron": k, "share": float(a[k]), "distance": float(d[k]), "local": local, "global": glob,
                     "distances": np.sort(d)[:3].tolist()}
        log(f"[{label}] DIAG posture '{pose}': its posture neuron {k} ({a[k]:.0%} of the activity; nearest three at "
            + "/".join(f"{x:.2f}" for x in np.sort(d)[:3]) + "); that neuron's reflex "
            f"links vs what the motors really do here, sign right: "
            + ", ".join(f"{nm.split()[0] if 'pressure' in nm else nm.split()[2]} {x['sign_right']:.0%}"
                        for nm, x in zip(GROUND_LINES, local))
            + " | the single global table: "
            + ", ".join(f"{x['sign_right']:.0%}" for x in glob))
    P = np.array(brain.ctx_p)
    pd = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).mean(2))[np.triu_indices(len(P), 1)] if len(P) > 1 else np.zeros(1)
    out["pairwise_distance"] = {"min": float(pd.min()), "median": float(np.median(pd)), "max": float(pd.max())}
    names = ([f"angle {n}" for n in body.names] + [f"speed {n}" for n in body.names] + [f"effort {n}" for n in body.names]
             + [f"limit {n}" for n in body.names] + [f"touch {f}" for f in ("FL", "FR", "RL", "RR")]
             + [f"gravity {a}" for a in "xyz"] + [f"turning {a}" for a in "xyz"])
    top = np.argsort(-brain.ctx_w)[:10]
    out["posture_weights_top"] = [[names[i], float(brain.ctx_w[i])] for i in top]
    log(f"[{label}] DIAG senses it reads posture from most (weight, mean 1; never told to the brain): "
        + ", ".join(f"{names[i]} {brain.ctx_w[i]:.1f}" for i in top))
    log(f"[{label}] DIAG grew {len(brain.ctx_p)} posture neurons; how far apart their preferred postures are (RMS adapted "
        f"units): min {pd.min():.2f}, median {np.median(pd):.2f}, max {pd.max():.2f} (they grow {CTX_GROW:g} apart, tuned "
        f"{CTX_SIGMA:g} wide)")
    brain.adapting = adapting
    return out


def skill_task(body, brain, rng, levels=None):
    """One practice request: {nerve: goal in its own units}, its kind, and the measure of success at the end."""
    J = len(body.joint)
    grav, touch = 4 * J + 4 + np.arange(3), 4 * J + np.arange(4)
    kinds = SKILL_MIX or (("tilt", "weight", "lift", "joint", "still") if SKILL_LIFT else ("tilt", "tilt", "weight", "joint", "still"))
    kind = kinds[rng.integers(len(kinds))]
    if kind == "tilt":
        a = rng.uniform(0.1, 0.3) * rng.choice([-1.0, 1.0])
        roll, pitch = (a, 0.0) if rng.random() < 0.5 else (0.0, a)
        g = np.array([np.sin(pitch), -np.sin(roll), -np.cos(roll) * np.cos(pitch)])
        return dict(zip(grav.tolist(), g.tolist())), kind, ("tilt", np.array([roll, pitch]))
    if kind == "weight":
        k = int(rng.integers(4))
        base = np.maximum(body.touch(), FOOT_FLOOR_N)                     # a varied set-down can start a foot unloaded
        L = (levels or {}).get(f"weight {k}", 0.5)                       # ask for this much more weight on the foot
        return {int(touch[k]): float((1 + L) * base[k])}, kind, ("weight", k, base[k] / max(base.sum(), 1e-6), L)
    if kind == "turn":                                                    # feel itself turning, left or right
        rate = TURN_RATE * rng.choice([-1.0, 1.0])
        return {4 * J + 9: rate}, kind, ("turn", rate, body.yaw())
    if kind == "walk":                                                    # feel itself move forward or back
        sign = float(rng.choice([-1.0, 1.0]))
        return {4 * J + 10: WALK_ACC * sign}, kind, ("walk", sign, body.d.qpos[:2].copy(), body.yaw())
    if kind == "lift":                                                    # take the weight off one paw
        k = int(rng.integers(4))
        base = np.maximum(body.touch(), FOOT_FLOOR_N)
        L = (levels or {}).get(f"lift {k}", 1.0)                         # take this share of the paw's weight off
        return {int(touch[k]): float((1 - L) * base[k])}, kind, ("lift", k, float(base[k]), L)
    if kind == "joint":
        j = int(rng.integers(J))
        q0 = body.angles()[j]
        goal = float(np.clip(q0 + rng.choice([-0.5, 0.5]), body.lo[j], body.hi[j]))
        return {j: goal}, kind, ("joint", j, q0, goal)
    return {}, kind, ("still",)


def skill_score(body, how, r0p0):
    if how[0] == "tilt":
        want = how[1]
        moved = np.array(body.tilt()) - np.array(r0p0)
        return float(moved @ want / (want @ want))
    if how[0] == "weight":
        k, s0, L = how[1], how[2], how[3]
        t = body.touch()
        return float((t[k] / max(t.sum(), 1e-6) - s0) / max(L * s0, 1e-6))
    if how[0] == "walk":                                                  # test only: true distance along its heading
        sign, p0, y0 = how[1], how[2], how[3]
        fwd = float((body.d.qpos[:2] - p0) @ np.array([np.cos(y0), np.sin(y0)]))
        return fwd / (sign * WALK_V * SKILL_TRIAL_S)
    if how[0] == "turn":
        rate, y0 = how[1], how[2]
        dy = (body.yaw() - y0 + np.pi) % (2 * np.pi) - np.pi
        return float(dy / (rate * SKILL_TRIAL_S))
    if how[0] == "lift":
        k, f0, L = how[1], how[2], how[3]
        return float((f0 - body.touch()[k]) / max(L * f0, 1e-6))
    if how[0] == "joint":
        j, q0, goal = how[1], how[2], how[3]
        return float((body.angles()[j] - q0) / (goal - q0)) if abs(goal - q0) > 1e-6 else 0.0
    return float(is_up(body))


def practice_skills(body, brain, minutes, rng, log, label):
    J = len(body.joint)
    brain.reflex_scale = max(float(np.percentile(np.abs(line_scores(brain)), 99)), 1e-12)
    brain.reflex = SKILL_REFLEX
    settle_reflex(body, brain)
    if brain.Q is None:
        brain.Q = np.zeros((len(brain.motor), len(brain.sense)))
    map_size = float(np.linalg.norm(brain.map)) / brain.reflex_scale
    trial = int(SKILL_TRIAL_S / BRAIN_DT)
    trials = int(minutes * 60 / SKILL_TRIAL_S)
    stats = {k: [0.0, 1e-4] for k in ("tilt", "weight", "lift", "joint", "still")}     # usual progress per step, per kind
    window, history = [], []
    levels, recent = {}, {}
    wp_stats = {}
    turn_log = []
    if SKILL_CURRICULUM:
        for k in range(4):
            levels[f"weight {k}"], levels[f"lift {k}"] = CURRICULUM["weight"][0], CURRICULUM["lift"][0]
    t0 = time.time()
    for n in range(trials):
        body.reset("stand", rng if SKILL_VARY else None)                  # --skill_vary: set down a little differently
        brain.fresh()
        targets, kind, how = skill_task(body, brain, rng, levels)
        task = kind
        if SKILL_PER_TASK:                                                # the exact request
            if how[0] == "tilt":
                task = f"tilt {'roll' if how[1][0] else 'pitch'} {'+' if how[1].sum() > 0 else '-'}"
            elif how[0] in ("weight", "lift"):
                task = f"{how[0]} {how[1]}"
            elif how[0] == "turn":
                task = f"turn {'+' if how[1] > 0 else '-'}"
            elif how[0] == "walk":
                task = f"walk {'+' if how[1] > 0 else '-'}"
            elif how[0] == "joint":
                task = f"joint {how[1]} {'+' if how[3] > how[2] else '-'}"
        if task not in stats:
            stats[task] = [0.0, 1e-4]
        for l, g in targets.items():
            brain.intention[l] = -(g - brain.mean[l]) / np.sqrt(brain.var[l] + 1e-6)
        lines = np.array(list(targets), int)
        goal = np.array([(targets[l] - brain.mean[l]) / np.sqrt(brain.var[l] + 1e-6) for l in lines])
        still = 4 * J + (np.r_[np.arange(7), 9] if STILL_TURN else np.arange(7))   # touch, gravity (, turning)
        watch = lines if len(lines) else still                            # "still": stay as it feels at rest
        aim = goal if len(lines) else np.zeros(len(still))
        r0p0 = body.tilt()
        noise = np.zeros(len(brain.motor))
        e = np.zeros_like(brain.Q)
        ec = None if brain.cpg is None or CPG_WP else np.zeros_like(brain.Qc)
        wp = None
        if CPG_WP and brain.cpg is not None and kind in ("turn", "walk"):  # synaptic noise on this direction's set
            k0 = (0 if kind == "turn" else 2) + (0 if how[1] > 0 else 1)
            cols = slice(k0 * CPG_N, (k0 + 1) * CPG_N)
            wp = np.zeros_like(brain.Qc)
            wp[:, cols] = rng.standard_normal((len(brain.motor), CPG_N))
            wp *= CPG_WP_SIZE * map_size / np.linalg.norm(wp)
            kept_qc = brain.Qc.copy()
            brain.Qc = kept_qc + wp
        dists, felt_turn, levels_felt = [], [], []
        v_felt = 0.0
        trial_q = STILL_TRIAL and kind == "still"                         # --still_trial: Q learns once, at trial end
        e_sum = np.zeros_like(brain.Q) if trial_q else None
        still_turn, still_level = [], []
        prev = None
        for _ in range(trial):
            noise += BRAIN_DT / BABBLE_TAU * -noise + np.sqrt(2 * BRAIN_DT / BABBLE_TAU) * rng.standard_normal(len(noise))
            xi = SKILL_NOISE * BABBLE_SIZE * noise
            pre = brain.rates()[brain.sense] - brain.rest_sense
            felt = body.feel()
            r = brain.step(felt, xi)
            cmd = brain.command(r)
            if SKILL_VARY:                                                # ...and with the +-0.1 motor wobble
                cmd = np.clip(cmd + rng.uniform(-0.1, 0.1, J), -1.0, 1.0)
            body.step(cmd)
            e *= SKILL_TRACE
            e += np.outer(xi, pre)
            if trial_q:
                e_sum += e
            if ec is not None:
                ec *= SKILL_TRACE
                ec += np.outer(xi, brain.cpg_f)
            zf = brain.zscore(body.feel())
            dist = float(np.sqrt(np.mean((zf[watch] - aim) ** 2)))
            if SKILL_LEVEL and kind not in ("tilt", "still") and (SKILL_LEVEL_JOINTS or kind != "joint"):  # ...and otherwise level
                dist += float(np.sqrt(np.mean(zf[4 * J + 4:4 * J + 7] ** 2)))
            dists.append(dist)
            if trial_q:
                still_turn.append(float(zf[4 * J + 9]))
                still_level.append(float(np.sqrt(np.mean(zf[4 * J + 4:4 * J + 7] ** 2))))
            if kind == "turn":
                felt_turn.append(float(zf[watch][0]))
                levels_felt.append(float(np.sqrt(np.mean(zf[4 * J + 4:4 * J + 7] ** 2))))
            if kind == "walk":                                            # its accelerometer, added up: felt speed
                v_felt += float(body.feel()[4 * J + 10]) * BRAIN_DT
                felt_turn.append(v_felt)
                levels_felt.append(float(np.sqrt(np.mean(zf[4 * J + 4:4 * J + 7] ** 2))))
            if prev is not None:
                D = prev - dist                                           # got closer to the asked feeling
                m, v = stats[task]
                m += 0.002 * (D - m)
                v += 0.002 * ((D - m) ** 2 - v)
                stats[task] = [m, v]
                dz = float(np.clip((D - m) / np.sqrt(v + 1e-12), -3.0, 3.0))
                if not (SKILL_NO_Q_TURN and kind in ("turn", "walk")) and not trial_q:  # --no_q_in_turn: turning is the rhythm's job
                    brain.Q += SKILL_RATE * dz * e
                if ec is not None:
                    brain.Qc += SKILL_RATE * dz * ec
                    qc = float(np.linalg.norm(brain.Qc))
                    if qc > SKILL_MAX * map_size:
                        brain.Qc *= SKILL_MAX * map_size / qc
                qn = float(np.linalg.norm(brain.Q))
                if qn > SKILL_MAX * map_size:
                    brain.Q *= SKILL_MAX * map_size / qn
            prev = dist
        brain.intention[:] = 0.0
        if trial_q:                                                       # still: no turning felt over the trial, level
            R = -abs(float(np.mean(still_turn))) - float(np.mean(still_level))
            ms = wp_stats.setdefault("still trial", [R, 0.0, 0])
            ms[2] += 1
            if ms[2] > 10:
                dzt = float(np.clip((R - ms[0]) / np.sqrt(ms[1] + 1e-12), -3.0, 3.0))
                brain.Q += SKILL_RATE * dzt * e_sum / np.sqrt(trial)       # 1/sqrt(T): the per-step rule's net change per
                qn = float(np.linalg.norm(brain.Q))                        # trial is a random walk ~sqrt(T)|e|; one dz on
                                                                           # the whole sum (~T|e|) ran away (stilltrial1)
                if qn > SKILL_MAX * map_size:
                    brain.Q *= SKILL_MAX * map_size / qn
            ms[0] += 0.05 * (R - ms[0])
            ms[1] += 0.05 * ((R - ms[0]) ** 2 - ms[1])
        if wp is not None:                                                # dopamine for the whole trial, vs the usual
            R = -float(np.mean(dists))                                    # for this exact request (felt, not seen)
            if kind == "walk":                                            # the speed it felt over the trial, and level
                R = -abs(float(np.mean(felt_turn)) - how[1] * WALK_V) - (float(np.mean(levels_felt)) if SKILL_LEVEL else 0.0)
            elif CPG_WP_MEAN:                                             # the turn it felt over the trial, and level
                R = -abs(float(np.mean(felt_turn)) - float(aim[0])) - (float(np.mean(levels_felt)) if SKILL_LEVEL else 0.0)
            ms = wp_stats.setdefault(task, [R, 0.0, 0])
            ms[2] += 1
            if ms[2] > 10:
                dzt = float(np.clip((R - ms[0]) / np.sqrt(ms[1] + 1e-12), -3.0, 3.0))
                kept_qc += CPG_WP_RATE * dzt * wp
                grp = slice(0, 2 * CPG_N) if kind == "turn" else slice(2 * CPG_N, 4 * CPG_N)   # turning and walking
                qc = float(np.linalg.norm(kept_qc[:, grp]))                # sets are capped separately, so walking
                if qc > CPG_WP_MAX * map_size:                             # cannot shrink the learned turning
                    kept_qc[:, grp] *= CPG_WP_MAX * map_size / qc
            ms[0] += 0.05 * (R - ms[0])
            ms[1] += 0.05 * ((R - ms[0]) ** 2 - ms[1])
            brain.Qc = kept_qc
        score = skill_score(body, how, r0p0)
        window.append((kind, score, is_up(body)))
        if kind == "turn":                                                # for the record only: does the dopamine
            turn_log.append((task, -float(np.mean(dists)),                # track the turn actually made?
                             -abs(float(np.mean(felt_turn)) - float(aim[0])), float(score), bool(is_up(body))))
        if SKILL_CURRICULUM and how[0] in ("weight", "lift"):              # the teacher's steps for this exact request
            key = f"{how[0]} {how[1]}"
            lo, hi = CURRICULUM[how[0]]
            recent.setdefault(key, []).append(score)
            if len(recent[key]) >= 10:
                med = float(np.median(recent[key]))
                if med >= 0.5:
                    levels[key] = min(hi, levels[key] * 1.25)
                elif med < 0.0:
                    levels[key] = max(lo, levels[key] / 1.25)
                recent[key] = []
        if (n + 1) % int(5 * 60 / SKILL_TRIAL_S) == 0 or n == trials - 1:
            by = {k: [w[1] for w in window if w[0] == k] for k in ("tilt", "weight", "lift", "joint", "still", "turn", "walk")}
            up = float(np.mean([w[2] for w in window]))
            qsize = float(np.linalg.norm(brain.Q)) / max(map_size, 1e-12)
            qcsize = [float(np.linalg.norm(brain.Qc[:, k * CPG_N:(k + 1) * CPG_N])) / max(map_size, 1e-12)
                      for k in range(brain.Qc.shape[1] // CPG_N)] if brain.cpg is not None else None
            history.append({"minute": (n + 1) * SKILL_TRIAL_S / 60, "still_up": up, "Q_vs_map": qsize, "Qc_vs_map": qcsize,
                            "levels": dict(levels),
                            **{k: (float(np.median(v)) if v else None) for k, v in by.items()}})
            log(f"[{label}] skill practice {(n + 1) * SKILL_TRIAL_S / 60:4.1f} min ({time.time() - t0:4.0f} s), last "
                f"{len(window)} trials: median share of the asked change: tilt "
                f"{np.median(by['tilt']) if by['tilt'] else float('nan'):+.0%}, weight "
                f"{np.median(by['weight']) if by['weight'] else float('nan'):+.0%}, lift a paw "
                f"{np.median(by['lift']) if by['lift'] else float('nan'):+.0%}, turn "
                f"{np.median(by['turn']) if by['turn'] else float('nan'):+.0%}, walk "
                f"{np.median(by['walk']) if by['walk'] else float('nan'):+.0%}, joint "
                f"{np.median(by['joint']) if by['joint'] else float('nan'):+.0%}; still up at the end {up:.0%}; "
                f"skill synapses {qsize:.2f} of the twitch map's size"
                + ("" if qcsize is None else "; rhythm synapses (turn left / right" + (", walk fwd / back" if len(qcsize) > 2 else "")
                   + ") " + " / ".join(f"{q:.2f}" for q in qcsize))
                + ("" if not levels else "; steps (weight / lift, FL FR RL RR): "
                   + " ".join(f"{levels[f'weight {k}']:.2f}" for k in range(4)) + " / "
                   + " ".join(f"{levels[f'lift {k}']:.2f}" for k in range(4))))
            window = []
    if turn_log:
        for t in sorted({x[0] for x in turn_log}):
            rows = np.array([x[1:4] for x in turn_log if x[0] == t])
            up = np.array([x[4] for x in turn_log if x[0] == t])
            c = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")
            rec = {"trials": int(len(rows)), "corr_reward_vs_turn": c(rows[:, 0], rows[:, 2]),
                   "corr_mean_felt_turn_vs_turn": c(rows[:, 1], rows[:, 2]), "up": float(up.mean()),
                   "turn_median": float(np.median(rows[:, 2]))}
            history.append({"turn_reward_check": t, **rec})
            log(f"[{label}] {t}: does the dopamine track the turn actually made? {rec['trials']} trials; correlation of the "
                f"trial's dopamine (closeness, moment by moment, plus level) with the true turn {rec['corr_reward_vs_turn']:+.2f}; "
                f"of the felt turning averaged over the trial {rec['corr_mean_felt_turn_vs_turn']:+.2f}; median turn "
                f"{rec['turn_median']:+.0%}; up {rec['up']:.0%}")
    return history


def feasibility(body, log, rng, samples=300, seconds=2.0, spread=0.6):
    """Physics only, no brain: from standing, hold random joint postures (within +-spread rad of the standing posture)
    for 2 s and record the best achieved per request. Says whether a weight shift or a paw lift is possible at all for
    this body. Nothing here is given to the brain."""
    J = len(body.joint)
    leak, body.leak = body.leak, False                                    # hold the posture asked for
    best_share, best_lift, ok = np.zeros(4), np.zeros(4), 0
    best_pose = {}
    for n in range(samples):
        body.reset("stand")
        stand = body.angles()
        f0 = body.touch()
        s0 = f0 / max(f0.sum(), 1e-6)
        body.target = np.clip(stand + rng.uniform(-spread, spread, J), body.lo, body.hi)
        for _ in range(int(seconds / BRAIN_DT)):
            body.step(np.zeros(J))
        if not is_up(body):
            continue
        ok += 1
        f = body.touch()
        share = f / max(f.sum(), 1e-6)
        for k in range(4):
            if share[k] - s0[k] > best_share[k]:
                best_share[k] = share[k] - s0[k]
                best_pose[f"weight {k}"] = (body.target - stand).round(2).tolist()
            lift = (f0[k] - f[k]) / max(f0[k], 1e-6)
            if lift > best_lift[k]:
                best_lift[k] = lift
                best_pose[f"lift {k}"] = (body.target - stand).round(2).tolist()
    body.leak = leak
    feet = ("FL", "FR", "RL", "RR")
    log(f"FEASIBILITY ({samples} random held postures within +-{spread:g} rad of standing, {ok} stayed up): best gain in "
        f"a foot's share of the weight: " + ", ".join(f"{f} +{b:.0%}" for f, b in zip(feet, best_share))
        + "; best lightening of a paw: " + ", ".join(f"{f} {b:.0%}" for f, b in zip(feet, best_lift)))
    return {"stayed_up": ok, "best_share_gain": best_share.tolist(), "best_lift": best_lift.tolist(), "poses": best_pose}


def rhythm_diag(body, brain, log, rng, gains=(0.0, 0.5, 1.0, 2.0, 4.0), seconds=20.0):
    """Measure only, no learning: does the brain produce rhythmic motor output by itself while it stands? For each
    adaptation strength: 20 s set down standing, the learned reflexes (x0.125) and skill synapses on, small motor
    variation (practice level); once held still (output cut) and once free. Power of each pool's output in the
    walking band (1-3 Hz) as a share of 0.25-10 Hz, the strongest frequency, and whether it is still up."""
    J = len(body.joint)
    brain.reflex_scale = max(float(np.percentile(np.abs(line_scores(brain)), 99)), 1e-12)
    out = {}
    n = int(seconds / BRAIN_DT)
    freqs = np.fft.rfftfreq(n, BRAIN_DT)
    band, wide = (freqs >= 1.0) & (freqs <= 3.0), (freqs >= 0.25) & (freqs <= 10.0)
    for g in gains:
        brain.adapt_g, brain.adapt = g, None
        brain.reflex = SKILL_REFLEX
        settle_reflex(body, brain)
        for how in ("held still", "free"):
            body.reset("stand")
            brain.fresh()
            noise = np.zeros(len(brain.motor))
            outs = np.zeros((n, J))
            for t in range(n):
                noise += BRAIN_DT / BABBLE_TAU * -noise + np.sqrt(2 * BRAIN_DT / BABBLE_TAU) * rng.standard_normal(len(noise))
                r = brain.step(body.feel(), SKILL_NOISE * BABBLE_SIZE * noise)
                o = brain.command(r)
                outs[t] = o
                body.step(o if how == "free" else np.zeros(J))
            x = outs - outs.mean(0)
            P = np.abs(np.fft.rfft(x, axis=0)) ** 2
            share = P[band].sum(0) / np.maximum(P[wide].sum(0), 1e-12)
            peak = freqs[wide][np.argmax(P[wide], axis=0)]
            res = {"band_share": float(share.mean()), "peak_hz_median": float(np.median(peak)),
                   "amplitude": float(x.std(0).mean()), "up_after": is_up(body)}
            out[f"adapt {g:g} {how}"] = res
            log(f"RHYTHM adaptation {g:g}, {how}: output swing {res['amplitude']:.3f}; share in 1-3 Hz "
                f"{res['band_share']:.0%} (random would be ~{band.sum() / wide.sum():.0%}); strongest frequency (median of "
                f"pools) {res['peak_hz_median']:.2f} Hz; still up {res['up_after']}")
    brain.adapt_g, brain.adapt, brain.reflex = 0.0, None, 0.0
    return out


def cpg_check(brain, log, seconds=10.0):
    """Does the grown pattern generator oscillate when driven? Hand-overs per second, period, share of time each leads."""
    saved = {k: v.copy() for k, v in brain.cpg.items()}
    brain.cpg["u"][:] = 0.0
    brain.cpg["u"][0] = 0.1
    brain.cpg["a"][:] = 0.0
    lead = []
    for _ in range(int(seconds / BRAIN_DT)):
        r = brain.cpg_step(1.0)
        lead.append(int(np.argmax(r)) if r.max() > 1e-3 else -1)
    brain.cpg.update(saved)
    lead = np.array(lead[int(1.0 / BRAIN_DT):])
    switches = int((np.diff(lead) != 0).sum())
    rate = switches / (seconds - 1.0)
    share = [float(np.mean(lead == i)) for i in range(CPG_N)]
    silent = float(np.mean(lead == -1))
    log(f"CPG check (driven, {seconds - 1:g} s): {rate:.2f} hand-overs/s, a full cycle every "
        f"{CPG_N / max(rate, 1e-9):.2f} s; share of time each neuron leads " + "/".join(f"{x:.0%}" for x in share)
        + f"; silent {silent:.0%}")
    return rate, share, silent


def turn_eval(body, brain, log, rng, lives=30, seconds=3.0):
    """Promotion check for turning, on fresh lives (seeds never used in practice): each life is set down standing with
    its joints a little different (the reset's own +-0.2 rad), runs with the standing motor wobble (+-0.1 on each
    command, AGENTS.md) on top of the practice-level motor variation, and is asked to turn left or right for 3 s.
    (Practice-level variation alone gave 30 identical lives, SE 0: not independent.) Three versions of this brain:
    as learned; with its rhythm synapses switched off (does the turning come from them?); with them replaced by random
    synapses of the same size per direction (the random floor). Reports mean share of the asked turn +- SE, k/n lives
    turning >= 30% the asked way and still up, and lives still up."""
    J = len(body.joint)
    brain.reflex_scale = max(float(np.percentile(np.abs(line_scores(brain)), 99)), 1e-12)
    brain.reflex = SKILL_REFLEX
    settle_reflex(body, brain)
    kept = brain.Qc.copy()
    rand = rng.standard_normal(kept.shape)
    for c in (slice(0, CPG_N), slice(CPG_N, 2 * CPG_N)):
        n_ = np.linalg.norm(rand[:, c])
        rand[:, c] *= (np.linalg.norm(kept[:, c]) / n_) if n_ > 0 else 0.0
    out = {}
    for name, Qc in (("as learned", kept), ("rhythm synapses off", np.zeros_like(kept)), ("random, same size", rand)):
        brain.Qc = Qc
        res = {}
        for side, rate in (("left", TURN_RATE), ("right", -TURN_RATE)):
            sh, up = [], []
            for _ in range(lives):
                body.reset("stand", rng)
                brain.fresh()
                brain.intention[4 * J + 9] = -(rate - brain.mean[4 * J + 9]) / np.sqrt(brain.var[4 * J + 9] + 1e-6)
                y0 = body.yaw()
                noise = np.zeros(len(brain.motor))
                for _ in range(int(seconds / BRAIN_DT)):
                    noise += BRAIN_DT / BABBLE_TAU * -noise + np.sqrt(2 * BRAIN_DT / BABBLE_TAU) * rng.standard_normal(len(noise))
                    r = brain.step(body.feel(), SKILL_NOISE * BABBLE_SIZE * noise)
                    body.step(np.clip(brain.command(r) + rng.uniform(-0.1, 0.1, J), -1.0, 1.0))
                brain.intention[:] = 0.0
                dy = (body.yaw() - y0 + np.pi) % (2 * np.pi) - np.pi
                sh.append(float(dy / (rate * seconds)))
                up.append(is_up(body))
            sh, up = np.array(sh), np.array(up)
            k = int(((sh >= 0.3) & up).sum())
            res[side] = {"share": sh.tolist(), "up": up.tolist(), "mean": float(sh.mean()),
                         "se": float(sh.std(ddof=1) / np.sqrt(len(sh))), "k_ge_30_up": k, "n": len(sh), "up_frac": float(up.mean())}
        out[name] = res
        log(f"TURN EVAL {name} ({lives} fresh lives per side: varied set-down, +-0.1 wobble): "
            + "; ".join(f"{side} {r['mean']:+.0%} +- {r['se']:.0%} (SE), >=30% and up {r['k_ge_30_up']}/{r['n']}, "
                        f"up {r['up_frac']:.0%}" for side, r in res.items()))
    brain.Qc = kept
    drift, up = [], []                                                    # nothing asked: does it drift by itself?
    for _ in range(lives):
        body.reset("stand", rng)
        brain.fresh()
        y0 = body.yaw()
        noise = np.zeros(len(brain.motor))
        for _ in range(int(seconds / BRAIN_DT)):
            noise += BRAIN_DT / BABBLE_TAU * -noise + np.sqrt(2 * BRAIN_DT / BABBLE_TAU) * rng.standard_normal(len(noise))
            r = brain.step(body.feel(), SKILL_NOISE * BABBLE_SIZE * noise)
            body.step(np.clip(brain.command(r) + rng.uniform(-0.1, 0.1, J), -1.0, 1.0))
        dy = (body.yaw() - y0 + np.pi) % (2 * np.pi) - np.pi
        drift.append(float(dy / (TURN_RATE * seconds)))
        up.append(is_up(body))
    drift, up = np.array(drift), np.array(up)
    out["nothing asked"] = {"drift_share_of_left_turn": drift.tolist(), "up": up.tolist(), "mean": float(drift.mean()),
                            "se": float(drift.std(ddof=1) / np.sqrt(len(drift))), "up_frac": float(up.mean())}
    log(f"TURN EVAL nothing asked ({lives} fresh lives, same conditions): drifts {drift.mean():+.0%} +- "
        f"{drift.std(ddof=1) / np.sqrt(len(drift)):.0%} (SE) of a left turn (+ is left), up {up.mean():.0%}")
    if brain.walk_line is not None:                                       # walking on request, same conditions
        wl = brain.walk_line
        for name, Qc in (("as learned", kept), ("rhythm synapses off", np.zeros_like(kept))):
            brain.Qc = Qc
            res = {}
            for side, sign in (("forward", 1.0), ("back", -1.0), ("nothing asked", 0.0)):
                sh, up = [], []
                for _ in range(lives):
                    body.reset("stand", rng)
                    brain.fresh()
                    if sign:
                        brain.intention[wl] = -(WALK_ACC * sign - brain.mean[wl]) / np.sqrt(brain.var[wl] + 1e-6)
                    p0, y0 = body.d.qpos[:2].copy(), body.yaw()
                    noise = np.zeros(len(brain.motor))
                    for _ in range(int(seconds / BRAIN_DT)):
                        noise += BRAIN_DT / BABBLE_TAU * -noise + np.sqrt(2 * BRAIN_DT / BABBLE_TAU) * rng.standard_normal(len(noise))
                        r = brain.step(body.feel(), SKILL_NOISE * BABBLE_SIZE * noise)
                        body.step(np.clip(brain.command(r) + rng.uniform(-0.1, 0.1, J), -1.0, 1.0))
                    brain.intention[:] = 0.0
                    fwd = float((body.d.qpos[:2] - p0) @ np.array([np.cos(y0), np.sin(y0)]))
                    sh.append(fwd / (WALK_V * seconds))                   # + is forward, as a share of the asked walk
                    up.append(is_up(body))
                sh, up = np.array(sh), np.array(up)
                res[side] = {"share_forward": sh.tolist(), "up": up.tolist(), "mean": float(sh.mean()),
                             "se": float(sh.std(ddof=1) / np.sqrt(len(sh))), "up_frac": float(up.mean())}
            out[f"walk, {name}"] = res
            log(f"TURN EVAL walk, {name} ({lives} lives each; distance along its heading as a share of {WALK_V:g} m/s x "
                f"{seconds:g} s, + is forward): " + "; ".join(f"{k} {v['mean']:+.0%} +- {v['se']:.0%}, up {v['up_frac']:.0%}"
                                                            for k, v in res.items()))
        brain.Qc = kept
    grav = 4 * J + 4 + np.arange(3)                                       # retention: tilt on request, same conditions
    tilt = {}
    for name, (roll, pitch) in {"roll left": (0.25, 0.0), "roll right": (-0.25, 0.0), "nose down": (0.0, 0.25),
                                "nose up": (0.0, -0.25)}.items():
        sh, up = [], []
        g = np.array([np.sin(pitch), -np.sin(roll), -np.cos(roll) * np.cos(pitch)])
        want = np.array([roll, pitch])
        for _ in range(lives):
            body.reset("stand", rng)
            brain.fresh()
            t0 = np.array(body.tilt())
            for l, gv in zip(grav.tolist(), g.tolist()):
                brain.intention[l] = -(gv - brain.mean[l]) / np.sqrt(brain.var[l] + 1e-6)
            noise = np.zeros(len(brain.motor))
            for _ in range(int(2.0 / BRAIN_DT)):
                noise += BRAIN_DT / BABBLE_TAU * -noise + np.sqrt(2 * BRAIN_DT / BABBLE_TAU) * rng.standard_normal(len(noise))
                r = brain.step(body.feel(), SKILL_NOISE * BABBLE_SIZE * noise)
                body.step(np.clip(brain.command(r) + rng.uniform(-0.1, 0.1, J), -1.0, 1.0))
            brain.intention[:] = 0.0
            sh.append(float((np.array(body.tilt()) - t0) @ want / (want @ want)))
            up.append(is_up(body))
        sh, up = np.array(sh), np.array(up)
        tilt[name] = {"share": sh.tolist(), "up": up.tolist(), "mean": float(sh.mean()),
                      "se": float(sh.std(ddof=1) / np.sqrt(len(sh))), "up_frac": float(up.mean())}
    out["tilt"] = tilt
    log(f"TURN EVAL retention, tilt 0.25 rad as asked ({lives} fresh lives each, same conditions): "
        + "; ".join(f"{n} {t['mean']:+.0%} +- {t['se']:.0%}, up {t['up_frac']:.0%}" for n, t in tilt.items()))
    brain.reflex = 0.0
    return out


def drift_diag(body, brain, log, rng, lives=30, seconds=3.0):
    """Measurement only: where does the spin with nothing asked come from? The nothing-asked drift (share of a 0.5 rad/s
    left turn, + is left) with each candidate switched off in turn: as is; skill synapses off; reflexes off; no motor
    wobble; no varied set-down. Same conditions otherwise (practice-level motor variation)."""
    J = len(body.joint)
    brain.reflex_scale = max(float(np.percentile(np.abs(line_scores(brain)), 99)), 1e-12)
    keptQ = brain.Q
    out = {}
    for name, q_on, reflex, wobble, vary in (("as is", True, SKILL_REFLEX, True, True),
                                             ("skill synapses off", False, SKILL_REFLEX, True, True),
                                             ("reflexes off", True, 0.0, True, True),
                                             ("no motor wobble", True, SKILL_REFLEX, False, True),
                                             ("no varied set-down", True, SKILL_REFLEX, True, False)):
        brain.Q = keptQ if q_on else np.zeros_like(keptQ)
        brain.reflex = reflex
        settle_reflex(body, brain)
        drift, up = [], []
        for _ in range(lives):
            body.reset("stand", rng if vary else None)
            brain.fresh()
            y0 = body.yaw()
            noise = np.zeros(len(brain.motor))
            for _ in range(int(seconds / BRAIN_DT)):
                noise += BRAIN_DT / BABBLE_TAU * -noise + np.sqrt(2 * BRAIN_DT / BABBLE_TAU) * rng.standard_normal(len(noise))
                r = brain.step(body.feel(), SKILL_NOISE * BABBLE_SIZE * noise)
                cmd = brain.command(r)
                if wobble:
                    cmd = np.clip(cmd + rng.uniform(-0.1, 0.1, J), -1.0, 1.0)
                body.step(cmd)
            dy = (body.yaw() - y0 + np.pi) % (2 * np.pi) - np.pi
            drift.append(float(dy / (TURN_RATE * seconds)))
            up.append(is_up(body))
        drift, up = np.array(drift), np.array(up)
        out[name] = {"drift": drift.tolist(), "up": up.tolist(), "mean": float(drift.mean()),
                     "se": float(drift.std(ddof=1) / np.sqrt(len(drift))), "up_frac": float(up.mean())}
        log(f"DRIFT {name} ({lives} lives): {drift.mean():+.0%} +- {drift.std(ddof=1) / np.sqrt(len(drift)):.0%} (SE) of a "
            f"left turn (+ is left), up {up.mean():.0%}")
    brain.Q = keptQ
    brain.reflex = 0.0
    return out


def cpg_trace(body, brain, log, rng, size=0.5, seconds=3.0):
    """Measurement only: where does the rhythm's signal go? One turn request with random rhythm-to-motor synapses of
    the given size, and the same with none. At each stage (the generator's drive and output, its input to the motor
    neurons against their other input, the motor neurons, the motor pools' raw output, the command, the set points,
    the joints), the swing over time, so the stage where the two runs stop differing is where the signal is lost."""
    J = len(body.joint)
    brain.reflex_scale = max(float(np.percentile(np.abs(line_scores(brain)), 99)), 1e-12)
    map_size = float(np.linalg.norm(brain.map)) / brain.reflex_scale
    brain.reflex = SKILL_REFLEX
    settle_reflex(body, brain)
    kept = brain.Qc.copy()
    Qc = rng.standard_normal(brain.Qc.shape)
    Qc *= size * map_size / np.linalg.norm(Qc)
    m = brain.motor
    out = {}
    for name, Q in (("with", Qc), ("none", np.zeros_like(Qc))):
        brain.Qc = Q
        body.reset("stand")
        brain.fresh()
        for l, g in {4 * J + 9: TURN_RATE}.items():
            brain.intention[l] = -(g - brain.mean[l]) / np.sqrt(brain.var[l] + 1e-6)
        want = float(-brain.intention[4 * J + 9])
        y0 = body.yaw()
        rec = {k: [] for k in ("cpg", "cpg_in", "rest_in", "v", "rate", "raw", "cmd", "target", "q")}
        for _ in range(int(seconds / BRAIN_DT)):
            r0, v0 = brain.rates(), brain.v.copy()
            wr = (brain.W @ r0)[m]
            r = brain.step(body.feel(), np.zeros(len(m)))
            I_m = (brain.v[m] - v0[m]) / ALPHA + v0[m] - wr
            ci = brain.Qc @ brain.cpg_f
            o = brain.command(r)
            body.step(o)
            rec["cpg"].append(brain.cpg_f.copy()); rec["cpg_in"].append(ci); rec["rest_in"].append(wr + I_m - ci)
            rec["v"].append(brain.v[m].copy()); rec["rate"].append(r[m].copy()); rec["raw"].append(brain.last_raw.copy())
            rec["cmd"].append(o); rec["target"].append(body.target.copy()); rec["q"].append(body.angles().copy())
        brain.intention[:] = 0.0
        a = {k: np.array(v) for k, v in rec.items()}
        sd = lambda x: float(x[25:].std(0).mean())                      # swing over time, after the first 0.5 s
        res = {"want_z": want, "cpg_drive": min(1.0, abs(want) / 0.25), "cpg_peak": float(np.abs(a["cpg"]).max()),
               "cpg_swing": sd(a["cpg"]), "cpg_in_swing": sd(a["cpg_in"]), "cpg_in_rms": float(np.sqrt((a["cpg_in"] ** 2).mean())),
               "rest_in_swing": sd(a["rest_in"]), "rest_in_rms": float(np.sqrt((a["rest_in"] ** 2).mean())),
               "v_pos_frac": float((a["v"] > 0).mean()), "v_sat_frac": float((a["v"] > 2.0).mean()),
               "v_median": float(np.median(a["v"])), "rate_swing": sd(a["rate"]), "raw_swing": sd(a["raw"]),
               "cmd_swing": sd(a["cmd"]), "cmd_abs": float(np.abs(a["cmd"]).mean()), "target_swing": sd(a["target"]),
               "q_swing": sd(a["q"]), "yaw_turned": float((body.yaw() - y0 + np.pi) % (2 * np.pi) - np.pi)}
        out[name] = res
        log(f"CPG TRACE {name} rhythm synapses (size {size if name == 'with' else 0:g}): turn request z {res['want_z']:.2f} "
            f"-> generator drive {res['cpg_drive']:.2f}, output peak {res['cpg_peak']:.3f} swing {res['cpg_swing']:.3f}; "
            f"its input to motor neurons swing {res['cpg_in_swing']:.4f} rms {res['cpg_in_rms']:.4f} vs their other input "
            f"swing {res['rest_in_swing']:.4f} rms {res['rest_in_rms']:.4f}; motor neurons above threshold "
            f"{res['v_pos_frac']:.0%}, saturated {res['v_sat_frac']:.0%}, median v {res['v_median']:.3f}; swing of motor "
            f"rates {res['rate_swing']:.4f}, pool raw {res['raw_swing']:.4f}, command {res['cmd_swing']:.4f} "
            f"(mean |cmd| {res['cmd_abs']:.3f}), set points {res['target_swing']:.4f} rad, joints {res['q_swing']:.4f} rad")
    brain.Qc = kept
    brain.reflex = 0.0
    return out


def g1_reflex_stand(body, brain, log, rng, gains=(0.0, 0.125, 0.25, 0.5, 1.0), lives=20, seconds=10.0):
    """Can the brain's own reflexes, from the map it learned in the G1, keep the G1 standing? Brain plugged into the G1,
    no hands, the +-0.1 motor wobble and practice-level motor variation, at each reflex strength. Per life: how long it
    stayed up. Compare: 2.2 s with no brain at all (§6.58)."""
    J = len(body.joint)
    brain.reflex_scale = max(float(np.percentile(np.abs(line_scores(brain)), 99)), 1e-12)
    out = {}
    for g in gains:
        brain.reflex = g
        settle_reflex(body, brain)
        ups, times = [], []
        for _ in range(lives):
            body.reset("stand")
            brain.fresh()
            noise = np.zeros(len(brain.motor))
            stayed = seconds
            for t in range(int(seconds / BRAIN_DT)):
                noise += BRAIN_DT / BABBLE_TAU * -noise + np.sqrt(2 * BRAIN_DT / BABBLE_TAU) * rng.standard_normal(len(noise))
                r = brain.step(body.feel(), SKILL_NOISE * BABBLE_SIZE * noise)
                body.step(np.clip(brain.command(r) + rng.uniform(-0.1, 0.1, J), -1.0, 1.0))
                if stayed == seconds and not is_up(body):
                    stayed = (t + 1) * BRAIN_DT
            ups.append(is_up(body)); times.append(stayed)
        out[f"x{g:g}"] = {"up": ups, "stayed_s": times, "up_frac": float(np.mean(ups)), "median_stayed_s": float(np.median(times))}
        log(f"G1 REFLEX STAND own reflexes x{g:g} ({lives} lives, {seconds:g} s, no hands, +-0.1 wobble): up at the end "
            f"{np.mean(ups):.0%}; stayed up median {np.median(times):.2f} s (min {np.min(times):.2f}, max {np.max(times):.2f})")
    brain.reflex = 0.0
    return out


def practice_g1_balance(body, brain, minutes, rng, log, label):
    """Balance in the G1, learned by reward with a parent's hands that fade. 2 s trials set down standing, +-0.1 motor
    wobble, practice-level motor variation (the exploration). The skill synapses learn once per trial (node
    perturbation, summed eligibility x 1/sqrt(T)) from the dopamine of staying level as felt over the whole trial,
    standardised. The parent (teacher) sees whether it is still up: help x0.9 after a run of successes, x1.1 after
    falls, between 0 and 0.5. Reflexes off (they topple it, §6.61). Nerve positions come from the body (2 feet)."""
    J, nf = len(body.joint), len(body.feet)
    grav = 4 * J + nf + np.arange(3)                                      # the inner ear's gravity lines
    brain.reflex_scale = max(float(np.percentile(np.abs(line_scores(brain)), 99)), 1e-12)
    brain.reflex = 0.0
    settle_reflex(body, brain)
    if brain.Q is None:
        brain.Q = np.zeros((len(brain.motor), len(brain.sense)))
    map_size = float(np.linalg.norm(brain.map)) / brain.reflex_scale
    T = int(SKILL_TRIAL_S / BRAIN_DT)
    help_, recent, stats, window, history = G1_HELP_START, [], [0.0, 0.0, 0], [], []
    t0 = time.time()
    for n in range(int(minutes * 60 / SKILL_TRIAL_S)):
        body.reset("stand")
        body.support = help_
        brain.fresh()
        noise = np.zeros(len(brain.motor))
        e = np.zeros_like(brain.Q)
        e_sum = np.zeros_like(brain.Q)
        lev = []
        for _ in range(T):
            noise += BRAIN_DT / BABBLE_TAU * -noise + np.sqrt(2 * BRAIN_DT / BABBLE_TAU) * rng.standard_normal(len(noise))
            xi = SKILL_NOISE * BABBLE_SIZE * noise
            pre = brain.rates()[brain.sense] - brain.rest_sense
            r = brain.step(body.feel(), xi)
            body.step(np.clip(brain.command(r) + rng.uniform(-0.1, 0.1, J), -1.0, 1.0))
            e *= SKILL_TRACE
            e += np.outer(xi, pre)
            e_sum += e
            zf = brain.zscore(body.feel())
            lev.append(float(np.sqrt(np.mean(zf[grav] ** 2))))
        R = -float(np.mean(lev))                                          # felt level over the trial
        if stats[2] > 10:
            dz = float(np.clip((R - stats[0]) / np.sqrt(stats[1] + 1e-12), -3.0, 3.0))
            brain.Q += SKILL_RATE * dz * e_sum / np.sqrt(T)
            qn = float(np.linalg.norm(brain.Q))
            if qn > SKILL_MAX * map_size:
                brain.Q *= SKILL_MAX * map_size / qn
        stats[0] += 0.05 * (R - stats[0]); stats[1] += 0.05 * ((R - stats[0]) ** 2 - stats[1]); stats[2] += 1
        up = is_up(body)
        window.append((up, help_, -R))
        recent.append(up)
        if len(recent) >= 10:                                             # the parent's judgement
            help_ = max(0.0, help_ * 0.9) if np.mean(recent) >= 0.8 else min(G1_HELP_START, help_ * 1.1 + 0.01)
            recent = []
        if (n + 1) % int(5 * 60 / SKILL_TRIAL_S) == 0:
            ups = [w[0] for w in window]
            history.append({"minute": (n + 1) * SKILL_TRIAL_S / 60, "up": float(np.mean(ups)), "help": help_,
                            "tilt_felt": float(np.mean([w[2] for w in window])),
                            "Q_vs_map": float(np.linalg.norm(brain.Q)) / max(map_size, 1e-12)})
            log(f"[{label}] G1 balance practice {(n + 1) * SKILL_TRIAL_S / 60:4.1f} min ({time.time() - t0:4.0f} s): up at the "
                f"end {np.mean(ups):.0%}; parent's help now {help_:.2f} (mean {np.mean([w[1] for w in window]):.2f}); felt "
                f"tilt {np.mean([w[2] for w in window]):.2f}; skill synapses {history[-1]['Q_vs_map']:.2f} of the map")
            window = []
    body.support = 0.0
    return history


def g1_support_check(body, brain, log, rng, supports=(0.0, 0.25, 0.5, 0.75, 1.0), lives=20, seconds=10.0):
    """How much of a parent's help does the G1 need to stay up with the brain plugged in (reflexes x0.125, +-0.1
    wobble, practice-level variation)? Plus: the brain alone, no wobble and no added variation, no help."""
    J = len(body.joint)
    brain.reflex_scale = max(float(np.percentile(np.abs(line_scores(brain)), 99)), 1e-12)
    brain.reflex = SKILL_REFLEX
    settle_reflex(body, brain)
    out = {}
    for name, sup, noisy in [(f"help {x:g}", x, True) for x in supports] + [("brain alone, no wobble, no help", 0.0, False)]:
        ups, times, heights = [], [], []
        for _ in range(lives if noisy else 1):
            body.reset("stand")
            body.support = sup
            brain.fresh()
            noise = np.zeros(len(brain.motor))
            stayed = seconds
            for t in range(int(seconds / BRAIN_DT)):
                if noisy:
                    noise += BRAIN_DT / BABBLE_TAU * -noise + np.sqrt(2 * BRAIN_DT / BABBLE_TAU) * rng.standard_normal(len(noise))
                r = brain.step(body.feel(), SKILL_NOISE * BABBLE_SIZE * noise)
                cmd = brain.command(r)
                body.step(np.clip(cmd + rng.uniform(-0.1, 0.1, J), -1.0, 1.0) if noisy else cmd)
                if stayed == seconds and not is_up(body):
                    stayed = (t + 1) * BRAIN_DT
            ups.append(is_up(body)); times.append(stayed); heights.append(float(body.d.qpos[2]))
        body.support = 0.0
        out[name] = {"up": ups, "stayed_s": times, "height_end": heights, "up_frac": float(np.mean(ups)),
                     "median_stayed_s": float(np.median(times))}
        log(f"G1 SUPPORT {name} ({len(ups)} lives, {seconds:g} s, brain on, reflexes x{SKILL_REFLEX:g}): up at the end "
            f"{np.mean(ups):.0%}; stayed up median {np.median(times):.2f} s; pelvis at the end median {np.median(heights):.2f} m")
    brain.reflex = 0.0
    return out


def g1_stand_check(body, log, rng, lives=20, seconds=10.0):
    """Physics premise for the humanoid, no brain: can the G1 stay up with its joints only held (position servos at the
    set-down posture, leaky set points as for the dog)? (a) held still; (b) with the +-0.1 motor wobble on every
    command; (c) set down a little differently (+-0.05 rad per joint) and with the wobble. Per life: up at the end, how
    long it stayed up, the pelvis height at the end."""
    J = len(body.joint)
    out = {}
    for name, wobble, vary in (("held", False, 0.0), ("+-0.1 wobble", True, 0.0), ("varied set-down +-0.05, wobble", True, 0.05)):
        ups, times, heights = [], [], []
        for _ in range(lives if (wobble or vary) else 1):
            body.reset("stand")
            if vary:
                body.d.qpos[body.qadr] += rng.uniform(-vary, vary, J)
                body.mj.mj_forward(body.m, body.d)
                body.target = body.angles()
                body.rest_q = body.target.copy()
            stayed = seconds
            for t in range(int(seconds / BRAIN_DT)):
                body.step(rng.uniform(-0.1, 0.1, J) if wobble else np.zeros(J))
                if stayed == seconds and not is_up(body):
                    stayed = (t + 1) * BRAIN_DT
            ups.append(is_up(body)); times.append(stayed); heights.append(float(body.d.qpos[2]))
        out[name] = {"up": ups, "stayed_s": times, "height_end": heights, "up_frac": float(np.mean(ups)),
                     "median_stayed_s": float(np.median(times))}
        log(f"G1 STAND {name} ({len(ups)} lives, {seconds:g} s, no brain): up at the end {np.mean(ups):.0%}; stayed up "
            f"median {np.median(times):.2f} s (min {np.min(times):.2f}); pelvis at the end median {np.median(heights):.2f} m")
    return out


def walk_feasibility(body, brain, log, rng, scales=(0.0, 1.0, 2.0, 4.0), samples=100, seconds=3.0):
    """Premise check for walking, no learning: random synapses in the walk-forward set only (the learned turning sets
    kept), of each size (x the map), asked to walk forward for 3 s; distance along the starting heading as a share of
    0.3 m/s x 3 s. A sign flip of a rhythm set is the same gait half a cycle later, so direction is not mirrored: report
    the signed forward share and how far it travels either way."""
    J = len(body.joint)
    brain.reflex_scale = max(float(np.percentile(np.abs(line_scores(brain)), 99)), 1e-12)
    map_size = float(np.linalg.norm(brain.map)) / brain.reflex_scale
    brain.reflex = SKILL_REFLEX
    settle_reflex(body, brain)
    kept = brain.Qc.copy()
    fwd = slice(2 * CPG_N, 3 * CPG_N)
    out = {}
    for sc in scales:
        sh, up = [], []
        for _ in range(samples if sc > 0 else 10):
            Qc = kept.copy()
            r_ = rng.standard_normal((len(brain.motor), CPG_N))
            Qc[:, fwd] = r_ * (sc * map_size / np.linalg.norm(r_)) if sc > 0 else 0.0
            brain.Qc = Qc
            body.reset("stand")
            brain.fresh()
            p0, y0 = body.d.qpos[:2].copy(), body.yaw()
            intend(body, brain, {brain.walk_line: WALK_ACC}, seconds)
            sh.append(float((body.d.qpos[:2] - p0) @ np.array([np.cos(y0), np.sin(y0)])) / (WALK_V * seconds))
            up.append(is_up(body))
        sh, up = np.array(sh), np.array(up)
        res = {"share_forward": sh.tolist(), "up": up.tolist(), "median": float(np.median(sh)),
               "best_forward_up": float((sh * up).max()), "best_either_up": float((np.abs(sh) * up).max()),
               "frac_fwd_ge_30_up": float(np.mean((sh >= 0.3) & up)), "frac_either_ge_30_up": float(np.mean((np.abs(sh) >= 0.3) & up)),
               "up_frac": float(up.mean())}
        out[f"{sc:g}"] = res
        log(f"WALK FEASIBILITY size {sc:g} of the map ({len(sh)} random draws in the walk-forward set): forward share median "
            f"{res['median']:+.0%}, best forward (up) {res['best_forward_up']:+.0%}, best either way (up) "
            f"{res['best_either_up']:.0%}; >=30% forward and up {res['frac_fwd_ge_30_up']:.0%}, >=30% either way and up "
            f"{res['frac_either_ge_30_up']:.0%}; still up {res['up_frac']:.0%}")
    brain.Qc = kept
    brain.reflex = 0.0
    return out


def cpg_feasibility(body, brain, log, rng, scales=(0.0, 0.01, 0.03, 0.1, 0.3, 0.5), samples=100, seconds=3.0):
    """Premise check, no learning: can the grown rhythm turn this body at all? For each size (norm of the rhythm-to-motor
    synapses as a share of the map's size; 0.5 is the cap learning is held to), draw random synapses, ask for a left turn
    for 3 s with the reflexes (x0.125) and skill synapses as learned, and measure the turn. The sign of random synapses
    is arbitrary (their mirror turns the other way), so |share| counts. The learned synapses are restored afterwards."""
    J = len(body.joint)
    brain.reflex_scale = max(float(np.percentile(np.abs(line_scores(brain)), 99)), 1e-12)
    map_size = float(np.linalg.norm(brain.map)) / brain.reflex_scale
    brain.reflex = SKILL_REFLEX
    settle_reflex(body, brain)
    kept = brain.Qc.copy()
    out = {}
    for sc in scales:
        rows = []
        for i in range(samples if sc > 0 else 10):
            Qc = rng.standard_normal(brain.Qc.shape)
            brain.Qc = Qc * (sc * map_size / np.linalg.norm(Qc)) if sc > 0 else np.zeros_like(Qc)
            body.reset("stand")
            brain.fresh()
            y0 = body.yaw()
            traj = intend(body, brain, {4 * J + 9: TURN_RATE}, seconds)
            dy = (body.yaw() - y0 + np.pi) % (2 * np.pi) - np.pi
            rows.append((float(dy / (TURN_RATE * seconds)), is_up(body), float(np.mean([t[7] for t in traj]))))
        sh = np.array([r[0] for r in rows])
        up = np.array([r[1] for r in rows])
        ok = np.abs(sh) * up
        res = {"share": sh.tolist(), "up": up.tolist(), "effort": [r[2] for r in rows],
               "median_abs": float(np.median(np.abs(sh))), "p90_abs_up": float(np.percentile(ok, 90)),
               "best_abs_up": float(ok.max()), "frac_ge_30_up": float(np.mean(ok >= 0.3)),
               "frac_ge_60_up": float(np.mean(ok >= 0.6)), "up_frac": float(up.mean())}
        out[f"{sc:g}"] = res
        log(f"CPG FEASIBILITY size {sc:g} of the map ({len(rows)} random draws): |turn| median {res['median_abs']:.0%}, "
            f"90th pct (up) {res['p90_abs_up']:.0%}, best (up) {res['best_abs_up']:.0%}; >=30% and up "
            f"{res['frac_ge_30_up']:.0%}, >=60% and up {res['frac_ge_60_up']:.0%}; still up {res['up_frac']:.0%}; "
            f"mean |command| {np.mean(res['effort']):.2f}")
    brain.Qc = kept
    brain.reflex = 0.0
    return out


def main():
    global BABBLE_TAU, REFLEX_GAINS, TILT_XY_TOO, CTX_WINNER, CTX_GROW, CTX_SIGMA, CTX_SLOW, CTX_LEARN, SKILL_RATE, SKILL_NOISE, SKILL_LIFT, SKILL_MIX, SKILL_PER_TASK, SKILL_LEVEL, SKILL_CURRICULUM, SKILL_MAX, SKILL_LEVEL_JOINTS, CPG_WP, SKILL_VARY, CPG_WP_MEAN, SKILL_NO_Q_TURN, STILL_TURN, ACTIVE_REST, STILL_TRIAL, GROW_NERVES
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0, help="minutes of babbling per dog")
    ap.add_argument("--dogs", type=int, default=3, help="independent dogs (same genome, own babbling)")
    ap.add_argument("--ground", action="store_true", help="free on the ground, with foot touch and an inner ear")
    ap.add_argument("--tone", action="store_true", help="muscle tone: the motor neurons move each joint's set point")
    ap.add_argument("--leak", action="store_true", help="with tone: set points relax back to the set-down posture")
    ap.add_argument("--leak_s", type=float, default=TONE_LEAK_S, help="relaxation time of the set points (s)")
    ap.add_argument("--set_speed", type=float, default=SET_SPEED,
                    help="fastest set-point speed (rad/s); with --leak the farthest reach from the set-down posture is "
                         "set_speed x leak_s, and the thighs' far ends are up to 3.6 rad away")
    ap.add_argument("--reflex_gains", default=",".join(f"{g:g}" for g in REFLEX_GAINS),
                    help="reflex strengths to test, comma-separated (0 = switched off)")
    ap.add_argument("--contexts", action="store_true", help="grow posture neurons, each with its own reflexes")
    ap.add_argument("--ctx_winner", action="store_true", help="only the best-matching posture neuron moves its posture")
    ap.add_argument("--ctx_learn", type=float, default=CTX_LEARN,
                    help="how fast a posture neuron's preferred posture follows what it feels, per step (0.01 = ~2 s: it "
                         "tracks the moment and no new ones grow; set-downs last 15 s)")
    ap.add_argument("--ctx_slow", action="store_true", help="read posture mostly from the slowly changing senses")
    ap.add_argument("--ctx_grow", type=float, default=CTX_GROW, help="grow a posture neuron this far from all others")
    ap.add_argument("--ctx_sigma", type=float, default=CTX_SIGMA, help="how broadly a posture neuron is tuned")
    ap.add_argument("--skill_rate", type=float, default=SKILL_RATE, help="skill synapse learning rate")
    ap.add_argument("--cpg_sizes", default="0,0.01,0.03,0.1,0.3,0.5",
                    help="--cpg_feasibility: sizes of the random rhythm-to-motor synapses, as a share of the map's size")
    ap.add_argument("--drift_diag", action="store_true",
                    help="measure the nothing-asked spin with skill synapses / reflexes / wobble / varied set-down off in turn")
    ap.add_argument("--eval_seed", type=int, default=7000,
                    help="--turn_eval: seed of its fresh lives (7000 was used to choose noqturn1; confirm on others)")
    ap.add_argument("--turn_eval", action="store_true",
                    help="with --cpg: turning on fresh lives, as learned vs rhythm synapses off vs random of the same size")
    ap.add_argument("--cpg_trace", action="store_true",
                    help="with --cpg: measure, stage by stage, where the rhythm's signal goes on its way to the joints")
    ap.add_argument("--cpg_feasibility", action="store_true",
                    help="with --cpg: premise check, can random rhythm-to-motor synapses of each size turn the body at all?")
    ap.add_argument("--g1_balance", type=float, default=0.0,
                    help="with --grow_g1: minutes of balance practice in the G1 (reward, fading parent's help)")
    ap.add_argument("--g1_support", action="store_true",
                    help="with --grow_g1: how much of a parent's help the G1 needs to stay up with the brain on")
    ap.add_argument("--g1_reflex", action="store_true",
                    help="with --grow_g1: can the brain's own reflexes from its G1 map keep the G1 standing (no hands)?")
    ap.add_argument("--g1_babble", type=float, default=0.0,
                    help="with --grow_g1: minutes of babbling in the G1 (held up by a parent's hands)")
    ap.add_argument("--grow_g1", action="store_true",
                    help="grow the G1 humanoid's own sensory and motor neurons by copying (the dog stays unchanged)")
    ap.add_argument("--body", default="go2", choices=("go2", "g1"), help="which body: the Go2 dog or the G1 humanoid")
    ap.add_argument("--g1_stand", action="store_true", help="physics premise, no brain: can the G1 stay up with joints held?")
    ap.add_argument("--walk_feasibility", action="store_true",
                    help="with --walk: premise check, can random walk-forward rhythm synapses move the body forward at all?")
    ap.add_argument("--walk", action="store_true",
                    help="with --cpg --accel: walk forward/back on request (rhythm synapses per direction, felt speed)")
    ap.add_argument("--accel", action="store_true",
                    help="on the ground: the IMU's accelerometer as three appended nerves (resumed brains grow them)")
    ap.add_argument("--still_trial", action="store_true",
                    help="still trials: skill synapses learn at trial end from the felt turning over the trial plus level")
    ap.add_argument("--active_rest", action="store_true",
                    help="record the at-rest senses while standing with the brain on (reflexes and skill synapses)")
    ap.add_argument("--still_turn", action="store_true",
                    help="the 'still' task also watches the turning sense (body z): staying still means not rotating")
    ap.add_argument("--no_q_in_turn", action="store_true",
                    help="the skill synapses do not learn on turn trials (the rhythm synapses learn turning)")
    ap.add_argument("--cpg_wp_mean", action="store_true",
                    help="with --cpg_wp: a turn trial's dopamine is the felt turning over the whole trial (plus level)")
    ap.add_argument("--skill_vary", action="store_true",
                    help="skill practice set down a little differently each trial and with the +-0.1 motor wobble")
    ap.add_argument("--cpg_wp", action="store_true",
                    help="with --cpg: learn the rhythm synapses by reward-gated synaptic noise on turn trials")
    ap.add_argument("--cpg", action="store_true", help="grow a 4-neuron pattern generator driven by turn requests")
    ap.add_argument("--rhythm", action="store_true", help="measure: does the brain make rhythmic motor output (adaptation sweep)?")
    ap.add_argument("--no_level_joints", action="store_true", help="joint-move trials do not count the stay-level term")
    ap.add_argument("--skill_max", type=float, default=SKILL_MAX,
                    help="cap on the skill synapses' size, as a share of the twitch map's (they sat at 0.25 in steps1)")
    ap.add_argument("--skill_curriculum", action="store_true", help="weight and lift requests start small, grow per foot")
    ap.add_argument("--skill_level", action="store_true", help="requests also mean: otherwise stay level")
    ap.add_argument("--skill_per_task", action="store_true", help="dopamine relative to what is usual for the exact request")
    ap.add_argument("--skill_mix", default="", help="practice task mix, comma-separated (tilt,weight,lift,joint,still)")
    ap.add_argument("--feasibility", action="store_true", help="physics only: can this body shift weight / lift each paw?")
    ap.add_argument("--resume", action="store_true",
                    help="continue the saved brain_dogN.npz from an attached kernel output (skips babbling)")
    ap.add_argument("--skill_lift", action="store_true", help="also practise lifting a paw; equal task mix")
    ap.add_argument("--skill_noise", type=float, default=SKILL_NOISE,
                    help="motor variation in skill practice, share of babbling's (0.3 at 9 rad/s moves a set point "
                         "~0.45 rad and toppled it in 77% of trials)")
    ap.add_argument("--skills", type=float, default=0.0, help="minutes of reward-based skill practice (tested before and after)")
    ap.add_argument("--held_babble", action="store_true", help="babble held standing by a parent in half the set-downs")
    ap.add_argument("--tilt_xy", action="store_true", help="diagnosis: also ask tilts on the two directional lines only")
    ap.add_argument("--babble_tau", type=float, default=BABBLE_TAU,
                    help="how long a spontaneous burst lasts (s); infants' general movements are slow sweeps of seconds")
    ap.add_argument("--reflex", action="store_true", help="after babbling, test with the brain's own reflexes only")
    ap.add_argument("--reflex_diag", action="store_true", help="after babbling: which reflexes knock it over, and how")
    ap.add_argument("--fair_dopamine", action="store_true", help="practice: dopamine relative to the usual settling; synaptic scaling")
    ap.add_argument("--support", action="store_true", help="practice with a parent's hands on the trunk, fading to none")
    ap.add_argument("--practice", type=float, default=0.0, help="minutes of standing practice (reflexes learn) before the tests")
    ap.add_argument("--diagnose", action="store_true", help="on the ground: record posture, the map, and the physical truth")
    ap.add_argument("--out", default="/kaggle/working" if Path("/kaggle/working").exists() else "results/babble")
    args = ap.parse_args([]) if JOB_ARGS else ap.parse_args()
    for k, v in JOB_ARGS.items():
        setattr(args, k, v)
    BABBLE_TAU = float(args.babble_tau)
    REFLEX_GAINS = tuple(float(g) for g in str(args.reflex_gains).split(","))
    TILT_XY_TOO = bool(args.tilt_xy)
    CTX_WINNER = bool(args.ctx_winner)
    CTX_GROW, CTX_SIGMA = float(args.ctx_grow), float(args.ctx_sigma)
    CTX_SLOW = bool(args.ctx_slow)
    CTX_LEARN = float(args.ctx_learn)
    SKILL_RATE = float(args.skill_rate)
    SKILL_NOISE = float(args.skill_noise)
    SKILL_LIFT = bool(args.skill_lift)
    SKILL_MIX = tuple(k for k in str(args.skill_mix).split(",") if k)
    SKILL_PER_TASK = bool(args.skill_per_task)
    SKILL_LEVEL = bool(args.skill_level)
    SKILL_CURRICULUM = bool(args.skill_curriculum)
    SKILL_MAX = float(args.skill_max)
    SKILL_LEVEL_JOINTS = not bool(args.no_level_joints)
    CPG_WP = bool(args.cpg_wp)
    SKILL_VARY = bool(args.skill_vary)
    CPG_WP_MEAN = bool(args.cpg_wp_mean)
    SKILL_NO_Q_TURN = bool(args.no_q_in_turn)
    STILL_TURN = bool(args.still_turn)
    ACTIVE_REST = bool(args.active_rest)
    STILL_TRIAL = bool(args.still_trial)
    GROW_NERVES = bool(args.accel)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ensure_packages()
    import tempfile
    lines_out = []

    def log(msg):
        print(msg, flush=True)
        lines_out.append(msg)

    if args.body == "g1" and not (args.ground and args.tone):
        raise SystemExit("--body g1 needs --ground and --tone (its joints are position servos)")
    body = Body(fetch_model(Path(tempfile.gettempdir()), "unitree_g1" if args.body == "g1" else "unitree_go2"),
                held=not args.ground, tone=bool(args.tone), leak=bool(args.leak), accel=bool(args.accel) and bool(args.ground))
    body.leak_s = float(args.leak_s)
    body.set_speed = float(args.set_speed)
    core = load_core()
    where = f"held at {HELD_AT} m like a baby" if body.held else (
        f"free on the ground, with foot touch and an inner ear; set down somewhere new every {REPOSITION_S:g} s")
    if body.tone:
        where += f"; muscle tone (stiffness {TONE_KP:g} N m/rad, damping {TONE_KD:g}), the brain moves the set points"
        if body.leak:
            where += (f", which relax back to the set-down posture over {body.leak_s:g} s (set points move up to "
                      f"{body.set_speed:g} rad/s, reach up to {body.set_speed * body.leak_s:g} rad)")
    log(f"body: {'G1' if body.kind == 'g1' else 'Go2'}, {len(body.joint)} joints ({', '.join(body.names)}); {where}; {len(body.feel())} nerves; "
        f"physics {body.subs} steps per brain step")
    log(f"brain: fly connectome, {len(core['role'])} neurons, {len(core['pre'])} connections; it is told nothing about the body")
    if args.g1_stand:
        res = g1_stand_check(body, log, np.random.default_rng(8))
        (out / "g1_stand.json").write_text(json.dumps(res, indent=1))
        (out / "babble_log.txt").write_text("\n".join(lines_out))
        print("DONE", flush=True)
        return
    if args.feasibility:
        feas = feasibility(body, log, np.random.default_rng(7))
        (out / "feasibility.json").write_text(json.dumps(feas, indent=1))
        (out / "babble_log.txt").write_text("\n".join(lines_out))
        print("DONE", flush=True)
        return
    results = []
    g1_body = None
    for dog in range(args.dogs):
        brain = Brain(core, len(body.feel()), len(body.joint), np.random.default_rng(GENOME))     # same genome
        if body.tone:
            brain.motor_baseline = np.zeros(len(body.joint))
        if args.contexts:
            brain.ctx_init()
        label = f"dog {dog + 1}/{args.dogs}"
        body.reset()
        before = None if args.reflex else control_test(body, brain, joint_scores(brain, body), log, label + ", newborn (map empty)")
        ground_before = ground_tests(body, brain, log, label + ", newborn (map empty)", None) if args.ground and not args.reflex else None
        body.reset()
        diag = None
        if args.ground and args.diagnose:
            diag = {"feet_loaded": [0] * 5, "standing": 0, "steps": 0, "up_since": None, "stand_lasted_s": [],
                    "abs_command": np.zeros(len(body.joint)),
                    "cols": brain.fibre_col[4 * len(body.joint):4 * len(body.joint) + 7].ravel(),
                    "part": np.zeros((2, len(brain.motor), 7 * NERVE_FIBRES))}
        if args.resume:
            found = sorted(Path("/kaggle/input").glob(f"**/brain_dog{dog + 1}.npz")) if Path("/kaggle/input").exists() else []
            if not found:
                raise SystemExit(f"--resume: no brain_dog{dog + 1}.npz in the attached inputs")
            load_brain(brain, found[0])
            log(f"[{label}] continuing the saved brain {found[0]} (it has felt {int(brain.seen)} brain steps; skill synapses "
                f"{'present' if brain.Q is not None else 'none'})")
            explored = {"coverage": [0.0] * len(body.joint), "seen_lo": body.lo.tolist(), "seen_hi": body.hi.tolist(),
                        "limit_felt_steps": [[0, 0]] * len(body.joint), "resumed": True}
        else:
            explored = babble(body, brain, args.minutes, np.random.default_rng(1000 + dog), log, label, diag,
                              held=bool(args.held_babble))
        if args.grow_g1:                                                  # the humanoid arrives: grow its periphery
            g1 = g1_body or Body(fetch_model(Path(tempfile.gettempdir()), "unitree_g1"), held=False, tone=True,
                                 leak=bool(args.leak), accel=bool(args.accel))
            g1_body = g1
            n_s, n_m, n_syn = brain.grow_body("g1", len(g1.feel()), len(g1.joint), np.random.default_rng(GENOME_G1))
            log(f"[{label}] grew the G1's periphery by copying: {n_s} sensory neurons ({len(g1.feel())} nerves x "
                f"{NERVE_FIBRES}), {n_m} motor neurons ({len(g1.joint)} motors), {n_syn} copied synapses; brain now "
                f"{brain.N} neurons. The dog's neurons, wiring and learned parts are unchanged.")
            if args.resume and load_body_parts(brain, found[0], "g1"):
                log(f"[{label}] restored what it learned in the G1 from {found[0]}")
            if args.g1_babble:                                            # the same brain, plugged into the humanoid
                brain.use("g1")
                brain.adapting, brain.adapt_var, brain.reflex = True, True, 0.0
                g1_explored = babble(g1, brain, args.g1_babble, np.random.default_rng(1500 + dog), log, label + ", in the G1")
                brain.adapting = False
                brain.use("go2")
                log(f"[{label}] back in the dog after babbling in the G1 (the G1's map and senses learned; the dog's untouched)")
            if args.g1_balance:                                           # learn to balance, hands fading
                brain.use("g1")
                practice_g1_balance(g1, brain, args.g1_balance, np.random.default_rng(1800 + dog), log, label + ", in the G1")
                brain.use("go2")
            if args.g1_support:                                           # how much help does it need?
                brain.use("g1")
                g1_support_check(g1, brain, log, np.random.default_rng(1700 + dog))
                brain.use("go2")
            if args.g1_reflex:                                            # can its new reflexes hold the G1 up?
                brain.use("g1")
                g1_reflex_stand(g1, brain, log, np.random.default_rng(1600 + dog))
                brain.use("go2")
        diag_report = diagnose(body, brain, diag, log, label, out, dog + 1) if diag is not None else None
        ctx_report = ctx_diag(body, brain, log, label) if args.contexts else None
        scores = joint_scores(brain, body)
        after = None if args.reflex else control_test(body, brain, scores, log, label + f", after {args.minutes:g} min of babbling")
        ground_after = ground_tests(body, brain, log, label + f", after {args.minutes:g} min of babbling", None) if args.ground and not args.reflex else None
        practised = None
        if args.cpg:
            if args.walk:
                if not body.accel:
                    raise SystemExit("--walk needs --accel (the request and its felt speed live on the accelerometer)")
                brain.walk_line = 4 * len(body.joint) + 10
            brain.cpg_init(np.random.default_rng(5000 + dog))
            brain.turn_line = 4 * len(body.joint) + 9
            rate, share, silent = cpg_check(brain, log)
            if not (1.0 <= rate <= 15.0 and silent < 0.2):
                log(f"[{label}] the pattern generator is not a usable rhythm ({rate:.2f} hand-overs/s, silent {silent:.0%}; "
                    f"needs 1-15/s and < 20% silent); skipping its practice")
                args.skills = 0.0
        if args.drift_diag:
            practised = {"drift_diag": drift_diag(body, brain, log, np.random.default_rng(int(args.eval_seed) + dog))}
            args.skills = 0.0
        if args.cpg and args.turn_eval:
            practised = {"turn_eval": turn_eval(body, brain, log, np.random.default_rng(int(args.eval_seed) + dog))}
            args.skills = 0.0
        if args.cpg and args.cpg_trace:
            practised = {"cpg_trace": {f"{sz:g}": cpg_trace(body, brain, log, np.random.default_rng(6100 + dog), size=sz)
                                       for sz in (0.1, 0.5)}}
            args.skills = 0.0
        if args.cpg and args.walk and args.walk_feasibility:
            practised = {"walk_feasibility": walk_feasibility(body, brain, log, np.random.default_rng(6200 + dog))}
            args.skills = 0.0
        if args.cpg and args.cpg_feasibility:
            practised = {"cpg_feasibility": cpg_feasibility(body, brain, log, np.random.default_rng(6000 + dog),
                                                            scales=tuple(float(x) for x in str(args.cpg_sizes).split(",")))}
            args.skills = 0.0
        if args.rhythm:
            practised = {"rhythm": rhythm_diag(body, brain, log, np.random.default_rng(4000 + dog))}
        if args.skills:
            before_skills = reflex_tests(body, brain, log, label + ", before skill practice")
            practised = {"before": before_skills,
                         "history": practice_skills(body, brain, args.skills, np.random.default_rng(3000 + dog), log, label)}
        if args.practice:
            practised = practice(body, brain, args.practice, np.random.default_rng(2000 + dog), log, label,
                                 support=bool(args.support), fair=bool(args.fair_dopamine))
        if args.reflex_diag:
            reflex = {f"x{g:g}": reflex_diag(body, brain, log, label, g) for g in (0.5, 1.0)}
            if args.reflex:
                reflex["tests"] = reflex_tests(body, brain, log, label + f", after babbling and {args.practice:g} min of practice")
        else:
            reflex = reflex_tests(body, brain, log, label + f", after {args.minutes:g} min of babbling"
                                  + (f" and {args.practice:g} min of practice" if args.practice else "")
                                  + (f" and {args.skills:g} min of skill practice" if args.skills else "")) if args.reflex else None
        acc = map_accuracy(brain, body)
        per_joint = []
        for j, name in enumerate(body.names):
            mine = brain.drives == j
            best = np.abs(scores[mine]).argmax(1)
            per_joint.append({"joint": name, "range_rad": [float(body.lo[j]), float(body.hi[j])],
                              "explored_rad": [explored["seen_lo"][j], explored["seen_hi"][j]],
                              "coverage": explored["coverage"][j], "limits_felt_steps": explored["limit_felt_steps"][j],
                              "its_neurons_mapped_to_it": float(np.mean(best == j))})
            log(f"  {name:9s} range {body.lo[j]:+.2f}..{body.hi[j]:+.2f} rad; babbling reached {explored['seen_lo'][j]:+.2f}.."
                f"{explored['seen_hi'][j]:+.2f} ({explored['coverage'][j]:.0%}); limits felt (low/high, steps) "
                f"{explored['limit_felt_steps'][j][0]}/{explored['limit_felt_steps'][j][1]}; its motor neurons mapped to it "
                f"{np.mean(best == j):.0%}")
        save_brain(brain, out / f"brain_dog{dog + 1}.npz")
        log(f"[{label}] saved this brain as brain_dog{dog + 1}.npz")
        results.append({"dog": dog + 1, "map": acc, "per_joint": per_joint, "control_newborn": before, "control_after": after,
                        "ground_newborn": ground_before, "ground_after": ground_after, "diagnosis": diag_report,
                        "reflex": reflex, "practice": practised, "contexts": ctx_report})
    reach = lambda rows: float(np.mean([np.clip(r_["reach"], -1, 1) for r_ in rows]))
    log("VERDICT babbling: " + "; ".join(
        f"dog {r_['dog']}: body map right joint {r_['map']['joint']:.0%} (direction too {r_['map']['both']:.0%}), "
        + (f"control {reach(r_['control_newborn']):.0%} newborn -> {reach(r_['control_after']):.0%} after"
           if r_["control_after"] else "")
        for r_ in results) + f" (chance for the map: {1 / len(body.joint):.0%})")
    (out / "babble.json").write_text(json.dumps({"minutes": args.minutes, "genome": GENOME, "dogs": results,
                                                 "settings": {k: globals()[k] for k in ("BRAIN_DT", "ALPHA", "GAIN",
                                                              "BABBLE_TAU", "BABBLE_SIZE", "NERVE_FIBRES", "TRACE",
                                                              "HEBB_RATE", "HELD_AT", "REPOSITION_S", "FOOT_LOADED_N",
                                                              "STANDING_HEIGHT", "TONE_KP", "TONE_KD", "SET_SPEED",
                                                              "MOTOR_TAU", "REFLEX_GAINS", "SETTLE_S", "PRACTICE_RATE",
                                                              "PRACTICE_TRACE", "PRACTICE_NOISE", "EPISODE_S",
                                                              "PRACTICE_GAIN0", "PRACTICE_GAIN_STEP",
                                                              "SUPPORT_FADE_SHARE", "SUPPORT_SAG", "TONE_LEAK_S")} | {"tone": bool(args.tone),
                                                              "ground": bool(args.ground), "reflex": bool(args.reflex),
                                                              "leak": bool(args.leak),
                                                              "fair_dopamine": bool(args.fair_dopamine),
                                                              "leak_s": float(args.leak_s),
                                                              "set_speed": float(args.set_speed),
                                                              "held_babble": bool(args.held_babble), "skills": float(args.skills),
                                                              "SKILL_REFLEX": SKILL_REFLEX, "SKILL_NOISE": SKILL_NOISE,
                                                              "SKILL_RATE": SKILL_RATE, "SKILL_TRACE": SKILL_TRACE, "SKILL_MAX": SKILL_MAX,
                                                              "SKILL_LIFT": SKILL_LIFT, "SKILL_MIX": list(SKILL_MIX), "resume": bool(args.resume),
                                                              "SKILL_PER_TASK": SKILL_PER_TASK, "SKILL_LEVEL": SKILL_LEVEL,
                                                              "SKILL_CURRICULUM": SKILL_CURRICULUM,
                                                              "SKILL_MAX": SKILL_MAX, "SKILL_LEVEL_JOINTS": SKILL_LEVEL_JOINTS,
                                                              "contexts": bool(args.contexts), "ctx_winner": bool(args.ctx_winner),
                                                              "CTX_GROW": CTX_GROW, "CTX_SIGMA": CTX_SIGMA, "ctx_slow": CTX_SLOW, "CTX_LEARN": CTX_LEARN,
                                                              "HELD_BABBLE_SHARE": HELD_BABBLE_SHARE,
                                                              "HELD_BABBLE_HELP": HELD_BABBLE_HELP}}, indent=1))
    (out / "babble_log.txt").write_text("\n".join(lines_out))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
