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
STANDING_HEIGHT = 0.16   # m: trunk above this counts as up on its legs (measured: lying 0.07-0.14; with tone it stands at 0.19)
# --tone: muscle tone. Each joint springs toward a set point (the Go2's standard stiffness and damping); the motor
# neurons move the set point; when the dog is set down, the set points take the posture it is in.
TONE_KP = 20.0           # N m / rad
TONE_KD = 0.5            # N m s / rad
SET_SPEED = 4.0          # rad/s: how fast a fully driven motor pool moves its joint's set point
MOTOR_TAU = 3.0          # s: the motor pool adapts away its own steady resting drive, as the senses do
# --reflex: the brain's own feedback. While babbling, synapses from sensory back to motor neurons are tuned by the
# twitches themselves (as in newborn rats' spinal reflexes, Petersson et al. 2003): a link grows when the motor
# neuron's activity is followed by a FALL in that sensory neuron, so the finished reflex opposes change. They stay
# silent while babbling (as reflexes are tuned during sleep twitches) and act afterwards. They are the body map's
# reverse direction with the opposite sign: the same co-activity, learned once.
REFLEX_GAINS = (0.0, 0.5, 1.0, 2.0, 4.0)   # reflex strength, in units of the map's own scale (0 = switched off)
SETTLE_S = 30.0          # s: before tests, set down standing at rest while senses and motor pools settle (~3 time constants)


def ensure_packages():
    try:
        import mujoco  # noqa: F401
        import scipy  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "mujoco==3.14.0", "scipy"])


def fetch_go2(root: Path) -> Path:
    d = root / "mujoco_menagerie"
    if not (d / "unitree_go2" / "scene.xml").exists():
        run = lambda *a: subprocess.check_call(list(a))
        run("git", "clone", "-q", "--filter=blob:none", "--no-checkout", MENAGERIE_URL, str(d))
        run("git", "-C", str(d), "sparse-checkout", "set", "unitree_go2")
        run("git", "-C", str(d), "checkout", "-q", MENAGERIE_COMMIT)
    return d / "unitree_go2"


def load_core():
    found = sorted(Path("/kaggle/input").glob("**/core_v2.npz")) if Path("/kaggle/input").exists() else []
    path = found[0] if found else Path(__file__).resolve().parent.parent / "data" / "malecns" / "core_v2.npz"
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


class Body:
    """The Go2: held in the air by its trunk, or free on the ground. Torque motors; every joint has a hard limit.
    On the ground it also has touch in each foot and an inner ear (which way gravity pulls, and how it is turning)."""

    def __init__(self, go2: Path, held: bool = True, tone: bool = False):
        import mujoco
        self.mj = mujoco
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
        self.base = self.m.body("base").id
        self.feet = [self.m.geom(f).id for f in ("FL", "FR", "RL", "RR")]
        self._force = np.zeros(6)
        self.reset()

    def reset(self, pose: str = "stand", rng=None):
        """Held: back in the parent's hands. On the ground: set down standing, on its belly, or on one side."""
        self.mj.mj_resetDataKeyframe(self.m, self.d, self.m.keyframe("home").id)
        if self.held:
            self.hold()
            self.mj.mj_forward(self.m, self.d)
            self.target = self.angles()
            return
        q = self.d.qpos
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
            return
        for _ in range(int(0.5 / self.m.opt.timestep)):                 # let it settle onto the floor, limp
            self.mj.mj_step(self.m, self.d)
        self.target = self.angles()                                      # the muscles take the posture it was set in

    def hold(self):
        self.d.qpos[0:3] = [0.0, 0.0, HELD_AT]
        self.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.d.qvel[0:6] = 0.0

    def step(self, out):
        """out: each motor pool's output in [-1, 1]. Without tone it is a torque (share of the motor's maximum); with
        tone it moves the joint's set point, and the muscle pulls toward it."""
        if self.tone:
            self.target = np.clip(self.target + SET_SPEED * BRAIN_DT * out, self.lo - 0.1, self.hi + 0.1)
        else:
            self.d.ctrl[:] = np.clip(self.tmax * out, -self.tmax, self.tmax)
        for _ in range(self.subs):
            if self.tone:
                pull = TONE_KP * (self.target - self.d.qpos[self.qadr]) - TONE_KD * self.d.qvel[self.vadr]
                self.d.ctrl[:] = np.clip(pull, -self.tmax, self.tmax)
            self.mj.mj_step(self.m, self.d)
            if self.held:
                self.hold()

    def angles(self):
        return self.d.qpos[self.qadr].copy()

    def feel(self):
        """What the joints' own sense organs report: angle, speed, effort, and the push-back at a limit. On the
        ground also: the pressure under each foot, and the inner ear."""
        joints = [self.d.qpos[self.qadr], self.d.qvel[self.vadr], self.d.actuator_force, self.d.qfrc_constraint[self.vadr]]
        if self.held:
            return np.concatenate(joints)
        return np.concatenate([*joints, self.touch(), self.gravity(), self.d.qvel[3:6]])

    def touch(self):
        f = np.zeros(4)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            for k, g in enumerate(self.feet):
                if c.geom1 == g or c.geom2 == g:
                    self.mj.mj_contactForce(self.m, self.d, i, self._force)
                    f[k] += abs(self._force[0])
        return f

    def gravity(self):
        """Which way gravity pulls, in the trunk's own frame: (0, 0, -1) when upright."""
        return self.d.xmat[self.base].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])

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

    def zscore(self, felt):
        return np.clip((felt - self.mean) / np.sqrt(self.var + 1e-6), -4.0, 4.0)

    def step(self, felt, motor_drive):
        if self.adapting:
            self.seen += 1
            a = max(1.0 / self.seen, 0.002)                              # adapts within ~10 s
            self.mean += a * (felt - self.mean)
            if self.adapt_var:
                self.var += a * ((felt - self.mean) ** 2 - self.var)
        zf = self.zscore(felt) + self.intention
        I = np.zeros(self.N)
        np.add.at(I, self.fibre_neuron.ravel(), (self.fibre_gain * zf[:, None]).ravel())
        I[self.motor] += motor_drive
        if self.reflex and self.rest_sense is not None:      # its own reflexes: oppose what changed since rest
            change = self.rates()[self.sense] - self.rest_sense
            if self.reflex_cols is None:
                I[self.motor] -= self.reflex / self.reflex_scale * (self.map @ change)
            else:
                c = self.reflex_cols
                I[self.motor] -= self.reflex / self.reflex_scale * (self.map[:, c] @ change[c])
        self.v += ALPHA * (-self.v + self.W @ self.rates() + I)
        return self.rates()

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

    def command(self, r):
        net = np.bincount(self.drives, weights=self.push * r[self.motor], minlength=self.n_motors)
        x = 1.5 * net / np.sqrt(np.bincount(self.drives, minlength=self.n_motors) + 1.0)
        self.last_raw = x
        if self.motor_baseline is not None:
            self.motor_baseline += BRAIN_DT / MOTOR_TAU * (x - self.motor_baseline)
            x = x - self.motor_baseline
        return np.tanh(x)


def babble(body, brain, minutes, rng, log, label, diag=None):
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
            pose = POSES[rng.integers(len(POSES))]
            body.reset(pose, rng)
            trace[:] = 0.0                                                # being moved is not something it did
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
        brain.map += HEBB_RATE * np.outer(trace, s - s_prev)            # activity, then what was felt shortly after
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


def line_scores(brain):
    """[motor neurons, nerves]: each motor neuron's learned link to each nerve (sign: which way it moves it)."""
    return np.einsum("mlf,lf->ml", brain.map[:, brain.fibre_col], brain.fibre_gain)


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
        + (f" (with tone: set points move {SET_SPEED * diag['abs_command'].mean() / n:.2f} rad/s on average)" if body.tone else ""))
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


def intend(body, brain, targets, seconds):
    """The test only names what the dog should come to feel: {nerve: value, in that nerve's own units}. It enters once,
    at the start, as a shift of that nerve's felt reference (it feels as if it were on the other side of the target).
    The brain's own reflexes move the body until the shift is cancelled. The test drives no motor neuron."""
    for l, g in targets.items():
        brain.intention[l] = -(g - brain.mean[l]) / np.sqrt(brain.var[l] + 1e-6)
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
    return bool(-body.gravity()[2] > 0.8 and body.d.qpos[2] > STANDING_HEIGHT and (body.touch() > FOOT_LOADED_N).sum() >= 3)


def reflex_tests(body, brain, log, label):
    """Everything by its own reflexes, at each strength (x0 = reflexes switched off, the baseline)."""
    J = len(body.joint)
    brain.reflex_scale = max(float(np.percentile(np.abs(line_scores(brain)), 99)), 1e-12)
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
        log(f"[{tag}] tilt 0.25 rad as asked (roll left/right, nose down/up): "
            + ", ".join(f"{a:+.0%}{'' if u else ' (fell)'}" for a, u in got) + " of the asked tilt")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0, help="minutes of babbling per dog")
    ap.add_argument("--dogs", type=int, default=3, help="independent dogs (same genome, own babbling)")
    ap.add_argument("--ground", action="store_true", help="free on the ground, with foot touch and an inner ear")
    ap.add_argument("--tone", action="store_true", help="muscle tone: the motor neurons move each joint's set point")
    ap.add_argument("--reflex", action="store_true", help="after babbling, test with the brain's own reflexes only")
    ap.add_argument("--reflex_diag", action="store_true", help="after babbling: which reflexes knock it over, and how")
    ap.add_argument("--diagnose", action="store_true", help="on the ground: record posture, the map, and the physical truth")
    ap.add_argument("--out", default="/kaggle/working" if Path("/kaggle/working").exists() else "results/babble")
    args = ap.parse_args([]) if JOB_ARGS else ap.parse_args()
    for k, v in JOB_ARGS.items():
        setattr(args, k, v)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ensure_packages()
    import tempfile
    lines_out = []

    def log(msg):
        print(msg, flush=True)
        lines_out.append(msg)

    body = Body(fetch_go2(Path(tempfile.gettempdir())), held=not args.ground, tone=bool(args.tone))
    core = load_core()
    where = f"held at {HELD_AT} m like a baby" if body.held else (
        f"free on the ground, with foot touch and an inner ear; set down somewhere new every {REPOSITION_S:g} s")
    if body.tone:
        where += f"; muscle tone (stiffness {TONE_KP:g} N m/rad, damping {TONE_KD:g}), the brain moves the set points"
    log(f"body: Go2, {len(body.joint)} joints ({', '.join(body.names)}); {where}; {len(body.feel())} nerves; "
        f"physics {body.subs} steps per brain step")
    log(f"brain: fly connectome, {len(core['role'])} neurons, {len(core['pre'])} connections; it is told nothing about the body")
    results = []
    for dog in range(args.dogs):
        brain = Brain(core, len(body.feel()), len(body.joint), np.random.default_rng(GENOME))     # same genome
        if body.tone:
            brain.motor_baseline = np.zeros(len(body.joint))
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
        explored = babble(body, brain, args.minutes, np.random.default_rng(1000 + dog), log, label, diag)
        diag_report = diagnose(body, brain, diag, log, label, out, dog + 1) if diag is not None else None
        scores = joint_scores(brain, body)
        after = None if args.reflex else control_test(body, brain, scores, log, label + f", after {args.minutes:g} min of babbling")
        ground_after = ground_tests(body, brain, log, label + f", after {args.minutes:g} min of babbling", None) if args.ground and not args.reflex else None
        if args.reflex_diag:
            reflex = {f"x{g:g}": reflex_diag(body, brain, log, label, g) for g in (0.5, 1.0)}
        else:
            reflex = reflex_tests(body, brain, log, label + f", after {args.minutes:g} min of babbling") if args.reflex else None
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
        results.append({"dog": dog + 1, "map": acc, "per_joint": per_joint, "control_newborn": before, "control_after": after,
                        "ground_newborn": ground_before, "ground_after": ground_after, "diagnosis": diag_report,
                        "reflex": reflex})
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
                                                              "MOTOR_TAU", "REFLEX_GAINS", "SETTLE_S")} | {"tone": bool(args.tone),
                                                              "ground": bool(args.ground), "reflex": bool(args.reflex)}}, indent=1))
    (out / "babble_log.txt").write_text("\n".join(lines_out))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
