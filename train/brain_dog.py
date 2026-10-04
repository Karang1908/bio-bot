"""Stage 1: the fly-wired brain discovers the Go2 dog's body by itself.

Every *life*, the brain is plugged into the dog as an unknown device:
  * inputs  whatever the dog's onboard sensors produce (joint encoders, IMU), handed over as a set
            of numbers whose size and meaning it is never told: shuffled, sign-flipped, rescaled and
            offset differently every life;
  * outputs the dog's actuators, a set of unknown size: every life each output line gets a fresh
            random tag (meaningless) and a random polarity. Like real robot firmware, a motor given
            no command holds where it woke up; a command in [-1, 1] moves it by up to half its range;
  * goal    only meaning: "move at (vx, vy) and turn at wz" (all zero = stand), scored by meaning
            code on the body's centre of mass and posture, the same for any body.

The brain's own machinery, identical for any body, with no parameter whose size depends on how
many lines the body has:
  * the MaleCNS sensorimotor core (brain/core.py): 2,500 cell-type units, 167,630 real connections;
    learned per-unit time constants and biases and per-connection gains;
  * adaptation: every input line is normalised to its own running range (built from nothing each life);
  * a causal self-map: running correlation between each output's recent change and each input's
    change right after it ("what changed after I did that"), built from zero each life;
  * attention that carries any number of lines into the sense neurons and out of the motor neurons.
If the dog falls it is put back; the brain keeps what it has discovered until the life ends.

It is taught the way a child is (the teacher, below, is the AI layer; the brain never sees inside it):
  * show    the teacher knows what walking is. It moves the dog's legs through a walk (a trot) so the
            brain feels it on its own anonymous lines, and after every step tells it what its lines
            should have done;
  * assist  it holds the trunk up like a hand under the belly, so trying costs no falls;
  * let go  it takes its hands off the legs, then the harness, then stops correcting, each only once
            the dog keeps walking with less help. What stays for good is the meaning reward.
The teacher practises its own walk on the body first (a small search over rhythm, stride and lift) and
refuses to teach if it cannot walk the body itself.

Learning: recurrent PPO with truncated backprop through time, data-parallel over the GPUs, plus the
teacher's corrections as an imitation loss that fades. The critic sees the true simulator state
(training scaffolding, stripped later).

  Kaggle:  scripts/kaggle_job.py push brain_dog --embed data/malecns/core_v1.npz
           scripts/kaggle_job.py push brain_dog --flag teacher_only     (the teacher practises and is filmed)
  local:   python train/brain_dog.py --selftest     (permutation checks, then a tiny training run)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

JOB_ARGS: dict = {}   # filled in by scripts/kaggle_job.py
EMBEDDED: dict = {}   # data files shipped inside the job (base64), filled in by scripts/kaggle_job.py

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_COMMIT = "367e3d9884401dcf6f9c27fa69f118992539039f"  # same as scripts/fetch_assets.py
PINS = {"jax": "0.7.2", "jaxlib": "0.7.2", "mujoco": "3.14.0", "mujoco-mjx": "3.14.0"}   # a set known to work together
REPO = Path(__file__).resolve().parent.parent

CTRL_DT, SIM_DT = 0.02, 0.004     # the brain acts at 50 Hz; physics runs at 250 Hz
TICK_DT = 0.02                    # one brain tick per control step
LIFE_STEPS = 1500                 # 30 s per life
LAGS = (1, 3)                     # self-map: input change 1 and 3 control steps after an output change
SELF_MAP_RATE = 0.01              # ~2 s memory
ONBOARD = ("jointpos", "jointvel", "gyro", "accelerometer", "framequat")   # what the robot itself measures
FEET = ["FL", "FR", "RL", "RR"]
FOOT_UP = 0.03                    # a foot centre this high (1.25 cm clear of the floor) is off the ground
CMD_MAX = [0.6, 0.0, 0.8]         # |vx|, |vy| m/s, |wz| rad/s (no side-stepping lesson yet)
D, DT, HEADS = 32, 16, 4          # token width, output-tag width, readout heads
NOISE_RHO = 0.9                   # exploration noise persists ~0.2 s, so a limb actually swings
STD_FLOOR = -1.9                  # exploration never drops below std 0.15
GAIN0 = 2.0                       # starting gain on the averaged connectome input (learned)
GOAL_GATE = 5e-4                  # training only starts if a held goal changes the commands at least this much:
                                  # a dead path measures 1e-4..3e-4; a live one 0.002..0.01 depending on the random draw
UPRIGHT = 0.65                    # the trunk counts as "up" above this share of its standing height (a walking
                                  # dog carries it at ~0.72; lower is collapsing)
# ---- the teacher
TROT = {"FR": 0.0, "RL": 0.0, "FL": math.pi, "RR": math.pi}   # diagonal legs step together
HARNESS = {"kp": 100.0, "kd": 10.0, "kr": 40.0, "dr": 4.0}     # lift per metre below UPRIGHT (x mass), N·m per unit tilt
WOBBLE = 0.1                      # while learning from the teacher, the brain's own movements vary this much
IMIT_WEIGHT = 10.0                # how much the teacher's corrections count against the reward's gradient (2 let them wash out)
ALPHA_STEP = 0.004                # help taken away per update once the dog copes (>= 250 updates from full to none)
PROGRESS_PASS = 0.7               # "copes": covers at least this share of the ground the teacher's own walk covers
                                  # when asked to move (standing still scores 0), without falling more
HOLD_UNTIL, FADE_BY = 0.3, 0.7    # help is never forced off before 30% of training, and is all gone by 70%
MIN_DEMO_SPEED = 0.08             # m/s: below this the teacher has no walk to show, and training does not start


def ensure_packages() -> None:
    """Install the pinned set unless it is already there, checking versions without importing anything (an
    import would load whatever the machine came with: Kaggle switched to jax 0.11.1 on Python 3.13 in Oct 2026,
    and installing jax 0.7.2 on top of its jaxlib 0.11.1 left a mix that cannot import)."""
    from importlib import metadata

    def have(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None
    gpu = Path("/kaggle").exists() or any(have(p) for p in ("jax-cuda12-plugin", "nvidia-cuda-runtime-cu12"))
    if all(have(p) == v for p, v in PINS.items()) and have("optax"):
        return
    jax_spec = f"jax[cuda12]=={PINS['jax']}" if gpu else f"jax=={PINS['jax']}"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", jax_spec, f"jaxlib=={PINS['jaxlib']}",
                           f"mujoco=={PINS['mujoco']}", f"mujoco-mjx=={PINS['mujoco-mjx']}", "optax"])


def fetch_go2(root: Path) -> Path:
    d = root / "mujoco_menagerie"
    if not (d / "unitree_go2" / "go2_mjx.xml").exists():
        run = lambda *a: subprocess.check_call(list(a))
        run("git", "clone", "-q", "--filter=blob:none", "--no-checkout", MENAGERIE_URL, str(d))
        run("git", "-C", str(d), "sparse-checkout", "set", "unitree_go2")
        run("git", "-C", str(d), "checkout", "-q", MENAGERIE_COMMIT)
    return d / "unitree_go2"


def load_core(version: str = "v1") -> dict:
    """v1: one unit per cell type (2,500). v2: one per cell type per side, with the senses of the full robot
    (brain/core.py build_v2)."""
    import numpy as np
    name = f"core_{version}.npz"
    found = sorted(Path("/kaggle/input").glob(f"**/{name}")) if Path("/kaggle/input").exists() else []
    raw = base64.b64decode(EMBEDDED[name]) if name in EMBEDDED else found[0].read_bytes() if found else \
        (REPO / "data" / "malecns" / name).read_bytes()          # embedded, a Kaggle dataset (bio-bot-cores), or local
    with np.load(io.BytesIO(raw)) as z:
        return {k: z[k] for k in z.files}


def build_model(go2_dir: Path, world: bool = False):
    """Go2 on a plane. The paws touch the ground, and so do the trunk and head when it is down: the Menagerie MJX
    Go2's only trunk shape is a 5.7 cm sphere, so a tipped dog used to sink through the floor (and could never get
    up); it now has a box the size of its real trunk. Its legs above the paws touch nothing (a physics speed-up).
    world: the park and every sense of the real robot (train/dog_world.py)."""
    import mujoco
    spec = mujoco.MjSpec.from_file(str(go2_dir / "scene_mjx.xml"))
    for g in spec.geoms:
        if g.name not in ("floor", *FEET):
            g.contype = 0
            g.conaffinity = 0
    # collision bits: 1 = paws and loose things on the ground, 2 = the trunk and head against the ground and things
    spec.geom("floor").conaffinity = 3
    base = spec.body("base")
    base.add_geom(name="trunk", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.19, 0.047, 0.057], group=3, contype=2,
                  conaffinity=0)                    # the base has its own inertia, so this adds no mass
    for g in spec.geoms:
        if g.parent.name == "base" and g.type == mujoco.mjtGeom.mjGEOM_SPHERE and g.pos[0] > 0.2:
            g.contype = 2                           # the head
    if world:
        import dog_world
        dog_world.add_world(spec)
    model = spec.compile()
    model.opt.timestep = SIM_DT
    return model


# ============================================================================ the body, as the brain meets it
def make_body(model, wiring: int | None = None, keep_fallen: bool = False, fallen_starts: float = 0.0, world=None):
    """wiring: one fixed random wiring for this body, the same in every life (a baby keeps its body): the lines
    are still shuffled, re-signed, rescaled and tagged at random, and the brain is never told which is which,
    but it is the same random draw every life. None: a new draw every life.
    keep_fallen: a fall does not put the body back; it stays down until it gets itself up (only a new life resets).
    fallen_starts: share of new bodies that start fallen, on their side or back, legs every which way.
    world: the park and the full robot's senses (dog_world.make_world on a build_model(world=True) model). The 34
    lines the brain already knows keep their wiring; the new lines (paws, skin, motor heat, ears, nose, battery)
    get their own fixed draw, and every line carries the nerve it arrives on (seen["mod"]: 0 body, 2 hearing,
    3 smell, 4 inner state), as a real animal's senses arrive each on its own nerve. The eye and the LiDAR arrive
    whole and in order (seen["eye"], seen["lidar"]): retinotopic, like a fly's eye."""
    import jax
    import jax.numpy as jp
    import mujoco
    import numpy as np
    from mujoco import mjx

    mx = mjx.put_model(model, impl="jax")
    lo, hi = (jp.array(x) for x in model.actuator_ctrlrange.T)
    home = jp.array(model.keyframe("home").qpos)
    base = model.body("base").id
    mass = float(model.body_subtreemass[base])
    n_sub = int(round(CTRL_DT / SIM_DT))
    cmd_max = jp.array(CMD_MAX)
    onboard = [s for s in range(model.nsensor)
               if mujoco.mjtSensor(model.sensor_type[s]).name.removeprefix("mjSENS_").lower() in ONBOARD
               and not model.sensor(s).name.startswith("global")]
    sidx = jp.array(np.concatenate([np.arange(model.sensor_adr[s], model.sensor_adr[s] + model.sensor_dim[s])
                                    for s in onboard]))
    K0, J = int(sidx.shape[0]), int(model.nu)
    # the new lines, in the order dog_world gives them, and the nerve each arrives on
    extra = [("paws", 0, 4), ("skin", 0, 1), ("heat", 0, J), ("ears", 2, 6), ("nose", 3, 4), ("battery", 4, 1)] if world else []
    E = sum(n for _, _, n in extra)
    K = K0 + E
    mod_of = jp.array([0] * K0 + [m for _, m, n in extra for _ in range(n)], jp.int32)
    feet = jp.array([model.geom(f).id for f in FEET])
    d0 = mujoco.MjData(model)
    d0.qpos[:] = model.keyframe("home").qpos
    mujoco.mj_forward(model, d0)
    h_stand = float(d0.subtree_com[base][2])             # standing height, the yardstick even for a fallen start

    def frame(d):
        R = d.xmat[base]
        h = jp.arctan2(R[1, 0], R[0, 0])
        return R, jp.cos(h), jp.sin(h)

    def sample_cmd(key):
        k1, k2 = jax.random.split(key)
        c = jax.random.uniform(k1, (3,), minval=-cmd_max, maxval=cmd_max)
        c = c.at[0].set(jp.abs(c[0]) * jp.where(c[0] < -0.5 * cmd_max[0], -0.5, 1.0))    # mostly forward
        return jp.where(jax.random.uniform(k2) < 0.15, jp.zeros(3), c)                     # some lives: stand

    def new_body(key, fallen=False):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        spread = jp.where(fallen, 0.5, 0.1)
        q = home.at[7:19].add(jax.random.uniform(k1, (12,), minval=-1.0, maxval=1.0) * spread)
        yaw = jax.random.uniform(k2, (), minval=-3.14, maxval=3.14)
        cy, sy = jp.cos(yaw / 2), jp.sin(yaw / 2)
        roll = jax.random.uniform(k3, (), minval=1.3, maxval=3.0) * jp.where(jax.random.bernoulli(k4), 1.0, -1.0)
        cr, sr = jp.cos(roll / 2), jp.sin(roll / 2)                        # on its side (~1.6) or back (~3)
        tipped = jp.array([cy * cr, cy * sr, sy * sr, sy * cr])            # yaw, then roll about its own long axis
        q = q.at[3:7].set(jp.where(fallen, tipped, jp.array([cy, 0.0, 0.0, sy])))
        q = q.at[2].set(jp.where(fallen, 0.25, q[2]))
        d = mjx.forward(mx, mjx.make_data(model, impl="jax").replace(qpos=q, ctrl=q[7:19]))
        ws = {}
        if world:
            d, ws = world["place"](k5, d)
        return d, d.subtree_com[base], ws       # d.ctrl: where each motor woke up (its "zero")

    def new_life(key):
        """How this life's body appears to the brain: shuffled, re-signed, rescaled, re-tagged."""
        k0 = key if wiring is None else jax.random.PRNGKey(wiring)
        ks = jax.random.split(k0, 6)
        life = {"perm": jax.random.permutation(ks[0], K0),
                "sign_in": jnp_sign(ks[1], K0), "scale_in": jp.exp(jax.random.uniform(ks[2], (K0,), minval=-1.4, maxval=1.4)),
                "off_in": jax.random.normal(ks[3], (K0,)),
                "sign_out": jnp_sign(ks[4], J), "tag": jax.random.normal(ks[5], (J, DT))}
        if E:                                   # the new lines: their own draw, so the old ones stay exactly as they were
            kn = jax.random.split(jax.random.fold_in(k0, 1), 4)
            life = {**life, "perm": jp.concatenate([life["perm"], K0 + jax.random.permutation(kn[0], E)]),
                    "sign_in": jp.concatenate([life["sign_in"], jnp_sign(kn[1], E)]),
                    "scale_in": jp.concatenate([life["scale_in"], jp.exp(jax.random.uniform(kn[2], (E,), minval=-1.4, maxval=1.4))]),
                    "off_in": jp.concatenate([life["off_in"], jax.random.normal(kn[3], (E,))])}
        return {**life, "mod": mod_of[life["perm"]]}

    def jnp_sign(key, n):
        return jp.where(jax.random.bernoulli(key, 0.5, (n,)), 1.0, -1.0)

    def present(d, life, ws):
        """The inputs as the brain sees them this life (the eye and the LiDAR come separately, in order)."""
        x = d.sensordata[sidx]
        if E:
            s = world["sense"](d, ws, ws["key"], sight=False)
            x = jp.concatenate([x, *[s[name] for name, _, _ in extra]])
        return (x[life["perm"]] * life["scale_in"] + life["off_in"]) * life["sign_in"]

    def fresh_discovery(x):
        return {"n": jp.zeros(()), "mean": x, "var": jp.ones(K), "z": jp.zeros(K), "vz": jp.ones(K) * 1e-2,
                "a": jp.zeros((4, J)), "va": jp.ones(J) * 1e-2, "C": jp.zeros((K, J, len(LAGS)))}

    def reset(key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        fallen = jax.random.uniform(k5) < fallen_starts
        d, com, ws = new_body(k1, fallen)
        life = new_life(k2)
        return {"d": d, "life": life, "disc": fresh_discovery(present(d, life, ws)), "world": ws,
                "info": {"cmd": sample_cmd(k3), "h0": jp.where(fallen, h_stand, com[2]), "com": com, "rest": d.ctrl,
                         "t": jp.zeros((), jp.int32), "phase": jp.zeros(()), "key": k4, "new_life": jp.ones((), bool),
                         "down": fallen, "spoke": jp.zeros((), bool)}}

    def sense(state):
        """Adapt, update the self-map, and return what the brain gets this step."""
        d, life, disc = state["d"], state["life"], state["disc"]
        x = present(d, life, state["world"])
        n = disc["n"] + 1.0
        rate = jp.maximum(1.0 / n, 0.02)                       # quick first estimate, then ~1 s adaptation
        mean = disc["mean"] + rate * (x - disc["mean"])
        var = disc["var"] + rate * ((x - mean) ** 2 - disc["var"])
        z = jp.clip((x - mean) / jp.sqrt(var + 1e-4), -5.0, 5.0)
        dz = jp.where(n > 1, z - disc["z"], 0.0)
        vz = disc["vz"] + 0.02 * (dz ** 2 - disc["vz"])
        da = disc["a"][:-1] - disc["a"][1:]                     # output changes, newest first: t-1, t-2, t-3
        C = disc["C"]
        for li, lag in enumerate(LAGS):
            prod = dz[:, None] * da[lag - 1][None, :] / jp.sqrt(vz[:, None] * disc["va"][None, :] + 1e-6)
            C = C.at[..., li].add(SELF_MAP_RATE * (jp.clip(prod, -3, 3) - C[..., li]))
        disc = {**disc, "n": n, "mean": mean, "var": var, "z": z, "vz": vz, "C": C}
        feats = jp.stack([z, dz], -1)                           # [K, 2]
        seen = {"feats": feats, "C": C, "a_hist": disc["a"][:2].T, "tag": life["tag"], "cmd": state["info"]["cmd"],
                "new_life": state["info"]["new_life"]}
        if world:
            sight = world["sense"](d, state["world"], state["world"]["key"])
            seen = {**seen, "mod": life["mod"], "eye": sight["eye"], "lidar": sight["lidar"]}
        return {**state, "disc": disc}, seen

    def line_commands(state, joint_target):
        """The command on each output line that puts its motor at `joint_target`: how the teacher's hands
        on the legs arrive, in this life's units and polarities."""
        a = (joint_target - state["info"]["rest"]) / (0.5 * (hi - lo)) * state["life"]["sign_out"]
        return jp.clip(a, -1.0, 1.0)

    def harness(d, h0, support):
        """The teacher's hand under the trunk: catches it when it collapses or tips, never pushes along or around."""
        R = d.xmat[base]
        lift = mass * (HARNESS["kp"] * (UPRIGHT * h0 - d.subtree_com[base][2]) - HARNESS["kd"] * d.qvel[2])
        spin = (R @ d.qvel[3:6]).at[2].set(0.0)
        tip = HARNESS["kr"] * jp.cross(R[:, 2], jp.array([0.0, 0.0, 1.0])) - HARNESS["dr"] * spin
        f = jp.concatenate([jp.array([0.0, 0.0, jp.maximum(lift, 0.0)]), tip.at[2].set(0.0)]) * support
        return jp.zeros((model.nbody, 6)).at[base].set(f)

    def act(state, a, key, support=0.0):
        """Apply the commands the motors receive (the brain's, or partly the teacher's hands on the legs),
        run physics with the teacher's harness at strength `support`, score by meaning code, handle falls
        and lives."""
        d, life, disc, info, ws = state["d"], state["life"], state["disc"], state["info"], state["world"]
        target = jp.clip(info["rest"] + a * life["sign_out"] * 0.5 * (hi - lo), lo, hi)   # zero = hold still
        if world:                               # things get thrown, the battery drains, hot or flat motors are weak
            d, ws = world["tick"](d, ws, CTRL_DT, info["spoke"])
            target = d.qpos[7:19] + world["strength"](ws) * (target - d.qpos[7:19])
        d = jax.lax.fori_loop(0, n_sub, lambda _, x: mjx.step(mx, x.replace(
            ctrl=target, xfrc_applied=harness(x, info["h0"], support))), d)
        R, ch, sh = frame(d)
        com = d.subtree_com[base]
        v = (com - info["com"]) / CTRL_DT
        v_fwd, v_side = ch * v[0] + sh * v[1], -sh * v[0] + ch * v[1]
        wz = (R @ d.qvel[3:6])[2]
        up = R[2, 2]
        cmd = info["cmd"]
        # ---- meaning code: what "move at (vx, vy), turn at wz" means, for any body
        # asked to move: only moving the asked way scores (standing still earns nothing);
        # asked to stand: stillness scores; asked to turn: turning the asked way scores.
        track = jp.exp(-((v_fwd - cmd[0]) ** 2 + (v_side - cmd[1]) ** 2 + (wz - cmd[2]) ** 2) / 0.25)   # reported, not rewarded
        asked = jp.sqrt(cmd[0] ** 2 + cmd[1] ** 2)
        moving = asked > 0.05
        along = (v_fwd * cmd[0] + v_side * cmd[1]) / jp.maximum(asked, 1e-6)       # speed the asked way
        across = jp.abs(v_fwd * cmd[1] - v_side * cmd[0]) / jp.maximum(asked, 1e-6)  # drift sideways to it
        move = jp.where(moving, jp.clip(along, -0.5, asked) - 0.5 * across, 0.0)
        still = jp.where(moving, 0.0, jp.exp(-(v_fwd ** 2 + v_side ** 2) / 0.05))
        turning = jp.where(jp.abs(cmd[2]) > 0.05, jp.clip(wz * jp.sign(cmd[2]), -0.5, jp.abs(cmd[2])),
                           -0.2 * jp.abs(wz))
        sag = jp.clip(UPRIGHT * info["h0"] - com[2], 0.0, None)
        effort = jp.sum(jp.abs(d.actuator_force * d.qvel[6:18])) / mass
        jitter = jp.sum((a - disc["a"][0]) ** 2)
        reward = (move + still + 0.5 * turning - 2.0 * (1.0 - up) - 10.0 * sag - 0.002 * effort - 0.02 * jitter) * CTRL_DT
        bad = ~(jp.all(jp.isfinite(d.qpos)) & jp.all(jp.isfinite(d.qvel)))
        down = (up < 0.2) | bad                                  # tipped over (on its side or back); lying down is not a fall
        fell = down & ~info["down"]                              # the moment it goes down
        reward = jp.where(bad, 0.0, reward) - 0.2 * fell                            # a fall costs 0.2 s of standing
        life_over = info["t"] + 1 >= LIFE_STEPS
        disc = {**disc, "a": jp.concatenate([a[None], disc["a"][:-1]]),
                "va": disc["va"] + 0.02 * ((a - disc["a"][0]) ** 2 - disc["va"])}
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        cmd = jp.where(jax.random.uniform(k1) < CTRL_DT / 4.0, sample_cmd(k2), cmd)       # new goal every ~4 s
        # fell: the body is put back, the brain keeps its discoveries; life over: everything is new
        pick = lambda new, old, cond: jax.tree_util.tree_map(lambda y, x: jp.where(cond, y, x), new, old)
        k7, k8 = jax.random.split(k5)
        fallen_new = jax.random.uniform(k7) < fallen_starts
        d_new, com_new, ws_new = new_body(k3, fallen_new)
        life_new = new_life(k4)
        over = (fell & ~keep_fallen) | bad | life_over           # keep_fallen: it has to get itself up
        d, com = pick(d_new, d, over), jp.where(over, com_new, com)
        if world:                               # a fall puts the body back, but not its hunger or its hot motors
            ws_new = {**ws_new, "battery": jp.where(life_over, ws_new["battery"], ws["battery"]),
                      "heat": jp.where(life_over, ws_new["heat"], ws["heat"])}
        ws = pick(ws_new, ws, over)
        life = pick(life_new, life, life_over)
        disc = pick(fresh_discovery(present(d, life, ws)), disc, life_over)
        info = {"cmd": jp.where(life_over, sample_cmd(k6), cmd),
                "h0": jp.where(over, jp.where(fallen_new, h_stand, com_new[2]), info["h0"]),
                "com": com, "rest": jp.where(over, d_new.ctrl, info["rest"]),
                "t": jp.where(life_over, 0, info["t"] + 1), "phase": jp.where(over, 0.0, info["phase"]),
                "key": k8, "new_life": life_over, "down": jp.where(over, fallen_new, down), "spoke": info["spoke"]}
        stats = {"v_fwd": v_fwd, "v_side": v_side, "wz": wz, "track": track, "fell": fell.astype(jp.float32),
                 "down": down.astype(jp.float32),
                 "height": com[2] / info["h0"], "along": jp.where(moving, along, 0.0), "asked": jp.where(moving, asked, 0.0)}
        return {"d": d, "life": life, "disc": disc, "info": info, "world": ws}, reward, over, stats

    def privileged(state):
        """For the critic only: the true, unshuffled state."""
        d, info = state["d"], state["info"]
        R, ch, sh = frame(d)
        v = d.qvel[0:3]
        out = [d.sensordata[sidx], info["cmd"],
               jp.array([ch * v[0] + sh * v[1], -sh * v[0] + ch * v[1], v[2],
                         d.subtree_com[base][2] / info["h0"], info["t"] / LIFE_STEPS]),
               state["disc"]["a"][0] * state["life"]["sign_out"]]
        if world:                               # where things are, in the dog's own frame, and how it is inside
            w, ws = world["where"](d), state["world"]
            rel = lambda p: jp.array([ch * (p[0] - d.qpos[0]) + sh * (p[1] - d.qpos[1]),
                                      -sh * (p[0] - d.qpos[0]) + ch * (p[1] - d.qpos[1]), p[2]])
            out += [rel(w["ball"]), rel(w["thrown"]), w["thrown_vel"], rel(w["owner"]), rel(w["pad"]),
                    ws["battery"][None], (ws["heat"] - 25.0) / 40.0]
        return jp.concatenate(out)

    P = K0 + 3 + 5 + J + (15 + 1 + J if world else 0)
    return {"reset": reset, "sense": sense, "act": act, "privileged": privileged, "line_commands": line_commands,
            "feet": feet, "K": K, "J": J, "P": P}


# ============================================================================ the teacher (the AI layer)
def make_teacher(model, body):
    """What the teacher knows and the brain does not: this is a four-legged dog, and a dog walks by
    trotting (diagonal legs swing together; a leg lifts while it swings forward, pushes back while
    down). Its walk is a few numbers: rhythm, stride, lift, and which way the joints turn."""
    import jax
    import jax.numpy as jp
    import numpy as np

    names = [model.actuator(i).name for i in range(model.nu)]            # FL_hip, FL_thigh, FL_calf, ...
    leg, part = [n.split("_")[0] for n in names], [n.split("_")[1] for n in names]
    joints = [model.actuator_trnid[i, 0] for i in range(model.nu)]
    qadr, dadr = jp.array(model.jnt_qposadr[joints]), jp.array(model.jnt_dofadr[joints])
    stand = jp.array(model.keyframe("home").qpos[model.jnt_qposadr[joints]])
    phase0 = jp.array([TROT[l] for l in leg])
    side = jp.array([1.0 if l.endswith("L") else -1.0 for l in leg])
    thigh = jp.array([p == "thigh" for p in part], jp.float32)
    calf = jp.array([p == "calf" for p in part], jp.float32)
    rear = jp.array([l.startswith("R") for l in leg], jp.float32)
    v_reset = jax.vmap(body["reset"])
    v_act = jax.vmap(body["act"], in_axes=(0, 0, 0, None))

    def gait(cmd, g):
        fwd = jp.clip(cmd[0] / g["v_ref"], -1.0, 1.0)
        turn = jp.clip(cmd[2] / g["w_ref"], -1.0, 1.0)
        stride = jp.clip(fwd - g["turn_sign"] * side * turn, -1.0, 1.0) * g["stride"]   # one side strides more to turn
        return stride, jp.clip((jp.abs(fwd) + jp.abs(turn)) / 0.2, 0.0, 1.0)

    def targets(info, g):
        """Joint targets of the teacher's walk for the current goal (stand = the standing pose)."""
        stride, stepping = gait(info["cmd"], g)
        th = info["phase"] + phase0
        swing = stepping * jp.maximum(jp.sin(th), 0.0)                  # 0 while the foot is down
        lift = rear * g["lift_rear"] + (1.0 - rear) * g["lift"]
        tall = g["tall"] * stepping                                     # straighter legs while walking
        return stand + thigh * (g["sign"] * stride * jp.cos(th) - g["tuck"] * swing - 0.5 * tall) + calf * (tall - lift * swing)

    def show(state, g):
        """The teacher's walk as commands on the brain's own output lines."""
        return body["line_commands"](state, targets(state["info"], g))

    phases = jp.linspace(0.0, 2 * jp.pi, 16, endpoint=False)             # where in the walk cycle, sampled
    step_phase = phases[1] - phases[0]

    def follow(state, g):
        """One step of the teacher's rhythm, taken from the brain itself: it looks at the commands the brain's
        lines just sent (the brain knows these too: they come back to it as its own recent outputs) and finds
        where in the walk cycle they are, preferring to carry on from where it was; its next instruction is
        the next moment of the walk from there. So what it asks is always learnable from what the brain
        knows. Asked to stand, it waits at the start of the cycle, so every walk starts the same way."""
        info = state["info"]
        last = state["disc"]["a"][0]                                          # commands the motors just received
        demos = jax.vmap(lambda ph: body["line_commands"](state, targets({**info, "phase": ph}, g)))(phases)   # [16, J]
        miss = jp.mean((demos - last) ** 2, -1)
        cost = miss / 0.002 + 2.0 * (1.0 - jp.cos(phases - info["phase"]))  # fit the commands, and carry on
        k = jp.argmin(cost)
        lo_, mid, hi_ = cost[(k - 1) % 16], cost[k], cost[(k + 1) % 16]
        shift = jp.clip(0.5 * (lo_ - hi_) / jp.maximum(lo_ - 2 * mid + hi_, 1e-6), -0.5, 0.5)   # between samples
        here = phases[k] + shift * step_phase
        _, stepping = gait(info["cmd"], g)
        phase = jp.where(stepping > 0, here + 2 * jp.pi * g["freq"] * CTRL_DT, 0.0)
        return {**state, "info": {**info, "phase": jp.mod(phase, 2 * jp.pi)}}, {"conf": jp.exp(-miss[k] / 0.01)}

    def rhythm(state, g):
        return jp.stack([jp.cos(state["info"]["phase"]), jp.sin(state["info"]["phase"])])

    def practise(g, cmds, key, steps, hold):
        """The teacher walks the dog alone (no brain, no harness). g: one walk per body (arrays)."""
        n = cmds.shape[0]
        k_body, k_run = jax.random.split(key)
        env = v_reset(jax.random.split(k_body, n))
        if hold:
            env["info"]["cmd"] = cmds

        def one(env, k):
            env, sync = jax.vmap(follow)(env, g)
            env, _, _, st = v_act(env, jax.vmap(show)(env, g), jax.random.split(k, n), 0.0)
            if hold:
                env["info"]["cmd"] = cmds
            return env, {**st, **sync, "foot_z": env["d"].geom_xpos[:, body["feet"], 2]}
        return jax.lax.scan(one, env, jax.random.split(k_run, steps))[1]
    practise = jax.jit(practise, static_argnames=("steps", "hold"))

    def calibrate(key, out: Path) -> tuple[dict, dict]:
        """Before teaching, the teacher finds a walk for this body: which rhythm, stride and foot lift carry
        it forward without falling with every foot really stepping, which way to stride to turn, and what
        that walk achieves on the goals the brain will be given. Stops everything if it finds no walk."""
        grid = [{"freq": f, "stride": s, "lift": l, "lift_rear": lr, "tuck": tk, "tall": ta, "sign": sg}
                for f in (2.0, 2.5, 3.0) for s in (0.25, 0.4) for l in (0.4, 0.7) for lr in (0.4, 0.8)
                for tk in (0.0, 0.3) for ta in (0.0, 0.25) for sg in (1.0, -1.0)]
        rep = 8
        n = len(grid) * rep
        per_env = lambda x: np.asarray(x)[100:].mean(0)                          # after the first 2 s
        per_walk = lambda x: x.reshape(-1, rep).mean(1)

        def walks(rows):
            rows = [rows[i // rep % len(rows)] for i in range(n)]
            return {k: jp.array([r[k] for r in rows], jp.float32) for k in rows[0]}

        ones = {"v_ref": 1.0, "w_ref": 1.0, "turn_sign": 1.0}
        k1, k2, k3, k4 = jax.random.split(key, 4)
        st = practise(walks([{**r, **ones} for r in grid]), jp.tile(jp.array([1.0, 0.0, 0.0]), (n, 1)), k1, steps=250, hold=True)
        fwd = per_walk(per_env(st["v_fwd"]))
        drift = per_walk(np.abs(per_env(st["v_side"])))
        spin = per_walk(np.abs(per_env(st["wz"])))
        falls = per_walk(np.asarray(st["fell"]).sum(0))
        height = per_walk(per_env(st["height"]))                                 # trunk height / standing height
        air = (np.asarray(st["foot_z"])[100:] > FOOT_UP).mean(0).reshape(-1, rep, len(FEET)).mean(1)   # [walk, foot]
        steps_all = air.min(1) >= 0.15                                           # every foot spends time in the air
        score = (fwd - 0.5 * drift - 0.3 * spin - 2.0 * falls + 0.5 * air.min(1) - 2.0 * np.maximum(0.9 - height, 0.0)
                 - 10.0 * ~steps_all)                                            # a teacher that stumbles or crouches teaches that
        order = np.argsort(score)[::-1]
        best = grid[int(order[0])]
        table = [{**grid[i], "forward": round(float(fwd[i]), 3), "drift": round(float(drift[i]), 3),
                  "spin": round(float(spin[i]), 3), "falls_per_5s": round(float(falls[i]), 2), "height": round(float(height[i]), 2),
                  "air": dict(zip(FEET, np.round(air[i], 2).tolist()))} for i in order[:12]]
        if not steps_all[order[0]]:
            print("teacher: no walk tried lifts every foot; using the best shuffle", flush=True)
        for row in table[:6]:
            print(f"teacher tries {row}", flush=True)
        v_ref = float(fwd[order[0]])
        if v_ref < MIN_DEMO_SPEED:
            (out / "teacher.json").write_text(json.dumps({"tried": table}, indent=1))
            raise SystemExit(f"STOP: the teacher found no walk for this body (best {v_ref:.3f} m/s < {MIN_DEMO_SPEED})")

        # turning: stride more on one side; which side turns which way is found, not assumed
        turns = [{**best, "v_ref": v_ref, "w_ref": 1.0, "turn_sign": ts} for ts in (1.0, -1.0)]
        st = practise(walks(turns), jp.tile(jp.array([0.0, 0.0, 1.0]), (n, 1)), k2, steps=250, hold=True)
        groups = per_walk(per_env(st["wz"]))                                    # walks alternate +1, -1 by group
        wz = np.array([groups[0::2].mean(), groups[1::2].mean()])
        ts = 1.0 if wz[0] >= wz[1] else -1.0
        w_ref = float(max(wz.max(), 0.1))
        g = {**best, "v_ref": v_ref, "w_ref": w_ref, "turn_sign": ts}
        st = practise(walks([g]), jp.tile(jp.array([-0.5 * v_ref, 0.0, 0.0]), (n, 1)), k3, steps=250, hold=True)
        v_back = float(per_env(st["v_fwd"]).mean())
        # what the teacher's walk achieves on the brain's own goal mix: the bar for letting go
        st = practise(walks([g]), jp.zeros((n, 3)), k4, steps=750, hold=False)
        lesson = {"progress_ref": float(np.asarray(st["along"])[100:].sum() / max(np.asarray(st["asked"])[100:].sum(), 1e-6)),
                  "track_ref": float(np.asarray(st["track"])[100:].mean()),
                  "falls_ref": float(np.asarray(st["fell"]).mean() * 1000),
                  "forward": v_ref, "turn": float(wz.max()), "back_at_half_speed": v_back}
        (out / "teacher.json").write_text(json.dumps({"walk": g, "lesson": lesson, "tried": table}, indent=1))
        print(f"teacher's walk: {g}\nit achieves: " + "  ".join(f"{k} {v:.3f}" for k, v in lesson.items()), flush=True)
        return g, lesson

    return {"show": show, "follow": follow, "rhythm": rhythm, "calibrate": calibrate}


# ============================================================================ the brain
READOUTS = ("rates", "vector", "line")
OL_CH = 16                        # channels of the optic lobe's column circuit


def make_brain(core: dict, key, bias0: float = 0.0, goal_scale: float = 0.3, gain0: float = None, readout: str = "rates",
               words: int = 0, eye_shape: tuple | None = None, lidar_shape: tuple | None = None, say: int = 0):
    """Connectome core + set-based interface. No parameter's shape depends on the body.

    readout, how an output line turns the motor neurons into its command:
      rates   a weighted average of motor-neuron firing rates, chosen by the line (one number per head)
      vector  the same choice, but each motor neuron contributes a learned vector and a small network reads
              the mix: richer, still only what the connectome's motor neurons say
      line    the rates mix plus the line's own token (its recent commands and self-map) into a small network:
              lets a line shape its own command, like the local circuits next to a real motor neuron
    words: size of the vocabulary the teacher speaks; heard words (seen["words"], one weight per word, 0..1)
           enter the central brain and the descending neurons like the goal does.
    step.predict(p, s, seen): what the brain expects each input line to do next, given the command it has just
           sent (seen["a_hist"][..., 0]): its model of its own body.
    A v2 core (core["modality"]): each input line reaches only the sense neurons of the nerve it arrives on
           (seen["mod"]; without it every line is a body line), and an old neuron's inputs keep the proportions
           they had in v1 while connections from new neurons start weak (their gain at ~1%) and grow by learning.
    eye_shape, lidar_shape: the optic lobe. One small circuit, the same at every point of the eye (as the fly
           repeats the same cell types in every column), mirror-symmetric between the left and right halves, sees
           each point's colour and how it changed since the last step (and the LiDAR's nearness); every visual
           projection neuron type pools its own side's circuit through a receptive field of its own. Its output
           starts at exactly zero, so a brain that grows eyes keeps doing what it did until it learns to see.
    say: words it can say about itself. step.say(p, s) -> [B, say] logits, read from its own central and descending
           neurons the way its motor readout drives the legs (the fly has no voice; this is the dog's)."""
    assert readout in READOUTS, readout
    import jax
    import jax.numpy as jp
    import numpy as np

    role = core["role"]
    N = len(role)
    pre, post = core["pre"].astype(np.int32), core["post"].astype(np.int32)
    syn = core["synapses"].astype(np.float32)
    # Each neuron's input is the synapse-weighted average of its inputs, so a signal keeps its size
    # from hop to hop. (Scaling the whole matrix to spectral radius 0.9 instead shrank every ordinary
    # connection ~70x and signals died within one hop: goals never reached the motor neurons.)
    z = np.bincount(post, weights=syn, minlength=N) + 1.0
    if "v1_unit" in core:                   # grown from v1: old neurons keep their input proportions
        old_pre = core["v1_unit"][pre] >= 0
        z_old = np.bincount(post, weights=np.where(old_pre, syn, 0.0), minlength=N)
        z = np.where((core["v1_unit"] >= 0) & (z_old > 0), z_old, z - 1.0) + 1.0
    base_w = core["sign"].astype(np.float32)[pre] * syn / z[post]
    rho = float("nan")                      # spectral radius, only reported (too slow to compute for big cores)
    if N <= 3000:
        M = np.zeros((N, N), np.float64)
        np.add.at(M, (pre, post), base_w)
        rho = float(np.max(np.abs(np.linalg.eigvals(M))))
    idx = {n: np.where(role == i)[0] for i, n in enumerate(["sense", "motor", "descending", "ascending", "cord", "central"])}
    sense_mod = core["modality"][idx["sense"]].astype(np.int32) if "modality" in core else np.zeros(len(idx["sense"]), np.int32)
    vis = np.where(sense_mod == 1)[0]                                     # visual projection neurons, among the senses
    names = core["name"][idx["sense"]][vis] if len(vis) else np.array([], str)
    vis_type_names, vis_type = np.unique([n.rsplit("|", 1)[0] for n in names], return_inverse=True)
    vis_left = np.array([n.endswith("|L") for n in names])
    eyes = eye_shape is not None and len(vis) > 0
    goal_units = np.concatenate([idx["central"], idx["descending"]])      # intent enters the central brain
    A, Mo, G = len(idx["sense"]), len(idx["motor"]), len(goal_units)        # and the fly's command neurons
    L = len(LAGS)

    ks = iter(jax.random.split(key, 64 if eye_shape else 32))       # (32: the v1 draws stay exactly as they were)
    nrm = lambda shape, s: jax.random.normal(next(ks), shape) * s
    mlp = lambda i, h, o: {"w1": nrm((i, h), (1 / i) ** 0.5), "b1": jp.zeros(h), "w2": nrm((h, o), (1 / h) ** 0.5), "b2": jp.zeros(o)}
    params = {
        "log_tau": jp.full((N,), np.log(0.05), jp.float32), "bias": jp.full((N,), bias0, jp.float32),   # tonic drive
        "gamma": jp.zeros((len(pre),), jp.float32), "log_gain": jp.array(np.log(gain0 or GAIN0), jp.float32),
        "h": mlp(2, 64, D), "g": mlp(DT + 2, 64, D),
        "w_in": nrm((L, D, D), (1 / D) ** 0.5), "w_out": nrm((L, D, D), (1 / D) ** 0.5),
        "i": mlp(2 * D, 64, D), "o": mlp(2 * D, 64, D),
        "q_aff": nrm((A, D), 1.0), "wk": nrm((D, D), (1 / D) ** 0.5), "wv": nrm((D,), (1 / D) ** 0.5),
        "goal_w": nrm((3, G), goal_scale),
        "wq": nrm((D, HEADS * D), (1 / D) ** 0.5), "k_motor": nrm((HEADS, Mo, D), 1.0),
        "out_w": nrm((HEADS,), 0.3), "out_b": jp.zeros((), jp.float32), "log_std": jp.array(-1.0, jp.float32),
    }
    if eyes:
        Tv, (eh, ew), (lh, lw) = len(vis_type_names), eye_shape, lidar_shape
        conv = lambda cin, cout: nrm((3, 3, cin, cout), (1 / (9 * cin)) ** 0.5)
        params.update({"ol_e1": conv(4, OL_CH), "ol_e2": conv(OL_CH, OL_CH), "ol_l1": conv(2, OL_CH), "ol_l2": conv(OL_CH, OL_CH),
                       "rf_e": nrm((Tv, eh, ew // 2), 1.0 / (eh * ew // 2)), "rf_l": nrm((Tv, lh, lw // 2), 1.0 / (lh * lw // 2)),
                       "vp_e": jp.zeros((Tv, OL_CH), jp.float32), "vp_l": jp.zeros((Tv, OL_CH), jp.float32),
                       # what it will see next: the optic lobe's columns plus a copy of what its motor neurons do
                       "sp_eff": nrm((len(idx["motor"]), 8), (1 / len(idx["motor"])) ** 0.5),
                       "sp_1": nrm((OL_CH + 8, 16), (1 / (OL_CH + 8)) ** 0.5), "sp_2": nrm((16, 2), 0.1)})
    if say:                                 # its voice: a readout from its own central and descending neurons
        params.update({"say_w": nrm((G, say), (1 / G) ** 0.5), "say_b": jp.zeros((say,), jp.float32)})
    if "v1_edge" in core:                   # connections the v1 brain never had start weak
        params["gamma"] = jp.where(jp.array(core["v1_edge"]) >= 0, 0.0, -5.0).astype(jp.float32)
    params.update({"word_w": nrm((max(words, 1), G), goal_scale),
                   "pq": nrm((D, D), (1 / D) ** 0.5), "pk": nrm((A, D), 1.0), "pv": nrm((A, D), 1.0), "pm": mlp(2 * D, 64, 1)})
    if readout == "vector":
        params.update({"v_motor": nrm((Mo, D), 1.0), "o2": mlp(HEADS * D, 64, 1)})
    elif readout == "line":
        params.update({"o2": mlp(HEADS + D, 64, 1)})
    static = {"pre": jp.array(pre), "post": jp.array(post), "base_w": jp.array(base_w),
              "sense": jp.array(idx["sense"]), "motor": jp.array(idx["motor"]), "goal": jp.array(goal_units)}
    sp1 = float(np.log(np.e - 1.0))

    def run_mlp(p, x):
        return jax.nn.gelu(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]

    def weights(p):
        w = static["base_w"] * jax.nn.softplus(p["gamma"] + sp1) * jp.exp(p["log_gain"])
        return jp.zeros((N, N), jp.float32).at[static["pre"], static["post"]].add(w)

    def init_state(batch: int):
        s = {"v": jp.zeros((batch, N)), "f": jp.zeros((batch, N))}
        if eyes:                            # the last frame, for the change at every point; and its own surprise:
            s.update({"eye": jp.zeros((batch, *eye_shape, 2)), "lidar": jp.zeros((batch, *lidar_shape)),
                      "sight_pred": jp.zeros((batch, *eye_shape, 2)), "has_pred": jp.zeros((batch,)),   # what it expected
                      "surprise_fast": jp.zeros((batch,)), "surprise_slow": jp.zeros((batch,))})       # (curiosity)
        return s

    def eye_halves(e, before):
        """Each side's half of the eye (colour and its change), seen from the midline outward: mirror images."""
        fe = jp.concatenate([e, e - before], -1)                             # [B, H, W, 4]
        half = eye_shape[1] // 2
        return {True: fe[:, :, :half][:, :, ::-1], False: fe[:, :, half:]}

    def predict_sight(p, s_before, s_after, seen):
        """What it expects to see next: the change of every eye point by the next step [B, H, W, 2], from each
        column of its optic lobe and a copy of what its motor neurons are doing (an efference copy)."""
        conv = lambda x, w: jax.lax.conv_general_dilated(x, w, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
        eff = jp.tanh(jax.nn.relu(s_after["v"]))[:, static["motor"]] @ p["sp_eff"]          # [B, 8]
        halves = eye_halves(seen["eye"], s_before["eye"])
        out = {}
        for left in (True, False):
            g = jax.nn.gelu(conv(jax.nn.gelu(conv(halves[left], p["ol_e1"])), p["ol_e2"]))
            g = jp.concatenate([g, jp.broadcast_to(eff[:, None, None], (*g.shape[:3], 8))], -1)
            out[left] = jax.nn.gelu(g @ p["sp_1"]) @ p["sp_2"]
        return jp.concatenate([out[True][:, :, ::-1], out[False]], 2)

    def route(mod):
        """[B, A, K]: which input lines reach which sense neurons (each line its own nerve)."""
        return jp.asarray(sense_mod)[None, :, None] == mod[:, None, :]

    def optic_lobe(p, s, seen):
        """The eye and the LiDAR -> each visual projection neuron's input [B, Nv], and the frames to remember."""
        e, l = seen["eye"], 1.0 / (0.3 + seen["lidar"])                  # colour; LiDAR nearness
        conv = lambda x, w: jax.lax.conv_general_dilated(x, w, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
        def lobe(x, w1, w2):
            return jax.nn.gelu(conv(jax.nn.gelu(conv(x, w1)), w2))
        e_side = eye_halves(e, s["eye"])
        fl = jp.stack([l, l - 1.0 / (0.3 + s["lidar"])], -1)               # [B, h, w, 2]
        half_l = lidar_shape[1] // 2
        l_side = {True: fl[:, :, :half_l], False: fl[:, :, ::-1][:, :, :half_l]}
        out = []
        for left in (True, False):
            ge = lobe(e_side[left], p["ol_e1"], p["ol_e2"])                # [B, H, W/2, C]
            gl = lobe(l_side[left], p["ol_l1"], p["ol_l2"])
            out.append(jp.einsum("bhwc,thw,tc->bt", ge, p["rf_e"], p["vp_e"]) +
                       jp.einsum("bhwc,thw,tc->bt", gl, p["rf_l"], p["vp_l"]))   # [B, Tv] per type
        per_unit = jp.where(jp.asarray(vis_left)[None], out[0][:, vis_type], out[1][:, vis_type])
        return per_unit, {"eye": e, "lidar": seen["lidar"]}

    def step(p, W, s, seen):
        """One control step for a batch. Works for any number of inputs K and outputs J."""
        feats, C, a_hist, tag, cmd = seen["feats"], seen["C"], seen["a_hist"], seen["tag"], seen["cmd"]
        B, K, J = feats.shape[0], feats.shape[1], tag.shape[1]
        h = run_mlp(p["h"], feats)                                                    # [B,K,D] each input line
        g = run_mlp(p["g"], jp.concatenate([tag, a_hist], -1))                        # [B,J,D] each output line
        agg_in = sum(jp.einsum("bkj,bjd->bkd", C[..., l], g @ p["w_in"][l]) for l in range(L)) / J
        agg_out = sum(jp.einsum("bkj,bkd->bjd", C[..., l], h @ p["w_out"][l]) for l in range(L)) / K
        tok_in = run_mlp(p["i"], jp.concatenate([h, agg_in], -1))                    # what this input does, for me
        tok_out = run_mlp(p["o"], jp.concatenate([g, agg_out], -1))                  # what this output does, for me
        logits = jp.einsum("ad,bkd->bak", p["q_aff"], tok_in @ p["wk"]) / np.sqrt(D)
        if "modality" in core:              # each line reaches only the sense neurons of its own nerve
            ok = route(seen.get("mod", jp.zeros((B, K), jp.int32)))
            att = jax.nn.softmax(jp.where(ok, logits, -1e9), -1) * jp.any(ok, -1, keepdims=True)
        else:
            att = jax.nn.softmax(logits, -1)
        u = jp.zeros((B, N)).at[:, static["sense"]].set(jp.einsum("bak,bk->ba", att, tok_in @ p["wv"]))
        new = {}
        if eyes and "eye" in seen:
            seen_now, new = optic_lobe(p, s, seen)
            u = u.at[:, static["sense"][vis]].add(seen_now)
        u = u.at[:, static["goal"]].add(cmd @ p["goal_w"])
        if words:
            u = u.at[:, static["goal"]].add(seen["words"] @ p["word_w"])
        alpha = jp.clip(TICK_DT / jp.exp(p["log_tau"]), 0.02, 1.0)
        r = jp.tanh(jax.nn.relu(s["v"]))
        rec = jp.dot(r.astype(jp.float16), W.astype(jp.float16), preferred_element_type=jp.float32)
        v = s["v"] + alpha * (-s["v"] + rec + p["bias"] + u - s["f"])
        fat = s["f"] + 0.04 * (3.0 * r - s["f"])                                       # neurons tire when active
        r_m = jp.tanh(jax.nn.relu(v))[:, static["motor"]]                              # [B,Mo]
        q = (tok_out @ p["wq"]).reshape(B, J, HEADS, D)
        read = jax.nn.softmax(jp.einsum("bjhd,hmd->bjhm", q, p["k_motor"]) / np.sqrt(D), -1)
        if readout == "rates":
            mu = jp.einsum("bjhm,bm->bjh", read, r_m) @ p["out_w"] + p["out_b"]        # [B,J]
        elif readout == "vector":
            mix = jp.einsum("bjhm,bm,md->bjhd", read, r_m, p["v_motor"]).reshape(B, J, HEADS * D)
            mu = run_mlp(p["o2"], mix)[..., 0] + p["out_b"]
        else:
            mix = jp.einsum("bjhm,bm->bjh", read, r_m)
            mu = run_mlp(p["o2"], jp.concatenate([mix, tok_out], -1))[..., 0] + p["out_b"]
        return {**s, "v": v, "f": fat, **new}, mu

    def predict(p, s, seen):
        feats, C, a_hist, tag = seen["feats"], seen["C"], seen["a_hist"], seen["tag"]
        K, J = feats.shape[1], tag.shape[1]
        h = run_mlp(p["h"], feats)
        g = run_mlp(p["g"], jp.concatenate([tag, a_hist], -1))
        agg_in = sum(jp.einsum("bkj,bjd->bkd", C[..., l], g @ p["w_in"][l]) for l in range(L)) / J
        tok_in = run_mlp(p["i"], jp.concatenate([h, agg_in], -1))
        r_s = jp.tanh(jax.nn.relu(s["v"]))[:, static["sense"]]                      # [B,A]
        logits_p = jp.einsum("bkd,ad->bka", tok_in @ p["pq"], p["pk"]) / np.sqrt(D)
        if "modality" in core:
            ok = jp.swapaxes(route(seen.get("mod", jp.zeros(feats.shape[:2], jp.int32))), 1, 2)   # [B, K, A]
            att_p = jax.nn.softmax(jp.where(ok, logits_p, -1e9), -1)
        else:
            att_p = jax.nn.softmax(logits_p, -1)
        read = jp.einsum("bka,ba,ad->bkd", att_p, r_s, p["pv"])
        return run_mlp(p["pm"], jp.concatenate([tok_in, read], -1))[..., 0]          # [B,K]
    step.predict = predict
    step.say = (lambda p, s: jp.tanh(jax.nn.relu(s["v"]))[:, static["goal"]] @ p["say_w"] + p["say_b"]) if say else None
    step.predict_sight = predict_sight if eyes else None

    return params, weights, init_state, step, {"N": N, "A": A, "Mo": Mo, "G": G, "edges": len(pre), "rho": rho,
                                               "motor_idx": idx["motor"].tolist(), "readout": readout}


def make_critic(P: int, key):
    import jax
    import jax.numpy as jp
    sizes = [P, 256, 256, 256, 1]
    ks = jax.random.split(key, len(sizes))
    params = [{"w": jax.random.normal(ks[i], (a, b)) * (1.0 / a) ** 0.5, "b": jp.zeros(b)}
              for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]))]

    def value(ps, x):
        for i, l in enumerate(ps):
            x = x @ l["w"] + l["b"]
            x = jax.nn.swish(x) if i < len(ps) - 1 else x
        return x[..., 0]
    return params, value


# ============================================================================ self-test: no identity can leak
def selftest(bstep, weights, init_state, bp, K: int, J: int) -> None:
    """Permuting inputs and outputs must permute the commands the same way; any K, J must work."""
    import jax
    import jax.numpy as jp
    import numpy as np
    key = jax.random.PRNGKey(3)
    ks = jax.random.split(key, 6)
    B, L = 2, len(LAGS)
    seen = {"feats": jax.random.normal(ks[0], (B, K, 2)), "C": jax.random.normal(ks[1], (B, K, J, L)) * 0.3,
            "a_hist": jax.random.uniform(ks[2], (B, J, 2), minval=-1, maxval=1), "tag": jax.random.normal(ks[3], (B, J, DT)),
            "cmd": jax.random.normal(ks[4], (B, 3)) * 0.3}
    W = weights(bp)
    s = init_state(B)
    s = {"v": jax.random.normal(ks[5], s["v"].shape) * 0.3, "f": s["f"]}
    _, mu = bstep(bp, W, s, seen)
    pk, pj = np.random.default_rng(0).permutation(K), np.random.default_rng(1).permutation(J)
    shuffled = {**seen, "feats": seen["feats"][:, pk], "C": seen["C"][:, pk][:, :, pj], "a_hist": seen["a_hist"][:, pj],
                "tag": seen["tag"][:, pj]}
    _, mu2 = bstep(bp, W, s, shuffled)
    err = float(jp.abs(mu2 - mu[:, pj]).max())
    fewer = {**seen, "feats": seen["feats"][:, : K - 5], "C": seen["C"][:, : K - 5, : J - 3], "a_hist": seen["a_hist"][:, : J - 3],
             "tag": seen["tag"][:, : J - 3]}
    _, mu3 = bstep(bp, W, s, fewer)
    print(f"selftest: shuffling inputs and outputs permutes the commands exactly (max error {err:.1e}); "
          f"a body with {K - 5} inputs and {J - 3} outputs gives {mu3.shape[1]} commands", flush=True)
    assert err < 1e-3 and mu3.shape == (B, J - 3)



def goal_influence(bp, weights, bstep, init_state, body, key, n: int = 128, warm: int = 150, hold: int = 25,
                   motor_idx=None) -> dict:
    """Let a brain live 3 s, then hold different goals for `hold` steps from that same moment:
    how much do its commands differ? Also: how active are its motor neurons."""
    import jax
    import jax.numpy as jp
    v_reset, v_sense = jax.vmap(body["reset"]), jax.vmap(body["sense"])
    v_act = jax.vmap(body["act"], in_axes=(0, 0, 0, None))

    @jax.jit
    def run(bp, key):
        W = weights(bp)
        env = v_reset(jax.random.split(key, n))

        def live(carry, k):
            env, s = carry
            env, seen = v_sense(env)
            s, mu = bstep(bp, W, s, seen)
            env, _, _, _ = v_act(env, jp.tanh(mu), jax.random.split(k, n), 0.0)
            return (env, s), mu
        (env, s), _ = jax.lax.scan(live, (env, init_state(n)), jax.random.split(key, warm))

        def held(goal):
            def one(carry, k):
                env, s = carry
                env, seen = v_sense(env)
                s, mu = bstep(bp, W, s, {**seen, "cmd": jp.broadcast_to(goal, (n, 3))})
                env, _, _, _ = v_act(env, jp.tanh(mu), jax.random.split(k, n), 0.0)
                return (env, s), mu
            (_, s_end), mus = jax.lax.scan(one, (env, s), jax.random.split(jax.random.PRNGKey(99), hold))
            return mus, s_end
        fwd, s_f = held(jp.array([0.5, 0.0, 0.0]))
        back, _ = held(jp.array([-0.5, 0.0, 0.0]))
        stand, _ = held(jp.zeros(3))
        turn, _ = held(jp.array([0.0, 0.0, 0.8]))
        rates = jp.tanh(jax.nn.relu(s_f["v"]))
        motor = rates[:, jp.array(motor_idx)] if motor_idx is not None else rates
        return {"goal_fwd_vs_back": jp.abs(fwd - back)[-10:].mean(), "goal_fwd_vs_stand": jp.abs(fwd - stand)[-10:].mean(),
                "goal_turn_vs_stand": jp.abs(turn - stand)[-10:].mean(), "command_size": jp.abs(stand).mean(),
                "units_active": (rates > 0.01).mean(), "motor_active": (motor > 0.01).mean(), "motor_rate": motor.mean()}
    return {k: float(v) for k, v in run(bp, key).items()}

# ============================================================================ watching the teacher
def watch(bp, brain, body, teacher, walk, envs: int, unroll: int, steps: float, lr: float, seed: int, label: str,
          fade_hands: bool = False, probe: bool = False):
    """The brain walks with the teacher: the teacher holds the trunk and moves the legs together with the brain
    (each body gets its own share of the teacher's hands, 30-100%, so the brain also learns while partly in
    control), and the brain learns to say on each of its own anonymous lines what the teacher's walk does
    next, in step with the legs (a copying error, backpropagated through `unroll` steps of the connectome).
    fade_hands: instead, every body starts fully guided and the teacher's hands fade to nothing by 70% of the
    steps, so the brain ends up moving the legs alone while the teacher only says what it should have done.
    probe: before and after, measure the brain moving the legs alone (hands off, harness on, no learning):
    its copying error and how far it gets when asked to move.
    Returns the brain and the error curve."""
    import jax
    import jax.numpy as jp
    import numpy as np
    import optax
    weights, init_state, bstep = brain
    t0 = time.time()
    nd = jax.local_device_count()
    B, T = envs // nd, unroll
    v_reset, v_sense = jax.vmap(body["reset"]), jax.vmap(body["sense"])
    v_act = jax.vmap(body["act"], in_axes=(0, 0, 0, None))
    v_show = jax.vmap(teacher["show"], in_axes=(0, None))
    v_follow = jax.vmap(teacher["follow"], in_axes=(0, None))
    fresh = lambda tree, new_life: jax.tree_util.tree_map(       # a new life forgets everything (any state shape)
        lambda x: jp.where(new_life.reshape(-1, *([1] * (x.ndim - 1))), 0.0, x), tree)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))

    def loss(bp, traj, bst0):
        W = weights(bp)

        def one(s, x):
            return bstep(bp, W, fresh(s, x["new_life"]), x)
        _, mu = jax.lax.scan(one, bst0, traj)
        return jp.mean((jp.tanh(mu) - traj["shown"]) ** 2)

    def rollout(bp, env, bst, hands, key):
        W = weights(bp)

        def one(carry, k):
            env, s = carry
            k_act, k_wobble = jax.random.split(k)
            env, seen = v_sense(env)
            env, _ = v_follow(env, walk)
            s, mu = bstep(bp, W, fresh(s, seen["new_life"]), seen)
            shown = v_show(env, walk)
            own = jp.clip(jp.tanh(mu) + WOBBLE * jax.random.normal(k_wobble, mu.shape), -1.0, 1.0)
            a = (1.0 - hands[:, None]) * own + hands[:, None] * shown
            env, _, _, st = v_act(env, a, jax.random.split(k_act, B), 1.0)
            return (env, s), ({**seen, "shown": shown}, st["along"], st["asked"])
        (env, bst_end), (traj, along, asked) = jax.lax.scan(one, (env, bst), jax.random.split(key, T))
        moved = jax.lax.psum(jp.stack([along.sum(), asked.sum()]), "d")
        return env, bst_end, traj, moved

    def iteration(bp, opt_state, env, bst, hands, key):
        env, bst_end, traj, moved = rollout(bp, env, bst, hands, key)
        err, g = jax.value_and_grad(loss)(bp, traj, bst)
        upd, opt_state = opt.update(jax.lax.pmean(g, "d"), opt_state, bp)
        base = jp.mean(traj["shown"] ** 2)
        return optax.apply_updates(bp, upd), opt_state, env, bst_end, jax.lax.pmean(err, "d"), jax.lax.pmean(base, "d")

    def alone(bp, env, bst, key):
        env, bst_end, traj, moved = rollout(bp, env, bst, jp.zeros(B), key)
        return env, bst_end, jax.lax.pmean(loss(bp, traj, bst), "d"), jax.lax.pmean(jp.mean(traj["shown"] ** 2), "d"), moved

    step = jax.pmap(iteration, axis_name="d")
    measure = jax.pmap(alone, axis_name="d")
    bp_r = jax.device_put_replicated(bp, jax.local_devices())
    opt_r = jax.device_put_replicated(opt.init(bp), jax.local_devices())
    keys = jax.random.split(jax.random.PRNGKey(seed + 1), nd + 1)
    env = jax.pmap(lambda k: v_reset(jax.random.split(k, B)))(keys[1:])
    bst = jax.pmap(lambda _: init_state(B))(jp.arange(nd))
    hands = jax.random.uniform(jax.random.PRNGKey(seed + 2), (nd, B), minval=0.3, maxval=1.0)
    key, curve = keys[0], []

    def probe_alone(when):
        nonlocal env, bst, key
        errs, bases, moved = [], [], np.zeros(2)
        for _ in range(10):                                        # 10 x unroll steps (~6 s) of the brain alone
            key, k = jax.random.split(key)
            env, bst, e, b, m = measure(bp_r, env, bst, jax.random.split(k, nd))
            errs.append(float(np.asarray(e)[0])), bases.append(float(np.asarray(b)[0]))
            moved += np.asarray(m)[0]
        row = {"when": when, "alone_copy_error": float(np.mean(errs)), "do_nothing": float(np.mean(bases)),
               "alone_progress": float(moved[0] / max(moved[1], 1e-6))}
        curve.append(row)
        print(f"brain alone {when} teaching [{label}]: copy error {row['alone_copy_error']:.4f} (do-nothing {row['do_nothing']:.4f}, "
              f"ratio {row['alone_copy_error'] / max(row['do_nothing'], 1e-9):.2f})  "
              f"progress when asked to move {row['alone_progress']:+.2f}", flush=True)
    if probe:
        probe_alone("before")
    iters = max(1, int(steps) // (envs * T))
    for it in range(iters):
        key, k = jax.random.split(key)
        if fade_hands:
            hands = jp.full((nd, B), max(0.0, 1.0 - it / (0.7 * iters)))
        bp_r, opt_r, env, bst, err, base = step(bp_r, opt_r, env, bst, hands, jax.random.split(k, nd))
        if it % max(1, iters // 12) == 0 or it == iters - 1:
            e, b = float(np.asarray(err)[0]), float(np.asarray(base)[0])
            curve.append({"steps": (it + 1) * envs * T, "error": e, "do_nothing": b, "ratio": e / max(b, 1e-9)})
            print(f"watching [{label}] {(it + 1) * envs * T:>11,} steps  hands {float(np.asarray(hands).mean()):.2f}  "
                  f"copy error {e:.4f}  do-nothing {b:.4f}  "
                  f"ratio {e / max(b, 1e-9):.2f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    if probe:
        probe_alone("after")
    return jax.tree_util.tree_map(lambda x: np.asarray(x)[0], bp_r), curve


# ============================================================================ recurrent PPO, data-parallel
def train(args, out: Path) -> dict:
    import jax
    import jax.numpy as jp
    import numpy as np
    import optax

    t0 = time.time()
    go2 = Path(args.menagerie) / "unitree_go2" if args.menagerie else fetch_go2(Path(tempfile.gettempdir()))
    model = build_model(go2)
    body = make_body(model)
    teacher = make_teacher(model, body)
    key = jax.random.PRNGKey(args.seed)
    key, kb, kc, kt = jax.random.split(key, 4)
    walk, lesson = teacher["calibrate"](kt, out)
    bp, weights, init_state, bstep, binfo = make_brain(load_core(), kb, args.bias0, args.goal_scale, args.gain0, args.readout)
    cp, value = make_critic(body["P"] + 4, kc)
    selftest(bstep, weights, init_state, bp, body["K"], body["J"])
    print(f"body: {body['K']} input lines, {body['J']} output lines (the brain is told neither). "
          f"brain: {binfo['N']} units, {binfo['edges']:,} connections", flush=True)
    gate = goal_influence(bp, weights, bstep, init_state, body, jax.random.PRNGKey(7), motor_idx=binfo["motor_idx"])
    print("gate (untrained brain, goals held 0.5 s): " + "  ".join(f"{k} {v:.4f}" for k, v in gate.items()), flush=True)
    (out / "gate.json").write_text(json.dumps(gate, indent=1))
    if gate["goal_fwd_vs_back"] < GOAL_GATE:
        raise SystemExit(f"STOP: goals barely reach the motor neurons ({gate['goal_fwd_vs_back']:.4f} < {GOAL_GATE}); "
                         "not spending GPU time on training")

    initial_brain = jax.tree_util.tree_map(np.asarray, bp)
    # first the brain watches the teacher walk its body and learns to copy it, the way a child watches first
    bp, curve = watch(bp, (weights, init_state, bstep), body, teacher, walk, args.envs // args.minibatches, args.unroll,
                      args.watch_steps, args.watch_lr, args.seed, binfo["readout"])
    (out / "watching.json").write_text(json.dumps(curve, indent=1))
    watched_brain = jax.tree_util.tree_map(np.asarray, bp)

    nd = jax.local_device_count()
    B = args.envs // nd                                  # envs per device
    T = args.unroll
    params = {"brain": bp, "critic": cp}
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr))
    rep = lambda x: jax.device_put_replicated(x, jax.local_devices())
    params_r, opt_r = rep(params), rep(opt.init(params))

    v_reset, v_sense, v_priv = (jax.vmap(body[k]) for k in ("reset", "sense", "privileged"))
    v_act = jax.vmap(body["act"], in_axes=(0, 0, 0, None))
    v_show = jax.vmap(teacher["show"], in_axes=(0, None))
    v_follow = jax.vmap(teacher["follow"], in_axes=(0, None))
    v_rhythm = jax.vmap(teacher["rhythm"], in_axes=(0, None))

    def critic_view(env, guide, support):
        n = env["info"]["t"].shape[0]
        return jp.concatenate([v_priv(env), v_rhythm(env, walk), jp.full((n, 1), guide), jp.full((n, 1), support)], -1)

    def norm_stats(x, axis_name):
        m = jax.lax.pmean(x.mean(0), axis_name)
        v = jax.lax.pmean(((x - m) ** 2).mean(0), axis_name)
        return m, jp.maximum(jp.sqrt(v), 0.05)

    shrink = float(np.sqrt(1.0 - NOISE_RHO ** 2))

    def logprob(u, mu, eps_prev, log_std):
        # u = mu + std * eps,  eps = rho * eps_prev + sqrt(1 - rho^2) * N(0, 1)
        log_std = jp.maximum(log_std, STD_FLOOR)
        std = jp.exp(log_std)
        return jp.sum(-0.5 * ((u - mu - std * NOISE_RHO * eps_prev) / (std * shrink)) ** 2 - log_std - np.log(shrink), -1)

    def rollout(params, env, bst, eps, key, guide, support):
        W = weights(params["brain"])
        std = jp.exp(jp.maximum(params["brain"]["log_std"], STD_FLOOR))

        def one(carry, _):
            env, bst, eps, key = carry
            key, ka, kact = jax.random.split(key, 3)
            env, seen = v_sense(env)
            env, sync = v_follow(env, walk)
            fresh = seen["new_life"][:, None]
            bst = jax.tree_util.tree_map(lambda x: jp.where(fresh, 0.0, x), bst)
            eps_prev = jp.where(fresh, 0.0, eps)
            priv = critic_view(env, guide, support)
            bst, mu = bstep(params["brain"], W, bst, seen)
            eps = NOISE_RHO * eps_prev + shrink * jax.random.normal(ka, mu.shape)
            u = mu + std * eps
            logp = logprob(u, mu, eps_prev, params["brain"]["log_std"])
            shown = v_show(env, walk)                                  # what the teacher's walk does on each line now
            a = (1.0 - guide) * jp.tanh(u) + guide * shown             # its hands on the legs, as strong as `guide`
            env, r, done, st = v_act(env, a, jax.random.split(kact, B), support)
            return (env, bst, eps, key), {**seen, "priv": priv, "u": u, "eps_prev": eps_prev, "logp": logp,
                                          "shown": shown, "r": r, "done": done, "in_step": sync["conf"], **st}

        (env, bst, eps, key), traj = jax.lax.scan(one, (env, bst, eps, key), None, length=T)
        return env, bst, eps, traj, critic_view(env, guide, support)

    def advantages(params, traj, priv_last, pstat):
        pm, ps = pstat
        val = lambda x: value(params["critic"], jp.clip((x - pm) / ps, -10, 10))
        vals, last = val(traj["priv"]), val(priv_last)
        keep = 1.0 - traj["done"].astype(jp.float32)

        def back(carry, x):
            adv, nxt = carry
            r, v, m = x
            delta = r * args.reward_scale + args.gamma * nxt * m - v
            adv = delta + args.gamma * args.lam * m * adv
            return (adv, v), adv
        _, adv = jax.lax.scan(back, (jp.zeros(B), last), (traj["r"], vals, keep), reverse=True)
        return adv, adv + vals

    def loss_fn(params, mb, bst0, pstat, imit_w):
        W = weights(params["brain"])
        pb = params["brain"]

        def one(s, x):
            s = jax.tree_util.tree_map(lambda a: jp.where(x["new_life"][:, None], 0.0, a), s)
            s, mu = bstep(pb, W, s, x)
            return s, (logprob(x["u"], mu, x["eps_prev"], pb["log_std"]), mu)
        _, (logp, mu) = jax.lax.scan(one, bst0, mb)
        ratio = jp.exp(logp - mb["logp"])
        adv = (mb["adv"] - mb["adv"].mean()) / (mb["adv"].std() + 1e-8)
        pg = -jp.mean(jp.minimum(ratio * adv, jp.clip(ratio, 1 - args.clip, 1 + args.clip) * adv))
        pm, ps = pstat
        vl = jp.mean((value(params["critic"], jp.clip((mb["priv"] - pm) / ps, -10, 10)) - mb["ret"]) ** 2)
        imit = jp.mean((jp.tanh(mu) - mb["shown"]) ** 2)                # the teacher: "your lines should have done this"
        return pg + 0.5 * vl - args.ent * pb["log_std"] + imit_w * imit, \
            {"pg": pg, "vl": vl, "imit": imit, "ratio_dev": jp.mean(jp.abs(ratio - 1))}

    def update(params, opt_state, traj, bst0, adv, ret, pstat, key, imit_w):
        data = {k: traj[k] for k in ("feats", "C", "a_hist", "tag", "cmd", "new_life", "priv", "u", "eps_prev", "logp", "shown")}
        data["adv"], data["ret"] = adv, ret
        nm, size = args.minibatches, B // args.minibatches
        perms = jax.vmap(lambda k: jax.random.permutation(k, B))(jax.random.split(key, args.epochs))
        sels = perms[:, : nm * size].reshape(args.epochs * nm, size)

        def one(carry, sel):
            params, opt_state = carry
            mb = jax.tree_util.tree_map(lambda a: a[:, sel], data)
            b0 = jax.tree_util.tree_map(lambda a: a[sel], bst0)
            (_, m), g = jax.value_and_grad(loss_fn, has_aux=True)(params, mb, b0, pstat, imit_w)
            g = jax.lax.pmean(g, "d")                              # average gradients over the GPUs
            upd, opt_state = opt.update(g, opt_state, params)
            return (optax.apply_updates(params, upd), opt_state), m
        (params, opt_state), ms = jax.lax.scan(one, (params, opt_state), sels)
        return params, opt_state, jax.tree_util.tree_map(lambda x: jax.lax.pmean(x.mean(), "d"), ms)

    def iteration(params, opt_state, env, bst, eps, pstat, key, help_):
        # one "help" number, taken away in order: hands on the legs (first half), then the harness and
        # the corrections together (second half)
        guide, support = jp.clip(2 * help_ - 1, 0.0, 1.0), jp.clip(2 * help_, 0.0, 1.0)
        kr, ku = jax.random.split(key)
        bst0 = bst
        env, bst, eps, traj, priv_last = rollout(params, env, bst, eps, kr, guide, support)
        adv, ret = advantages(params, traj, priv_last, pstat)
        params, opt_state, m = update(params, opt_state, traj, bst0, adv, ret, pstat, ku, IMIT_WEIGHT * support)
        pstat = norm_stats(traj["priv"].reshape(-1, traj["priv"].shape[-1]), "d")
        asked = traj["cmd"][..., 0] > 0.3
        report = {"reward_per_s": jax.lax.pmean(traj["r"].mean(), "d") / CTRL_DT,
                  "track": jax.lax.pmean(traj["track"].mean(), "d"),
                  "progress": jax.lax.psum(traj["along"].sum(), "d") / jp.maximum(jax.lax.psum(traj["asked"].sum(), "d"), 1e-6),
                  "in_step": jax.lax.pmean(traj["in_step"].mean(), "d"),
                  "falls_per_1k": jax.lax.pmean(traj["fell"].mean(), "d") * 1000,
                  "v_fwd_asked": jax.lax.psum((traj["v_fwd"] * asked).sum(), "d") / jp.maximum(jax.lax.psum(asked.sum(), "d"), 1),
                  **m}
        return params, opt_state, env, bst, eps, pstat, report

    step = jax.pmap(iteration, axis_name="d")
    keys = jax.random.split(key, nd + 1)
    key = keys[0]
    env = jax.pmap(lambda k: v_reset(jax.random.split(k, B)))(keys[1:])
    bst = jax.pmap(lambda _: init_state(B))(jp.arange(nd))
    eps = jp.zeros((nd, B, body["J"]))
    pstat = jax.pmap(lambda e: norm_stats(critic_view(e, 1.0, 1.0), "d"), axis_name="d")(env)

    per_iter = args.envs * T
    iters = max(1, int(args.steps) // per_iter)
    every = max(1, iters // max(args.evals, 1))
    history = []
    help_, ema = 1.0, None
    print(f"training on {nd} GPU(s): {iters} iterations x {per_iter:,} steps; setup {time.time() - t0:.0f} s", flush=True)
    for it in range(iters):
        key, k = jax.random.split(key)
        params_r, opt_r, env, bst, eps, pstat, rep_ = step(params_r, opt_r, env, bst, eps, pstat, jax.random.split(k, nd),
                                                            jp.full((nd,), help_))
        # the teacher lets go a little whenever the dog keeps up with less help, and steps back in if it
        # collapses; whatever happens, all help is gone by FADE_BY of training
        now = np.array([float(np.asarray(rep_["progress"])[0]), float(np.asarray(rep_["falls_per_1k"])[0])])
        ema = now if ema is None else 0.95 * ema + 0.05 * now
        if ema[0] >= PROGRESS_PASS * lesson["progress_ref"] and ema[1] <= max(2 * lesson["falls_ref"], 1.0):
            help_ -= ALPHA_STEP
        elif ema[0] < 0.5 * PROGRESS_PASS * lesson["progress_ref"]:
            help_ += ALPHA_STEP
        help_ = float(np.clip(help_, 0.0, np.clip((FADE_BY * iters - it) / ((FADE_BY - HOLD_UNTIL) * iters), 0.0, 1.0)))
        if it % every == 0 or it == iters - 1:
            row = {"iter": it, "steps": (it + 1) * per_iter, "minutes": round((time.time() - t0) / 60, 2),
                   **{k: float(np.asarray(v)[0]) for k, v in rep_.items()},
                   "std": float(np.exp(np.asarray(params_r["brain"]["log_std"])[0])), "help": help_}
            history.append(row)
            (out / "metrics.json").write_text(json.dumps(history, indent=1))
            print(f"[{row['minutes']:6.1f} min] {row['steps']:>12,} steps  help {help_:.2f}  reward/s {row['reward_per_s']:.3f}  "
                  f"progress {row['progress']:.2f} (teacher alone {lesson['progress_ref']:.2f})  falls/1k {row['falls_per_1k']:5.2f}  "
                  f"forward when asked {row['v_fwd_asked']:+.2f}  legs stepping {row['in_step']:.2f}  copies teacher (err) {row['imit']:.4f}  "
                  f"std {row['std']:.2f}  pg {row['pg']:+.3f}  vl {row['vl']:.3f}", flush=True)
            export(jax.tree_util.tree_map(lambda x: np.asarray(x)[0], params_r["brain"]), binfo, out / "brain_dog.npz")
    trained = jax.tree_util.tree_map(lambda x: np.asarray(x)[0], params_r["brain"])
    return {"trained": trained, "initial": initial_brain, "watched": watched_brain, "model": model, "body": body, "teacher": teacher, "walk": walk,
            "bstep": bstep, "weights": weights, "init_state": init_state, "binfo": binfo,
            "minutes": (time.time() - t0) / 60}


def export(brain_params, binfo, path: Path) -> None:
    import numpy as np
    flat = {}
    for k, v in brain_params.items():
        if isinstance(v, dict):
            flat.update({f"{k}/{kk}": np.asarray(vv) for kk, vv in v.items()})
        else:
            flat[k] = np.asarray(v)
    np.savez(path, **flat, meta=json.dumps({**binfo, "tick_dt": TICK_DT, "ctrl_dt": CTRL_DT, "lags": LAGS,
                                            "self_map_rate": SELF_MAP_RATE, "D": D, "DT": DT, "heads": HEADS}))


# ============================================================================ evaluation
GOALS = {"stand": [0.0, 0.0, 0.0], "walk": [0.3, 0.0, 0.0], "trot": [0.6, 0.0, 0.0], "back": [-0.3, 0.0, 0.0],
         "turn left": [0.0, 0.0, 0.8]}


def evaluate(res, brains: dict, out: Path, lives: int = 16, steps: int = LIFE_STEPS - 1) -> dict:
    """New lives (new shuffles), one goal held all life, no help at all: does the brain discover its body
    and use it? The teacher's own walk on the same goals is the yardstick. Films the teacher and the
    trained brain so they can be watched."""
    import jax
    import jax.numpy as jp
    import numpy as np
    body, teacher, walk = res["body"], res["teacher"], res["walk"]
    v_reset, v_sense = jax.vmap(body["reset"]), jax.vmap(body["sense"])
    v_act = jax.vmap(body["act"], in_axes=(0, 0, 0, None))
    v_show = jax.vmap(teacher["show"], in_axes=(0, None))
    v_follow = jax.vmap(teacher["follow"], in_axes=(0, None))

    def life(bp, goal, key, by_teacher):
        env = v_reset(jax.random.split(key, lives))
        env["info"]["cmd"] = jp.broadcast_to(goal, (lives, 3))
        W = None if by_teacher else res["weights"](bp)

        def one(carry, k):
            env, s = carry
            env, seen = v_sense(env)
            env, _ = v_follow(env, walk)
            if by_teacher:
                a = v_show(env, walk)
            else:
                s, mu = res["bstep"](bp, W, s, seen)
                a = jp.tanh(mu)
            env, _, _, st = v_act(env, a, jax.random.split(k, lives), 0.0)
            env["info"]["cmd"] = jp.broadcast_to(goal, (lives, 3))             # hold the goal for the whole life
            return (env, s), (st["v_fwd"], st["wz"], st["fell"], st["track"], env["d"].qpos[0],
                              env["d"].geom_xpos[0, body["feet"], 2])
        s0 = jp.zeros(()) if by_teacher else res["init_state"](lives)
        return jax.lax.scan(one, (env, s0), jax.random.split(key, steps))[1]
    life = jax.jit(life, static_argnames=("by_teacher",))

    results = []
    who = [("teacher", None), *brains.items()]
    for name, bp in who:
        for goal_name, goal in GOALS.items():
            vf, wz, fell, track, qpos, foot_z = (np.asarray(x) for x in life(bp, jp.array(goal), jax.random.PRNGKey(21),
                                                                             by_teacher=name == "teacher"))
            win = min(250, steps)                                               # 5 s windows
            late = slice(min(500, steps // 2), min(600, steps // 2 + 100))      # 2 s of footfalls, one life
            row = {"who": name, "goal": goal_name, "cmd": goal,
                   "forward_by_5s": [round(float(vf[i:i + win].mean()), 2) for i in range(0, len(vf), win)],
                   "turn_by_5s": [round(float(wz[i:i + win].mean()), 2) for i in range(0, len(wz), win)],
                   "track": round(float(track.mean()), 3),
                   "falls_per_life": round(float(fell.sum(0).mean()), 2),
                   "lives_spread_forward": round(float(vf[-500:].mean(0).std()), 2),
                   "footfalls": {f: "".join("#" if z < FOOT_UP else "." for z in foot_z[late, i]) for i, f in enumerate(FEET)}}
            results.append(row)
            print(f"eval {name:9s} {goal_name:9s}: forward by 5 s {row['forward_by_5s']}  turn {row['turn_by_5s']}  "
                  f"track {row['track']}  falls/life {row['falls_per_life']}  spread across lives {row['lives_spread_forward']}",
                  flush=True)
            if goal_name in ("trot", "turn left") and name in ("teacher", "trained", "hands fade", "taught"):
                print("\n".join(f"      {f} {s}" for f, s in row["footfalls"].items()), flush=True)
                start = min(250, len(qpos) // 2)
                try:
                    film(res["model"], qpos[start:start + 200], out / f"film_{name}_{goal_name.replace(' ', '_')}")
                except Exception as e:                   # rendering needs an OpenGL context; the numbers stand without it
                    print(f"film {name} {goal_name}: skipped ({e!r})", flush=True)
    (out / "eval.json").write_text(json.dumps(results, indent=1))
    return {"eval": results}


def film(model, qpos, path: Path) -> None:
    """A GIF (25 frames/s) and a contact sheet (12 frames, 0.12 s apart) of a recorded stretch of one life."""
    import mujoco
    from PIL import Image
    d = mujoco.MjData(model)
    r = mujoco.Renderer(model, 240, 320)
    cam = mujoco.MjvCamera()                              # from the side, following the trunk, close enough to see feet
    cam.type, cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, model.body("base").id
    w, x, y, z = qpos[0][3:7]
    heading = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))   # the dog starts facing a random way
    cam.distance, cam.azimuth, cam.elevation = 1.1, heading + 90.0, -12.0
    frames = []
    try:
        for q in qpos[::2]:
            d.qpos[:] = q
            mujoco.mj_forward(model, d)
            r.update_scene(d, camera=cam)
            frames.append(Image.fromarray(r.render()))
    finally:
        r.close()
    frames[0].save(path.with_suffix(".gif"), save_all=True, append_images=frames[1:], duration=40, loop=0)
    sheet = Image.new("RGB", (320 * 4, 240 * 3))
    for i, f in enumerate(frames[::3][:12]):
        sheet.paste(f, (320 * (i % 4), 240 * (i // 4)))
    sheet.save(path.with_suffix(".png"))


def teach_only(args, out: Path) -> None:
    """The teacher practises its walk on the body and is filmed doing it: no brain, no training."""
    import jax
    go2 = Path(args.menagerie) / "unitree_go2" if args.menagerie else fetch_go2(Path(tempfile.gettempdir()))
    model = build_model(go2)
    body = make_body(model)
    teacher = make_teacher(model, body)
    walk, _ = teacher["calibrate"](jax.random.PRNGKey(args.seed), out)
    evaluate({"model": model, "body": body, "teacher": teacher, "walk": walk}, {}, out, lives=16, steps=500)


# ============================================================================ diagnosis
def copy_test(args, out: Path) -> dict:
    """Can the brain learn from the teacher at all? Each readout design watches the teacher walk the dog for
    the same number of steps (see `watch`). A design that cannot get well below the do-nothing error
    (always commanding zero) cannot be taught to walk this way."""
    import jax
    go2 = Path(args.menagerie) / "unitree_go2" if args.menagerie else fetch_go2(Path(tempfile.gettempdir()))
    model = build_model(go2)
    body = make_body(model)
    teacher = make_teacher(model, body)
    walk, _ = teacher["calibrate"](jax.random.PRNGKey(args.seed), out)
    core = load_core()
    report = {}
    for readout in READOUTS:
        bp, weights, init_state, bstep, _ = make_brain(core, jax.random.PRNGKey(args.seed), args.bias0, args.goal_scale,
                                                       args.gain0, readout)
        selftest(bstep, weights, init_state, bp, body["K"], body["J"])
        _, report[readout] = watch(bp, (weights, init_state, bstep), body, teacher, walk, args.envs, args.unroll,
                                   args.watch_steps, args.watch_lr, args.seed, readout)
        (out / "copy_test.json").write_text(json.dumps(report, indent=1))
    return report


def teach_check(args, out: Path) -> None:
    """Is the brain learning from the teaching at all? Measure the brain moving the dog's legs by itself, teach
    it (the teacher's hands on the legs fading to nothing, then only its corrections), and measure again the
    same way. Then the taught brain and the untaught one walk new lives with no help, next to the teacher."""
    import jax
    go2 = Path(args.menagerie) / "unitree_go2" if args.menagerie else fetch_go2(Path(tempfile.gettempdir()))
    model = build_model(go2)
    body = make_body(model)
    teacher = make_teacher(model, body)
    walk, _ = teacher["calibrate"](jax.random.PRNGKey(args.seed), out)
    if args.fixed_body:
        body = make_body(model, wiring=12345)             # the teacher's walk is the same; only the brain's view changes
        teacher = make_teacher(model, body)
        print("DIAGNOSTIC CONTROL: the body is presented the same way every life (no reshuffle)", flush=True)
    bp, weights, init_state, bstep, _ = make_brain(load_core(), jax.random.PRNGKey(args.seed), args.bias0, args.goal_scale,
                                                   args.gain0, args.readout)
    taught, curve = watch(bp, (weights, init_state, bstep), body, teacher, walk, args.envs, args.unroll, args.watch_steps,
                          args.watch_lr, args.seed, "hands fade", fade_hands=True, probe=True)
    (out / "teach_check.json").write_text(json.dumps(curve, indent=1))
    evaluate({"model": model, "body": body, "teacher": teacher, "walk": walk, "bstep": bstep, "weights": weights,
              "init_state": init_state}, {"before": bp, "taught": taught}, out, lives=16, steps=500)


def watch_test(args, out: Path) -> None:
    """Does the brain walk alone after learning only from the teacher's corrections? Two brains watch for the
    same steps: one with the teacher's hands always partly on (30-100%), one whose hands fade to nothing so it
    ends up moving the legs itself while being corrected. Then both walk new lives with no help at all."""
    import jax
    go2 = Path(args.menagerie) / "unitree_go2" if args.menagerie else fetch_go2(Path(tempfile.gettempdir()))
    model = build_model(go2)
    body = make_body(model)
    teacher = make_teacher(model, body)
    walk, _ = teacher["calibrate"](jax.random.PRNGKey(args.seed), out)
    bp, weights, init_state, bstep, _ = make_brain(load_core(), jax.random.PRNGKey(args.seed), args.bias0, args.goal_scale,
                                                   args.gain0, args.readout)
    brains, curves = {}, {}
    for name, fade in (("hands on", False), ("hands fade", True)):
        brains[name], curves[name] = watch(bp, (weights, init_state, bstep), body, teacher, walk, args.envs, args.unroll,
                                           args.watch_steps, args.watch_lr, args.seed, name, fade_hands=fade)
        (out / "watching.json").write_text(json.dumps(curves, indent=1))
    evaluate({"model": model, "body": body, "teacher": teacher, "walk": walk, "bstep": bstep, "weights": weights,
              "init_state": init_state}, brains, out)


def diagnose(args, out: Path) -> dict:
    """For a grid of starting settings (untrained): does a held goal reach the motor neurons, and are they active?"""
    import jax
    go2 = Path(args.menagerie) / "unitree_go2" if args.menagerie else fetch_go2(Path(tempfile.gettempdir()))
    body = make_body(build_model(go2))
    core = load_core()
    report = {}
    for bias0 in (0.0, 0.1, 0.3):
        for goal_scale in (0.3, 1.0):
            for gain0 in (2.0, 4.0):
                bp, weights, init_state, bstep, binfo = make_brain(core, jax.random.PRNGKey(args.seed), bias0, goal_scale, gain0)
                name = f"bias {bias0} goal {goal_scale} gain {gain0}"
                report[name] = goal_influence(bp, weights, bstep, init_state, body, jax.random.PRNGKey(5),
                                              motor_idx=binfo["motor_idx"])
                print(f"diagnose {name:28s}: " + "  ".join(f"{k} {v:.4f}" for k, v in report[name].items()), flush=True)
    (out / "diagnose.json").write_text(json.dumps(report, indent=1))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=float, default=150e6)
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--unroll", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=3e-3)
    ap.add_argument("--reward_scale", type=float, default=10.0)
    ap.add_argument("--evals", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/kaggle/working" if Path("/kaggle/working").exists() else "results/brain_dog")
    ap.add_argument("--menagerie", default="")
    ap.add_argument("--selftest", action="store_true", help="local: permutation checks and a tiny run")
    ap.add_argument("--diagnose", action="store_true", help="probe goal -> motor signal flow for a grid of start settings")
    ap.add_argument("--bias0", type=float, default=0.0, help="tonic drive on every unit at the start")
    ap.add_argument("--goal_scale", type=float, default=0.3, help="initial strength of the goal input")
    ap.add_argument("--gain0", type=float, default=None, help="initial connectome gain (default GAIN0)")
    ap.add_argument("--teacher_only", action="store_true", help="the teacher practises and is filmed; no brain")
    ap.add_argument("--readout", default="vector", choices=READOUTS, help="how output lines read the motor neurons")
    ap.add_argument("--copy_test", action="store_true", help="can each readout design learn to copy the teacher?")
    ap.add_argument("--watch_steps", type=float, default=15e6, help="steps spent watching the teacher before practising")
    ap.add_argument("--watch_test", action="store_true", help="does a brain taught only by corrections walk alone?")
    ap.add_argument("--teach_check", action="store_true", help="quick: does the brain learn from the teaching at all?")
    ap.add_argument("--fixed_body", action="store_true", help="diagnostic control: no reshuffle between lives")
    ap.add_argument("--watch_lr", type=float, default=1e-3)
    args = ap.parse_args([]) if JOB_ARGS else ap.parse_args()
    for k, v in JOB_ARGS.items():
        setattr(args, k, v)
    if args.selftest:
        args.steps, args.envs, args.unroll, args.minibatches, args.evals, args.watch_steps = 512, 8, 8, 2, 4, 128
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if Path("/kaggle").exists():
        os.environ.setdefault("MUJOCO_GL", "egl")        # headless GPU rendering for the films
    ensure_packages()
    import jax
    print("jax", jax.__version__, "devices", jax.devices(), flush=True)
    if args.diagnose:
        diagnose(args, out)
        print("DONE diagnose", flush=True)
        return
    if args.teach_check:
        teach_check(args, out)
        print("DONE teach check", flush=True)
        return
    if args.watch_test:
        watch_test(args, out)
        print("DONE watch test", flush=True)
        return
    if args.copy_test:
        copy_test(args, out)
        print("DONE copy test", flush=True)
        return
    if args.teacher_only:
        teach_only(args, out)
        print("DONE teacher", flush=True)
        return
    res = train(args, out)
    brains = {"untrained": res["initial"], "watched only": res["watched"], "trained": res["trained"]}
    summary = evaluate(res, brains, out, lives=4, steps=100) if args.selftest else evaluate(res, brains, out)
    summary.update({"minutes": round(res["minutes"], 1), "steps": int(args.steps), "envs": args.envs})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("DONE", json.dumps({"minutes": summary["minutes"], "steps": summary["steps"]}), flush=True)


if __name__ == "__main__":
    main()
