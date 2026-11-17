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
    """The Go2, held in the air by its trunk. Torque motors; every joint has a hard limit."""

    def __init__(self, go2: Path):
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
        self.reset()

    def reset(self):
        self.mj.mj_resetDataKeyframe(self.m, self.d, self.m.keyframe("home").id)
        self.hold()
        self.mj.mj_forward(self.m, self.d)

    def hold(self):
        self.d.qpos[0:3] = [0.0, 0.0, HELD_AT]
        self.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.d.qvel[0:6] = 0.0

    def step(self, torque):
        self.d.ctrl[:] = np.clip(torque, -self.tmax, self.tmax)
        for _ in range(self.subs):
            self.mj.mj_step(self.m, self.d)
            self.hold()

    def angles(self):
        return self.d.qpos[self.qadr].copy()

    def feel(self):
        """What the joints' own sense organs report: angle, speed, effort, and the push-back at a limit."""
        return np.concatenate([self.d.qpos[self.qadr], self.d.qvel[self.vadr],
                               self.d.actuator_force, self.d.qfrc_constraint[self.vadr]])


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
        # sensory adaptation: each nerve settles to what it usually reports
        self.mean, self.var, self.seen = np.zeros(n_lines), np.ones(n_lines), 0
        # the plastic motor-to-sensory synapses: its body map, all zero at birth
        self.map = np.zeros((len(self.motor), len(self.sense)))
        self.fibre_col = pick                                                         # [lines, fibres] -> column in map

    def rates(self):
        return np.tanh(np.maximum(self.v, 0.0))

    def step(self, felt, motor_drive):
        self.seen += 1
        a = max(1.0 / self.seen, 0.002)                                  # adapts within ~10 s
        self.mean += a * (felt - self.mean)
        self.var += a * ((felt - self.mean) ** 2 - self.var)
        zf = np.clip((felt - self.mean) / np.sqrt(self.var + 1e-6), -4.0, 4.0)
        I = np.zeros(self.N)
        np.add.at(I, self.fibre_neuron.ravel(), (self.fibre_gain * zf[:, None]).ravel())
        I[self.motor] += motor_drive
        self.v += ALPHA * (-self.v + self.W @ self.rates() + I)
        return self.rates()

    def torque(self, r, tmax):
        net = np.bincount(self.drives, weights=self.push * r[self.motor], minlength=self.n_motors)
        return tmax * np.tanh(1.5 * net / np.sqrt(np.bincount(self.drives, minlength=self.n_motors) + 1.0))


def babble(body, brain, minutes, rng, log, label):
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
    for t in range(steps):
        burst += BRAIN_DT / BABBLE_TAU * -burst + np.sqrt(2 * BRAIN_DT / BABBLE_TAU) * rng.standard_normal(len(burst))
        r = brain.step(body.feel(), BABBLE_SIZE * burst)
        body.step(brain.torque(r, body.tmax))
        m = r[brain.motor]
        m_mean += 0.01 * (m - m_mean)
        s = r[brain.sense]
        brain.map += HEBB_RATE * np.outer(trace, s - s_prev)            # activity, then what was felt shortly after
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


def control_test(body, brain, scores, log, label, seconds=1.5):
    """Ask the brain to drive each joint to each end, using only its own body map."""
    J = len(body.joint)
    best = np.abs(scores).argmax(1)
    rows = []
    for j in range(J):
        for d in (1.0, -1.0):
            body.reset()
            brain.v[:] = 0.0
            mine = best == j
            drive = np.where(mine, np.maximum(d * scores[:, j], 0.0), 0.0)
            drive = 3.0 * drive / (drive.max() + 1e-12)
            q0 = body.angles()
            for _ in range(int(seconds / BRAIN_DT)):
                r = brain.step(body.feel(), drive)
                body.step(brain.torque(r, body.tmax))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0, help="minutes of babbling per dog")
    ap.add_argument("--dogs", type=int, default=3, help="independent dogs (same genome, own babbling)")
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

    body = Body(fetch_go2(Path(tempfile.gettempdir())))
    core = load_core()
    log(f"body: Go2, {len(body.joint)} joints ({', '.join(body.names)}); held at {HELD_AT} m like a baby; "
        f"physics {body.subs} steps per brain step")
    log(f"brain: fly connectome, {len(core['role'])} neurons, {len(core['pre'])} connections; it is told nothing about the body")
    results = []
    for dog in range(args.dogs):
        brain = Brain(core, len(body.feel()), len(body.joint), np.random.default_rng(GENOME))     # same genome
        label = f"dog {dog + 1}/{args.dogs}"
        body.reset()
        before = control_test(body, brain, joint_scores(brain, body), log, label + ", newborn (map empty)")
        body.reset()
        explored = babble(body, brain, args.minutes, np.random.default_rng(1000 + dog), log, label)
        scores = joint_scores(brain, body)
        after = control_test(body, brain, scores, log, label + f", after {args.minutes:g} min of babbling")
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
        results.append({"dog": dog + 1, "map": acc, "per_joint": per_joint, "control_newborn": before, "control_after": after})
    reach = lambda rows: float(np.mean([np.clip(r_["reach"], -1, 1) for r_ in rows]))
    log("VERDICT babbling: " + "; ".join(
        f"dog {r_['dog']}: body map right joint {r_['map']['joint']:.0%} (direction too {r_['map']['both']:.0%}), "
        f"control {reach(r_['control_newborn']):.0%} newborn -> {reach(r_['control_after']):.0%} after"
        for r_ in results) + f" (chance for the map: {1 / len(body.joint):.0%})")
    (out / "babble.json").write_text(json.dumps({"minutes": args.minutes, "genome": GENOME, "dogs": results,
                                                 "settings": {k: globals()[k] for k in ("BRAIN_DT", "ALPHA", "GAIN",
                                                              "BABBLE_TAU", "BABBLE_SIZE", "NERVE_FIBRES", "TRACE",
                                                              "HEBB_RATE", "HELD_AT")}}, indent=1))
    (out / "babble_log.txt").write_text("\n".join(lines_out))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
