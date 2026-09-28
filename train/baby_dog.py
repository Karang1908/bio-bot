"""The fly-wired brain learns the Go2 dog from scratch, the way a baby learns its body, in stages.

One baby = one brain and one body, kept for its whole childhood. The body is wired to the brain at random
(its sensor and motor lines shuffled, re-signed and rescaled by one fixed random draw) and the brain is
never told which line is which. Stage by stage:

  1 babble   the lines twitch by themselves while the trunk is held (a baby's spontaneous twitches); the
             brain learns to predict what each twitch does to every sensor: its own model of its body
  2 name     (next) the teacher moves a joint and says its name; the brain links words to parts it knows
  3 act      (next) "lift front-left leg": the brain tries, the teacher says what it should have done
  4 walk     (next) the teacher talks it through steps, then only says "walk"

Each stage is a short job that ends with a number saying whether the brain learned, and saves the brain for
the next stage.

  Kaggle:  scripts/kaggle_job.py push baby_dog --embed train/brain_dog.py --embed data/malecns/core_v1.npz
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import time
from pathlib import Path

JOB_ARGS: dict = {}   # filled in by scripts/kaggle_job.py
EMBEDDED: dict = {}   # data files shipped inside the job (base64), filled in by scripts/kaggle_job.py

WIRING = 20260927        # this baby's one random wiring
BABBLE_ACTIVE = 0.25     # share of lines twitching at any moment
BABBLE_SWITCH = 0.02     # per step: chance the twitching set changes (~every 1 s)
BABBLE_RHO = 0.95        # twitches are smooth (~0.4 s)
BABBLE_SIZE = 0.5        # command size of a twitch
# the words the teacher speaks (the brain hears them as a set of active words; it is never told what they mean)
VOCAB = ["front-left", "front-right", "rear-left", "rear-right", "hip", "thigh", "knee",
         "move", "lift", "swing-forward", "swing-back", "put-down", "stand", "walk",
         "turn-left", "turn-right", "back", "sit", "lie-down"]
LEGS, JOINTS = VOCAB[0:4], VOCAB[4:7]
LEG_OF = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}
JOINT_OF = {"hip": 0, "thigh": 1, "calf": 2}
HELD_OUT = [("rear-left", "knee"), ("front-right", "hip")]   # never taught: understood only if the words are
MOVE_SIZE = 0.5          # "move X": command X's line this much, the others not at all
WOBBLE = 0.1             # the brain's own movements vary this much while it learns
CTRL_SWITCH = 0.02       # per step: chance the teacher gives a new instruction (~every 1 s)
WALK_STOP = 0.01         # per step while talking the dog through a walk: chance it stops (~2 s walks)
HOLDS = (5, 7, 10, 14)   # steps per spoken step of the walk the teacher tries (0.1-0.28 s)
WALK_CMD = (0.3, 0.0, 0.0)   # what "walk" means to the teacher: forward at 0.3 m/s (the brain only hears "walk")
FALL_COST = 2.0          # practice: a fall costs this much reward (about 2 s of doing everything right)
FEET_AIR = 0.3           # practice: while moving, each foot should be off the ground at least this share of the time
STEP_REWARD = 0.5        # practice: reward per second for all four feet stepping (the rear legs too)
POSE_REWARD = 1.0        # practice: reward per second for holding the asked rest pose
GUIDE_REST = 5.0         # practice: the teacher's guide counts this much for rest poses (new words are learned fast)
PUSH_P, PUSH_SIZE = 0.004, 0.5   # practice: a random shove about every 5 s, up to 0.5 m/s sideways/forward
GUIDE = 0.5              # practice: how much the teacher's corrections still count (a light guide)
# the teacher finds each rest pose on the body by trying these (radians from standing, per joint group)
POSE_TRIES = {
    "sit": [{"rear_thigh": rt, "rear_calf": rc, "front_thigh": ft, "front_calf": 0.0}
            for rt in (-0.6, -0.3, 0.3, 0.6, 0.9) for rc in (-0.3, -0.6, -0.8) for ft in (-0.3, 0.0, 0.3)],
    "lie-down": [{"rear_thigh": t, "rear_calf": c, "front_thigh": t, "front_calf": c}
                 for t in (-0.3, 0.0, 0.3, 0.6, 0.9) for c in (-0.4, -0.6, -0.8)],
}
POSE_GOAL = {"sit": "nose up ~0.5 rad, chest up (trunk >= half height), not tipped",
             "lie-down": "trunk low (~0.4 of standing) and level, not tipped"}
GAITS = {"walk": WALK_CMD, "turn-left": (0.0, 0.0, 0.6), "turn-right": (0.0, 0.0, -0.6), "back": (-0.25, 0.0, 0.0)}
KEEP_BEAT = 0.1          # the teacher keeps its own beat, nudged this much per step toward the brain's rhythm


def import_brain_dog():
    """The body, the brain and the teacher live in brain_dog.py (shipped inside the job on Kaggle)."""
    if "brain_dog.py" in EMBEDDED:
        d = Path(tempfile.mkdtemp())
        (d / "brain_dog.py").write_bytes(base64.b64decode(EMBEDDED["brain_dog.py"]))
        sys.path.insert(0, str(d))
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    import brain_dog
    brain_dog.EMBEDDED = EMBEDDED                        # so it finds the embedded connectome
    return brain_dog


def save_brain(bp, path: Path, meta: dict) -> None:
    import numpy as np
    flat = {}
    for k, v in bp.items():
        if isinstance(v, dict):
            flat.update({f"{k}/{kk}": np.asarray(vv) for kk, vv in v.items()})
        else:
            flat[k] = np.asarray(v)
    np.savez(path, **flat, meta=json.dumps(meta))


def find_brain(path: str) -> Path:
    """The previous stage's brain: a local file, or the one in the attached Kaggle dataset."""
    if path and Path(path).exists():
        return Path(path)
    found = sorted(Path("/kaggle/input").glob("**/baby_dog.npz")) if Path("/kaggle/input").exists() else []
    if not found:
        raise SystemExit(f"STOP: no previous-stage brain found (looked at {path!r} and /kaggle/input)")
    return found[0]


def carry_over(fresh: dict, learned: dict) -> dict:
    """Start from the fresh brain's shapes and take every learned part that fits. A table that has grown (new
    words appended to the vocabulary) keeps its learned rows and gets fresh ones for the new entries."""
    import numpy as np
    out, new = {}, []

    def take(f, l, name):
        if l is None:
            new.append(name)
            return f
        f, l = np.asarray(f), np.asarray(l)
        if f.shape == l.shape:
            return l
        if f.ndim == l.ndim and f.shape[1:] == l.shape[1:] and f.shape[0] > l.shape[0]:
            new.append(f"{name} (+{f.shape[0] - l.shape[0]} rows)")
            return np.concatenate([l, f[l.shape[0]:]])
        new.append(name)
        return f
    for k, v in fresh.items():
        if isinstance(v, dict):
            out[k] = {kk: take(vv, learned.get(k, {}).get(kk), f"{k}/{kk}") for kk, vv in v.items()}
        else:
            out[k] = take(v, learned.get(k), k)
    print(f"carried over the learned brain; new parts: {new or 'none'}", flush=True)
    return out


def load_brain(path: Path) -> dict:
    import numpy as np
    bp = {}
    with np.load(path) as z:
        for k in z.files:
            if k == "meta":
                continue
            if "/" in k:
                a, b = k.split("/", 1)
                bp.setdefault(a, {})[b] = z[k]
            else:
                bp[k] = z[k]
    return bp


# ============================================================================ stage 1: babbling
def babble(args, bd, out: Path) -> dict:
    """The lines twitch by themselves (a few at a time, smoothly) while the harness holds the trunk; the brain
    only watches and feels, and learns to predict what each twitch does to each sensor line next step."""
    import jax
    import jax.numpy as jp
    import numpy as np
    import optax

    t0 = time.time()
    go2 = Path(args.menagerie) / "unitree_go2" if args.menagerie else bd.fetch_go2(Path(tempfile.gettempdir()))
    model = bd.build_model(go2)
    body = bd.make_body(model, wiring=WIRING)
    bp, weights, init_state, bstep, binfo = bd.make_brain(bd.load_core(), jax.random.PRNGKey(args.seed), args.bias0,
                                                          args.goal_scale, args.gain0, args.readout)
    predict = bstep.predict
    nd = jax.local_device_count()
    B, T, J = args.envs // nd, args.unroll, body["J"]
    v_reset, v_sense = jax.vmap(body["reset"]), jax.vmap(body["sense"])
    v_act = jax.vmap(body["act"], in_axes=(0, 0, 0, None))
    fresh = lambda tree, new_life: jax.tree_util.tree_map(lambda x: jp.where(new_life[:, None], 0.0, x), tree)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr))
    print(f"baby: {body['K']} sensor lines, {J} motor lines, one fixed random wiring (never told which is which)", flush=True)

    def twitch(tw, key):
        k1, k2, k3 = jax.random.split(key, 3)
        switch = jax.random.uniform(k1, (B, 1)) < BABBLE_SWITCH
        mask = jp.where(switch, jax.random.bernoulli(k2, BABBLE_ACTIVE, (B, J)).astype(jp.float32), tw["mask"])
        x = BABBLE_RHO * tw["x"] + np.sqrt(1 - BABBLE_RHO ** 2) * jax.random.normal(k3, (B, J))
        return {"x": x, "mask": mask}, jp.clip(BABBLE_SIZE * x * mask, -1.0, 1.0)

    def rollout(bp, env, s, tw, key):
        """T+1 steps. At each step the brain takes in what it feels, then the next twitch is sent, and the brain
        is asked what it expects each sensor line to do next; the answer is checked one step later."""
        W = weights(bp)

        def one(carry, k):
            env, s, tw = carry
            k_tw, k_act = jax.random.split(k)
            env, seen = v_sense(env)
            s = fresh(s, seen["new_life"])
            s, _ = bstep(bp, W, s, seen)
            tw, a = twitch(tw, k_tw)
            ask = {**seen, "a_hist": jp.stack([a, seen["a_hist"][..., 0]], -1)}     # "I just sent a"
            env, _, _, _ = v_act(env, a, jax.random.split(k_act, B), 1.0)
            return (env, s, tw), (seen, ask)
        (env, s_end, tw), (seens, asks) = jax.lax.scan(one, (env, s, tw), jax.random.split(key, T + 1))
        return env, s_end, tw, seens, asks

    def loss(bp, seens, asks, s0):
        W = weights(bp)

        def one(s, x):
            seen, ask = x
            s = fresh(s, seen["new_life"])
            s, _ = bstep(bp, W, s, seen)
            return s, predict(bp, s, ask)
        _, pred = jax.lax.scan(one, s0, (seens, asks))
        got = seens["feats"][1:, ..., 1]                                  # what each line actually did next step
        keep = 1.0 - seens["new_life"][1:, :, None].astype(jp.float32)    # not across a new life
        err = jp.sum(keep * (pred[:-1] - got) ** 2) / jp.maximum(keep.sum() * got.shape[-1], 1.0)
        base = jp.sum(keep * got ** 2) / jp.maximum(keep.sum() * got.shape[-1], 1.0)   # guessing "no change"
        return err, base

    def iteration(bp, opt_state, env, s, tw, key, learn):
        s0 = s
        env, s, tw, seens, asks = rollout(bp, env, s, tw, key)
        (err, base), g = jax.value_and_grad(loss, has_aux=True)(bp, seens, asks, s0)
        upd, new_opt = opt.update(jax.lax.pmean(g, "d"), opt_state, bp)
        bp = jax.tree_util.tree_map(lambda old, new: jp.where(learn, new, old), bp, optax.apply_updates(bp, upd))
        opt_state = jax.tree_util.tree_map(lambda old, new: jp.where(learn, new, old), opt_state, new_opt)
        return bp, opt_state, env, s, tw, jax.lax.pmean(err, "d"), jax.lax.pmean(base, "d")

    step = jax.pmap(iteration, axis_name="d")
    rep = lambda x: jax.device_put_replicated(x, jax.local_devices())
    bp_r, opt_r = rep(bp), rep(opt.init(bp))
    keys = jax.random.split(jax.random.PRNGKey(args.seed + 1), nd + 1)
    env = jax.pmap(lambda k: v_reset(jax.random.split(k, B)))(keys[1:])
    s = jax.pmap(lambda _: init_state(B))(jp.arange(nd))
    tw = {"x": jp.zeros((nd, B, J)), "mask": jp.ones((nd, B, J)) * BABBLE_ACTIVE}
    key = keys[0]
    curve = []

    def measure(label, n=8):
        """The brain predicting, not learning: its error and the no-change guess, over n rollouts."""
        nonlocal bp_r, opt_r, env, s, tw, key
        errs, bases = [], []
        for _ in range(n):
            key, k = jax.random.split(key)
            bp_r, opt_r, env, s, tw, e, b = step(bp_r, opt_r, env, s, tw, jax.random.split(k, nd), jp.zeros(nd, bool))
            errs.append(float(np.asarray(e)[0])), bases.append(float(np.asarray(b)[0]))
        row = {"when": label, "minutes": round((time.time() - t0) / 60, 1), "error": float(np.mean(errs)),
               "guess_no_change": float(np.mean(bases))}
        row["ratio"] = row["error"] / max(row["guess_no_change"], 1e-9)
        curve.append(row)
        print(f"[{row['minutes']:5.1f} min] {label:>22s}: predicts its body with error {row['error']:.4f} "
              f"(guessing 'no change': {row['guess_no_change']:.4f}) -> ratio {row['ratio']:.2f}", flush=True)
        (out / "babble.json").write_text(json.dumps(curve, indent=1))

    measure("before babbling")
    iters = max(1, int(args.steps) // (args.envs * (T + 1)))
    every = max(1, iters // 8)
    for it in range(iters):
        key, k = jax.random.split(key)
        bp_r, opt_r, env, s, tw, e, b = step(bp_r, opt_r, env, s, tw, jax.random.split(k, nd), jp.ones(nd, bool))
        if (it + 1) % every == 0:
            measure(f"after {(it + 1) * args.envs * (T + 1) / 1e6:.1f}M steps")
    trained = jax.tree_util.tree_map(lambda x: np.asarray(x)[0], bp_r)
    save_brain(trained, out / "baby_dog.npz", {"stage": "babble", "wiring": WIRING, "readout": args.readout,
                                                "steps": int(args.steps), "curve": curve})
    first, last = curve[0], curve[-1]
    verdict = "LEARNING" if last["ratio"] < 0.8 * first["ratio"] else "NOT LEARNING"
    print(f"VERDICT stage 1 (babbling): {verdict}. prediction error went from {first['ratio']:.2f} to {last['ratio']:.2f} "
          f"of the no-change guess", flush=True)
    return {"curve": curve, "verdict": verdict}


# ============================================================================ lessons: the teacher speaks
# An instruction is a few words and what they mean for this body, known only to the teacher:
#   line    "move <leg> <joint>": this motor line, commanded MOVE_SIZE in the brain's own units (-1: none)
#   offset  joint-space pose change from standing, per motor (radians), or None
#   up      which feet should be off the ground (a physical check), or None
#   held    never taught: only understood if the words themselves are
ACTIONS = {"lift": {"thigh": -0.3, "calf": -0.8}, "swing-forward": {"thigh": -0.4}, "swing-back": {"thigh": 0.4},
           "put-down": {}}
PAIRS = [("front-left", "rear-right"), ("front-right", "rear-left")]      # the trot's diagonal pairs
HELD_OUT_ACTIONS = [("lift", ("rear-left",))]


def naming_instructions(names):
    out = []
    for i, n in enumerate(names):
        leg, joint = LEGS[LEG_OF[n.split("_")[0]]], JOINTS[JOINT_OF[n.split("_")[1]]]
        out.append({"kind": "move", "words": ["move", leg, joint], "line": i, "offset": None, "up": None,
                    "held": (leg, joint) in HELD_OUT})
    return out


def action_instructions(names):
    out = []
    for act, change in ACTIONS.items():
        for legs in [(l,) for l in LEGS] + PAIRS:
            offset = [change.get(n.split("_")[1], 0.0) if LEGS[LEG_OF[n.split("_")[0]]] in legs else 0.0 for n in names]
            up = [LEGS[i] in legs for i in range(4)] if act == "lift" else None
            out.append({"kind": act, "words": [act, *legs], "line": -1, "offset": offset, "up": up,
                        "held": (act, legs) in HELD_OUT_ACTIONS})
    return out


def lesson(args, bd, out: Path, stage: str, instructions: list, rehearse: list, cycle: list | None = None,
           walk_skill: bool = False, start_stop: bool = False, parent_leaves: bool = False,
           pose_checks: dict | None = None) -> dict:
    """The teacher gives instructions (a new one about every second, "stand" often). At first its hands put the
    body where the words mean while it says them; its hands fade away by 60% of the lesson, and after that it
    only corrects: "this is what your lines should have done". Earlier lessons are rehearsed now and then. The
    brain keeps predicting its body, as in babbling. Instructions marked held are never taught, only tested.
    cycle: instructions (indices into `instructions`) the teacher speaks over and over to talk the dog through
    a walk, one every `hold` steps; it first tries the spoken walk on the body itself to find a pace that
    carries the dog forward, and refuses to teach one that does not.
    walk_skill: "walk" means the teacher's own smooth trot (found by practising on the body first). The teacher
    keeps the beat (nudged a little toward the brain's own rhythm) and counts the steps aloud with words the
    brain knows ("lift front-right rear-left", "swing-forward front-right rear-left", then the other pair);
    the counting fades until only "walk" is said. Checked with the harness on and off.
    start_stop: stands are long (the dog really stops) and walks start from rest, on the same beat each time;
    every checkpoint runs the real test (fresh start, stand, <gait>, stand) as it was taught.
    Gaits are instructions with a "cmd" (what the teacher's trot does for that word: walk, turn, back).
    parent_leaves: the harness (the parent holding the dog up) fades out by half the lesson; the dog keeps
    learning without it, and every real test is without it.
    pose_checks: {pose word: offsets}: every checkpoint runs the real pose test (stand, <pose>, stand) without the
    harness: does the trunk really pitch / lower, and does it get back up."""
    import jax
    import jax.numpy as jp
    import numpy as np
    import optax

    t0 = time.time()
    go2 = Path(args.menagerie) / "unitree_go2" if args.menagerie else bd.fetch_go2(Path(tempfile.gettempdir()))
    model = bd.build_model(go2)
    body = bd.make_body(model, wiring=WIRING)
    fresh_bp, weights, init_state, bstep, _ = bd.make_brain(bd.load_core(), jax.random.PRNGKey(args.seed), args.bias0,
                                                            args.goal_scale, args.gain0, args.readout, words=len(VOCAB))
    bp = carry_over(fresh_bp, load_brain(find_brain(args.brain)))
    bp = jax.tree_util.tree_map(jp.asarray, bp)
    predict = bstep.predict
    nd = jax.local_device_count()
    B, T, J, V = args.envs // nd, args.unroll, body["J"], len(VOCAB)
    v_reset, v_sense = jax.vmap(body["reset"]), jax.vmap(body["sense"])
    v_act = jax.vmap(body["act"], in_axes=(0, 0, 0, None))
    v_lines = jax.vmap(body["line_commands"])
    fresh = lambda tree, new_life: jax.tree_util.tree_map(lambda x: jp.where(new_life[:, None], 0.0, x), tree)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr))
    joints = [model.actuator_trnid[i, 0] for i in range(model.nu)]
    stand = jp.array(model.keyframe("home").qpos[model.jnt_qposadr[joints]])
    if walk_skill or pose_checks:
        teacher = bd.make_teacher(model, body)
        walk_g, _ = teacher["calibrate"](jax.random.PRNGKey(args.seed), out)
        v_show = jax.vmap(teacher["show"], in_axes=(0, None))
        v_follow = jax.vmap(teacher["follow"], in_axes=(0, None))
        count_words = np.zeros((4, len(VOCAB)), np.float32)            # the count, a quarter of the beat each
        for q, (act_, legs) in enumerate([("lift", PAIRS[1]), ("swing-forward", PAIRS[1]),
                                          ("lift", PAIRS[0]), ("swing-forward", PAIRS[0])]):
            count_words[q, [VOCAB.index(w) for w in (act_, *legs)]] = 1.0
        count_words = jp.array(count_words)

    stand_instr = {"kind": "stand", "words": ["stand"], "line": -1, "offset": [0.0] * J, "up": [False] * 4, "held": False}
    table = instructions + rehearse + [stand_instr]
    n_instr = len(table)
    kinds = sorted({i["kind"] for i in table})
    words_of = np.zeros((n_instr, V), np.float32)
    for i, ins in enumerate(table):
        words_of[i, [VOCAB.index(w) for w in ins["words"]]] = 1.0
    line_t = np.array([[MOVE_SIZE if ins["line"] == j else 0.0 for j in range(J)] for ins in table], np.float32)
    posed = np.array([ins["offset"] is not None for ins in table])
    offset = np.array([ins["offset"] if ins["offset"] is not None else [0.0] * J for ins in table], np.float32)
    has_up = np.array([ins["up"] is not None for ins in table])
    up = np.array([ins["up"] if ins["up"] is not None else [False] * 4 for ins in table])
    held = np.array([ins["held"] for ins in table])
    kind = np.array([kinds.index(ins["kind"]) for ins in table])
    line = np.array([ins["line"] for ins in table])
    is_walk = np.array([ins.get("cmd") is not None for ins in table])           # a gait: the teacher's trot shows it
    gait_cmd = np.array([ins.get("cmd") or (0.0, 0.0, 0.0) for ins in table], np.float32)
    says_walk = np.array([ins["kind"] == "walk" for ins in table])              # the count goes with "walk" only
    # how often each comes up while teaching: this lesson's instructions, earlier ones sometimes, "stand" often
    teach_p = np.array([0.0 if ins["held"] else (1.0 if k < len(instructions) else 0.3 * len(instructions) / max(len(rehearse), 1))
                        for k, ins in enumerate(table)])
    teach_p[-1] = 0.15 * teach_p.sum() / 0.85
    if start_stop:                                  # walk 45%, stand 40%, earlier lessons 15%
        rest = teach_p[len(instructions):-1]
        teach_p[:len(instructions)] = 0.45 / len(instructions)
        teach_p[len(instructions):-1] = 0.15 * rest / max(rest.sum(), 1e-9)
        teach_p[-1] = 0.40
    test_p = np.where(np.arange(n_instr) < len(instructions), 1.0, 0.0)
    test_p[-1] = 0.15 * test_p.sum() / 0.85
    words_of, line_t, posed, offset, has_up, up, held, kind, line, is_walk, gait_cmd, says_walk = (
        jp.array(x) for x in (words_of, line_t, posed, offset, has_up, up, held, kind, line, is_walk, gait_cmd, says_walk))
    teach_p, test_p = jp.array(teach_p / teach_p.sum()), jp.array(test_p / test_p.sum())
    v_reset1 = jax.vmap(body["reset"])
    hold, walk_ref = 1, 0.0
    if cycle:
        cycle_arr = jp.array(cycle)

        def spoken_walk(hold, support, key):
            """The teacher alone: says the walk's steps and puts the body there itself (no brain)."""
            k_body, k_run = jax.random.split(key)
            env = v_reset1(jax.random.split(k_body, B))

            def one(env, x):
                t, k = x
                i = jp.full((B,), cycle_arr[(t // hold) % len(cycle)])
                target = line_t[i] + jp.where(posed[i][:, None], v_lines(env, stand + offset[i]), 0.0)
                env, _, _, st = v_act(env, target, jax.random.split(k, B), support)
                return env, (st["v_fwd"], st["fell"])
            _, (vf, fell) = jax.lax.scan(one, env, (jp.arange(400), jax.random.split(k_run, 400)))
            return vf[100:].mean(), fell.sum() / B
        spoken_walk = jax.jit(spoken_walk)
        tried = {}
        for h in HOLDS:
            for sup in (1.0, 0.0):
                v, f = spoken_walk(h, sup, jax.random.PRNGKey(3))
                tried[(h, sup)] = (float(v), float(f))
            print(f"teacher says the walk's steps every {h} steps ({h * bd.CTRL_DT:.2f} s): forward {tried[(h, 1.0)][0]:+.3f} m/s "
                  f"held up, {tried[(h, 0.0)][0]:+.3f} m/s alone ({tried[(h, 0.0)][1]:.1f} falls in 8 s)", flush=True)
        hold = max(HOLDS, key=lambda h: tried[(h, 1.0)][0])
        walk_ref = tried[(hold, 1.0)][0]
        if walk_ref < 0.05:
            raise SystemExit(f"STOP: the teacher's spoken walk does not carry the dog forward (best {walk_ref:+.3f} m/s)")
        print(f"teacher will say a step every {hold} steps; its own spoken walk, held up: {walk_ref:+.3f} m/s", flush=True)
    else:
        cycle_arr = jp.zeros(1, jp.int32)

    print(f"lesson {stage}: {len(instructions)} instructions ({int(held.sum())} never taught), "
          f"{len(rehearse)} from earlier lessons, plus stand; words heard: {sorted({w for i in table for w in i['words']})}",
          flush=True)

    walk_share = (0.6, 0.8) if cycle else (0.0, 0.0)          # of new instructions: start a spoken walk (teach, test)

    def rollout(bp, env, s, ins, wk, hands, support, counting, key, testing):
        W = weights(bp)
        p = jp.where(testing, test_p, teach_p)
        share = jp.where(testing, walk_share[1], walk_share[0])

        def one(carry, k):
            env, s, ins, wk = carry
            k_new, k_switch, k_mode, k_count, k_wob, k_act = jax.random.split(k, 6)
            p_switch = jp.where(wk["on"], WALK_STOP, CTRL_SWITCH)
            if start_stop:                               # walks ~3 s, stands ~2.5 s (long enough to really stop)
                p_switch = jp.where(is_walk[ins], 0.006, jp.where(ins == n_instr - 1, 0.008, p_switch))
            if pose_checks:                              # rest poses and stand held ~2.5 s, long enough to settle
                p_switch = jp.where((ins < len(instructions)) | (ins == n_instr - 1), 0.008, p_switch)
            switch = jax.random.uniform(k_switch, (B,)) < p_switch
            on = jp.where(switch, jax.random.uniform(k_mode, (B,)) < share, wk["on"])
            timer = jp.where(switch, 0, wk["timer"] + 1)
            advance = on & (timer >= hold)
            pos = jp.where(switch, 0, jp.where(advance, (wk["pos"] + 1) % max(len(cycle or [0]), 1), wk["pos"]))
            count = jp.where(switch, jax.random.uniform(k_count, (B,)) < counting, wk["count"])
            wk = {"on": on, "pos": pos, "timer": jp.where(advance, 0, timer), "count": count}
            ins = jp.where(switch, jax.random.choice(k_new, n_instr, (B,), p=p), ins)
            ins = jp.where(on, cycle_arr[pos], ins)
            walking_now = is_walk[ins]
            heard = words_of[ins]
            if walk_skill:                                   # the teacher's own meaning of "walk" (the brain hears words only)
                env = {**env, "info": {**env["info"], "cmd": gait_cmd[ins]}}
                beat = env["info"]["phase"] + 2 * jp.pi * walk_g["freq"] * bd.CTRL_DT
                theirs = v_follow(env, walk_g)[0]["info"]["phase"]         # where the brain's commands are, one step on
                beat = beat + KEEP_BEAT * jp.arctan2(jp.sin(theirs - beat), jp.cos(theirs - beat))
                phase = jp.where(walking_now, jp.mod(beat, 2 * jp.pi), 0.0)
                env = {**env, "info": {**env["info"], "phase": phase}}
                quarter = jp.floor(phase / (jp.pi / 2)).astype(jp.int32) % 4
                heard = heard + jp.where((says_walk[ins] & wk["count"])[:, None], count_words[quarter], 0.0)
            env, seen = v_sense(env)
            seen = {**seen, "words": heard, "cmd": jp.zeros_like(seen["cmd"])}
            s = fresh(s, seen["new_life"])
            s, mu = bstep(bp, W, s, seen)
            target = line_t[ins] + jp.where(posed[ins][:, None], v_lines(env, stand + offset[ins]), 0.0)
            if walk_skill:
                target = jp.where(walking_now[:, None], v_show(env, walk_g), target)
            own = jp.clip(jp.tanh(mu) + WOBBLE * jax.random.normal(k_wob, mu.shape), -1.0, 1.0)
            a = (1.0 - hands) * own + hands * target
            ask = {**seen, "a_hist": jp.stack([a, seen["a_hist"][..., 0]], -1)}
            env, _, _, st = v_act(env, a, jax.random.split(k_act, B), support)
            feet_up = env["d"].geom_xpos[:, body["feet"], 2] > bd.FOOT_UP
            return (env, s, ins, wk), (seen, ask, target, ins, jp.tanh(mu), feet_up, st["v_fwd"], on | walking_now, st["fell"])
        (env, s_end, ins, wk), traj = jax.lax.scan(one, (env, s, ins, wk), jax.random.split(key, T + 1))
        return env, s_end, ins, wk, traj

    def loss(bp, traj, s0):
        seens, asks, target = traj[0], traj[1], traj[2]
        W = weights(bp)

        def one(s, x):
            seen, ask = x
            s = fresh(s, seen["new_life"])
            s, mu = bstep(bp, W, s, seen)
            return s, (jp.tanh(mu), predict(bp, s, ask))
        _, (a, pred) = jax.lax.scan(one, s0, (seens, asks))
        copy = jp.mean((a - target) ** 2)                                        # the teacher's correction
        got = seens["feats"][1:, ..., 1]
        keep = 1.0 - seens["new_life"][1:, :, None].astype(jp.float32)
        body_err = jp.sum(keep * (pred[:-1] - got) ** 2) / jp.maximum(keep.sum() * got.shape[-1], 1.0)
        return copy + body_err, copy

    def iteration(bp, opt_state, env, s, ins, wk, hands, support, counting, key, learn, testing):
        s0 = s
        env, s, ins, wk, traj = rollout(bp, env, s, ins, wk, hands, support, counting, key, testing)
        (_, copy), g = jax.value_and_grad(loss, has_aux=True)(bp, traj, s0)
        upd, new_opt = opt.update(jax.lax.pmean(g, "d"), opt_state, bp)
        bp = jax.tree_util.tree_map(lambda old, new: jp.where(learn, new, old), bp, optax.apply_updates(bp, upd))
        opt_state = jax.tree_util.tree_map(lambda old, new: jp.where(learn, new, old), opt_state, new_opt)
        # per kind of instruction, taught vs never taught: pose error and the do-nothing error (sums), the named
        # joint moved most ("move"), the right feet up and the others down ("lift"), and counts
        _, _, target, ins_t, a, feet_up, v_fwd, walking, fell = traj
        walk_st = jp.stack([jp.sum(v_fwd * walking), jp.sum(walking), jp.sum(fell * walking)])
        k_, h_ = kind[ins_t], held[ins_t]
        err, base = jp.mean((a - target) ** 2, -1), jp.mean(target ** 2, -1)
        right = jp.argmax(jp.abs(a), -1) == line[ins_t]
        feet_ok = jp.all(feet_up == up[ins_t], -1)
        rows = []
        for kk in range(len(kinds)):
            for hh in (False, True):
                m = ((k_ == kk) & (h_ == hh)).astype(jp.float32)
                rows.append(jp.stack([jp.sum(m * err), jp.sum(m * base), jp.sum(m * right), jp.sum(m * feet_ok), jp.sum(m)]))
        return bp, opt_state, env, s, ins, wk, jax.lax.psum(jp.stack(rows), "d"), jax.lax.psum(walk_st, "d"), \
            jax.lax.pmean(copy, "d")

    step = jax.pmap(iteration, axis_name="d", in_axes=(0, 0, 0, 0, 0, 0, None, None, None, 0, None, None))
    rep = lambda x: jax.device_put_replicated(x, jax.local_devices())
    bp_r, opt_r = rep(bp), rep(opt.init(bp))
    keys = jax.random.split(jax.random.PRNGKey(args.seed + 1), nd + 1)
    env = jax.pmap(lambda k: v_reset(jax.random.split(k, B)))(keys[1:])
    s = jax.pmap(lambda _: init_state(B))(jp.arange(nd))
    ins = jp.full((nd, B), n_instr - 1)
    wk = {"on": jp.zeros((nd, B), bool), "pos": jp.zeros((nd, B), jp.int32), "timer": jp.zeros((nd, B), jp.int32),
          "count": jp.zeros((nd, B), bool)}
    key, curve = keys[0], []

    def measure(label, n=10, hands=0.0, support=1.0, walk_only=False, counting=0.0):
        """The brain alone (no hands, no learning) on this lesson's instructions, including the never-taught ones.
        hands=1: the teacher alone instead. support: the harness (1 held up, 0 on its own). counting: share of walks
        the teacher counts aloud (0: it only says "walk")."""
        nonlocal bp_r, opt_r, env, s, ins, wk, key
        tot, walked = 0.0, 0.0
        for _ in range(n):
            key, k = jax.random.split(key)
            bp_r, opt_r, env, s, ins, wk, st, ws, _ = step(bp_r, opt_r, env, s, ins, wk, hands, support, counting,
                                                           jax.random.split(k, nd), False, True)
            tot, walked = tot + np.asarray(st)[0], walked + np.asarray(ws)[0]
        row = {"when": label, "minutes": round((time.time() - t0) / 60, 1)}
        parts = []
        if walk_skill:
            row["walk_speed"] = float(walked[0] / max(walked[1], 1.0))
            row["falls_per_min_walking"] = float(walked[2] / max(walked[1], 1.0) * 3000)
            parts.append(f"told \"walk\": forward {row['walk_speed']:+.3f} m/s, {row['falls_per_min_walking']:.1f} falls/min")
            if walk_only:
                curve.append(row)
                print(f"[{row['minutes']:5.1f} min] {label}: " + parts[0], flush=True)
                (out / f"{stage}.json").write_text(json.dumps(curve, indent=1))
                return row
        if cycle:
            row["walk_speed"] = float(walked[0] / max(walked[1], 1.0))
            parts.append(f"talked through the walk: forward {row['walk_speed']:+.3f} m/s "
                         f"(the teacher's own spoken walk: {walk_ref:+.3f} m/s)")
        for kk, name in enumerate(kinds):
            for hh in (False, True):
                e, b, r, f, c = (float(x) for x in tot[2 * kk + hh])
                if c == 0:
                    continue
                tag = f"{name}{' (never taught)' if hh else ''}"
                row[tag] = {"pose_error_vs_doing_nothing": e / max(b, 1e-9), "count": int(c)}
                text = f"{tag}: pose error {e / max(b, 1e-9):.2f}x doing nothing"
                if name == "move":
                    row[tag]["right_joint"] = r / c
                    text += f", right joint {r / c:.0%}"
                if name in ("lift", "stand"):
                    row[tag]["feet_right"] = f / c
                    text += f", feet right {f / c:.0%}"
                parts.append(text)
        curve.append(row)
        print(f"[{row['minutes']:5.1f} min] {label}\n      " + "\n      ".join(parts), flush=True)
        (out / f"{stage}.json").write_text(json.dumps(curve, indent=1))

    if pose_checks:
        sc_p = make_scripted(bd, model, body, weights, init_state, bstep, teacher, walk_g, n=64)
        host_p = lambda: jax.tree_util.tree_map(lambda x: np.asarray(x)[0], bp_r)
        pose_ref = {k_: sc_p["check_pose"](host_p(), k_, off, teacher=True) for k_, off in pose_checks.items()}

        def pose_test(label):
            rows = {}
            for k_, off in pose_checks.items():
                r = sc_p["check_pose"](host_p(), k_, off)
                rows[k_] = r
                ref = pose_ref[k_]
                print(f"[{(time.time() - t0) / 60:5.1f} min] POSE TEST (no harness) {label}, \"{k_}\": nose-up "
                      f"{r['pitch_in_pose']:+.2f} rad (teacher {ref['pitch_in_pose']:+.2f}), trunk at {r['height_in_pose']:.2f} "
                      f"(teacher {ref['height_in_pose']:.2f}, standing {r['height_standing_before']:.2f}), tilt {r['roll_in_pose']:.2f};"
                      f"  then \"stand\": trunk back at {r['height_back_up']:.2f}, nose {r['pitch_back_up']:+.2f};  falls {r['falls']:.2f}",
                      flush=True)
            curve.append({"when": label, **rows})
            (out / f"{stage}.json").write_text(json.dumps(curve, indent=1))
            return rows
        first_pose = pose_test("before this lesson")
    if start_stop:
        sc = make_scripted(bd, model, body, weights, init_state, bstep, teacher, walk_g, n=64)
        host = lambda: jax.tree_util.tree_map(lambda x: np.asarray(x)[0], bp_r)
        gaits = [i["kind"] for i in instructions if i.get("cmd") is not None]
        test_support = 0.0 if parent_leaves else 1.0
        harness = "no harness" if parent_leaves else "harness on"

        def gait_score(r, g):                        # how much of the asked movement it makes: forward/back or turning
            return r["turn_rate"] * np.sign(GAITS[g][2]) if GAITS[g][2] else r["walks_from_rest"] * np.sign(GAITS[g][0])

        def real_test(label):
            rows = {}
            for g in gaits:
                r = sc["check"](host(), gait=g, support=test_support)
                rows[g] = r
                print(f"[{(time.time() - t0) / 60:5.1f} min] REAL TEST ({harness}) {label}, told \"{g}\" from rest: "
                      f"forward {r['walks_from_rest']:+.3f} m/s, turning {r['turn_rate']:+.2f} rad/s "
                      f"(teacher {teacher_ref[g]['walks_from_rest']:+.3f} m/s, {teacher_ref[g]['turn_rate']:+.2f} rad/s);  "
                      f"after \"stand\" {r['moves_after_stand']:.3f} (standing before: {r['moves_before_walk']:.3f});  "
                      f"falls {r['falls']:.2f}", flush=True)
            curve.append({"when": label, "harness": test_support, **rows})
            (out / f"{stage}.json").write_text(json.dumps(curve, indent=1))
            return rows
        teacher_ref = {g: sc["check"](host(), gait=g, support=test_support, teacher=True) for g in gaits}
        first_real = real_test("before this lesson")
    if walk_skill and not start_stop:
        ref_held = measure("the teacher's own walk, harness on", hands=1.0, walk_only=True)
        ref_alone = measure("the teacher's own walk, no harness", hands=1.0, support=0.0, walk_only=True)
        before_alone = measure("before this lesson, no harness", support=0.0, walk_only=True)
        measure("before this lesson, no harness, steps counted aloud", support=0.0, walk_only=True, counting=1.0)
    measure("before this lesson")
    iters = max(1, int(args.steps) // (args.envs * (T + 1)))
    every = max(1, iters // 8)
    for it in range(iters):
        key, k = jax.random.split(key)
        if walk_skill:                                   # hands gone by 40%; counting aloud fades from 20% to 80%
            hands = max(0.0, 1.0 - it / (0.4 * iters))
            counting = float(np.clip((0.8 * iters - it) / (0.6 * iters), 0.0, 1.0))
        else:
            hands, counting = max(0.0, 1.0 - it / (0.6 * iters)), 0.0          # hands on, fading, gone by 60%
        support = max(0.0, 1.0 - it / (0.5 * iters)) if parent_leaves else 1.0  # the parent lets go by half the lesson
        if pose_checks:
            support = 0.0                                # it already lives without the harness
        bp_r, opt_r, env, s, ins, wk, _, _, copy = step(bp_r, opt_r, env, s, ins, wk, hands, support, counting,
                                                        jax.random.split(k, nd), True, False)
        if (it + 1) % every == 0:
            print(f"          teaching: hands {hands:.2f}  counting aloud {counting:.2f}  harness {support:.2f}  "
                  f"correction error {float(np.asarray(copy)[0]):.4f}", flush=True)
            if walk_skill and not start_stop:
                measure("                 no harness, steps counted aloud", support=0.0, walk_only=True, counting=1.0)
            measure(f"after {(it + 1) * args.envs * (T + 1) / 1e6:.1f}M steps")
            if pose_checks:
                last_pose = pose_test(f"after {(it + 1) * args.envs * (T + 1) / 1e6:.1f}M steps")
            if start_stop:
                last_real = real_test(f"after {(it + 1) * args.envs * (T + 1) / 1e6:.1f}M steps")
            elif walk_skill:
                last_alone = measure("                 no harness, only \"walk\"", support=0.0, walk_only=True)
    trained = jax.tree_util.tree_map(lambda x: np.asarray(x)[0], bp_r)
    save_brain(trained, out / "baby_dog.npz", {"stage": stage, "wiring": WIRING, "readout": args.readout, "vocab": VOCAB,
                                                "curve": curve})
    if pose_checks:
        for k_ in pose_checks:
            r0, r1, ref = first_pose[k_], last_pose[k_], pose_ref[k_]
            key_ = "pitch_in_pose" if k_ == "sit" else "height_in_pose"
            gap0, gap1 = abs(r0[key_] - ref[key_]), abs(r1[key_] - ref[key_])
            print(f"VERDICT {stage} \"{k_}\" (no harness): {'DOES IT' if gap1 < 0.4 * gap0 else 'NOT YET'}. {key_} "
                  f"{r0[key_]:+.2f} -> {r1[key_]:+.2f} (teacher {ref[key_]:+.2f}); back up after: trunk {r1['height_back_up']:.2f}; "
                  f"falls {r1['falls']:.2f}", flush=True)
        return {"curve": curve}
    if start_stop:
        verdicts = {}
        for g in gaits:
            got, ref, was = gait_score(last_real[g], g), gait_score(teacher_ref[g], g), gait_score(first_real[g], g)
            stops = last_real[g]["moves_after_stand"] < last_real[g]["moves_before_walk"] + 0.02
            verdicts[g] = "DOES IT" if got > max(0.5 * ref, was + 0.05) and stops else "NOT YET"
            print(f"VERDICT {stage} \"{g}\" ({harness}, from rest): {verdicts[g]}. {was:+.3f} -> {got:+.3f} "
                  f"(teacher {ref:+.3f}; {'m/s' if not GAITS[g][2] else 'rad/s'}); stops on \"stand\": {'yes' if stops else 'no'}; "
                  f"falls {last_real[g]['falls']:.2f}", flush=True)
        return {"curve": curve, "verdict": verdicts}
    first, last = next(r for r in curve if r["when"] == "before this lesson"), curve[-1]
    if walk_skill:
        last = curve[-2]                                                         # the harness-on row
        verdict = "LEARNING" if last_alone["walk_speed"] > max(before_alone["walk_speed"] + 0.05,
                                                               0.3 * ref_alone["walk_speed"]) else "NOT LEARNING"
        print(f"VERDICT {stage} (told \"walk\", no harness): {verdict}. forward {before_alone['walk_speed']:+.3f} -> "
              f"{last_alone['walk_speed']:+.3f} m/s, {last_alone['falls_per_min_walking']:.1f} falls/min "
              f"(teacher's own: {ref_alone['walk_speed']:+.3f} m/s; with the harness the brain now does "
              f"{last['walk_speed']:+.3f}, the teacher {ref_held['walk_speed']:+.3f})", flush=True)
    main_kind = instructions[0]["kind"] if len({i["kind"] for i in instructions}) == 1 else None
    scores = {k: v["pose_error_vs_doing_nothing"] for k, v in last.items() if isinstance(v, dict) and "never" not in k}
    before = {k: v["pose_error_vs_doing_nothing"] for k, v in first.items() if isinstance(v, dict) and "never" not in k}
    learned = [k for k in scores if k != "stand" and scores[k] < 0.5 * before.get(k, 1.0)]
    verdict = "LEARNING" if learned and len(learned) >= (1 if main_kind else len([k for k in scores if k != "stand"]) // 2) \
        else "NOT LEARNING"
    if cycle:
        verdict = "LEARNING" if last["walk_speed"] > max(first["walk_speed"] + 0.05, 0.3 * walk_ref) else "NOT LEARNING"
        print(f"VERDICT {stage} (walking when talked through it): {verdict}. forward {first['walk_speed']:+.3f} -> "
              f"{last['walk_speed']:+.3f} m/s (teacher's own {walk_ref:+.3f})", flush=True)
    print(f"VERDICT {stage}: {verdict}. pose error vs doing nothing, before -> after: " +
          ", ".join(f"{k} {before.get(k, float('nan')):.2f} -> {v:.2f}" for k, v in scores.items()), flush=True)
    return {"curve": curve, "verdict": verdict}


def name(args, bd, out: Path) -> dict:
    """Stage 2: "move <leg> <joint>"."""
    names = [f"{l}_{j}" for l in ("FL", "FR", "RL", "RR") for j in ("hip", "thigh", "calf")]   # Go2 motor order
    return lesson(args, bd, out, "name", naming_instructions(names), [])


def act(args, bd, out: Path) -> dict:
    """Stage 3: "lift / swing-forward / swing-back / put-down <leg(s)>", rehearsing "move <leg> <joint>"."""
    names = [f"{l}_{j}" for l in ("FL", "FR", "RL", "RR") for j in ("hip", "thigh", "calf")]
    return lesson(args, bd, out, "act", action_instructions(names), [i for i in naming_instructions(names) if not i["held"]])


def walk_words(args, bd, out: Path) -> dict:
    """Stage 4a: the teacher talks the dog through a trot with words it knows: lift one diagonal pair, swing it
    forward, lift the other pair, swing it forward, again and again (each new step puts the other pair back to
    standing while its feet are down: that push is what carries the body forward)."""
    names = [f"{l}_{j}" for l in ("FL", "FR", "RL", "RR") for j in ("hip", "thigh", "calf")]
    acts = action_instructions(names)
    at = {(i["kind"], tuple(i["words"][1:])): k for k, i in enumerate(acts)}
    cycle = [at[("lift", PAIRS[0])], at[("swing-forward", PAIRS[0])], at[("lift", PAIRS[1])], at[("swing-forward", PAIRS[1])]]
    return lesson(args, bd, out, "walk_words", acts, [i for i in naming_instructions(names) if not i["held"]], cycle=cycle)


def walk(args, bd, out: Path) -> dict:
    """Stage 4b: the word "walk", taught with the teacher's smooth trot, rehearsing every earlier lesson."""
    names = [f"{l}_{j}" for l in ("FL", "FR", "RL", "RR") for j in ("hip", "thigh", "calf")]
    walk_instr = {"kind": "walk", "words": ["walk"], "line": -1, "offset": None, "up": None, "held": False}
    earlier = action_instructions(names) + [i for i in naming_instructions(names) if not i["held"]]
    return lesson(args, bd, out, "walk", [walk_instr], [i for i in earlier if not i["held"]], walk_skill=True)


def make_scripted(bd, model, body, weights, init_state, bstep, teacher, walk_g, n: int) -> dict:
    """Scripted lives for checking and filming, exactly as the brain is taught (wobble, harness): step by step
    walk[t] ("walk" said, else "stand"), count[t] (the teacher counts the steps aloud), hands[t] (share of the
    teacher's hands on the legs; 1 = the teacher walks the dog itself)."""
    import jax
    import jax.numpy as jp
    import numpy as np
    v_reset, v_sense = jax.vmap(body["reset"]), jax.vmap(body["sense"])
    v_act = jax.vmap(body["act"], in_axes=(0, 0, 0, None))
    v_lines = jax.vmap(body["line_commands"])
    v_show = jax.vmap(teacher["show"], in_axes=(0, None))
    joints = [model.actuator_trnid[i, 0] for i in range(model.nu)]
    stand = jp.array(model.keyframe("home").qpos[model.jnt_qposadr[joints]])
    said = {w: jp.zeros(len(VOCAB)).at[VOCAB.index(w)].set(1.0) for w in ("stand", "walk")}
    count_words = np.zeros((4, len(VOCAB)), np.float32)                # the teacher's count, a quarter beat each
    for q, (act_, legs) in enumerate([("lift", PAIRS[1]), ("swing-forward", PAIRS[1]), ("lift", PAIRS[0]),
                                      ("swing-forward", PAIRS[0])]):
        count_words[q, [VOCAB.index(w) for w in (act_, *legs)]] = 1.0
    count_words = jp.array(count_words)

    base = model.body("base").id

    def life(bp, key, walk, count, hands, wobble, support, word, cmd, pose=None, gait=True):
        """pose: joint offsets from standing of a rest pose (when gait is False, the active word means that pose)."""
        pose = jp.zeros(stand.shape) if pose is None else pose
        W = weights(bp)
        k_body, k_run = jax.random.split(key)
        env = v_reset(jax.random.split(k_body, n))

        def one(carry, x):
            env, s = carry
            walking, counting, h, k = x
            beat = jp.where(walking & gait, env["info"]["phase"] + 2 * jp.pi * walk_g["freq"] * bd.CTRL_DT, 0.0)
            phase = jp.mod(beat, 2 * jp.pi)
            env = {**env, "info": {**env["info"], "cmd": jp.broadcast_to(jp.where(walking & gait, cmd, 0.0), (n, 3)),
                                   "phase": phase}}
            env, seen = v_sense(env)
            quarter = jp.floor(phase / (jp.pi / 2)).astype(jp.int32) % 4
            heard = jp.where(walking, word, said["stand"]) + jp.where(walking & counting, count_words[quarter], 0.0)
            seen = {**seen, "words": jp.broadcast_to(heard, (n, len(VOCAB))), "cmd": jp.zeros_like(seen["cmd"])}
            s, mu = bstep(bp, W, s, seen)
            posed_target = v_lines(env, jp.broadcast_to(stand + jp.where(walking, pose, 0.0), (n, stand.shape[0])))
            target = jp.where(walking & gait, v_show(env, walk_g), posed_target)
            k, k_wob = jax.random.split(k)
            own = jp.clip(jp.tanh(mu) + wobble * jax.random.normal(k_wob, mu.shape), -1.0, 1.0)
            env, _, _, st = v_act(env, (1.0 - h) * own + h * target, jax.random.split(k, n), support)
            R = env["d"].xmat[:, base]
            pitch = jp.arcsin(jp.clip(R[:, 2, 0], -1.0, 1.0)).mean()           # nose up > 0
            roll = jp.abs(jp.arcsin(jp.clip(R[:, 2, 1], -1.0, 1.0))).mean()
            height = (env["d"].subtree_com[:, base, 2] / env["info"]["h0"]).mean()
            air = (env["d"].geom_xpos[:, body["feet"], 2] > bd.FOOT_UP).mean(0)
            h_each = env["d"].subtree_com[:, base, 2] / env["info"]["h0"]
            upright = ((R[:, 2, 2] > 0.9) & (h_each > 0.7)).mean()             # standing on its feet
            return (env, s), (env["d"].qpos[0], env["d"].geom_xpos[0, body["feet"], 2], st["v_fwd"], st["fell"], st["wz"],
                              pitch, roll, height, air, upright)
        xs = (walk, count, hands, jax.random.split(k_run, walk.shape[0]))
        return jax.lax.scan(one, (env, init_state(n)), xs)[1]

    sec = lambda x: int(round(x / bd.CTRL_DT))
    steps = sec(13)

    def script(parts):
        """parts: (seconds, walk, count, hands) in order; 13 s in all."""
        w, c, h = [], [], []
        for dur, walk_, count_, hands_ in parts:
            w += [walk_] * sec(dur)
            c += [count_] * sec(dur)
            h += [hands_] * sec(dur)
        return jp.array(w[:steps]), jp.array(c[:steps]), jp.array(h[:steps], jp.float32)

    run = jax.jit(life)
    word_of = lambda g: jp.zeros(len(VOCAB)).at[VOCAB.index(g)].set(1.0)

    def check(bp, gait="walk", support=1.0, teacher=False, key=None):
        """The real test, as taught (wobble on): fresh start, "stand" 2 s, <gait> 8 s, "stand" 3 s; nobody counts
        or helps. teacher=True: the teacher's own trot instead, as the yardstick."""
        h_ = 1.0 if teacher else 0.0
        w, c, h = script([(2, False, False, h_), (8, True, False, h_), (3, False, False, h_)])
        _, _, vf, fell, wz, _, _, _, air, _ = (np.asarray(x) for x in run(bp, jax.random.PRNGKey(11) if key is None else key, w, c,
                                                                       h, 0.0 if teacher else WOBBLE, support, word_of(gait),
                                                                       jp.array(GAITS[gait])))
        return {"walks_from_rest": float(vf[sec(4):sec(10)].mean()), "turn_rate": float(wz[sec(4):sec(10)].mean()),
                "feet_air": [round(float(x), 2) for x in air[sec(4):sec(10)].mean(0)],
                "moves_after_stand": float(np.abs(vf[sec(11):sec(13)]).mean()),
                "moves_before_walk": float(np.abs(vf[sec(0.5):sec(2)]).mean()), "falls": float(fell.sum() / n)}
    def check_pose(bp, name, offset, support=0.0, teacher=False, key=None):
        """Fresh start, "stand" 2 s, <pose word> 5 s, "stand" 4 s (back up), as taught (wobble on)."""
        h_ = 1.0 if teacher else 0.0
        w, c, h = script([(2, False, False, h_), (5, True, False, h_), (4, False, False, h_), (2, False, False, h_)])
        _, _, vf, fell, _, pitch, roll, height, _, _ = (np.asarray(x) for x in run(
            bp, jax.random.PRNGKey(11) if key is None else key, w, c, h, 0.0 if teacher else WOBBLE, support, word_of(name),
            jp.zeros(3), jp.array(offset, jp.float32), False))
        return {"pitch_in_pose": float(pitch[sec(5):sec(7)].mean()), "height_in_pose": float(height[sec(5):sec(7)].mean()),
                "roll_in_pose": float(roll[sec(5):sec(7)].mean()), "height_standing_before": float(height[sec(1):sec(2)].mean()),
                "height_back_up": float(height[sec(10):sec(13)].mean()), "pitch_back_up": float(pitch[sec(10):sec(13)].mean()),
                "falls": float(fell.sum() / n)}
    def check_getup(bp, key=None):
        """For a body that starts fallen and stays down until it gets itself up: "stand" for 10 s, as taught
        (wobble on, no harness). How many dogs are on their feet after 1, 2, 4 and 8 s."""
        w, c, h = script([(10, False, False, 0.0), (3, False, False, 0.0)])
        *_, upright = (np.asarray(x) for x in run(bp, jax.random.PRNGKey(13) if key is None else key, w, c, h, WOBBLE, 0.0,
                                                   word_of("stand"), jp.zeros(3)))
        return {f"up_after_{t}s": float(upright[sec(t) - 1]) for t in (1, 2, 4, 8)}
    return {"run": run, "script": script, "sec": sec, "steps": steps, "check": check, "check_pose": check_pose,
            "check_getup": check_getup, "word_of": word_of}


def film_walk(args, bd, out: Path) -> dict:
    """Watch it: the brain alone, no harness, no help, told "stand" for 2 s and then "walk" for 8 s; the same for
    the teacher's own trot. Films (GIF + contact sheet), footfalls and speeds."""
    import jax
    import jax.numpy as jp
    import numpy as np
    go2 = Path(args.menagerie) / "unitree_go2" if args.menagerie else bd.fetch_go2(Path(tempfile.gettempdir()))
    model = bd.build_model(go2)
    body = bd.make_body(model, wiring=WIRING)
    fresh_bp, weights, init_state, bstep, _ = bd.make_brain(bd.load_core(), jax.random.PRNGKey(args.seed), args.bias0,
                                                            args.goal_scale, args.gain0, args.readout, words=len(VOCAB))
    bp = jax.tree_util.tree_map(jp.asarray, carry_over(fresh_bp, load_brain(find_brain(args.brain))))
    teacher = bd.make_teacher(model, body)
    walk_g, _ = teacher["calibrate"](jax.random.PRNGKey(args.seed), out)
    sc = make_scripted(bd, model, body, weights, init_state, bstep, teacher, walk_g, n=8)
    run, script, sec, n = sc["run"], sc["script"], sc["sec"], 8

    # as it will run now: wobble on, no harness (the parent is gone), nobody counts or helps
    plain = script([(2, False, False, 0.0), (8, True, False, 0.0), (3, False, False, 0.0)])
    cases = {f"brain, {g}": (plain, g, False) for g in ("walk", "turn-left", "back")}
    cases["teacher, walk"] = (script([(2, False, False, 1.0), (8, True, False, 1.0), (3, False, False, 1.0)]), "walk", True)
    report = {}
    for label, ((w, c, h), gait, by_teacher) in cases.items():
        qpos, feet_z, vf, fell, wz, _, _, _, _, _ = (np.asarray(x) for x in run(bp, jax.random.PRNGKey(5), w, c, h,
                                                                 0.0 if by_teacher else WOBBLE, 0.0,
                                                                 sc["word_of"](gait), jp.array(GAITS[gait])))
        report[label] = {"forward": float(vf[sec(4):sec(10)].mean()), "turning": float(wz[sec(4):sec(10)].mean()),
                         "after_told_stand": float(np.abs(vf[sec(11):sec(13)]).mean()),
                         "falls": float(fell.sum() / n)}
        print(f"{label:22s}: told \"{gait}\" (s 4-10) forward {report[label]['forward']:+.3f} m/s, turning "
              f"{report[label]['turning']:+.2f} rad/s;  after \"stand\" moving {report[label]['after_told_stand']:.3f} m/s;  "
              f"falls {report[label]['falls']:.2f}", flush=True)
        for i, f in enumerate(bd.FEET):
            print(f"      {f} " + "".join("#" if z < bd.FOOT_UP else "." for z in feet_z[sec(6):sec(8), i]), flush=True)
        try:
            bd.film(model, qpos[sec(1.5):sec(1.5) + 300], out / ("film_" + label.replace(", ", "_").replace(" ", "_")))
        except Exception as e:                           # rendering needs an OpenGL context; the numbers stand without it
            print(f"film {label}: skipped ({e!r})", flush=True)
    # the rest poses: "stand" 2 s, <pose> 5 s, "stand" 4 s (getting back up), as it runs now (no harness)
    pose_off, _ = find_poses(bd, model, body, out)
    for pz, off in pose_off.items():
        w, c, h = script([(2, False, False, 0.0), (5, True, False, 0.0), (4, False, False, 0.0), (2, False, False, 0.0)])
        qpos, _, _, fell, _, pitch, _, height, _, _ = (np.asarray(x) for x in run(bp, jax.random.PRNGKey(5), w, c, h, WOBBLE, 0.0,
                                                                              sc["word_of"](pz), jp.zeros(3),
                                                                              jp.array(off, jp.float32), False))
        report[f"brain, {pz}"] = {"pitch": float(pitch[sec(5):sec(7)].mean()), "height": float(height[sec(5):sec(7)].mean()),
                                   "height_back_up": float(height[sec(10):sec(13)].mean()), "falls": float(fell.sum() / n)}
        print(f"brain, {pz:14s}: nose-up {report[f'brain, {pz}']['pitch']:+.2f}, trunk {report[f'brain, {pz}']['height']:.2f}, "
              f"back up {report[f'brain, {pz}']['height_back_up']:.2f}, falls {report[f'brain, {pz}']['falls']:.2f}", flush=True)
        try:
            bd.film(model, qpos[sec(1.5):sec(1.5) + 450], out / f"film_brain_{pz}_then_stand")
        except Exception as e:
            print(f"film {pz}: skipped ({e!r})", flush=True)
    (out / "film_walk.json").write_text(json.dumps(report, indent=1))
    return report


def start_stop(args, bd, out: Path) -> dict:
    """Stage 4c: start walking from rest and stop on "stand", rehearsing every earlier lesson."""
    names = [f"{l}_{j}" for l in ("FL", "FR", "RL", "RR") for j in ("hip", "thigh", "calf")]
    walk_instr = {"kind": "walk", "words": ["walk"], "line": -1, "offset": None, "up": None, "held": False}
    earlier = action_instructions(names) + [i for i in naming_instructions(names) if not i["held"]]
    return lesson(args, bd, out, "start_stop", [walk_instr], [i for i in earlier if not i["held"]], walk_skill=True,
                  start_stop=True)


def find_poses(bd, model, body, out: Path) -> dict:
    """The teacher tries each rest pose on the body before teaching it (1 s to get there, then 4 s held, no harness,
    no brain) and keeps, for each, the try that best reaches the goal without tipping over. Returns
    {pose: joint offsets from standing}."""
    import jax
    import jax.numpy as jp
    import numpy as np
    v_reset, v_act, v_lines = jax.vmap(body["reset"]), jax.vmap(body["act"], in_axes=(0, 0, 0, None)), jax.vmap(body["line_commands"])
    names = [model.actuator(i).name for i in range(model.nu)]
    joints = [model.actuator_trnid[i, 0] for i in range(model.nu)]
    stand = jp.array(model.keyframe("home").qpos[model.jnt_qposadr[joints]])
    base = model.body("base").id
    rep = 4

    def offset_of(t):
        o = []
        for n in names:
            leg, part = n.split("_")
            end = "rear" if leg.startswith("R") else "front"
            o.append(t.get(f"{end}_{part}", 0.0) if part in ("thigh", "calf") else 0.0)
        return o

    @jax.jit
    def hold(offsets, key):
        m = offsets.shape[0]
        env = v_reset(jax.random.split(key, m))

        def one(env, x):
            t, k = x
            target = v_lines(env, stand + offsets * jp.minimum(1.0, t / 50.0))
            env, _, _, st = v_act(env, target, jax.random.split(k, m), 0.0)
            R = env["d"].xmat[:, base]
            return env, (jp.arcsin(jp.clip(R[:, 2, 0], -1, 1)), jp.abs(jp.arcsin(jp.clip(R[:, 2, 1], -1, 1))),
                         env["d"].subtree_com[:, base, 2] / env["info"]["h0"], st["fell"])
        _, (pitch, roll, height, fell) = jax.lax.scan(one, env, (jp.arange(250), jax.random.split(key, 250)))
        return pitch[-50:].mean(0), roll[-50:].mean(0), height[-50:].mean(0), fell.sum(0)

    found, table = {}, {}
    for name, tries in POSE_TRIES.items():
        offs = np.repeat(np.array([offset_of(t) for t in tries], np.float32), rep, axis=0)
        pitch, roll, height, fell = (np.asarray(x).reshape(len(tries), rep).mean(1) for x in hold(jp.array(offs), jax.random.PRNGKey(4)))
        if name == "sit":                         # chest up (trunk at least half height), nose up ~0.5 rad
            score = -np.abs(pitch - 0.5) - 2 * roll - 3 * fell - 2 * np.maximum(0.5 - height, 0)
        else:                                     # low and level, but lying, not flattened
            score = -np.abs(height - 0.4) - 0.5 * np.abs(pitch) - 2 * roll - 3 * fell
        best = int(np.argmax(score))
        found[name] = offset_of(tries[best])
        table[name] = {"try": tries[best], "pitch": float(pitch[best]), "roll": float(roll[best]), "height": float(height[best]),
                       "falls": float(fell[best])}
        print(f"teacher's {name} ({POSE_GOAL[name]}): {tries[best]} -> nose-up {pitch[best]:+.2f} rad, tilt {roll[best]:.2f}, "
              f"trunk at {height[best]:.2f} of standing height, falls {fell[best]:.2f}", flush=True)
    (out / "poses.json").write_text(json.dumps(table, indent=1))
    return found, table


def poses(args, bd, out: Path) -> dict:
    """Stage 7: "sit", "lie-down", and "stand" (getting back up), taught by the teacher's corrections with its hands
    fading, no harness; the earlier still instructions are rehearsed (the gaits are left alone so the corrections
    do not drag them back toward the teacher's slower trot: practice round 2 brings everything together)."""
    import jax
    names = [f"{l}_{j}" for l in ("FL", "FR", "RL", "RR") for j in ("hip", "thigh", "calf")]
    go2 = Path(args.menagerie) / "unitree_go2" if args.menagerie else bd.fetch_go2(Path(tempfile.gettempdir()))
    model = bd.build_model(go2)
    found, _ = find_poses(bd, model, bd.make_body(model, wiring=WIRING), out)
    instrs = [{"kind": name, "words": [name], "line": -1, "offset": off, "up": None, "held": False} for name, off in found.items()]
    earlier = action_instructions(names) + [i for i in naming_instructions(names) if not i["held"]]
    return lesson(args, bd, out, "poses", instrs, [i for i in earlier if not i["held"]], pose_checks=found)


def gait_instr(g):
    return {"kind": g, "words": [g], "line": -1, "offset": None, "up": None, "held": False, "cmd": GAITS[g]}


def gaits(args, bd, out: Path) -> dict:
    """Stage 4d: "walk", "turn-left", "turn-right", "back", each started from rest and stopped on "stand"
    (harness on, as the walk was taught), rehearsing every earlier lesson."""
    names = [f"{l}_{j}" for l in ("FL", "FR", "RL", "RR") for j in ("hip", "thigh", "calf")]
    earlier = action_instructions(names) + [i for i in naming_instructions(names) if not i["held"]]
    return lesson(args, bd, out, "gaits", [gait_instr(g) for g in GAITS], [i for i in earlier if not i["held"]],
                  walk_skill=True, start_stop=True)


def no_parent(args, bd, out: Path) -> dict:
    """Stage 5: the parent lets go. The harness fades out by half the lesson and the dog keeps practising the gaits
    it knows (args.gaits) without it; every real test is without the harness."""
    names = [f"{l}_{j}" for l in ("FL", "FR", "RL", "RR") for j in ("hip", "thigh", "calf")]
    earlier = action_instructions(names) + [i for i in naming_instructions(names) if not i["held"]]
    return lesson(args, bd, out, "no_parent", [gait_instr(g) for g in args.gaits.split(",")],
                  [i for i in earlier if not i["held"]], walk_skill=True, start_stop=True, parent_leaves=True)


# ============================================================================ stage 6: practice with rewards
def practice(args, bd, out: Path) -> dict:
    """Stage 6: the dog practises everything it knows on its own (no harness, no hands, no counting), scored by
    meaning (the body's reward: going where asked, standing still when asked, staying upright; a fall costs
    FALL_COST) with random shoves now and then, like a toddler in a real room. The teacher's corrections stay on
    as a light guide (GUIDE) and the brain's wobble is its exploration, exactly as it learned with. Recurrent PPO
    with an asymmetric critic; the critic first learns alone for 10% of the practice so it cannot mislead the
    brain. Every checkpoint runs the real test of every gait with no harness."""
    import jax
    import jax.numpy as jp
    import numpy as np
    import optax

    t0 = time.time()
    go2 = Path(args.menagerie) / "unitree_go2" if args.menagerie else bd.fetch_go2(Path(tempfile.gettempdir()))
    model = bd.build_model(go2)
    body = bd.make_body(model, wiring=WIRING, keep_fallen=args.get_up, fallen_starts=0.25 if args.get_up else 0.0)
    body_test = bd.make_body(model, wiring=WIRING)                          # the usual tests: a fall resets
    body_down = bd.make_body(model, wiring=WIRING, keep_fallen=True, fallen_starts=1.0)   # the get-up test
    fresh_bp, weights, init_state, bstep, _ = bd.make_brain(bd.load_core(), jax.random.PRNGKey(args.seed), args.bias0,
                                                            args.goal_scale, args.gain0, args.readout, words=len(VOCAB))
    bp = jax.tree_util.tree_map(jp.asarray, carry_over(fresh_bp, load_brain(find_brain(args.brain))))
    teacher = bd.make_teacher(model, body)
    walk_g, _ = teacher["calibrate"](jax.random.PRNGKey(args.seed), out)
    if args.get_up:
        print("practice with falls that stay down: a quarter of lives start fallen; nobody picks it up", flush=True)
    nd = jax.local_device_count()
    B, T, J, V = args.envs // nd, args.unroll, body["J"], len(VOCAB)
    v_reset, v_sense, v_priv = jax.vmap(body["reset"]), jax.vmap(body["sense"]), jax.vmap(body["privileged"])
    v_act = jax.vmap(body["act"], in_axes=(0, 0, 0, None))
    v_lines = jax.vmap(body["line_commands"])
    v_show = jax.vmap(teacher["show"], in_axes=(0, None))
    v_follow = jax.vmap(teacher["follow"], in_axes=(0, None))
    fresh = lambda tree, new_life: jax.tree_util.tree_map(lambda x: jp.where(new_life[:, None], 0.0, x), tree)
    joints = [model.actuator_trnid[i, 0] for i in range(model.nu)]
    stand = jp.array(model.keyframe("home").qpos[model.jnt_qposadr[joints]])

    gaits = list(GAITS)
    pose_off, pose_goal = find_poses(bd, model, body, out)                  # the teacher's sit and lie-down
    rests = list(pose_off)
    said = ["stand", *gaits, *rests]                                        # what it practises
    n_i = len(said)
    words_of = jp.array(np.stack([np.eye(V, dtype=np.float32)[VOCAB.index(w)] for w in said]))
    cmd_of = jp.array([(0.0, 0.0, 0.0), *[GAITS[g] for g in gaits], *[(0.0, 0.0, 0.0)] * len(rests)], jp.float32)
    is_gait = jp.array([w in GAITS for w in said])
    is_rest = jp.array([w in pose_off for w in said])
    offs = jp.array([pose_off.get(w, [0.0] * J) for w in said], jp.float32)
    goal_pitch = jp.array([pose_goal[w]["pitch"] if w in pose_goal else 0.0 for w in said], jp.float32)
    goal_height = jp.array([pose_goal[w]["height"] if w in pose_goal else 0.0 for w in said], jp.float32)
    choose = jp.array([0.2] + [0.5 / len(gaits)] * len(gaits) + [0.3 / len(rests)] * len(rests))
    base = model.body("base").id

    # the critic: the true state, what was said, and whether a shove just happened
    P = body["P"] + n_i
    sizes = [P, 256, 256, 256, 1]
    ks = jax.random.split(jax.random.PRNGKey(args.seed + 7), len(sizes))
    cp = [{"w": jax.random.normal(ks[i], (a, b)) * (1.0 / a) ** 0.5, "b": jp.zeros(b)} for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]))]

    def value(ps, x):
        for i, l in enumerate(ps):
            x = x @ l["w"] + l["b"]
            x = jax.nn.swish(x) if i < len(ps) - 1 else x
        return x[..., 0]

    params = {"brain": bp, "critic": cp}
    lr_brain = optax.inject_hyperparams(optax.adam)(learning_rate=args.lr)
    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.multi_transform({"brain": lr_brain, "critic": optax.adam(3e-4)},
                                            {"brain": "brain", "critic": "critic"}))

    def logprob(a, mean):                        # the wobble is its exploration: a ~ N(tanh(mu), WOBBLE)
        return jp.sum(-0.5 * ((a - mean) / WOBBLE) ** 2 - np.log(WOBBLE), -1)

    def rollout(params, env, s, ins, air_ema, key):
        bp = params["brain"]
        W = weights(bp)

        def one(carry, k):
            env, s, ins, air_ema = carry
            k_new, k_sw, k_wob, k_push, k_size, k_act = jax.random.split(k, 6)
            switch = jax.random.uniform(k_sw, (B,)) < jp.where(is_gait[ins], 0.006, 0.008)  # ~3 s gaits, ~2.5 s the rest
            ins = jp.where(switch, jax.random.choice(k_new, n_i, (B,), p=choose), ins)
            moving = is_gait[ins]
            env = {**env, "info": {**env["info"], "cmd": cmd_of[ins]}}
            beat = env["info"]["phase"] + 2 * jp.pi * walk_g["freq"] * bd.CTRL_DT
            theirs = v_follow(env, walk_g)[0]["info"]["phase"]
            beat = beat + KEEP_BEAT * jp.arctan2(jp.sin(theirs - beat), jp.cos(theirs - beat))
            env = {**env, "info": {**env["info"], "phase": jp.where(moving, jp.mod(beat, 2 * jp.pi), 0.0)}}
            # a shove now and then (the world is not a lab)
            shove = (jax.random.uniform(k_push, (B,)) < PUSH_P)[:, None] * \
                jax.random.uniform(k_size, (B, 2), minval=-PUSH_SIZE, maxval=PUSH_SIZE)
            env = {**env, "d": env["d"].replace(qvel=env["d"].qvel.at[:, 0:2].add(shove))}
            env, seen = v_sense(env)
            seen = {**seen, "words": words_of[ins], "cmd": jp.zeros_like(seen["cmd"])}
            s = fresh(s, seen["new_life"])
            priv = jp.concatenate([v_priv(env), jax.nn.one_hot(ins, n_i)], -1)
            s, mu = bstep(bp, W, s, seen)
            mean = jp.tanh(mu)
            a = mean + WOBBLE * jax.random.normal(k_wob, mean.shape)
            guide = jp.where(moving[:, None], v_show(env, walk_g), v_lines(env, stand + offs[ins]))
            env, r, done, st = v_act(env, a, jax.random.split(k_act, B), 0.0)
            r = r - (FALL_COST - 0.2) * st["fell"]                                          # the body's own 0.2 plus this
            # all four feet stepping while it moves (a real dog does not drag its hind legs)
            air_now = (env["d"].geom_xpos[:, body["feet"], 2] > bd.FOOT_UP).astype(jp.float32)
            air_ema = jp.where(done[:, None], 0.0, 0.97 * air_ema + 0.03 * air_now)
            r = r + moving * STEP_REWARD * jp.min(jp.minimum(air_ema, FEET_AIR) / FEET_AIR, -1) * bd.CTRL_DT   # the laziest foot
            # rest poses: the trunk where the pose puts it (the body's "keep the trunk up and level" does not apply)
            R = env["d"].xmat[:, base]
            up, pitch = R[:, 2, 2], jp.arcsin(jp.clip(R[:, 2, 0], -1.0, 1.0))
            com_z = env["d"].subtree_com[:, base, 2]
            height = com_z / env["info"]["h0"]
            sag = jp.clip(bd.UPRIGHT * env["info"]["h0"] - com_z, 0.0, None)
            resting = is_rest[ins]
            held = jp.exp(-(pitch - goal_pitch[ins]) ** 2 / 0.02 - (height - goal_height[ins]) ** 2 / 0.005)
            r = r + resting * (POSE_REWARD * held + 2.0 * (1.0 - up) + 10.0 * sag) * bd.CTRL_DT
            gw = jp.where(is_rest[ins], GUIDE_REST, GUIDE)
            gw = jp.where(env["info"]["down"], 0.0, gw)           # the teacher cannot show how to get up
            return (env, s, ins, air_ema), {**seen, "a": a, "logp": logprob(a, mean), "guide": guide, "gw": gw, "priv": priv, "r": r,
                                            "done": done, "fell": st["fell"], "v_fwd": st["v_fwd"], "wz": st["wz"], "ins": ins}
        (env, s_end, ins, air_ema), traj = jax.lax.scan(one, (env, s, ins, air_ema), jax.random.split(key, T))
        last = jp.concatenate([v_priv(env), jax.nn.one_hot(ins, n_i)], -1)
        return env, s_end, ins, air_ema, traj, last

    def advantages(params, traj, last, pstat):
        pm, ps = pstat
        val = lambda x: value(params["critic"], jp.clip((x - pm) / ps, -10, 10))
        vals, v_last = val(traj["priv"]), val(last)
        keep = 1.0 - traj["done"].astype(jp.float32)

        def back(carry, x):
            adv, nxt = carry
            r, v, m = x
            delta = r * 10.0 + 0.97 * nxt * m - v
            adv = delta + 0.97 * 0.95 * m * adv
            return (adv, v), adv
        _, adv = jax.lax.scan(back, (jp.zeros(B), v_last), (traj["r"], vals, keep), reverse=True)
        return adv, adv + vals

    def loss_fn(params, mb, s0, pstat, brain_on):
        bp = params["brain"]
        W = weights(bp)

        def one(s, x):
            s = fresh(s, x["new_life"])
            s, mu = bstep(bp, W, s, x)
            return s, jp.tanh(mu)
        _, mean = jax.lax.scan(one, s0, mb)
        ratio = jp.exp(logprob(mb["a"], mean) - mb["logp"])
        adv = (mb["adv"] - mb["adv"].mean()) / (mb["adv"].std() + 1e-8)
        pg = -jp.mean(jp.minimum(ratio * adv, jp.clip(ratio, 0.8, 1.2) * adv))
        guide = jp.mean(mb["gw"][..., None] * (mean - mb["guide"]) ** 2) / GUIDE
        pm, ps = pstat
        vl = jp.mean((value(params["critic"], jp.clip((mb["priv"] - pm) / ps, -10, 10)) - mb["ret"]) ** 2)
        return brain_on * (pg + GUIDE * guide) + 0.5 * vl, {"pg": pg, "guide": guide, "vl": vl}   # GUIDE * guide = mean(gw * err)

    def update(params, opt_state, traj, s0, adv, ret, pstat, key, brain_on):
        data = {k: traj[k] for k in ("feats", "C", "a_hist", "tag", "cmd", "words", "new_life", "priv", "a", "logp", "guide", "gw")}
        data["adv"], data["ret"] = adv, ret
        nm = 4
        size = B // nm
        perms = jax.vmap(lambda k: jax.random.permutation(k, B))(jax.random.split(key, 2))
        sels = perms[:, : nm * size].reshape(2 * nm, size)

        def one(carry, sel):
            params, opt_state = carry
            mb = jax.tree_util.tree_map(lambda a: a[:, sel], data)
            b0 = jax.tree_util.tree_map(lambda a: a[sel], s0)
            (_, m), g = jax.value_and_grad(loss_fn, has_aux=True)(params, mb, b0, pstat, brain_on)
            upd, opt_state = opt.update(jax.lax.pmean(g, "d"), opt_state, params)
            return (optax.apply_updates(params, upd), opt_state), m
        (params, opt_state), ms = jax.lax.scan(one, (params, opt_state), sels)
        return params, opt_state, jax.tree_util.tree_map(lambda x: jax.lax.pmean(x.mean(), "d"), ms)

    def norm_stats(x):
        m = jax.lax.pmean(x.mean(0), "d")
        v = jax.lax.pmean(((x - m) ** 2).mean(0), "d")
        return m, jp.maximum(jp.sqrt(v), 0.05)

    def iteration(params, opt_state, env, s, ins, air_ema, pstat, key, brain_on):
        kr, ku = jax.random.split(key)
        s0 = s
        env, s, ins, air_ema, traj, last = rollout(params, env, s, ins, air_ema, kr)
        adv, ret = advantages(params, traj, last, pstat)
        params, opt_state, m = update(params, opt_state, traj, s0, adv, ret, pstat, ku, brain_on)
        pstat = norm_stats(traj["priv"].reshape(-1, P))
        stats = {"reward_per_s": jax.lax.pmean(traj["r"].mean(), "d") / bd.CTRL_DT,
                 "falls_per_min": jax.lax.pmean(traj["fell"].mean(), "d") * 3000, **m}
        return params, opt_state, env, s, ins, air_ema, pstat, stats

    step = jax.pmap(iteration, axis_name="d", in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None))
    rep = lambda x: jax.device_put_replicated(x, jax.local_devices())
    params_r, opt_r = rep(params), rep(opt.init(params))
    keys = jax.random.split(jax.random.PRNGKey(args.seed + 1), nd + 1)
    env = jax.pmap(lambda k: v_reset(jax.random.split(k, B)))(keys[1:])
    s = jax.pmap(lambda _: init_state(B))(jp.arange(nd))
    ins = jp.zeros((nd, B), jp.int32)
    air_ema = jp.zeros((nd, B, 4))
    pstat = jax.pmap(lambda e: norm_stats(jp.concatenate([v_priv(e), jax.nn.one_hot(jp.zeros(B, jp.int32), n_i)], -1)),
                     axis_name="d")(env)
    key, curve = keys[0], []

    sc = make_scripted(bd, model, body_test, weights, init_state, bstep, teacher, walk_g, n=64)
    sc_down = make_scripted(bd, model, body_down, weights, init_state, bstep, teacher, walk_g, n=64)
    host = lambda: jax.tree_util.tree_map(lambda x: np.asarray(x)[0], params_r["brain"])
    teacher_ref = {g: sc["check"](host(), gait=g, support=0.0, teacher=True) for g in gaits}

    def score(r, g):
        return r["turn_rate"] * np.sign(GAITS[g][2]) if GAITS[g][2] else r["walks_from_rest"] * np.sign(GAITS[g][0])

    def real_test(label):
        rows = {}
        for g in gaits:
            r = sc["check"](host(), gait=g, support=0.0)
            rows[g] = r
            print(f"[{(time.time() - t0) / 60:5.1f} min] REAL TEST (no harness) {label}, \"{g}\" from rest: "
                  f"{score(r, g):+.3f} {'rad/s' if GAITS[g][2] else 'm/s'} (teacher {score(teacher_ref[g], g):+.3f});  "
                  f"feet off the ground FL/FR/RL/RR {r['feet_air']};  falls {r['falls']:.2f} per 13 s;  "
                  f"after \"stand\" {r['moves_after_stand']:.3f} (standing before {r['moves_before_walk']:.3f})", flush=True)
        for pz in rests:
            r = sc["check_pose"](host(), pz, pose_off[pz])
            rows[pz] = r
            print(f"[{(time.time() - t0) / 60:5.1f} min] REAL TEST (no harness) {label}, \"{pz}\": nose-up {r['pitch_in_pose']:+.2f} "
                  f"(teacher {pose_goal[pz]['pitch']:+.2f}), trunk at {r['height_in_pose']:.2f} (teacher {pose_goal[pz]['height']:.2f}); "
                  f"then \"stand\": trunk back at {r['height_back_up']:.2f};  falls {r['falls']:.2f}", flush=True)
        rows["get_up"] = sc_down["check_getup"](host())
        print(f"[{(time.time() - t0) / 60:5.1f} min] REAL TEST (no harness) {label}, fallen on its side or back, told \"stand\": "
              + ", ".join(f"{int(v * 100)}% on its feet after {k.split('_')[2]}" for k, v in rows["get_up"].items()), flush=True)
        curve.append({"when": label, **rows})
        (out / "practice.json").write_text(json.dumps(curve, indent=1))
        return rows

    first = real_test("before practice")
    iters = max(1, int(args.steps) // (args.envs * T))
    every = max(1, iters // 8)
    warm = max(1, iters // 10)
    print(f"practice: {iters} updates x {args.envs * T:,} steps; the critic learns alone for the first {warm}", flush=True)
    for it in range(iters):
        key, k = jax.random.split(key)
        params_r, opt_r, env, s, ins, air_ema, pstat, st = step(params_r, opt_r, env, s, ins, air_ema, pstat,
                                                                jax.random.split(k, nd), float(it >= warm))
        if (it + 1) % every == 0 or it == warm - 1:
            m = {k_: float(np.asarray(v)[0]) for k_, v in st.items()}
            print(f"          practising: reward {m['reward_per_s']:+.3f}/s  falls {m['falls_per_min']:.2f}/min  "
                  f"guide {m['guide']:.4f}  value error {m['vl']:.3f}", flush=True)
            if (it + 1) % every == 0:
                last = real_test(f"after {(it + 1) * args.envs * T / 1e6:.1f}M steps")
    trained = host()
    save_brain(trained, out / "baby_dog.npz", {"stage": "practice", "wiring": WIRING, "readout": args.readout, "vocab": VOCAB,
                                                "curve": curve})
    for g in gaits:
        print(f"VERDICT practice \"{g}\" (no harness, from rest): {score(first[g], g):+.3f} -> {score(last[g], g):+.3f} "
              f"(teacher {score(teacher_ref[g], g):+.3f}); falls per 13 s {first[g]['falls']:.2f} -> {last[g]['falls']:.2f}; "
              f"feet off the ground {first[g]['feet_air']} -> {last[g]['feet_air']}", flush=True)
    print(f"VERDICT practice getting up (fallen, told \"stand\"): " + ", ".join(
        f"{k.split('_')[2]}: {int(first['get_up'][k] * 100)}% -> {int(last['get_up'][k] * 100)}%" for k in first["get_up"]), flush=True)
    for pz in rests:
        print(f"VERDICT practice \"{pz}\" (no harness): nose-up {first[pz]['pitch_in_pose']:+.2f} -> {last[pz]['pitch_in_pose']:+.2f}, "
              f"trunk {first[pz]['height_in_pose']:.2f} -> {last[pz]['height_in_pose']:.2f} (teacher {pose_goal[pz]['pitch']:+.2f}, "
              f"{pose_goal[pz]['height']:.2f}); back up {last[pz]['height_back_up']:.2f}; falls {last[pz]['falls']:.2f}", flush=True)
    return {"curve": curve}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="babble", choices=["babble", "name", "act", "walk_words", "walk", "start_stop",
                                                        "gaits", "no_parent", "practice", "poses", "film_walk"])
    ap.add_argument("--gaits", default="walk", help="no_parent: the gaits it already knows, comma-separated")
    ap.add_argument("--get_up", action="store_true", help="practice: falls stay down, some lives start fallen")
    ap.add_argument("--brain", default="", help="the previous stage's brain (default: the attached Kaggle dataset)")
    ap.add_argument("--steps", type=float, default=15e6)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--unroll", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bias0", type=float, default=0.3)
    ap.add_argument("--goal_scale", type=float, default=2.0)
    ap.add_argument("--gain0", type=float, default=4.0)
    ap.add_argument("--readout", default="vector")
    ap.add_argument("--out", default="/kaggle/working" if Path("/kaggle/working").exists() else "results/baby_dog")
    ap.add_argument("--menagerie", default="")
    args = ap.parse_args([]) if JOB_ARGS else ap.parse_args()
    for k, v in JOB_ARGS.items():
        setattr(args, k, v)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if Path("/kaggle").exists():
        os.environ.setdefault("MUJOCO_GL", "egl")        # headless GPU rendering for the films
    bd = import_brain_dog()
    bd.ensure_packages()
    import jax
    print("jax", jax.__version__, "devices", jax.devices(), flush=True)
    {"babble": babble, "name": name, "act": act, "walk_words": walk_words, "walk": walk, "start_stop": start_stop,
     "gaits": gaits, "no_parent": no_parent, "practice": practice, "poses": poses,
     "film_walk": film_walk}[args.stage](args, bd, out)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
