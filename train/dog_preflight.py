"""Free Kaggle CPU checks before spending GPU quota; never execute this on the laptop."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

JOB_ARGS: dict = {}
EMBEDDED: dict = {}


def sensor_path_probe(flat, meta):
    """Counterfactual neural response on Kaggle CPU; synthetic steady senses, no body or weight updates."""
    import ast
    import sys
    import tempfile
    import numpy as np
    import jax
    import jax.numpy as jp
    module_dir = Path(tempfile.mkdtemp())
    for name in ("brain_dog.py", "dog_world.py"):
        (module_dir / name).write_bytes(base64.b64decode(EMBEDDED[name]))
    sys.path.insert(0, str(module_dir))
    import brain_dog as bd
    import dog_world as dw
    bd.EMBEDDED = EMBEDDED
    assert meta.get("absolute_drives"), "charge probe requires the candidate's existing level nerves"
    core = bd.load_core("v2")
    bp = {}
    for name, value in flat.items():
        if "/" in name:
            parent, child = name.split("/", 1)
            bp.setdefault(parent, {})[child] = jp.asarray(value)
        else:
            bp[name] = jp.asarray(value)
    fresh, weights, init_state, tick, _ = bd.make_brain(
        core, jax.random.PRNGKey(0), readout=meta["readout"], words=len(meta["vocab"]), say=11,
        eye_shape=dw.eye_dirs().shape[:2], lidar_shape=dw.lidar_dirs().shape[:2])
    assert jax.tree_util.tree_structure(fresh) == jax.tree_util.tree_structure(bp), "probe would omit learned parameters"
    assert all(a.shape == b.shape for a,b in zip(jax.tree_util.tree_leaves(fresh), jax.tree_util.tree_leaves(bp))), "probe brain shape mismatch"
    brain_tree = ast.parse(base64.b64decode(EMBEDDED["brain_dog.py"]))
    body_fn = next(n for n in brain_tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_body")
    wiring_fns = [n for n in body_fn.body if isinstance(n, ast.FunctionDef) and n.name in ("new_life", "jnp_sign")]
    scope = {"jax": jax, "jp": jp, "K0": 34, "E0": 28, "EL": 13, "E": 41, "J": 12, "DT": 16,
             "wiring": meta["wiring"], "mod_of": jp.array([0]*51 + [2]*6 + [3]*4 + [4,4] + [0]*12, jp.int32)}
    exec(compile(ast.Module(body=wiring_fns, type_ignores=[]), "brain_dog.py", "exec"), scope)
    life = scope["new_life"](jax.random.PRNGKey(0))
    charges = np.array([0.1, 0.2, 0.4, 1.0], np.float32)
    physical = jp.zeros((4, 75)).at[:, 62].set(jp.asarray(charges)).at[:, 63:].set(dw.AMBIENT / 100.0)
    anonymous = (physical[:, life["perm"]] * life["scale_in"] + life["off_in"]) * life["sign_in"]
    levels = bd.level_features(anonymous, jp.zeros_like(anonymous), life["perm"] >= 62, jp)
    seen = {"feats": jp.stack([levels, jp.zeros_like(levels)], -1), "C": jp.zeros((4,75,12,2)),
            "a_hist": jp.zeros((4,12,2)), "tag": jp.broadcast_to(life["tag"], (4,12,16)),
            "mod": jp.broadcast_to(life["mod"], (4,75)), "cmd": jp.zeros((4,3)),
            "eye": jp.zeros((4,*dw.eye_dirs().shape[:2],2)),
            "lidar": jp.full((4,*dw.lidar_dirs().shape[:2]), dw.LIDAR_RANGE)}
    inner = np.flatnonzero((core["role"] == 0) & (core["modality"] == 4))
    goal = np.concatenate([np.flatnonzero(core["role"] == 5), np.flatnonzero(core["role"] == 2)])
    assert len(inner), "no internal-state sensory neurons"
    @jax.jit
    def run(params, words):
        W = weights(params)
        sample = {**seen, "words": words}
        def one(state, _):
            state, mu = tick(params, W, state, sample)
            rates = jp.tanh(jax.nn.relu(state["v"]))
            return state, (rates[:, inner], rates[:, goal], tick.say(params, state), jp.tanh(mu))
        return jax.lax.scan(one, init_state(4), None, length=200)[1]
    result = {"method": "actual frozen recurrent brain on anonymous synthetic steady inputs; no physics/actions/training",
              "limitation": "does not measure perception in moving real scenes or prove a causal learning defect",
              "charges": charges.tolist(), "inner_sense_units": int(len(inner)),
              "input_contrast_max": float(np.max(np.abs(np.asarray(levels[0]-levels[-1])))), "contexts": {}}
    assert result["input_contrast_max"] > 0, "probe presented identical charge inputs"
    for context in ("stand", "quiet"):
        words = jp.zeros((4,len(meta["vocab"])))
        if context == "stand":
            words = words.at[:, meta["vocab"].index("stand")].set(1.0)
        print(f"SENSOR PATH: running frozen neural response, context={context}, 200 ticks", flush=True)
        sense, central, voice, motor = (np.asarray(x) for x in run(bp, words))
        assert all(np.isfinite(x).all() for x in (sense, central, voice, motor)), "non-finite probe response"
        snapshots = []
        for t in (0,9,49,99,199):
            delta = central[t,0] - central[t,-1]
            snapshots.append({"tick": t+1, "seconds": (t+1)*bd.CTRL_DT,
                              "active_inner_units": (sense[t] > 1e-6).sum(-1).tolist(),
                              "inner_charge_contrast_max": float(np.max(np.abs(sense[t,0]-sense[t,-1]))),
                              "central_charge_contrast_max": float(np.max(np.abs(delta))),
                              "central_charge_contrast_rms": float(np.sqrt(np.mean(delta**2))),
                              "hungry_logits": voice[t,:,6].tolist(),
                              "motor_charge_contrast_max": float(np.max(np.abs(motor[t,0]-motor[t,-1])))})
        result["contexts"][context] = snapshots
    return result


def main() -> None:
    if not Path("/kaggle/working").is_dir():
        raise SystemExit("STOP: this check must run on Kaggle, never on the laptop")
    import numpy as np

    report = {"sources": {}, "checks": []}
    for name in ("baby_dog.py", "brain_dog.py", "dog_world.py"):
        source = base64.b64decode(EMBEDDED[name])
        compile(source, name, "exec")
        report["sources"][name] = hashlib.sha256(source).hexdigest()
        report["checks"].append(f"{name}: syntax compiles")
    import ast
    brain_source = base64.b64decode(EMBEDDED["brain_dog.py"])
    tree = ast.parse(brain_source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "level_features")
    scope = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "brain_dog.py", "exec"), scope)
    levels = scope["level_features"]
    # At a fresh start adaptive centring returns zero for both charge levels. The additional nerves must differ.
    raw = np.array([0.1, 1.0, 0.25, 0.65], dtype=np.float32)
    zero = np.zeros_like(raw)
    retained = levels(raw, zero, np.ones_like(raw, dtype=bool), np)
    assert retained[0] < retained[1] and retained[2] < retained[3], "steady charge or heat was hidden"
    still = levels(np.tile(raw, (200, 1)), np.zeros((200, 4), np.float32), np.ones(4, bool), np)
    assert np.all(still == retained), "steady levels adapted away"
    adaptive = np.array([-0.7, 0.2, 1.3, -2.0], dtype=np.float32)
    mixed = levels(raw, adaptive, np.array([False, False, True, True]), np)
    assert np.array_equal(mixed[:2], adaptive[:2]), "established adaptive nerves changed"
    reversed_ = levels(-3.0 * raw + 0.7, zero, np.ones(4, bool), np)
    assert reversed_[0] > reversed_[1] and reversed_[2] > reversed_[3], "anonymous signed/scaled levels lost information"
    report["checks"].append("additional level nerves distinguish steady charge/heat; original adaptive values unchanged")
    if any(JOB_ARGS.get(k) for k in ("check_wiring", "check_guard", "check_voice_optimizer", "check_vision_optimizer", "check_reset", "check_sensor_path")):
        import os
        import subprocess
        import sys
        os.environ["JAX_PLATFORMS"] = "cpu"
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "jax==0.7.2", "jaxlib==0.7.2"])
        import jax
        import jax.numpy as jp
        assert all(d.platform == "cpu" for d in jax.devices()), "CPU preflight must not use GPU"
        report["jax"] = jax.__version__
    if JOB_ARGS.get("check_reset"):
        baby_tree = ast.parse(base64.b64decode(EMBEDDED["baby_dog.py"]))
        fn = next(n for n in baby_tree.body if isinstance(n, ast.FunctionDef) and n.name == "reset_brain_state")
        scope = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "baby_dog.py", "exec"), scope)
        reset = jax.jit(lambda s, n: scope["reset_brain_state"](s, n, jp, jax.tree_util))
        state = {"v": jp.arange(6.0).reshape(2, 3), "eye": jp.ones((2, 2, 3, 2)),
                 "nested": {"memory": jp.array([4.0, 5.0])}}
        got = reset(state, jp.array([True, False]))
        for before, after in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(got)):
            assert np.array_equal(after[0], np.zeros_like(before[0])), "new life retained recurrent memory"
            assert np.array_equal(after[1], before[1]), "ongoing life lost recurrent memory"
        scripted = next(n for n in baby_tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_scripted")
        life = next(n for n in scripted.body if isinstance(n, ast.FunctionDef) and n.name == "life")
        inner = next(n for n in life.body if isinstance(n, ast.FunctionDef) and n.name == "one")
        reset_index = next(i for i,n in enumerate(inner.body) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                           and isinstance(n.value.func, ast.Name) and n.value.func.id == "reset_brain_state")
        step_index = next(i for i,n in enumerate(inner.body) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                          and isinstance(n.value.func, ast.Name) and n.value.func.id == "bstep")
        assert reset_index < step_index, "scripted brain tick ran before the new-life reset"
        fn = next(n for n in baby_tree.body if isinstance(n, ast.FunctionDef) and n.name == "mean_test_rows")
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "baby_dog.py", "exec"), scope)
        rows = [{"falls": 0.2, "feet": [0.1, 0.3], "voice": {"balanced_accuracy": None, "positive_states": 0, "negative_states": 8}},
                {"falls": 0.4, "feet": [0.3, 0.5], "voice": {"balanced_accuracy": 0.8, "positive_states": 2, "negative_states": 6}}]
        mean = scope["mean_test_rows"](rows, np)
        np.testing.assert_allclose(mean["falls"], 0.3)
        np.testing.assert_allclose(mean["feet"], [0.2, 0.4])
        assert mean["voice"] == {"balanced_accuracy": 0.8, "positive_states": 2, "negative_states": 14}
        assert scope["mean_test_rows"]([None, None], np) is None, "unsampled label became a measured score"
        report["checks"] += ["actual recurrent reset under jit clears new lives and retains ongoing nested states",
                             "scripted movement tick applies the new-life reset before the brain step",
                             "fresh-cohort aggregation averages metrics, sums class counts and preserves unsampled labels"]
    if JOB_ARGS.get("check_voice_optimizer"):
        # The actual optimizer must preserve all non-voice weights, including nested motor interfaces,
        # while learning the readout and critic. No model or physics is run here.
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "jax==0.7.2", "jaxlib==0.7.2", "optax"])
        import optax
        baby_tree = ast.parse(base64.b64decode(EMBEDDED["baby_dog.py"]))
        fn = next(n for n in baby_tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_practice_optimizer")
        scope = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "baby_dog.py", "exec"), scope)
        make_opt = scope["make_practice_optimizer"]
        ps = {"brain": {"gamma": jp.array([0.2, -0.3]), "o2": {"w1": jp.array([[0.4, -0.5]])},
                        "say_w": jp.array([[0.1, 0.2]]), "say_b": jp.array([0.0, 0.0])},
              "critic": [{"w": jp.array([0.7]), "b": jp.array([0.0])}]}
        grad = jax.tree_util.tree_map(lambda x: jp.ones_like(x) * 2, ps)
        opt = make_opt(ps, 0.0, 3e-4, optax)
        state = opt.init(ps)
        @jax.jit
        def step(p, state, g):
            u, state = opt.update(g, state, p)
            return optax.apply_updates(p, u), state
        trained = ps
        for _ in range(4):
            trained, state = step(trained, state, grad)
        for name in ("gamma", "o2"):
            assert all(np.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(ps["brain"][name]),
                                                         jax.tree_util.tree_leaves(trained["brain"][name]))), "voice lesson changed motor weights"
        for name in ("say_w", "say_b"):
            assert not np.array_equal(ps["brain"][name], trained["brain"][name]), "voice head did not learn"
        assert not np.array_equal(ps["critic"][0]["w"], trained["critic"][0]["w"]), "critic did not learn"
        # Omitting the option must reproduce the established two-group optimizer exactly.
        usual = make_opt(ps, 1e-5, None, optax)
        original = optax.chain(optax.clip_by_global_norm(1.0),
                              optax.multi_transform({"brain": optax.inject_hyperparams(optax.adam)(learning_rate=1e-5),
                                                     "critic": optax.adam(3e-4)}, {"brain": "brain", "critic": "critic"}))
        u, _ = usual.update(grad, usual.init(ps), ps)
        old_u, _ = original.update(grad, original.init(ps), ps)
        assert all(np.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(u), jax.tree_util.tree_leaves(old_u))), "default optimizer changed"
        try:
            make_opt({"brain": {"gamma": jp.ones(1)}, "critic": jp.ones(1)}, 0.0, 3e-4, optax)
        except ValueError:
            pass
        else:
            raise AssertionError("separate voice rate accepted a missing head")
        report["optax"] = optax.__version__
        report["checks"] += ["actual voice optimizer under jit learns both voice weights and critic",
                             "zero shared learning rate preserves nested non-voice parameters across repeated updates",
                             "omitted voice option exactly matches established optimizer; missing voice aborts"]
    if JOB_ARGS.get("check_vision_optimizer"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "jax==0.7.2", "jaxlib==0.7.2", "optax"])
        import optax
        baby_tree = ast.parse(base64.b64decode(EMBEDDED["baby_dog.py"]))
        fn = next(n for n in baby_tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_practice_optimizer")
        scope = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "baby_dog.py", "exec"), scope)
        make_opt = scope["make_practice_optimizer"]
        v_keys = {"ol_e1", "ol_e2", "ol_l1", "ol_l2", "rf_e", "rf_l", "vp_e", "vp_l", "sp_eff", "sp_1", "sp_2"}
        ps = {"brain": {"gamma": jp.array([0.2, -0.3]), "o2": {"w1": jp.array([[0.4, -0.5]])},
                        "say_w": jp.array([[0.1, 0.2]]), "say_b": jp.array([0.0, 0.0]),
                        "word_w": jp.ones((25, 4)) * 0.1,
                        **{k: jp.ones((2, 2)) * 0.1 for k in v_keys}},
              "critic": [{"w": jp.array([0.7]), "b": jp.array([0.0])}]}
        grad = jax.tree_util.tree_map(lambda x: jp.ones_like(x) * 2, ps)
        opt = make_opt(ps, 0.0, 3e-4, 3e-4, optax)
        state = opt.init(ps)
        @jax.jit
        def step(p, state, g):
            u, state = opt.update(g, state, p)
            return optax.apply_updates(p, u), state
        trained = ps
        for _ in range(4):
            trained, state = step(trained, state, grad)
        for name in ("gamma", "o2", "word_w"):
            assert all(np.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(ps["brain"][name]),
                                                             jax.tree_util.tree_leaves(trained["brain"][name]))), f"vision/voice lesson changed motor weight {name}"
        for name in ("say_w", "say_b"):
            assert not np.array_equal(ps["brain"][name], trained["brain"][name]), "voice head did not learn"
        for name in v_keys:
            assert not np.array_equal(ps["brain"][name], trained["brain"][name]), f"vision param {name} did not learn"
        assert not np.array_equal(ps["critic"][0]["w"], trained["critic"][0]["w"]), "critic did not learn"

        # Also test task_scale with non-zero brain_lr
        opt_scale = make_opt(ps, 5e-6, 3e-4, 3e-4, optax, task_scale=120.0)
        state_s = opt_scale.init(ps)
        u_s, _ = opt_scale.update(grad, state_s, ps)
        trained_s = optax.apply_updates(ps, u_s)
        diff_base = float(np.max(np.abs(ps["brain"]["word_w"][:23] - trained_s["brain"]["word_w"][:23])))
        diff_task = float(np.max(np.abs(ps["brain"]["word_w"][23:] - trained_s["brain"]["word_w"][23:])))
        assert diff_base > 0, "base words did not update at brain_lr"
        assert diff_task > 50 * diff_base, "task words did not scale by task_scale"
        report["optax"] = optax.__version__
        report["checks"] += ["actual vision optimizer under jit learns vision frontend, voice and critic while freezing motor weights",
                             "task word scaling verified under JIT: task words 23+ scaled by task_scale relative to base words"]
    if JOB_ARGS.get("check_wiring"):
        body_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_body")
        wiring_fns = [n for n in body_fn.body if isinstance(n, ast.FunctionDef) and n.name in ("new_life", "jnp_sign")]
        original_mods = [0] * 34 + [0] * 17 + [2] * 6 + [3] * 4 + [4]
        def draw(added):
            scope = {"jax": jax, "jp": jp, "K0": 34, "E0": 28, "EL": added, "E": 28 + added,
                     "J": 12, "DT": 16, "wiring": 20260927,
                     "mod_of": jp.array(original_mods + ([4] + [0] * 12 if added else []), jp.int32)}
            exec(compile(ast.Module(body=wiring_fns, type_ignores=[]), "brain_dog.py", "exec"), scope)
            return {k: np.asarray(v) for k, v in scope["new_life"](jax.random.PRNGKey(0)).items()}
        old, grown = draw(0), draw(13)
        for name in ("perm", "sign_in", "scale_in", "off_in", "mod"):
            assert np.array_equal(old[name], grown[name][:62]), f"adding level nerves rewired {name}"
        for name in ("sign_out", "tag"):
            assert np.array_equal(old[name], grown[name]), f"adding level nerves changed motor {name}"
        assert np.array_equal(np.sort(grown["perm"][62:]), np.arange(62, 75)), "new nerves overlap old ones"
        for charge, temp in ((0.1, 0.25), (1.0, 0.65)):
            physical = np.concatenate([np.zeros(62), [charge], np.full(12, temp)]).astype(np.float32)
            anonymous = (physical[grown["perm"]] * grown["scale_in"] + grown["off_in"]) * grown["sign_in"]
            sensed = levels(anonymous, np.zeros(75, np.float32), grown["perm"] >= 62, np)
            assert np.array_equal(sensed[:62], np.zeros(62)), "original adaptive values changed"
            if charge == 0.1:
                before = sensed
            else:
                assert np.all(sensed[62:] != before[62:]), "a steady drive level was lost after actual wiring"
        try:
            jax.random.PRNGKey(1.0)
        except TypeError as exc:
            report["decimal_seed_error"] = str(exc)
        else:
            raise AssertionError("expected JAX 0.7.2 to reject the original decimal seed")
        assert jax.random.PRNGKey(1).shape == (2,), "corrected integer seed failed"
        report["jax"] = jax.__version__
        report["checks"] += ["actual JAX wiring keeps all 62 established nerves and motor lines bit-for-bit",
                             "actual appended anonymous nerves retain steady drive levels",
                             "JAX decimal-seed failure reproduced; corrected integer seed works"]
    if JOB_ARGS.get("check_guard"):
        baby_tree = ast.parse(base64.b64decode(EMBEDDED["baby_dog.py"]))
        fn = next(n for n in baby_tree.body if isinstance(n, ast.FunctionDef) and n.name == "protect_outcome_update")
        scope = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "baby_dog.py", "exec"), scope)
        guard = jax.jit(lambda u, g: scope["protect_outcome_update"](u, g, jp, jax.tree_util))
        # Actual helper: nested trees, tangent learning retained, descent unchanged, zero gradient safe.
        gradient = {"x": jp.array([1.0, 0.0]), "head": {"b": jp.array(0.0)}}
        raw = {"x": jp.array([0.2, -0.1]), "head": {"b": jp.array(0.3)}}
        protected, metrics = guard(raw, gradient)
        np.testing.assert_allclose(protected["x"], [0.0, -0.1], atol=1e-6)
        np.testing.assert_allclose(protected["head"]["b"], 0.3, atol=1e-6)
        assert float(metrics["skill_dot_before"]) > 0 and abs(float(metrics["skill_dot_after"])) < 1e-6
        for g, u in ((gradient, {"x": jp.array([-0.2, -0.1]), "head": {"b": jp.array(0.3)}}),
                     (jax.tree_util.tree_map(jp.zeros_like, gradient), raw)):
            unchanged, _ = guard(u, g)
            assert all(np.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(unchanged), jax.tree_util.tree_leaves(u)))
        # Exercise the actual two-pullback API pattern, including the critic-only warm-up.
        def losses(ps, brain_on):
            x = ps["brain"]["x"]
            outcome = brain_on * 0.5 * jp.sum(x ** 2) + 0.5 * jp.sum(ps["critic"] ** 2)
            perception = brain_on * 0.5 * jp.sum((x - jp.array([4.0, 8.0])) ** 2)
            return jp.stack([outcome, perception]), {"outcome": outcome}
        ps = {"brain": {"x": jp.array([1.0, 2.0])}, "critic": jp.array([1.0])}
        for on in (0.0, 1.0):
            _, pullback, _ = jax.vjp(lambda p: losses(p, on), ps, has_aux=True)
            total_g = pullback(jp.ones(2, jp.float32))[0]
            outcome_g = pullback(jp.array([1.0, 0.0], jp.float32))[0]
            direct_g = jax.grad(lambda p: losses(p, on)[0].sum())(ps)
            assert all(np.allclose(a, b) for a, b in zip(jax.tree_util.tree_leaves(total_g), jax.tree_util.tree_leaves(direct_g)))
            updates = jax.tree_util.tree_map(lambda g: -0.1 * g, total_g)
            protected, metrics = guard(updates["brain"], outcome_g["brain"])
            assert float(metrics["skill_dot_after"]) <= 1e-6, "guarded step increases the outcome loss to first order"
            assert np.array_equal(updates["critic"], np.array([-0.1], np.float32)), "critic was frozen by brain warm-up"
            if not on:
                assert np.array_equal(protected["x"], np.zeros(2)), "critic-only warm-up changed the brain"
        report["checks"] += ["actual JAX update guard removes conflicting components and retains tangent learning",
                             "descent/zero-gradient updates unchanged; nested parameter trees work under jit",
                             "dual VJP matches total gradient; critic-only warm-up leaves brain frozen"]
    if "kaggle_job.py" in EMBEDDED:
        import re
        source = base64.b64decode(EMBEDDED["kaggle_job.py"])
        compile(source, "kaggle_job.py", "exec")
        tree = ast.parse(source)
        fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("parse_status", "parse_value")]
        scope = {"re": re}
        exec(compile(ast.Module(body=fns, type_ignores=[]), "kaggle_job.py", "exec"), scope)
        for response, expected in [
            ('owner/job has status "KernelWorkerStatus.QUEUED"', "QUEUED"),
            ('owner/job has status "COMPLETE"', "COMPLETE"),
            ('owner/job has status "KernelWorkerStatus.ERROR"\nFailure message: "ImportError"', "ERROR"),
            ('owner/job has status "KernelWorkerStatus.CANCELLED"', "CANCELLED"),
        ]:
            assert scope["parse_status"](response) == expected, "status parser hid a terminal state"
        report["sources"]["kaggle_job.py"] = hashlib.sha256(source).hexdigest()
        report["checks"].append("waiter recognises ERROR even with a trailing failure message")
        for raw, expected, kind in [("1", 1, int), ("-2", -2, int), ("5e-5", 5e-5, float),
                                    ("2e6", 2e6, float), ("perception", "perception", str)]:
            actual = scope["parse_value"](raw)
            assert actual == expected and type(actual) is kind, f"launcher changed the type of {raw!r}"
        report["checks"].append("launcher preserves integer seeds and numeric/string settings")
    chosen = JOB_ARGS.get("brain")
    if chosen:
        p = Path(chosen)
        found = [p] if p.is_file() else sorted(Path("/kaggle/input").glob(f"**/{p.name}"))
    else:
        found = sorted(Path("/kaggle/input").glob("**/baby_dog.npz"))
    if len(found) != 1 or not found[0].is_file():
        raise SystemExit(f"STOP: expected exactly one input brain, found {found}")
    cores = sorted(Path("/kaggle/input").glob("**/core_v2.npz"))
    if len(cores) != 1:
        raise SystemExit(f"STOP: expected exactly one v2 core, found {cores}")
    with np.load(cores[0], allow_pickle=False) as z:
        units, edges = len(z["name"]), len(z["pre"])
    with np.load(found[0], allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        assert meta.get("stage") == "let_go", "teacher has not left this brain"
        assert meta.get("wiring") == 20260927, "wrong body's wiring"
        assert meta.get("readout") == "vector", "wrong motor readout"
        last = meta["curve"][-1]
        assert last["when"] == JOB_ARGS["expect_checkpoint"], f"wrong checkpoint: {last['when']}"
        assert z["bias"].shape == (units,), "brain/core unit mismatch"
        assert z["log_tau"].shape == (units,), "brain/core time constant mismatch"
        assert z["gamma"].shape == (edges,), "brain/core connection mismatch"
        assert z["say_b"].shape == (11,), "voice head missing or wrong size"
        assert z["word_w"].shape[0] == 25, "word vocabulary changed"
        assert all(h == 0 for h in last["help"].values()), "last checkpoint had teacher help"
        for name in z.files:
            if name != "meta":
                assert np.isfinite(z[name]).all(), f"non-finite learned parameter: {name}"
        if JOB_ARGS.get("check_sensor_path"):
            probe_flat = {name: np.asarray(z[name]) for name in z.files if name != "meta"}
        report["checks"] += ["teacher already zero", "fixed wiring preserved", "v2 parameter shapes match",
                             "voice and words preserved", "all learned parameters finite"]
        report.update({"brain": str(found[0]), "brain_sha256": hashlib.sha256(found[0].read_bytes()).hexdigest(),
                       "stage": meta["stage"], "checkpoint": last["when"], "units": units, "edges": edges,
                       "absolute_drives": meta.get("absolute_drives", False), "partial": meta.get("partial"),
                       "baseline": {k: v for k, v in last.items() if k != "help"}})
    if JOB_ARGS.get("check_sensor_path"):
        print("SENSOR PATH: input verified; building the frozen CPU neural probe", flush=True)
        report["sensor_path"] = sensor_path_probe(probe_flat, meta)
        report["checks"].append("actual frozen-brain counterfactual charge response measured on CPU")
    Path("/kaggle/working/preflight.json").write_text(json.dumps(report, indent=2))
    print(f"PASS: {units:,} units, {edges:,} connections, checkpoint {report['checkpoint']}; teacher zero", flush=True)
    print(f"Input SHA256: {report['brain_sha256']}", flush=True)
    print("Sources compile; " + ("JAX CPU helpers checked; " if "jax" in report else "no JAX; ")
          + "no simulation or training executed in this CPU check.", flush=True)


if __name__ == "__main__":
    main()
