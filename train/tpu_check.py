"""Test whether JAX and MuJoCo MJX run on Kaggle TPU; never execute on the laptop."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

JOB_ARGS: dict = {}
EMBEDDED: dict = {}


def main() -> None:
    print("=== KAGGLE TPU MJX COMPATIBILITY CHECK ===", flush=True)
    report: dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "python": sys.version,
    }

    try:
        import jax
        print(f"JAX version: {jax.__version__}", flush=True)
        devices = jax.devices()
        print(f"JAX devices ({len(devices)}): {devices}", flush=True)
        backend = jax.default_backend()
        print(f"JAX default backend: {backend}", flush=True)
        report["jax_version"] = jax.__version__
        report["devices"] = [str(d) for d in devices]
        report["default_backend"] = backend
    except Exception as e:
        print(f"Failed to initialize JAX on TPU: {e}", flush=True)
        report["jax_error"] = str(e)
        Path("/kaggle/working/tpu_check.json").write_text(json.dumps(report, indent=2))
        sys.exit(1)

    # Ensure mujoco and mujoco-mjx are available
    try:
        import mujoco
        from mujoco import mjx
        print(f"Pre-installed MuJoCo: {mujoco.__version__}", flush=True)
    except ImportError:
        print("Installing mujoco and mujoco-mjx...", flush=True)
        res = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mujoco==3.14.0", "mujoco-mjx==3.14.0"], capture_output=True, text=True)
        print("Pip install output:", res.stdout, res.stderr, flush=True)
        import mujoco
        from mujoco import mjx
        print(f"Installed MuJoCo: {mujoco.__version__}", flush=True)

    report["mujoco_version"] = mujoco.__version__

    # Step 1: Minimal MJX test on TPU
    print("Step 1: Testing minimal MJX forward step...", flush=True)
    xml = """
    <mujoco>
      <worldbody>
        <body name="floor" pos="0 0 0">
          <geom type="plane" size="1 1 0.1"/>
        </body>
        <body name="box" pos="0 0 1">
          <joint type="free"/>
          <geom type="box" size="0.1 0.1 0.1" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    try:
        m = mujoco.MjModel.from_xml_string(xml)
        mx = mjx.put_model(m)
        dx = mjx.make_data(mx)
        step_fn = jax.jit(mjx.step)
        print("Compiling mjx.step on TPU...", flush=True)
        t0 = time.time()
        dx2 = step_fn(mx, dx)
        dx2.qpos.block_until_ready()
        compile_time = round(time.time() - t0, 3)
        print(f"Compiled and executed mjx.step in {compile_time}s! New qpos: {dx2.qpos}", flush=True)
        report["minimal_step_success"] = True
        report["compile_time_s"] = compile_time
    except Exception as e:
        import traceback
        print(f"Minimal MJX step on TPU failed: {e}", flush=True)
        traceback.print_exc()
        report["minimal_step_success"] = False
        report["minimal_step_error"] = str(e)
        Path("/kaggle/working/tpu_check.json").write_text(json.dumps(report, indent=2))
        sys.exit(1)

    print("Step 2: Testing batched mjx.step across TPU cores...", flush=True)
    try:
        batch_size = 64
        dx_batch = jax.vmap(lambda _: mjx.make_data(mx))(jax.numpy.arange(batch_size))
        vstep = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
        t0 = time.time()
        out = vstep(mx, dx_batch)
        out.qpos.block_until_ready()
        batch_time = round(time.time() - t0, 4)
        print(f"Batched step (B={batch_size}) completed in {batch_time}s!", flush=True)
        report["batch_step_success"] = True
        report["batch_time_s"] = batch_time
    except Exception as e:
        import traceback
        print(f"Batched MJX step failed: {e}", flush=True)
        traceback.print_exc()
        report["batch_step_success"] = False
        report["batch_step_error"] = str(e)

    report["status"] = "COMPLETE"
    Path("/kaggle/working/tpu_check.json").write_text(json.dumps(report, indent=2))
    print("=== TPU MJX CHECK FINISHED SUCCESSFULLY ===", flush=True)


if __name__ == "__main__":
    main()
