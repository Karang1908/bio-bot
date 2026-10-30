"""Runner for Colab Round 57 that streams stdout/stderr live to keep kernel busy.
Trains visual target approach from all-time multi-cohort champion baby_dog_step_0000327680.npz
with widened proximity capture funnel (brake_scale 2.5, reach_pay 250.0, stay_pay 150.0, prox_pay 60.0)
and proven postural stabilization (task_fall_cost 15.0, roll_cost 5.0, upright_cost 2.5, side_cost 1.5)
to break the >15% arrival barrier while preserving 100% nominal motor skills.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import threading
import time

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

res_dir = Path("/content/bio-bot/results/colab_round57")
res_dir.mkdir(parents=True, exist_ok=True)

log_fp = res_dir / "log.txt"
log_file = open(log_fp, "w")

cmd = [
    sys.executable, "-u", "train/baby_dog.py",
    "--stage", "practice",
    "--core", "v2",
    "--readout", "vector",
    "--lesson", "approach",
    "--dog",
    "--park",
    "--throws",
    "--tasks",
    "--drives",
    "--voice",
    "--self_learning",
    "--let_go",
    "--absolute_drives",
    "--protect_skills",
    "--lr", "3e-6",
    "--voice_lr", "0.0003",
    "--vision_lr", "0.0003",
    "--task_scale", "60.0",
    "--task_pay", "90.0",
    "--reach_pay", "250.0",
    "--stay_pay", "150.0",
    "--prox_pay", "60.0",
    "--face_pay", "40.0",
    "--fwd_pay", "60.0",
    "--sprint_pay", "45.0",
    "--sprint_floor", "0.38",
    "--align_floor", "0.35",
    "--yaw_pay", "35.0",
    "--search_pay", "45.0",
    "--misalign_cost", "3.0",
    "--brake_scale", "2.5",
    "--brake_floor", "0.60",
    "--roll_cost", "5.0",
    "--side_cost", "1.5",
    "--upright_cost", "2.5",
    "--task_fall_cost", "15.0",
    "--steps", "131072",
    "--envs", "256",
    "--evals", "2",
    "--eval_repeats", "3",
    "--seed", "0",
    "--brain", "checkpoints/baby_dog_step_0000327680.npz",
    "--expect_checkpoint", "after 0.3M steps",
    "--out", str(res_dir),
    "--menagerie", "/content/bio-bot/mujoco_menagerie",
    "--skip_baseline",
]

print("Starting Round 57 training in /content/bio-bot on Colab Tesla T4 GPU...", flush=True)

stop_hb = threading.Event()

def heartbeat_worker():
    while not stop_hb.wait(15.0):
        t_str = time.strftime("%H:%M:%S")
        sys.stdout.write(f"[{t_str}] [colab-heartbeat] active kernel liveness keepalive\n")
        sys.stdout.flush()

hb_thread = threading.Thread(target=heartbeat_worker, daemon=True)
hb_thread.start()

proc = subprocess.Popen(
    cmd,
    cwd="/content/bio-bot",
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

with open(res_dir / "pid.txt", "w") as f:
    f.write(str(proc.pid))

for line in proc.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()
    log_file.write(line)
    log_file.flush()

proc.wait()
stop_hb.set()
hb_thread.join(timeout=1.0)
log_file.close()
print(f"Training finished with exit code {proc.returncode}", flush=True)
sys.exit(proc.returncode)
