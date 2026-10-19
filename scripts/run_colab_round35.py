"""Runner for Colab Round 35 that streams stdout/stderr live to keep kernel busy.
Trains visual target approach with extended 1.2m deceleration zone and continuous brake scaling.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

res_dir = Path("/content/bio-bot/results/colab_round35")
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
    "--lr", "5e-6",
    "--voice_lr", "0.0003",
    "--vision_lr", "0.0003",
    "--task_scale", "60.0",
    "--task_pay", "50.0",
    "--reach_pay", "100.0",
    "--stay_pay", "50.0",
    "--prox_pay", "25.0",
    "--face_pay", "20.0",
    "--fwd_pay", "45.0",
    "--sprint_pay", "40.0",
    "--sprint_floor", "0.42",
    "--align_floor", "0.25",
    "--yaw_pay", "25.0",
    "--misalign_cost", "4.0",
    "--roll_cost", "3.0",
    "--side_cost", "1.0",
    "--upright_cost", "0.0",
    "--task_fall_cost", "3.0",
    "--steps", "131072",
    "--envs", "256",
    "--evals", "2",
    "--eval_repeats", "3",
    "--seed", "0",
    "--brain", "checkpoints/baby_dog_step_0000393216.npz",
    "--expect_checkpoint", "after 0.4M steps",
    "--out", str(res_dir),
]

print("Starting Round 35 training in /content/bio-bot on Colab Tesla T4 GPU...", flush=True)
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
log_file.close()
print(f"Training finished with exit code {proc.returncode}", flush=True)
sys.exit(proc.returncode)
