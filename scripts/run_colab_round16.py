"""Runner for Colab Round 16 that streams stdout/stderr live to keep kernel busy."""
import os
import subprocess
import sys
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

res_dir = Path("/content/bio-bot/results/colab_round16")
res_dir.mkdir(parents=True, exist_ok=True)

log_fp = res_dir / "log.txt"
log_file = open(log_fp, "w")

cmd = [
    sys.executable, "-u", "train/baby_dog.py",
    "--stage", "practice",
    "--core", "v2",
    "--readout", "vector",
    "--lesson", "approach",
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
    "--task_scale", "800.0",
    "--task_pay", "16.0",
    "--fwd_pay", "16.0",
    "--face_pay", "4.0",
    "--yaw_pay", "0.0",
    "--reach_pay", "25.0",
    "--steps", "1500000",
    "--envs", "256",
    "--evals", "3",
    "--eval_repeats", "3",
    "--seed", "1",
    "--brain", "checkpoints/baby_dog_step_0000491520.npz",
    "--expect_checkpoint", "after 0.5M steps",
    "--out", str(res_dir),
]

print("Starting Round 16 training in /content/bio-bot...", flush=True)
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
