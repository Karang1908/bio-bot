"""Automated environment setup for Colab session round36.
Configures pinned bio-bot stack on Tesla T4 GPU VM.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

print(f"[{sys.executable}] Setting up bio-bot on Colab VM...", flush=True)

# 1. Pinned packages
print("1. Installing pinned ML stack...", flush=True)
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "jax==0.7.2", "jaxlib==0.7.2", "mujoco==3.14.0", "mujoco-mjx==3.14.0", "optax==0.2.8"
], check=True)

# Remove incompatible pre-installed cuda13 plugin if present
subprocess.run([
    sys.executable, "-m", "pip", "uninstall", "-y", "-q",
    "jax-cuda13-pjrt", "jax-cuda13-plugin"
], check=False)

# 2. Clone/update repo
repo_dir = Path("/content/bio-bot")
if not repo_dir.exists():
    print("2. Cloning repo to /content/bio-bot...", flush=True)
    subprocess.run(["git", "clone", "https://github.com/Karang1908/bio-bot.git", str(repo_dir)], check=True)
else:
    print("2. Updating repo at /content/bio-bot...", flush=True)
    subprocess.run(["git", "pull", "--ff-only"], cwd=str(repo_dir), check=True)

# 3. Clone mujoco menagerie
menagerie_dir = Path("/content/bio-bot/mujoco_menagerie")
if not menagerie_dir.exists():
    print("3. Cloning mujoco_menagerie sparse checkout for unitree_go2...", flush=True)
    subprocess.run(["git", "clone", "--filter=blob:none", "--sparse", "https://github.com/google-deepmind/mujoco_menagerie.git", str(menagerie_dir)], check=True)
    subprocess.run(["git", "sparse-checkout", "set", "unitree_go2"], cwd=str(menagerie_dir), check=True)
else:
    print("3. mujoco_menagerie already exists.", flush=True)

# Ensure checkpoint and data directories exist
(repo_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
(repo_dir / "data" / "malecns").mkdir(parents=True, exist_ok=True)
(repo_dir / "data" / "dog_mocap").mkdir(parents=True, exist_ok=True)
(repo_dir / "results" / "colab_round36").mkdir(parents=True, exist_ok=True)

print("Environment setup script completed successfully.", flush=True)
