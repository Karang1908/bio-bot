"""Active monitoring and real-time snapshot synchronization for Colab Round 16."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

LOCAL_OUT = Path(__file__).resolve().parent.parent / "results" / "baby_dog_vision_approach_round16"
LOCAL_OUT.mkdir(parents=True, exist_ok=True)
SESSION = "round16"


def run_cmd(cmd: list[str] | str, capture: bool = True) -> tuple[int, str]:
    if isinstance(cmd, str):
        res = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    else:
        res = subprocess.run(cmd, capture_output=capture, text=True)
    return res.returncode, (res.stdout + res.stderr).strip()


def check_liveness() -> bool:
    code, out = run_cmd(["colab", "sessions"])
    return SESSION in out


def inspect_remote() -> dict:
    code = """
import os, subprocess, json

pid_file = "/content/bio-bot/results/colab_round16/pid.txt"
pid = open(pid_file).read().strip() if os.path.exists(pid_file) else None
running = False
if pid:
    res = subprocess.run(f"ps -p {pid}", shell=True, capture_output=True, text=True)
    running = (res.returncode == 0)

files = {}
res_dir = "/content/bio-bot/results/colab_round16"
if os.path.exists(res_dir):
    for f in os.listdir(res_dir):
        fp = os.path.join(res_dir, f)
        if os.path.isfile(fp):
            files[f] = os.path.getsize(fp)

log_tail = []
log_file = os.path.join(res_dir, "log.txt")
if os.path.exists(log_file):
    lines = open(log_file).readlines()
    log_tail = [l.strip() for l in lines[-15:]]

print(json.dumps({"pid": pid, "running": running, "files": files, "log_tail": log_tail}))
"""
    cmd = f"echo '{code}' | colab exec -s {SESSION} --timeout 30.0"
    ret, out = run_cmd(cmd)
    try:
        # Find json line in output
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)
    except Exception as e:
        print(f"Error parsing remote state: {e}\nRaw output:\n{out}", file=sys.stderr)
    return {"raw": out}


def sync_files(remote_files: dict[str, int]) -> list[str]:
    downloaded = []
    for fname, size in remote_files.items():
        if fname in ("pid.txt",):
            continue
        local_fp = LOCAL_OUT / fname
        need_download = False
        if not local_fp.exists():
            need_download = True
        elif local_fp.stat().st_size != size:
            need_download = True
        
        if need_download:
            remote_path = f"/content/bio-bot/results/colab_round16/{fname}"
            print(f"[{time.strftime('%H:%M:%S')}] Downloading {fname} ({size:,} bytes)...", flush=True)
            ret, out = run_cmd(["colab", "download", "-s", SESSION, remote_path, str(local_fp)])
            if ret == 0 and local_fp.exists():
                print(f"[{time.strftime('%H:%M:%S')}] ✅ Saved {local_fp.name} ({local_fp.stat().st_size:,} bytes)", flush=True)
                downloaded.append(fname)
            else:
                print(f"[{time.strftime('%H:%M:%S')}] ⚠️ Download failed for {fname}: {out}", flush=True)
    return downloaded


def main() -> None:
    if not check_liveness():
        print(f"Session {SESSION} is not active!", file=sys.stderr)
        sys.exit(1)
    
    state = inspect_remote()
    if "files" in state:
        files = state["files"]
        sync_files(files)
        print(f"[{time.strftime('%H:%M:%S')}] Remote PID: {state.get('pid')} (running: {state.get('running')}) | Remote files: {list(files.keys())}")
        if state.get("log_tail"):
            print("--- Log tail ---")
            for l in state["log_tail"][-8:]:
                print("  ", l)
            print("----------------")
    else:
        print(f"State check returned:\n{state}")


if __name__ == "__main__":
    main()
