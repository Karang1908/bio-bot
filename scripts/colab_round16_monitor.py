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


def list_remote_files() -> list[str]:
    ret, out = run_cmd(["colab", "ls", "-s", SESSION, "/content/bio-bot/results/colab_round16"])
    if ret != 0:
        return []
    lines = [l.strip() for l in out.splitlines() if l.strip() and not l.strip().endswith("/")]
    return lines


def sync_files(remote_files: list[str]) -> list[str]:
    downloaded = []
    for fname in remote_files:
        if fname in ("pid.txt",):
            continue
        local_fp = LOCAL_OUT / fname
        remote_path = f"/content/bio-bot/results/colab_round16/{fname}"
        
        # Always sync log.txt; for checkpoints/json sync if missing or modified
        need_download = (fname == "log.txt") or (not local_fp.exists())
        
        if need_download:
            ret, out = run_cmd(["colab", "download", "-s", SESSION, remote_path, str(local_fp)])
            if ret == 0 and local_fp.exists():
                downloaded.append(fname)
            else:
                print(f"[{time.strftime('%H:%M:%S')}] ⚠️ Download failed for {fname}: {out}", flush=True)
    return downloaded


def main() -> None:
    if not check_liveness():
        print(f"Session {SESSION} is not active!", file=sys.stderr)
        sys.exit(1)
    
    files = list_remote_files()
    if files:
        sync_files(files)
        print(f"[{time.strftime('%H:%M:%S')}] Active session {SESSION} | Remote files: {files}")
        local_log = LOCAL_OUT / "log.txt"
        if local_log.exists():
            lines = local_log.read_text().splitlines()
            if lines:
                print("--- Log tail ---")
                for l in lines[-8:]:
                    print("  ", l)
                print("----------------")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] Active session {SESSION} | No results files generated yet.")


if __name__ == "__main__":
    main()
