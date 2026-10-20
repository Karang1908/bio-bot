"""Active monitoring and real-time snapshot synchronization for Colab Round 36.
Uses non-blocking HTTP REST endpoints (colab ls, colab download) to monitor
progress and sync checkpoints and logs from Colab VM to local workspace.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

LOCAL_OUT = Path(__file__).resolve().parent.parent / "results" / "baby_dog_vision_approach_round36"
LOCAL_OUT.mkdir(parents=True, exist_ok=True)
SESSION = "round36"
REMOTE_DIR = "/content/bio-bot/results/colab_round36"


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
    ret, out = run_cmd(["colab", "ls", "-s", SESSION, REMOTE_DIR])
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
        remote_path = f"{REMOTE_DIR}/{fname}"

        # Always sync log.txt, practice.json, and baby_dog.npz; for snapshots sync if missing
        need_download = (fname in ("log.txt", "practice.json", "baby_dog.npz")) or (not local_fp.exists())

        if need_download:
            ret, out = run_cmd(["colab", "download", "-s", SESSION, remote_path, str(local_fp)])
            if ret == 0 and local_fp.exists():
                downloaded.append(fname)
            else:
                print(f"[{time.strftime('%H:%M:%S')}] ⚠️ Download failed for {fname}: {out}", flush=True)
    return downloaded


def check_and_sync() -> None:
    if not check_liveness():
        print(f"[{time.strftime('%H:%M:%S')}] Session {SESSION} is not active!", file=sys.stderr)
        return

    files = list_remote_files()
    if files:
        downloaded = sync_files(files)
        print(f"[{time.strftime('%H:%M:%S')}] Active session {SESSION} | Remote files: {len(files)} | Downloaded: {downloaded}")
        local_log = LOCAL_OUT / "log.txt"
        if local_log.exists():
            lines = local_log.read_text().splitlines()
            if lines:
                print("--- Log tail (latest 8 lines) ---")
                for l in lines[-8:]:
                    print("  ", l)
                print("---------------------------------")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] Active session {SESSION} | No results files generated yet.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor Colab Round 36 session")
    parser.add_argument("--loop", action="store_true", help="Run continuous monitoring loop")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval in seconds (default: 60)")
    args = parser.parse_args()

    if args.loop:
        print(f"Starting continuous monitor for Colab session '{SESSION}' (poll every {args.interval}s)...")
        while True:
            check_and_sync()
            time.sleep(args.interval)
    else:
        check_and_sync()


if __name__ == "__main__":
    main()
