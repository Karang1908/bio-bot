"""Run a training script from train/ as a private GPU job on Kaggle, then fetch its outputs.

    .venv/bin/python scripts/kaggle_job.py push brain_dog --steps 100e6 --embed data/malecns/core_v1.npz
    .venv/bin/python scripts/kaggle_job.py wait brain_dog        # poll until it finishes
    .venv/bin/python scripts/kaggle_job.py fetch brain_dog       # -> results/brain_dog/

A Kaggle script job is a single file, so the job's settings are written into the script's
JOB_ARGS line, and data files it needs (--embed) into its EMBEDDED line as base64. Needs the Kaggle CLI signed in once (`.venv/bin/kaggle auth login`).
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KAGGLE = str(REPO / ".venv" / "bin" / "kaggle")


def kaggle(*args: str) -> str:
    out = subprocess.run([KAGGLE, *args], capture_output=True, text=True)
    text = out.stdout + out.stderr
    # the CLI can print an HTTP error and still exit 0 (seen: 400 on an oversized kernel push)
    if out.returncode != 0 or "Client Error" in text or "Server Error" in text:
        sys.exit(f"kaggle {' '.join(args)} failed:\n{text}")
    return out.stdout


def username() -> str:
    m = re.search(r"username:\s*(\S+)", kaggle("config", "view"))
    if not m or m.group(1) == "None":
        sys.exit("Not signed in to Kaggle: run  .venv/bin/kaggle auth login")
    return m.group(1)


def slug(name: str) -> str:
    return f"bio-bot-{name.replace('_', '-')}"


def push(name: str, job_args: dict, embed: list[str], datasets: list[str], machine: str = "", job: str = "") -> None:
    """name: the script in train/; job: the Kaggle job it runs as (default: name). Different jobs run side by side."""
    src = REPO / "train" / f"{name}.py"
    code = src.read_text()
    if "JOB_ARGS: dict = {}" not in code:
        sys.exit(f"{src} has no `JOB_ARGS: dict = {{}}` line to fill in")
    code = code.replace("JOB_ARGS: dict = {}", f"JOB_ARGS: dict = {job_args!r}", 1)   # a Python literal (True, not true)
    if embed:
        if "EMBEDDED: dict = {}" not in code:
            sys.exit(f"{src} has no `EMBEDDED: dict = {{}}` line for --embed")
        files = {Path(f).name: base64.b64encode((REPO / f).read_bytes()).decode() for f in embed}
        code = code.replace("EMBEDDED: dict = {}", f"EMBEDDED: dict = {json.dumps(files)}", 1)
    job_dir = REPO / "results" / "kaggle" / (job or name)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / f"{name}.py").write_text(code)
    meta = {
        "id": f"{username()}/{slug(job or name)}", "title": slug(job or name), "code_file": f"{name}.py",
        "language": "python", "kernel_type": "script", "is_private": "true",
        "enable_gpu": "false" if machine.startswith("Tpu") else "true", "enable_tpu": "true" if machine.startswith("Tpu") else "false",
        "enable_internet": "true", "machine_shape": machine, "dataset_sources": [f"{username()}/{d}" for d in datasets], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    (job_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(kaggle("kernels", "push", "-p", str(job_dir)).strip())


def status(name: str) -> str:
    line = kaggle("kernels", "status", f"{username()}/{slug(name)}").strip().splitlines()[-1]
    m = re.search(r'status "(?:KernelWorkerStatus\.)?(\w+)"', line)
    return (m.group(1) if m else line).upper()


def wait(name: str, every: float = 30.0) -> str:
    t0 = time.time()
    misses = 0
    while True:
        try:
            s = status(name)
            misses = 0
        except SystemExit as e:              # a dropped connection mid-run is common; give up only if it persists
            misses += 1
            if misses >= 10:
                raise
            print(f"{time.strftime('%H:%M:%S')}  status check failed ({misses}/10), retrying: {str(e).strip()[-120:]}", flush=True)
            time.sleep(every)
            continue
        print(f"{time.strftime('%H:%M:%S')}  {s}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if s in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED"):
            return s
        time.sleep(every)


def fetch(name: str) -> Path:
    out = REPO / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    # -o: always download. Without it the CLI skips a file that exists locally with the same size, and a new brain
    # of the same shape silently kept the previous run's file (seen twice: stage 4b, practice round 3)
    kaggle("kernels", "output", f"{username()}/{slug(name)}", "-p", str(out), "-o")
    log = next(out.glob("*.log"), None)
    if log:  # the log is a JSON list of {stream, data} chunks; keep a plain-text copy
        try:
            text = "".join(e.get("data", "") for e in json.loads(log.read_text()))
            (out / "log.txt").write_text(text)
        except json.JSONDecodeError:
            pass
    print(f"outputs in {out}: {', '.join(sorted(p.name for p in out.iterdir()))}")
    return out


def upload(name: str) -> None:
    """Upload results/<name>/<name>.npz as a new version of the private dataset bio-bot-brains."""
    src = REPO / "results" / name / f"{name}.npz"
    ds = REPO / "results" / "kaggle" / "datasets" / "bio-bot-brains"
    ds.mkdir(parents=True, exist_ok=True)
    (ds / f"{name}.npz").write_bytes(src.read_bytes())
    meta = {"title": "bio-bot-brains", "id": f"{username()}/bio-bot-brains", "licenses": [{"name": "CC0-1.0"}]}
    (ds / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    exists = subprocess.run([KAGGLE, "datasets", "files", f"{username()}/bio-bot-brains"], capture_output=True, text=True)
    if exists.returncode == 0 and "Error" not in exists.stdout + exists.stderr:
        print(kaggle("datasets", "version", "-p", str(ds), "-m", f"{name}", "-q").strip())
    else:
        print(kaggle("datasets", "create", "-p", str(ds), "-q").strip())       # private by default


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["push", "status", "wait", "fetch", "upload"])
    ap.add_argument("name", help="script name in train/, e.g. brain_dog")
    ap.add_argument("--job", default="", help="run as this Kaggle job (default: the script name); jobs run side by side, "
                                               "and status/wait/fetch take the job name")
    ap.add_argument("--steps", type=float, help="environment steps to train")
    ap.add_argument("--envs", type=int, help="parallel environments")
    ap.add_argument("--evals", type=int, help="evaluations during training")
    ap.add_argument("--embed", action="append", default=[], help="repo file to ship inside the job (repeatable)")
    ap.add_argument("--flag", action="append", default=[], help="boolean option to switch on in the job (repeatable)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="option for the job, e.g. --set bias0=0.3 or --set readout=line (repeatable)")
    ap.add_argument("--machine", default="", choices=["", "NvidiaTeslaT4", "NvidiaTeslaP100", "Tpu1VmV38"],
                    help="Kaggle machine (default: Kaggle's GPU default, 2x T4); another shape has its own queue")
    ap.add_argument("--dataset", action="append", default=[],
                    help="one of your Kaggle datasets to attach; it appears under /kaggle/input/<name>/ (repeatable)")
    a = ap.parse_args()
    if a.action == "push":
        job = {k: v for k, v in (("steps", a.steps), ("envs", a.envs), ("evals", a.evals)) if v is not None}
        def value(v: str):
            try:
                return float(v)
            except ValueError:
                return v
        settings = {k: value(v) for k, v in (kv.split("=", 1) for kv in a.set)}
        push(a.name, {**job, **{f: True for f in a.flag}, **settings}, a.embed, a.dataset, a.machine, a.job)
    elif a.action == "status":
        print(status(a.job or a.name))
    elif a.action == "wait":
        sys.exit(0 if wait(a.job or a.name) == "COMPLETE" else 1)
    elif a.action == "upload":
        upload(a.name)
    else:
        fetch(a.job or a.name)


if __name__ == "__main__":
    main()
