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


def ensure_kaggle_auth() -> None:
    """Work around kagglesdk's bug where access_token_has_expired uses minus instead of plus 30 min."""
    try:
        from datetime import datetime, timezone, timedelta
        from kagglesdk import KaggleClient, KaggleCredentials
        client = KaggleClient()
        creds = KaggleCredentials.load(client)
        if creds and creds._refresh_token:
            if not creds._access_token_expiration or creds._access_token_expiration < datetime.now(timezone.utc) + timedelta(minutes=10):
                creds.refresh_access_token()
    except Exception:
        pass


def kaggle(*args: str) -> str:
    ensure_kaggle_auth()
    out = subprocess.run([KAGGLE, *args], capture_output=True, text=True)
    text = out.stdout + out.stderr
    # the CLI can print an HTTP error and still exit 0 (seen: 400 on an oversized kernel push)
    if out.returncode != 0 or "Client Error" in text or "Server Error" in text or "Kernel push error:" in text:
        sys.exit(f"kaggle {' '.join(args)} failed:\n{text}")
    return out.stdout


def username() -> str:
    m = re.search(r"username:\s*(\S+)", kaggle("config", "view"))
    if not m or m.group(1) == "None":
        sys.exit("Not signed in to Kaggle: run  .venv/bin/kaggle auth login")
    return m.group(1)


def slug(name: str) -> str:
    return f"bio-bot-{name.replace('_', '-')}"


def parse_value(value: str):
    # Integer options such as seed and unroll must stay integers when JOB_ARGS bypasses argparse.
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def push(name: str, job_args: dict, embed: list[str], datasets: list[str], machine: str = "", job: str = "",
         kernel_sources: list[str] | None = None, timeout: int | None = None) -> None:
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
        "enable_gpu": "false" if machine == "CPU" or machine.startswith("Tpu") else "true", "enable_tpu": "true" if machine.startswith("Tpu") else "false",
        "enable_internet": "true", "machine_shape": "" if machine == "CPU" else machine, "dataset_sources": [f"{username()}/{d}" for d in datasets], "competition_sources": [],
        "kernel_sources": kernel_sources or [], "model_sources": [],
    }
    (job_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    command = ["kernels", "push", "-p", str(job_dir)]
    if timeout is not None:
        command += ["--timeout", str(timeout)]
    print(kaggle(*command).strip())


def parse_status(output: str) -> str:
    # A failed job prints its failure message *after* its status. Reading the last line would hide ERROR
    # and leave the waiter running forever. Parse the actual status line wherever the CLI places it.
    m = re.search(r'status "(?:KernelWorkerStatus\.)?(\w+)"', output)
    if not m:
        raise SystemExit(f"Unrecognised Kaggle status response:\n{output}")
    return m.group(1).upper()


def status(name: str) -> str:
    output = kaggle("kernels", "status", f"{username()}/{slug(name)}").strip()
    state = parse_status(output)
    if state == "ERROR" and "Failure message:" in output:
        print(output.split("Failure message:", 1)[1].strip(), file=sys.stderr, flush=True)
    return state


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
                hint = (" The Kaggle login has probably expired (it lasts ~3 h): run `.venv/bin/kaggle auth login --force`."
                        if "denied" in str(e) or "401" in str(e) or "nauthenticated" in str(e) else "")
                sys.exit(f"{time.strftime('%H:%M:%S')}  GAVE UP: status checks failed 10 times in a row.{hint}")
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
    ap.add_argument("--timeout", type=int, help="maximum remote runtime in seconds; useful for bounded free-compute trials")
    ap.add_argument("--embed", action="append", default=[], help="repo file to ship inside the job (repeatable)")
    ap.add_argument("--flag", action="append", default=[], help="boolean option to switch on in the job (repeatable)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="option for the job, e.g. --set bias0=0.3 or --set readout=line (repeatable)")
    ap.add_argument("--machine", default="", choices=["", "CPU", "NvidiaTeslaT4", "NvidiaTeslaP100", "Tpu1VmV38"],
                    help="Kaggle machine (default: Kaggle's GPU default, 2x T4); CPU uses no GPU quota")
    ap.add_argument("--dataset", action="append", default=[],
                    help="one of your Kaggle datasets to attach; it appears under /kaggle/input/<name>/ (repeatable)")
    ap.add_argument("--kernel-source", action="append", default=[],
                    help="existing Kaggle job outputs to attach directly, e.g. username/kernel-slug (repeatable)")
    a = ap.parse_args()
    if a.action == "push":
        job = {k: v for k, v in (("steps", a.steps), ("envs", a.envs), ("evals", a.evals)) if v is not None}
        settings = {k: parse_value(v) for k, v in (kv.split("=", 1) for kv in a.set)}
        push(a.name, {**job, **{f: True for f in a.flag}, **settings}, a.embed, a.dataset, a.machine, a.job, a.kernel_source, a.timeout)
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
