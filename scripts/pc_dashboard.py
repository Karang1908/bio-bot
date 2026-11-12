#!/usr/bin/env python3
"""Live monitor for the bio-bot GPU PC (WSL2 on Windows) and its training runs.

Run on the GPU PC (standard library only, read-only):
    python3 scripts/pc_dashboard.py            # binds to this machine's Tailscale IP, port 8765
Open from any device on the tailnet:  http://<tailscale-ip>:8765

It reads GPU/CPU/memory/disk counters, process lists, Windows host stats (through WSL interop) and the run logs
under ~/runs/<run>/run.log. It never starts, stops or changes anything.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

RUNS = Path.home() / "runs"
NVSMI = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
TASKLIST = "/mnt/c/Windows/System32/tasklist.exe"
SEC_PER_UPDATE = 7.9          # measured 12 Nov (LOGBOOK §6.12); a run's own measured rate replaces it when available
HISTORY = 300                 # samples kept for the sparklines (one every 2 s = 10 min)


def sh(cmd: list[str], timeout: float = 10.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def num(s: str):
    s = s.strip()
    return float(s) if re.fullmatch(r"[+-]?\d+(\.\d+)?", s) else None


# ------------------------------------------------------------------ system sampling (background threads)
STATE = {"gpu": None, "cpu": None, "mem": None, "windows": None, "windows_time": 0.0, "sample_time": 0.0}
HIST: collections.deque = collections.deque(maxlen=HISTORY)
LOCK = threading.Lock()
GPU_FIELDS = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,clocks.sm,pstate"


def read_gpu():
    out = sh([NVSMI, f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits"]).strip()
    if not out:
        return None
    f = [x.strip() for x in out.splitlines()[0].split(",")]
    return {"name": f[0], "util": num(f[1]), "mem_used": num(f[2]), "mem_total": num(f[3]), "temp": num(f[4]),
            "power": num(f[5]), "power_limit": num(f[6]), "clock": num(f[7]), "pstate": f[8]}


def cpu_times():
    v = list(map(int, open("/proc/stat").readline().split()[1:]))
    return sum(v), v[3] + v[4]


def read_mem():
    m = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        m[k] = int(v.split()[0]) * 1024
    return {"total": m["MemTotal"], "used": m["MemTotal"] - m["MemAvailable"],
            "swap_total": m["SwapTotal"], "swap_used": m["SwapTotal"] - m["SwapFree"]}


def read_windows():
    ps = ("$o=Get-CimInstance Win32_OperatingSystem;"
          "$c=(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average;"
          "Write-Output ('{0} {1} {2} {3}' -f $c,$o.TotalVisibleMemorySize,$o.FreePhysicalMemory,"
          "$o.LastBootUpTime.ToString('o'))")
    out = sh([POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", ps], timeout=20).replace("\r", "").strip()
    w = {"ok": False}
    parts = out.split()
    if len(parts) >= 4 and num(parts[1]) is not None:
        total, free = float(parts[1]) * 1024, float(parts[2]) * 1024
        boot = parts[3]
        try:
            up = time.time() - datetime.fromisoformat(re.sub(r"(\.\d{6})\d*", r"\1", boot)).timestamp()
        except ValueError:
            up = None
        w = {"ok": True, "cpu": num(parts[0]), "mem_total": total, "mem_used": total - free, "uptime": up}
    w["top"] = windows_processes(8)
    return w


def windows_processes(n: int):
    rows = []
    for line in sh([TASKLIST, "/FO", "CSV", "/NH"], timeout=20).replace("\r", "").splitlines():
        f = [x.strip('"') for x in re.findall(r'"[^"]*"', line)]
        if len(f) >= 5:
            rows.append({"name": f[0], "pid": f[1], "mem": int(re.sub(r"\D", "", f[4]) or 0) * 1024})
    agg: dict[str, dict] = {}
    for r in rows:                                          # group by image name (e.g. many chrome.exe)
        a = agg.setdefault(r["name"], {"name": r["name"], "count": 0, "mem": 0})
        a["count"] += 1
        a["mem"] += r["mem"]
    return sorted(agg.values(), key=lambda a: -a["mem"])[:n]


def sampler():
    prev = cpu_times()
    while True:
        time.sleep(2)
        cur = cpu_times()
        dt, didle = cur[0] - prev[0], cur[1] - prev[1]
        prev = cur
        cpu = 100.0 * (1 - didle / dt) if dt > 0 else None
        g, m = read_gpu(), read_mem()
        with LOCK:
            STATE.update(gpu=g, mem=m, sample_time=time.time(),
                         cpu={"util": cpu, "cores": os.cpu_count(), "load": os.getloadavg()})
            HIST.append({"t": time.time(), "gpu": (g or {}).get("util"), "cpu": cpu,
                         "gmem": (g or {}).get("mem_used"), "power": (g or {}).get("power")})


def windows_sampler():
    while True:
        w = read_windows()
        with LOCK:
            STATE.update(windows=w, windows_time=time.time())
        time.sleep(30)


# ------------------------------------------------------------------ run logs
LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d|\d\d:\d\d:\d\d) (.*)$")
TEST = re.compile(r"(REAL TEST \(no harness\)|PARK TEST|HELD-OUT VOICE) (before practice|after [\d.]+M steps)"
                  r"(?:, cohort (\d+)/(\d+))?[,:]? ?(.*)$")
_cache: dict[str, tuple] = {}


def parse_ts(stamp: str, day: str) -> float:
    if len(stamp) == 8:
        stamp = f"{day} {stamp}"
    return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").timestamp()


def python_procs():
    out = []
    for line in sh(["ps", "-eo", "pid,etimes,pcpu,args", "--no-headers"]).splitlines():
        p = line.split(None, 3)
        if len(p) == 4 and "python" in p[3]:
            out.append({"pid": int(p[0]), "etime": int(p[1]), "cpu": float(p[2]), "args": p[3]})
    return out


def parse_log(d: Path):
    log = d / "run.log"
    st = log.stat()
    key = (st.st_mtime, st.st_size)
    if _cache.get(str(d), (None,))[0] == key:
        return _cache[str(d)][1]
    day = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d")
    lines, practising, groups = [], [], []
    info = {"iters": None, "steps_per_update": None, "warm": None, "practice_start": None, "exit": None,
            "done": False, "traceback": False, "nan": False, "stop": None, "first": None}
    prev_kind = None
    for raw in log.read_text(errors="replace").splitlines():
        m = LINE.match(raw)
        if not m:
            continue
        ts, text = parse_ts(m.group(1), day), m.group(2)
        lines.append((ts, text))
        info["first"] = info["first"] or ts
        kind = "other"
        if (p := re.search(r"practice: (\d+) updates x ([\d,]+) steps; the critic learns alone for the first (\d+)", text)):
            info.update(iters=int(p.group(1)), steps_per_update=int(p.group(2).replace(",", "")),
                        warm=int(p.group(3)), practice_start=ts)
        elif (p := re.search(r"practising: reward ([+-]?[\d.na]+)/s\s+falls ([\d.na]+)/min.*?value error ([\d.na]+)", text)):
            practising.append({"t": ts, "reward": num(p.group(1)), "falls": num(p.group(2)), "value_error": num(p.group(3))})
            kind = "practising"
        elif (t := TEST.search(text)):
            kind = "test"
            if prev_kind != "test":
                groups.append({"label": t.group(2), "t0": ts, "t1": ts, "after_update": len(practising), "m": {}})
            g = groups[-1]
            g["t1"] = ts
            add_metrics(g["m"], t.group(1), t.group(5))
        if text.startswith("EXIT "):
            info["exit"] = int(text.split()[1]) if text.split()[1].lstrip("-").isdigit() else None
        if text.strip() == "DONE":
            info["done"] = True
        if "Traceback" in text:
            info["traceback"] = True
        if re.search(r"(reward|value error|guide) nan", text):
            info["nan"] = True
        if text.startswith("STOP"):
            info["stop"] = text
        prev_kind = kind if kind != "other" else prev_kind if text.startswith(" ") else kind
    for g in groups:                                        # mean over cohorts
        g["m"] = {k: sum(v) / len(v) for k, v in g["m"].items() if v}
    result = {"lines": lines, "practising": practising, "groups": groups, "info": info}
    _cache[str(d)] = (key, result)
    return result


def add_metrics(m: dict, kind: str, rest: str):
    def put(k, v):
        m.setdefault(k, []).append(float(v))
    w = re.match(r'(?:told )?"([\w-]+)"(?: from rest)?: (.*)', rest)
    if kind == "HELD-OUT VOICE" or not w:
        if (g := re.search(r'(\d+)% on its feet after 8s', rest)):
            put("getup8", g.group(1))
        return
    word, body = w.group(1), w.group(2)
    if (a := re.search(r"(\d+)% got there.*?distance ([\d.]+) -> ([\d.]+) m", body)) and word in ("ball", "come"):
        put(f"{word}_arrive", a.group(1)); put(f"{word}_d0", a.group(2)); put(f"{word}_d1", a.group(3))
    elif (h := re.match(r"hit (\d+)%", body)):
        put("hit", h.group(1))
    elif (v := re.match(r"([+-][\d.]+) (m/s|rad/s)", body)) and word in ("walk", "run", "turn-left", "back"):
        put(word, v.group(1))
        if word == "walk" and (f := re.search(r"falls ([\d.]+) per 13 s", body)):
            put("walk_falls", f.group(1))


def run_summary(d: Path, procs):
    p = parse_log(d)
    info, practising, groups, lines = p["info"], p["practising"], p["groups"], p["lines"]
    now = time.time()
    proc = next((x for x in procs if re.search(re.escape(str(d)) + r"(\s|$)", x["args"])), None)
    if proc:
        status = "running"
    elif info["traceback"] or (info["exit"] not in (None, 0)):
        status = "crashed"
    elif info["done"] or info["exit"] == 0:
        status = "finished"
    else:
        status = "stopped"
    iters, warm = info["iters"], info["warm"]
    evals = None
    if proc and (e := re.search(r"--evals (\d+)", proc["args"])):
        evals = int(e.group(1))
    snaps = sorted(int(re.search(r"(\d+)", f.name).group(1)) for f in d.glob("baby_dog_step_*.npz"))
    every = None
    if iters and evals:
        every = max(1, iters // evals)
    elif iters and snaps and info["steps_per_update"]:
        every = snaps[0] // info["steps_per_update"]
    marks = sorted({warm} | {every * j for j in range(1, iters // every + 1)} | {iters}) if iters and every and warm else []
    k_last = marks[len(practising) - 1] if practising and len(practising) <= len(marks) else (iters if practising and marks else 0)
    spu = None
    for a, b in zip(range(len(practising) - 1), range(1, len(practising))):
        if b < len(marks):
            ta = max([t for t, _ in lines if practising[a]["t"] <= t < practising[b]["t"]], default=practising[a]["t"])
            if marks[b] > marks[a]:
                spu = (practising[b]["t"] - ta) / (marks[b] - marks[a])
    spu = spu or SEC_PER_UPDATE
    last_t = lines[-1][0] if lines else None
    est = None
    if iters and status == "running" and info["practice_start"]:
        nxt = next((x for x in marks if x > k_last), iters)
        est = min(nxt - 1, k_last + max(0.0, now - last_t) / spu) if marks else min(iters, (now - info["practice_start"]) / spu)
    elif iters and status == "finished":
        est = iters
    done_groups = [g for g in groups if g["label"] != "before practice"]
    durs = [g["t1"] - practising[g["after_update"] - 1]["t"] for g in done_groups if g["after_update"] >= 1]
    base = next((g for g in groups if g["label"] == "before practice"), None)
    eval_dur = (sum(durs) / len(durs)) if durs else ((base["t1"] - base["t0"] + 60) if base else 15 * 60)
    remaining = None
    if est is not None and iters and status == "running":
        evals_left = len([x for x in marks if x > k_last and (x % every == 0 or x == iters)])
        remaining = (iters - est) * spu + evals_left * eval_dur
    last = practising[-1] if practising else None
    return {"name": d.name, "status": status, "pid": proc["pid"] if proc else None, "nan": info["nan"],
            "stop": info["stop"], "iters": iters, "est_update": est, "sec_per_update": spu,
            "steps_per_update": info["steps_per_update"], "evals": evals, "every": every, "snapshots": snaps,
            "started": info["first"], "practice_start": info["practice_start"], "last_line_time": last_t,
            "last_line": lines[-1][1][:220] if lines else "", "last_practising": last, "remaining_s": remaining,
            "eta": now + remaining if remaining is not None else None, "exit": info["exit"],
            "groups": [{"label": g["label"], "t": g["t0"], "after_update": marks[g["after_update"] - 1]
                        if 0 < g["after_update"] <= len(marks) else 0, "m": g["m"]} for g in groups],
            "mtime": (d / "run.log").stat().st_mtime}


def all_runs():
    procs = python_procs()
    dirs = [d for d in RUNS.iterdir() if d.is_dir() and (d / "run.log").exists()] if RUNS.exists() else []
    out = []
    for d in sorted(dirs, key=lambda x: (x / "run.log").stat().st_mtime, reverse=True):
        try:
            out.append(run_summary(d, procs))
        except Exception as e:               # a half-written or odd log must not break the dashboard
            out.append({"name": d.name, "status": "unreadable", "error": str(e)[:200],
                        "mtime": (d / "run.log").stat().st_mtime})
    return out


def top_processes(n=12):
    rows = []
    for line in sh(["ps", "-eo", "pid,pcpu,pmem,etimes,args", "--sort=-pcpu", "--no-headers"]).splitlines()[:n]:
        p = line.split(None, 4)
        if len(p) == 5:
            rows.append({"pid": p[0], "cpu": num(p[1]), "mem": num(p[2]), "etime": int(p[3]), "cmd": p[4][:160]})
    return rows


def gpu_apps():
    rows = []
    for line in sh([NVSMI, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]).splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) == 3:
            rows.append({"pid": f[0], "name": f[1], "mem": f[2]})
    return rows


def disks():
    out = []
    for label, path in (("Windows D: (holds the WSL disk)", "/mnt/d"), ("Windows C:", "/mnt/c"),
                        ("WSL virtual disk (its real limit is D:)", "/")):
        try:
            u = shutil.disk_usage(path)
            out.append({"label": label, "path": path, "total": u.total, "used": u.used, "free": u.free})
        except OSError:
            pass
    return out


def health(runs, gpu):
    active = next((r for r in runs if r.get("status") == "running"), None)
    util = (gpu or {}).get("util")
    if active:
        if active.get("nan"):
            return {"level": "bad", "text": f"NaN in {active['name']}: check it"}
        if util is not None and util >= 50:
            return {"level": "ok", "text": f"Training running · GPU {util:.0f}%"}
        return {"level": "info", "text": f"Run alive (testing or compiling) · GPU {util if util is not None else '?'}%"}
    latest = runs[0] if runs else None
    if latest and latest.get("status") == "crashed":
        return {"level": "bad", "text": f"Latest run {latest['name']} crashed"}
    if latest and latest.get("status") == "finished":
        return {"level": "info", "text": f"No run active · {latest['name']} finished"}
    return {"level": "idle", "text": "No training running"}


def status_payload():
    with LOCK:
        s = {k: STATE[k] for k in STATE}
        hist = list(HIST)
    runs = all_runs()
    active = next((r for r in runs if r.get("status") == "running"), runs[0] if runs else None)
    up = float(open("/proc/uptime").read().split()[0])
    return {"time": time.time(), "host": {"hostname": socket.gethostname(), "uptime": up},
            "health": health(runs, s["gpu"]), "gpu": s["gpu"], "cpu": s["cpu"], "mem": s["mem"],
            "sample_age": time.time() - s["sample_time"] if s["sample_time"] else None,
            "windows": s["windows"], "windows_age": time.time() - s["windows_time"] if s["windows_time"] else None,
            "disks": disks(), "history": hist[-150:], "active": active,
            "runs": [{k: r.get(k) for k in ("name", "status", "mtime", "iters", "est_update", "nan", "pid", "error")}
                     for r in runs],
            "procs": top_processes(), "gpu_apps": gpu_apps()}


def details(q):
    kind = q.get("kind", ["run"])[0]
    if kind == "gpu":
        return {"title": "GPU · nvidia-smi", "text": sh([NVSMI]) + "\n" + sh([NVSMI, "-q", "-d", "PERFORMANCE,CLOCK,POWER,TEMPERATURE,MEMORY"])}
    if kind == "windows":
        w = read_windows()
        rows = windows_processes(25)
        text = "\n".join(f"{r['name']:<32} x{r['count']:<3} {r['mem'] / 2**20:9.0f} MB" for r in rows)
        return {"title": "Windows host", "windows": w, "text": "Top Windows processes by memory:\n" + text}
    name = q.get("run", [""])[0]
    names = {d.name: d for d in RUNS.iterdir() if d.is_dir()} if RUNS.exists() else {}
    if name not in names:
        return {"error": "unknown run"}
    d = names[name]
    r = run_summary(d, python_procs())
    p = parse_log(d)
    proc = next((x for x in python_procs() if x["pid"] == r.get("pid")), None)
    files = sorted(({"name": f.name, "size": f.stat().st_size, "mtime": f.stat().st_mtime}
                    for f in d.iterdir() if f.is_file() and not f.name.startswith(".")), key=lambda f: f["name"])
    extra = {}
    for fn in ("resume.json", "commit.txt"):
        if (d / fn).exists():
            extra[fn] = (d / fn).read_text(errors="replace")[:4000]
    tail = [datetime.fromtimestamp(t).strftime("%H:%M:%S") + " " + x for t, x in p["lines"]
            if not x.startswith("teacher tries")][-120:]
    return {"title": f"Run {name}", "summary": r, "command": proc["args"] if proc else None, "files": files,
            "extra": extra, "log_tail": "\n".join(tail)}


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path == "/":
                self.send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif u.path == "/api/status":
                self.send(200, json.dumps(status_payload()).encode(), "application/json")
            elif u.path == "/api/details":
                self.send(200, json.dumps(details(parse_qs(u.query))).encode(), "application/json")
            else:
                self.send(404, b"not found", "text/plain")
        except Exception as e:
            self.send(500, json.dumps({"error": str(e)[:300]}).encode(), "application/json")


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>bio-bot monitor</title>
<style>
:root{--bg:#f5f6f8;--card:#fff;--ink:#15181d;--muted:#5d6572;--line:#e2e5ea;--ok:#17794a;--okbg:#e3f4ea;--warn:#8a5d00;
--warnbg:#fff3d1;--bad:#b42318;--badbg:#fde9e7;--info:#1d4ed8;--infobg:#e7eeff;--bar:#2f6fed;--track:#e8ecf2}
@media (prefers-color-scheme: dark){:root{--bg:#0e1014;--card:#161920;--ink:#e7e9ee;--muted:#98a1b0;--line:#252a34;
--ok:#4ade80;--okbg:#0f2a1a;--warn:#facc15;--warnbg:#2a2408;--bad:#f87171;--badbg:#2d1313;--info:#86a8ff;--infobg:#141f38;
--bar:#6c9bff;--track:#232835}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{display:flex;flex-wrap:wrap;gap:10px;align-items:center;justify-content:space-between;padding:14px 16px;
border-bottom:1px solid var(--line);background:var(--card);position:sticky;top:0;z-index:2}
h1{font-size:17px;margin:0}h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px;font-weight:600}
main{padding:16px;display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));max-width:1400px;margin:0 auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;min-width:0}.wide{grid-column:1/-1}
.pill{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;font-weight:600;font-size:13px}
.ok{color:var(--ok);background:var(--okbg)}.warn{color:var(--warn);background:var(--warnbg)}.bad{color:var(--bad);background:var(--badbg)}
.info{color:var(--info);background:var(--infobg)}.idle{color:var(--muted);background:var(--track)}
.dot{width:8px;height:8px;border-radius:50%;background:currentColor;flex:none}.live .dot{animation:pulse 1.6s infinite}
@keyframes pulse{50%{opacity:.25}}@media (prefers-reduced-motion:reduce){.live .dot{animation:none}}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 12px}.kv>span:nth-child(odd){color:var(--muted)}
.meter{height:8px;background:var(--track);border-radius:4px;overflow:hidden;margin:3px 0 9px}.meter>i{display:block;height:100%;background:var(--bar)}
.big{font-size:28px;font-weight:650;font-variant-numeric:tabular-nums;line-height:1.1}
.row{display:flex;justify-content:space-between;gap:8px;align-items:center;flex-wrap:wrap}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:5px 7px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--muted);font-weight:500;font-size:12px}.scroll{overflow-x:auto}.mono,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
td.cmd{white-space:normal;word-break:break-all;min-width:240px}
pre{white-space:pre-wrap;word-break:break-word;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px;max-height:46vh;overflow:auto;margin:0}
button{font:inherit;font-size:13px;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:5px 10px;cursor:pointer}
button:hover{border-color:var(--bar)}button:focus-visible{outline:2px solid var(--bar);outline-offset:2px}
.good{color:var(--ok)}.badt{color:var(--bad)}.muted{color:var(--muted)}.sub{font-size:12px;color:var(--muted)}
dialog{width:min(1000px,94vw);border:1px solid var(--line);border-radius:12px;background:var(--card);color:var(--ink);padding:0}
dialog::backdrop{background:#0009}.dhead{display:flex;justify-content:space-between;align-items:center;padding:12px 14px;border-bottom:1px solid var(--line)}
.dbody{padding:14px;display:grid;gap:12px;max-height:80vh;overflow:auto}svg.spark{width:100%;height:44px;display:block;margin:4px 0 8px}
.prog{height:12px;background:var(--track);border-radius:6px;overflow:hidden;margin:8px 0}.prog>i{display:block;height:100%;background:var(--bar)}
@media (max-width:520px){main{padding:10px}header{padding:12px}}
</style></head><body>
<header><div><h1>bio-bot · GPU PC monitor</h1><div class="sub" id="host">connecting…</div></div>
<div class="row"><span id="health" class="pill idle"><span class="dot"></span><span>…</span></span>
<span id="conn" class="pill idle"><span class="dot"></span><span>connecting</span></span></div></header>
<main>
<section class="card wide"><div class="row"><h2>Training run</h2><button id="runBtn" hidden>Fetch details</button></div><div id="run">…</div></section>
<section class="card"><div class="row"><h2>GPU</h2><button data-kind="gpu">Fetch details</button></div><div id="gpu">…</div></section>
<section class="card"><h2>WSL (Linux) and disks</h2><div id="wsl">…</div></section>
<section class="card"><div class="row"><h2>Windows host</h2><button data-kind="windows">Fetch details</button></div><div id="win">…</div></section>
<section class="card wide"><h2>Evaluations of this run (mean over cohorts)</h2><div class="scroll" id="evals">…</div></section>
<section class="card wide"><h2>Processes</h2><div class="scroll" id="procs">…</div></section>
<section class="card wide"><h2>All runs on the PC</h2><div class="scroll" id="runs">…</div></section>
</main>
<dialog id="dlg"><div class="dhead"><strong id="dtitle">Details</strong><button id="dclose">Close</button></div><div class="dbody" id="dbody"></div></dialog>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const f=(v,d=0,u='')=>v==null||isNaN(v)?'—':Number(v).toFixed(d)+u;
const gb=b=>b==null?'—':(b/2**30).toFixed(b>=100*2**30?0:1)+' GB';
const meter=p=>`<div class="meter"><i style="width:${Math.max(0,Math.min(100,p||0)).toFixed(1)}%"></i></div>`;
const dur=s=>{if(s==null||isNaN(s))return'—';s=Math.max(0,Math.round(s));const d=Math.floor(s/86400),h=Math.floor(s%86400/3600),m=Math.floor(s%3600/60);return d?`${d} d ${h} h`:h?`${h} h ${m} min`:`${m} min`;};
const clock=t=>t?new Date(t*1000).toLocaleString([], {weekday:'short',hour:'2-digit',minute:'2-digit'}):'—';
const ago=t=>t?dur(Date.now()/1000-t)+' ago':'—';
let last=0, cur=null;
function pill(el,cls,text,live){el.className='pill '+cls+(live?' live':'');el.lastElementChild.textContent=text;}
function spark(hist,key,max){const p=hist.filter(h=>h[key]!=null);if(p.length<2)return'';const w=300,h=44;
 const pts=p.map((x,i)=>`${(i/(p.length-1)*w).toFixed(1)},${(h-2-Math.min(max,x[key])/max*(h-4)).toFixed(1)}`).join(' ');
 return `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-label="${key} history, last ${dur(p.length*2)}"><polyline fill="none" stroke="var(--bar)" stroke-width="1.6" vector-effect="non-scaling-stroke" points="${pts}"/></svg>`;}
const statusCls={running:'ok',finished:'info',crashed:'bad',stopped:'idle',unreadable:'warn'};
function evalTable(groups){
 if(!groups||!groups.length)return '<p class="muted">No evaluation yet.</p>';
 const dist=(m,w)=>{const a=m[w+'_d0'],b=m[w+'_d1'];if(a==null)return'—';const c=b<a?'good':'badt';return `${f(a,2)} → <span class="${c}">${f(b,2)} m</span>`;};
 let h='<table><thead><tr><th>evaluation</th><th>at update</th><th>walk m/s</th><th>walk falls/13 s</th><th>turn-left rad/s</th><th>ball arrive</th><th>ball distance</th><th>come arrive</th><th>come distance</th><th>get up 8 s</th><th>hit by throw</th></tr></thead><tbody>';
 for(const g of groups){const m=g.m||{};h+=`<tr><td>${esc(g.label)}</td><td>${g.after_update||0}</td><td>${f(m.walk,3)}</td><td>${f(m.walk_falls,2)}</td><td>${f(m['turn-left'],2)}</td><td>${f(m.ball_arrive,1,'%')}</td><td>${dist(m,'ball')}</td><td>${f(m.come_arrive,1,'%')}</td><td>${dist(m,'come')}</td><td>${f(m.getup8,0,'%')}</td><td>${f(m.hit,0,'%')}</td></tr>`;}
 return h+'</tbody></table><p class="sub">Green end distance = ended closer than it started. Arrival rates of one 64-life cohort carry about ±4 points of noise.</p>';}
function renderRun(r){
 const b=$('runBtn');
 if(!r){$('run').innerHTML='<p class="muted">No runs found in ~/runs.</p>';b.hidden=true;return;}
 b.hidden=false;b.onclick=()=>openDetails({run:r.name});
 const pct=r.iters&&r.est_update!=null?100*r.est_update/r.iters:null;
 const lp=r.last_practising||{};
 $('run').innerHTML=`<div class="row"><div><div class="big">${esc(r.name)}</div><div class="sub">pid ${r.pid??'—'} · started ${clock(r.started)}</div></div>
 <span class="pill ${statusCls[r.status]||'idle'} ${r.status==='running'?'live':''}"><span class="dot"></span><span>${esc(r.status)}${r.nan?' · NaN!':''}</span></span></div>
 ${r.iters?`<div class="prog"><i style="width:${f(pct,1)}%"></i></div><div class="row"><span><b>${r.est_update!=null?Math.floor(r.est_update):'—'}</b> / ${r.iters} updates ${r.status==='running'?'(estimated)':''} · ${f(pct,1,'%')}</span><span class="muted">${r.status==='running'?`about ${dur(r.remaining_s)} left · ETA ${clock(r.eta)}`:''}</span></div>`:''}
 <div class="kv" style="margin-top:10px"><span>speed</span><span>${f(r.sec_per_update,2)} s per update · ${r.steps_per_update?(r.steps_per_update/r.sec_per_update).toFixed(0)+' steps/s':''}</span>
 <span>evaluations</span><span>${r.evals??'—'} planned (every ${r.every??'—'} updates) · snapshots saved: ${(r.snapshots||[]).length}</span>
 <span>last training stats</span><span>${lp.reward!=null?`reward ${f(lp.reward,2)}/s · falls ${f(lp.falls,2)}/min · value error ${f(lp.value_error,0)} (${ago(lp.t)})`:'—'}</span>
 <span>last log line</span><span class="mono">${esc(r.last_line)} <span class="muted">(${ago(r.last_line_time)})</span></span>
 ${r.stop?`<span>stopped</span><span class="badt">${esc(r.stop)}</span>`:''}</div>`;
 $('evals').innerHTML=evalTable(r.groups);}
function render(d){
 cur=d;
 $('host').textContent=`${d.host.hostname} · WSL up ${dur(d.host.uptime)} · PC time ${new Date(d.time*1000).toLocaleTimeString()}`;
 pill($('health'),d.health.level,d.health.text,d.health.level==='ok');
 const g=d.gpu;
 $('gpu').innerHTML=g?`<div class="row"><div class="big">${f(g.util,0,'%')}</div><span class="sub">${esc(g.name)}</span></div>${spark(d.history,'gpu',100)}
 <div class="kv"><span>memory</span><span>${f(g.mem_used/1024,1)} / ${f(g.mem_total/1024,1)} GB</span></div>${meter(100*g.mem_used/g.mem_total)}
 <div class="kv"><span>power</span><span>${f(g.power,0)} / ${f(g.power_limit,0)} W</span><span>temperature</span><span>${f(g.temp,0,' °C')}</span><span>clock</span><span>${f(g.clock,0,' MHz')} · ${esc(g.pstate)}</span></div>`:'<p class="badt">nvidia-smi not responding</p>';
 const c=d.cpu||{},m=d.mem||{};
 $('wsl').innerHTML=`<div class="kv"><span>CPU</span><span>${f(c.util,0,'%')} of ${c.cores??'—'} threads · load ${c.load?c.load.map(x=>x.toFixed(1)).join(' / '):'—'}</span></div>${meter(c.util)}
 <div class="kv"><span>memory</span><span>${gb(m.used)} / ${gb(m.total)} · swap ${gb(m.swap_used)}</span></div>${meter(100*m.used/m.total)}
 ${(d.disks||[]).map(k=>`<div class="kv"><span>${esc(k.label)}</span><span>${gb(k.free)} free of ${gb(k.total)}</span></div>${meter(100*k.used/k.total)}`).join('')}`;
 const w=d.windows;
 $('win').innerHTML=!w?'<p class="muted">sampling…</p>':!w.ok?'<p class="warn">Windows stats unavailable (WSL interop)</p>':
 `<div class="kv"><span>CPU</span><span>${f(w.cpu,0,'%')}</span></div>${meter(w.cpu)}<div class="kv"><span>memory</span><span>${gb(w.mem_used)} / ${gb(w.mem_total)}</span></div>${meter(100*w.mem_used/w.mem_total)}
 <div class="kv"><span>Windows up</span><span>${dur(w.uptime)}</span><span>sampled</span><span>${f(d.windows_age,0)} s ago (every 30 s)</span></div>
 <table style="margin-top:8px"><thead><tr><th>top by memory</th><th>count</th><th>memory</th></tr></thead><tbody>${(w.top||[]).slice(0,6).map(p=>`<tr><td>${esc(p.name)}</td><td>${p.count}</td><td>${gb(p.mem)}</td></tr>`).join('')}</tbody></table>`;
 renderRun(d.active);
 $('procs').innerHTML=`<table><thead><tr><th>pid</th><th>CPU %</th><th>mem %</th><th>running for</th><th>command (WSL, by CPU)</th></tr></thead><tbody>${d.procs.map(p=>`<tr><td>${esc(p.pid)}</td><td>${f(p.cpu,0)}</td><td>${f(p.mem,1)}</td><td>${dur(p.etime)}</td><td class="cmd mono">${esc(p.cmd)}</td></tr>`).join('')}</tbody></table>
 <p class="sub" style="margin-top:8px">On the GPU: ${d.gpu_apps.length?d.gpu_apps.map(a=>`pid ${esc(a.pid)} ${esc(a.name)}`).join(', '):'no compute process'}</p>`;
 $('runs').innerHTML=`<table><thead><tr><th>run</th><th>status</th><th>progress</th><th>last activity</th><th></th></tr></thead><tbody>${d.runs.map(r=>`<tr><td class="mono">${esc(r.name)}</td><td><span class="pill ${statusCls[r.status]||'idle'}"><span class="dot"></span><span>${esc(r.status)}${r.nan?' · NaN':''}</span></span></td><td>${r.iters?`${r.est_update!=null?Math.floor(r.est_update):'?'} / ${r.iters}`:'—'}</td><td>${ago(r.mtime)}</td><td><button data-run="${esc(r.name)}">Fetch details</button></td></tr>`).join('')}</tbody></table>`;}
async function poll(){try{const r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw new Error(r.status);render(await r.json());last=Date.now();}
 catch(e){pill($('conn'),'bad','PC unreachable · retrying',false);}}
setInterval(()=>{if(!last)return;const a=(Date.now()-last)/1000;if(a<12)pill($('conn'),'ok',`live · updated ${Math.round(a)} s ago`,true);else pill($('conn'),'bad',`no update for ${Math.round(a)} s`,false);},1000);
async function openDetails(q){const dlg=$('dlg');$('dtitle').textContent='Fetching…';$('dbody').innerHTML='<p class="muted">Fetching details from the PC…</p>';if(!dlg.open)dlg.showModal();
 try{const r=await fetch('/api/details?'+new URLSearchParams(q),{cache:'no-store'});const d=await r.json();if(d.error)throw new Error(d.error);$('dtitle').textContent=d.title;
 let h='';
 if(d.summary){const s=d.summary;h+=`<div class="kv"><span>status</span><span>${esc(s.status)}${s.exit!=null?' · exit '+s.exit:''}</span><span>updates</span><span>${s.est_update!=null?Math.floor(s.est_update):'—'} / ${s.iters??'—'}</span><span>snapshots</span><span>${(s.snapshots||[]).join(', ')||'none'}</span><span>fetched</span><span>${new Date().toLocaleTimeString()}</span></div>`;
  h+=`<div class="scroll">${evalTable(s.groups)}</div>`;}
 if(d.command)h+=`<div><h2>command</h2><pre>${esc(d.command)}</pre></div>`;
 for(const [k,v] of Object.entries(d.extra||{}))h+=`<div><h2>${esc(k)}</h2><pre>${esc(v)}</pre></div>`;
 if(d.files)h+=`<div><h2>files</h2><table><tbody>${d.files.map(x=>`<tr><td class="mono">${esc(x.name)}</td><td>${(x.size/2**20).toFixed(2)} MB</td><td>${clock(x.mtime)}</td></tr>`).join('')}</tbody></table></div>`;
 if(d.log_tail)h+=`<div><h2>log (last 120 lines)</h2><pre>${esc(d.log_tail)}</pre></div>`;
 if(d.text)h+=`<pre>${esc(d.text)}</pre>`;
 $('dbody').innerHTML=h;}catch(e){$('dtitle').textContent='Could not fetch details';$('dbody').innerHTML=`<p class="badt">${esc(e.message)}</p>`;}}
document.addEventListener('click',e=>{const t=e.target.closest('button');if(!t)return;if(t.dataset.run)openDetails({run:t.dataset.run});else if(t.dataset.kind)openDetails({kind:t.dataset.kind});});
$('dclose').onclick=()=>$('dlg').close();
poll();setInterval(poll,3000);
</script></body></html>"""


def tailscale_ip() -> str:
    out = sh(["tailscale", "ip", "-4"]).strip().splitlines()
    return out[0] if out else "127.0.0.1"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="", help="address to bind (default: this machine's Tailscale IPv4)")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    host = a.host or tailscale_ip()
    threading.Thread(target=sampler, daemon=True).start()
    threading.Thread(target=windows_sampler, daemon=True).start()
    print(f"bio-bot monitor on http://{host}:{a.port}", flush=True)
    ThreadingHTTPServer((host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
