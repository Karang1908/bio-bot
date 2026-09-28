# One Brain, Many Bodies: Infrastructure and Requirements

_What we use to build, train and run the project · 23 September 2026 · Budget: **$0**_

Companion docs: [README](README.md) · [Overview](Overview.md) · [TechnicalDesign](TechnicalDesign.md) · [LegacyIdea](LegacyIdea.md)

---

## 1. What this infrastructure has to support

Two research questions (details in [TechnicalDesign.md](TechnicalDesign.md)):

1. **Self-learning:** a brain that configures itself for whatever it's plugged into (robot, software, IoT), with no manual setup. This needs **many simulated bodies** for its "childhood", plus a way to connect **real systems** (smart home, homelab) safely.
2. **Coordination:** one brain running several bodies at once. This needs **multi-body simulations** and enough compute to compare one brain against several separate brains.

Applications this setup is meant to reach:
- self-configuring robots
- smart home / IoT that learns what each device does
- self-managing homelabs and servers
- building systems
- mixed robot teams for search, home assistance, warehouses and inspection

---

## 2. Where everything runs

| Work | Machine | Why |
|---|---|---|
| Bio-Bot Studio: sandbox + open world, body inspector, command box | MacBook Air M3 | open world with all five bodies has 1.24× real-time headroom, the sandbox 2.35×; the browser renders at 60 fps (measured) |
| Writing code, debugging, unit tests | MacBook Air M3 | always available; JAX on CPU is fine at tiny scale |
| Coarse-scale experiments (~2k-unit brain), quick checks | MacBook Air M3 | minutes per run (measured, §3.1) |
| Main training runs (mesoscale and larger, 3D bodies, many seeds) | **Kaggle free GPUs** | GPU-parallel physics (MJX) + brain |
| Paired comparisons (connectome vs baseline) | Kaggle, **one T4 per condition** | two T4s per session means side-by-side runs with identical settings |
| Live demos, full MaleCNS brain | MacBook Air M3 | full brain runs in real time (measured) |
| Long CPU-only jobs (evolution-strategy workers) | Linux machine | has fans; can run for days |
| Edge deployment target | Raspberry Pi | proves the stripped brain runs on low-spec hardware |
| Real "digital bodies" | Mac + Linux + Pi (homelab), Home Assistant (smart home) | RQ1 on real systems, behind the fence |

---

## 3. Hardware

### 3.1 MacBook Air M3 (development + demos)

- **Spec:** Apple M3, 4 performance + 4 efficiency CPU cores, 10-core GPU, 16 GB unified memory, fanless.
- **Measured on this machine** (23 Sep 2026, on battery at ~47%, two runs each). Random graphs with matching sizes; only the full-size counts are real MaleCNS counts, the smaller edge counts are assumed.

| What | Run 1 | Run 2 |
|---|---|---|
| Physics, simple quadruped (MuJoCo), 1 core | 58k steps/s | 59k steps/s |
| Physics, batched, 4 threads | 255k | 172k–257k |
| Physics, batched, 8 threads | 255k | 303k–337k |
| Full MaleCNS (166.7k neurons, 25.5M connections), one brain step on GPU (MLX) | 12.2 ms | 10.8 ms |
| Pruned (~70k, 10M assumed), GPU, batch 8 | 391 steps/s | 442 steps/s |
| Mesoscale (~10k, 1M assumed), dense GPU, batch 64 | 8.9k steps/s | 9.2k steps/s |
| Coarse (~2k, 200k assumed), dense GPU, batch 256 | 156k steps/s | 144k steps/s |

What that means:
- The full MaleCNS brain **runs in real time** on the Mac (~40–80 Hz), but is too slow to *train* here (~2 days per 10M steps).
- Coarse-scale training is ~3 min per 10M steps, mesoscale ~40 min. (Estimated from the numbers above, assuming 2 brain ticks per physics step, forward passes only.)

**Operating notes for a fanless laptop:**
- It throttles under sustained load. Run long jobs overnight, plugged in, on a hard surface.
- `caffeinate -i <command>` prevents sleep. `nohup` keeps jobs alive after the terminal closes. Background jobs started from Claude Code have been killed after 5–15 minutes before.
- Save checkpoints every few minutes.

### 3.2 Kaggle (free GPU training)

| Resource | Free allowance |
|---|---|
| GPU | **2× NVIDIA T4** (16 GB each) per session. Kaggle's API docs also list an L4 option |
| GPU quota | **≈30 hours/week** ("floating", sometimes more) |
| TPU | TPU v5e-8, **≈20 hours/week** |
| Session length | **12 h** GPU / CPU, **9 h** TPU |
| Quota reset | Saturday 00:00 UTC |
| Account | phone verification needed to enable GPU and internet (believed; confirm in settings) |

Rules of use:
- **Train through pushed jobs**, not interactive notebooks. Open sessions burn quota while idle.
- **Checkpoint every 15–30 minutes** and chain sessions. Each job reads the previous job's output.
- **Each job starts from a fresh machine**, so pin package versions and install them at the top of the script.
- **Budget:** 30 GPU-hours is about two full sessions a week. Anything that doesn't need a GPU runs on the Mac.
- **TPU hours are a bonus.** Test once whether MJX physics runs well on the TPU before planning around it.

### 3.3 Linux machine

- **Role:** long CPU-only jobs (e.g. evolution-strategy workers) and one of the homelab "bodies".
- **Specs:** to be recorded here (CPU cores, RAM, OS).

### 3.4 Raspberry Pi

- **Roles:**
  - edge deployment target: can the stripped brain run and keep learning on low-spec hardware?
  - homelab body, including GPIO (LED, sensors)
- **Not yet measured.** Expected to run the mesoscale brain in real time but probably not the full one. Measure before claiming either.

---

## 4. Software

### 4.1 Core stack (one stack everywhere: JAX)

| Package | Purpose |
|---|---|
| Python | the project `.venv` uses **3.12** (created with `uv`); confirm it matches Kaggle's image (`python --version` in a Kaggle notebook) before the first Kaggle job |
| `jax` | CPU build on the Mac, CUDA build on Kaggle |
| `flax`, `optax` | network modules and optimisers |
| `brax` | PPO training (what MuJoCo Playground uses) |
| `mujoco`, `mujoco-mjx` | physics (CPU on the Mac, GPU via MJX on Kaggle) |
| `mujoco_playground` | ready-made bodies: Go1, Spot, G1, H1, Berkeley Humanoid, and more |
| `evosax` | evolution strategies for plasticity rules |
| `numpy`, `scipy`, `pandas`, `pyarrow` | graph building; reading the `.feather` connectome tables |
| `networkx` (or custom code) | degree-preserving rewiring for baseline graphs |
| `rliable` | statistics (interquartile mean, bootstrap confidence intervals) |
| `matplotlib`, `tensorboard` | plots and training curves (logs saved to job output) |

On the Mac: `mlx` runs the Stage 0 brain on the GPU (installed; also used for the benchmarks above). `pandas` + `pyarrow` read the MaleCNS tables.

### 4.1b Studio (installed and in use)

| Package / asset | Purpose |
|---|---|
| `fastapi`, `uvicorn[standard]` | studio server: REST commands, WebSocket pose stream, brain activity stream (`studio/server`) |
| `lucide-react` | the studio's icons |
| `pytest`, `httpx` | tests (`tests/`) |
| Vite 8, React 19, TypeScript 5.9 | studio web app (`studio/web`) |
| `three` 0.186, `@react-three/fiber` 9, `@react-three/drei` 10 | 3D rendering of the MuJoCo world |
| Poly Haven sky HDRI (CC0), fetched by `scripts/fetch_assets.py` | lighting and background |

### 4.2 Intent layer (all local, free)

| Tool | Purpose |
|---|---|
| `mlx-whisper` or `whisper.cpp` | speech → text on the Mac (whisper tiny/base on the Pi) |
| `ollama` or `mlx-lm` + a small open model (3–8B, quantised) | command → goal spec |
| `open_clip` or `transformers` (SigLIP-class model) | vision-language grounding of object words |
| Meaning code in `intent/` | what each verb means as outcomes: written once, no model at training time |

### 4.3 Real systems (Phase 6)

| Tool | Purpose |
|---|---|
| Home Assistant (REST / WebSocket API, long-lived access token) | smart-home devices become sensor and actuator channels |
| `psutil` / a metrics exporter on each machine | homelab sensor channels |
| `fence/` config + shield | whitelisted actions, limits, forbidden states; enforced outside the brain |

### 4.4 Tooling

`git`; `uv` or `venv`; Kaggle CLI 2.2 (`uv pip install -e '.[train]'`, then sign in once with `kaggle auth login`).

---

## 5. Data

| Item | Size | Source | Licence |
|---|---|---|---|
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1.1 GB | [male-cns.janelia.org/download](https://male-cns.janelia.org/download/) | CC-BY 4.0 |
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 13 MB | same | CC-BY 4.0 |
| `body-neurotransmitters-male-cns-v1.0.feather` | 42 MB | same | CC-BY 4.0 |
| Robot and fly body models | small | MuJoCo Menagerie / MuJoCo Playground / flybody | per model (mostly Apache 2.0) |

Storage:
- Locally in `data/` (git-ignored).
- On Kaggle as a **private dataset**, uploaded once with `kaggle datasets create`.
- CC-BY requires attribution in any publication or demo.

---

## 6. Workflow

### 6.1 Loop

1. Write and test on the Mac at tiny scale (`python train/<run>.py --steps 20000 --envs 16`).
2. `scripts/kaggle_job.py push <run>` writes the job's settings into the script and sends it to Kaggle.
3. `scripts/kaggle_job.py wait <run>` polls until it finishes.
4. `scripts/kaggle_job.py fetch <run>` pulls the log, training curve, evaluation and policy into `results/<run>/`.
5. A trained result that passes its checks is promoted into the repo, and `pytest` re-checks it in the studio's physics.

Measured (2× T4, jax 0.7.2, MJX, Brax 0.14.2, 8192 envs, MLP policy): Go2 physics with feet-only contacts runs 200M
control steps (50 Hz, 4 ms physics) in 38 min on flat ground and 61 min on a heightfield, plus ~3.5 min of install and
compilation; a run costs ~1 of the 30 weekly GPU hours. MJX's JAX backend needs zero contact margin against heightfields,
and a rare solver blow-up on terrain turns a whole run to NaN unless the environment ends such episodes and zeroes their
observations (Playground's auto-reset keeps `info`, so that must be cleaned too).
The Kaggle CLI's OAuth login (`kaggle auth login`) expired within ~3 h in use; re-run it with `--force` before fetching.

Gotchas measured on 27–28 Sep 2026:
- `kaggle kernels output` skips any file that already exists locally with the same size, unless it is given `-o`. Two
  trained brains of the same shape silently kept the previous run's file. `fetch` now always passes `-o`.
- A free account runs at most 2 batch GPU jobs at once ("Maximum batch GPU session count of 2 reached"). A cancelled or
  replaced job can hold a slot for a while. `push --job <name>` runs one script as separate jobs side by side, and
  `push --machine NvidiaTeslaP100` uses the P100 queue instead of the default 2× T4.
- Kaggle keeps a job's log to itself until the job ends, so a long job cannot be checked mid-run. The baby curriculum
  (`train/baby_dog.py`) therefore runs as short stages, each ending with its own verdict. Brains move between stages
  through the private dataset `bio-bot-brains` (`upload`, then `push --dataset bio-bot-brains`).

### 6.2 Job metadata (`jobs/<run>/kernel-metadata.json`)

```json
{
  "id": "<kaggle-username>/obmb-run-01",
  "title": "obmb-run-01",
  "code_file": "train.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": ["<kaggle-username>/malecns-v1"],
  "kernel_sources": ["<kaggle-username>/obmb-run-00"]
}
```

Field names are from Kaggle's API docs (`kernels_metadata.md`).

### 6.3 Experiment hygiene

- One folder per run with its config, seed, git commit hash, logs and checkpoints.
- At least 5 seeds per condition; predictions written in [TechnicalDesign.md §9](TechnicalDesign.md#9-experiments) before running.

---

## 7. Repository layout

Built so far: `world/` (free world, fly scaling, built-in controllers), `studio/server` and `studio/web` (Bio-Bot Studio), `scripts/` (`fetch_assets.py`, `studio.sh`), `tests/`. The rest is planned.

```
deep-learning/
├── README.md  Overview.md  TechnicalDesign.md  Requirements.md  LegacyIdea.md  PRODUCT.md  DESIGN.md
├── world/          free world, unit scaling, built-in controllers           (built)
├── studio/         server/ (FastAPI) + web/ (React + three.js)               (built)
├── scripts/        fetch_assets.py, studio.sh                                (built)
├── tests/          world, controllers, parser, API                           (built)
├── brain/          graph building, four size variants, baseline graphs, dynamics, plasticity
├── interface/      sensor tokens, motor readout, per-episode channel randomisation
├── bodies/         procedural bodies, digital bodies
├── coordination/   nerve-cord slots, shared world memory, merge / hot-plug
├── intent/         meaning code (one file per verb), speech, command parsing, vocabulary
├── training/       PPO, evolution strategies, childhood / joint / strip pipelines
├── experiments/    E1–E13 configs and analysis
├── jobs/           Kaggle job folders (kernel-metadata.json + entry script)
├── fence/          safety rules for real systems
├── data/           (git-ignored) MaleCNS tables
└── results/        (git-ignored) checkpoints, logs, plots
```

This folder is its own git repository (github.com/Karang1908/bio-bot). `third_party/`, `data/`, `results/`, the HDRI and build output are git-ignored.

---

## 8. Secrets and safety

- **Kaggle credentials:** from `kaggle auth login` (or a legacy `~/.kaggle/kaggle.json` with `chmod 600`). Never committed, never inside a job folder.
- **Home Assistant token:** in an environment variable or the macOS Keychain, never in the repo.
- **`.gitignore`:** `data/`, `results/`, `*.ckpt`, `.env`, `kaggle.json`.
- **Real systems only run behind the fence:**
  - whitelisted actions
  - rate and value limits
  - forbidden states enforced outside the brain
  - watch first, then reversible probes, then goal-directed use

---

## 9. Requirements per application

| To connect… | You need | Research question |
|---|---|---|
| A simulated body | a MuJoCo model (or JAX dynamics) exposed as unnamed sensor and actuator arrays | 1 |
| A real robot | a driver exposing joint / IMU / contact readings and commands as arrays; hard limits in `fence/` | 1 |
| A smart home | Home Assistant with the devices added; an access token; a fence listing forbidden entities (locks, heaters) | 1 |
| A homelab | a metrics exporter on each machine; a list of approved jobs, containers and GPIO pins; no shell access | 1 |
| Several bodies at once | a shared coordinate source (GPS-like channel in simulation); one nerve-cord slot per body | 2 |
| Voice commands | microphone, local Whisper, local LLM | both |

---

## 10. Not used, and why

| Option | Reason |
|---|---|
| Paid cloud GPUs | budget is $0; Kaggle's free tier covers training |
| Laya as a training teacher | reads only text; judging every step would cost 3–8 CPU-weeks or 20–44 T4-hours per 10M steps; near chance without fine-tuning. Its calibration *idea* is borrowed |
| Jev (OpenCode free tier) | free "for a limited time"; reports from Sept 2026 say it now rejects clients outside the OpenCode app; text-only and remote |
| Spiking brain simulators for training | ~0.015× real time for full MaleCNS; too slow for learning |
| jax-metal | unmaintained; the Mac runs JAX on CPU for debugging only |

---

## 11. Open items (to measure or confirm)

- [ ] Brain step time on a **Kaggle T4** (decides the largest trainable brain size)
- [ ] Kaggle image's Python version; match it on the Mac
- [ ] Kaggle phone verification done; GPU + internet enabled
- [ ] Whether MJX runs acceptably on Kaggle's TPU v5e-8
- [ ] Linux machine specs (§3.3)
- [ ] Raspberry Pi step time at mesoscale and full scale
- [ ] Mac benchmarks repeated on mains power

---

## 12. Claim status

- **Verified:** Mac specs and benchmark numbers (measured twice); MaleCNS file names, sizes and licence; Kaggle job-metadata fields (Kaggle API docs).
- **From secondary sources:** Kaggle quotas and session limits (Kaggle's own docs page did not render); the phone-verification requirement.
- **Estimates:** training times per 10M steps; Pi performance.
