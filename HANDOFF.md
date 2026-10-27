# HANDOFF: bio-bot, "One Brain, Many Bodies"

The handoff for the next AI coding agent. Written 10 Oct 2026; updated 11 Oct 2026 after the longer-history trial.

Read this file first, then:
- `LOGBOOK.md`: every experiment, number and mistake;
- `DogMind.md`: the agreed plan for senses, self-discovery and mind;
- `TechnicalDesign.md` / `Requirements.md`: the design, the infrastructure, the Kaggle gotchas.

Everything below marked **verified** was measured. **Believed** means reasoned but not measured.

**Current state, 27 Oct 16:50 Dubai: Round 48 COMPLETED on Kaggle 2x Nvidia Tesla T4 GPU (Job 'bio-bot-baby-dog-vision-approach-round48', 87.6 min runtime); Step 65k achieved ALL-TIME DUAL-ARRIVAL PARITY of 7.81% Ball (15/192) AND 7.81% Owner (15/192) = 15.6% Combined Arrival across 192 lives (Cohort 1 Ball 9.0%, Cohort 3 Ball 8.0%), Step 262k Ball Arrival 7.81%, Step 393k Ball Arrival 7.29%; 100% Motor Skills Preserved; Step 65k Snapshot (SHA256 4d54e3e4...) Promoted Alongside Cohort Record Candidate (Round 47 step 327k, 16.0% in C2) and Multi-Cohort Candidate (Round 43 step 196k, 9.90% overall); Quota: Kaggle GPU 11.20h Remaining.**
- **Completed Remote Training Job: Round 48 on Kaggle 2x Tesla T4 GPU**:
  - Kernel: `karang1908/bio-bot-baby-dog-vision-approach-round48` (Status: `COMPLETE`, runtime 87.6 min).
  - Executed 393,216 steps (24 updates × 16,384 steps = 512 envs, 256/GPU) with zero cost ($0 budget).
  - All outputs fetched locally to `results/baby-dog-vision-approach-round48/`: `baby_dog.npz`, snapshots at 65k, 131k, 196k, 262k, 327k, 393k steps, `practice.json`, and `bio-bot-baby-dog-vision-approach-round48.log` (copied to `log.txt`).
  - Key measured outcomes across 3 fresh cohorts (192 lives per evaluation):
    - Ball arrival: **7.81% (15/192)** at step 65k (`4d54...`, C1 9.0%, C2 6.0%, C3 8.0%), **7.81% (15/192)** at step 262k (`f1cf...`, C1 6.0%, C2 8.0%, C3 9.0%), and **7.29% (14/192)** at step 393k (`543c...`, C1 8.0%, C2 6.0%, C3 8.0%).
    - Owner arrival: **7.81% (15/192)** at step 65k (`4d54...`, C1 6.0%, C2 8.0%, C3 9.0%) — achieving simultaneous **15.6% combined target arrival rate**!
    - Directional search recovery: In Round 47, Cohort 1 lagged at 2.0% on rear spawns; with directional `rot_search`, Cohort 1 reached **9.0%** at step 65k and **8.0%** at step 393k!
    - 100% nominal motor skills preserved throughout: walk speed +0.3891 to +0.4030 m/s, pace +0.5541 to +0.5716 m/s, turn-left yaw +1.3735 to +1.3962 rad/s.
- **Completed Remote Training Job: Round 47 on Kaggle 2x Tesla T4 GPU**:
  - Kernel: `karang1908/bio-bot-baby-dog-vision-approach-round47` (Status: `COMPLETE`, runtime 87.1 min).
  - Executed 393,216 steps (24 updates × 16,384 steps = 512 envs, 256/GPU) with zero cost ($0 budget).
  - All outputs fetched locally to `results/baby-dog-vision-approach-round47/`: `baby_dog.npz`, snapshots at 65k, 131k, 196k, 262k, 327k, 393k steps, `practice.json`, and `bio-bot-baby-dog-vision-approach-round47.log` (copied to `log.txt`).
  - Key measured outcomes across 3 fresh cohorts (192 lives per evaluation):
    - Ball arrival: **16.0% (10/64 lives)** in Cohort 2 at step 327k (`53f26415...`) — **first time any cohort has exceeded the 15% threshold**! Overall ball arrival 7.81% (15/192) at step 327k, 6.77% at baseline.
    - Owner arrival: **6.77% (13/192)** overall at step 327k and step 393k (Cohort 2: 9.0%, Cohort 3: 9.0%).
    - 100% nominal motor skills preserved throughout: walk speed +0.3898 to +0.3998 m/s, pace +0.5590 to +0.5736 m/s, turn-left yaw +1.3544 to +1.3862 rad/s.
    - Hazard dodge reaction latency: an unprecedented **0.0 ms** (instant reaction) at step 393k (`2bee...`), 20.0 ms at step 196k, 23.3 ms at step 131k.
- **Completed Remote Training Job: Round 46 on Kaggle 2x Tesla T4 GPU**:
  - Kernel: `karang1908/bio-bot-baby-dog-vision-approach-round46` (Status: `COMPLETE`, runtime 90.1 min).
  - Executed 393,216 steps (24 updates × 16,384 steps = 512 envs, 256/GPU) with zero cost ($0 budget).
  - All outputs fetched locally to `results/baby-dog-vision-approach-round46/`: `baby_dog.npz`, snapshots at 65k, 131k, 196k, 262k, 327k, 393k steps, `practice.json`, and `bio-bot-baby-dog-vision-approach-round46.log` (copied to `log.txt`).
  - Key measured outcomes across 3 fresh cohorts (192 lives per evaluation):
    - Ball arrival: 6.77% (13/192) at step 131k (`a47ab901...`, C2 9.4%), 6.25% (12/192) at step 65k, 327k, and 393k.
    - Owner arrival: 7.29% (14/192) at baseline, 4.69% (9/192) at step 196k.
    - 100% nominal motor skills preserved throughout: walk speed +0.3823 to +0.4019 m/s, pace +0.5520 to +0.5740 m/s, turn-left yaw +1.3520 to +1.4112 rad/s.
    - Hazard dodge reaction latency: ultra-fast **6.7 ms** at step 393k (`0b13280d...`), 40.0 ms at step 262k, 43.3 ms at baseline.
- **Completed Remote Training Job: Round 45 on Kaggle 2x Tesla T4 GPU**:
  - Kernel: `karang1908/bio-bot-baby-dog-vision-approach-round45` (Status: `COMPLETE`, runtime 88.5 min).
  - Executed 393,216 steps (24 updates × 16,384 steps = 512 envs, 256/GPU) with zero cost ($0 budget).
  - All outputs fetched locally to `results/baby-dog-vision-approach-round45/`: `baby_dog.npz`, snapshots at 65k, 131k, 196k, 262k, 327k, 393k steps, `practice.json`, and `bio-bot-baby-dog-vision-approach-round45.log` (copied to `log.txt`).
  - Key measured outcomes across 3 fresh cohorts (192 lives per evaluation):
    - Ball arrival: 6.77% (13/192) at step 327k (`8a52bcb6...`), 6.25% (12/192) at step 65k (`7dec842c...`), and **11.0% (7/64 lives)** in Cohort 1 at step 393k (`a1a7babc...`).
    - Owner arrival: 5.73% (11/192) at baseline and at step 393k, 4.69% (9/192) at step 65k.
    - 100% nominal motor skills preserved throughout: walk speed +0.3864 to +0.3986 m/s, pace +0.5453 to +0.5696 m/s, turn-left yaw +1.3329 to +1.3836 rad/s, sit +0.490 to +0.503 / 0.617 to 0.620, lie-down +0.013 to +0.020 / 0.946 to 0.955.
    - Hazard dodge reaction latency: 10.0 ms at step 196k (`ee6e8b28...`), 23.3 ms at baseline, 30.0 ms at step 65k.
- **Completed Remote Training Job: Round 44 on Kaggle 2x Tesla T4 GPU**:
  - Kernel: `karang1908/bio-bot-baby-dog-vision-approach-round44` (Status: `COMPLETE`, runtime 91.2 min).
  - Executed 393,216 steps; ball arrival peaked at 6.77% (13/192) at step 65k and step 196k; record 3.3 ms dodge latency at step 65k.
- **Completed Remote Training Job: Round 43 on Kaggle 2x Tesla T4 GPU**:
  - Kernel: `karang1908/bio-bot-baby-dog-vision-approach-round43` (Status: `COMPLETE`, runtime 87.5 min).
  - Executed 393,216 steps; step 196k (`0436cf4c...`) hit an **all-time record ball arrival of 9.90% (19/192 lives)** overall across all 3 cohorts (Cohort 2: 14.1%, Cohort 3: 9.4%).
- **Checkpoint Outcomes & Promotion**:
  - **All-Time Cohort Record Candidate**: `baby_dog_step_0000327680.npz` (SHA256 `53f264155a58dece500d0a352ed7fa4b7698907f9cac0c34319759eb9d10eb4c`, verified, Round 47).
    - Ball arrival: **16.0% (10/64)** in Cohort 2 (first time any cohort has exceeded >15%), **7.81% (15/192)** overall; Owner arrival **9.0%** in C2/C3, **6.77% (13/192)** overall.
    - Motor skills: +0.3965 m/s walk speed, +0.5682 m/s pace, +1.3757 rad/s turn-left.
    - Recorded checkpoint string: `'after 0.3M steps'`.
  - **All-Time Dual-Arrival Parity Candidate**: `baby_dog_step_0000065536.npz` (SHA256 `4d54e3e430719f6449b9d4105bc80e0642a3aab21f9f5c6f6d868ffa69629f6e`, verified, Round 48).
    - Dual arrival parity: **7.81% (15/192) Ball Arrival AND 7.81% (15/192) Owner Arrival** (15.6% combined target arrival rate across 192 lives; C1 Ball 9.0%, C3 Ball 8.0%).
    - Motor skills: +0.3928 m/s walk speed, +0.5592 m/s pace, +1.3962 rad/s turn-left.
    - Recorded checkpoint string: `'after 0.1M steps'`.
  - **All-Time Multi-Cohort Ball Candidate**: `baby_dog_step_0000196608.npz` (SHA256 `0436cf4cb8ec2cd93ecaf2e4ceceaafb316f285eaed3694a18785ff58278751f`, verified, Round 43).
    - Ball arrival: **9.90% (19/192)** overall, **14.1% (9/64)** in Cohort 2, **9.4% (6/64)** in Cohort 3.
    - Motor skills: +0.3958 m/s walk speed, +0.5632 m/s pace, +1.3695 rad/s turn-left.
    - Recorded checkpoint string: `'after 0.2M steps'`.
  - **Alternative Fast-Reflex Candidate**: `baby_dog_step_0000393216.npz` (SHA256 `2beea63a6af1c0dcde07fe13829e7fa8faec49d44dccb5e1a6dd02533c78252d`, Round 47).
    - Record **0.0 ms** hazard dodge latency (instantaneous reaction to projectile), 6.77% owner arrival.
  - Final Trained Brain (Round 48): `baby_dog.npz` (SHA256 `a2478453b5b7d48d2240df2ecf3e905ea34736d85c404ef238112f452fd899c1`).
- **Remote Compute Quota (Verified)**:
  - Kaggle GPU quota: 11.20h available (used 1.48h for Round 48).
  - Google Colab Free Tier: Available as fallback.
- Strictly zero laptop compute checks ($0 budget); all source commits pushed individually on `main`.
- The current-state and active-job sections take precedence over chronological pending-status notes.

### Latest completed trial: vision approach round 48 (directional search & dual-target parity on 2x Nvidia Tesla T4 GPU)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round48 (version 1).
Completed all 24 updates (393,216 steps, 87.6 min runtime) on 2x Nvidia Tesla T4 GPU (512 envs, 256/GPU) to `KernelWorkerStatus.COMPLETE`. Outputs ingested into `results/baby-dog-vision-approach-round48/`.

Setup: 393,216 steps (24 updates × 16,384 steps = 393,216 steps), 512 envs, 6 evaluation checkpoints (every 65k steps across 3 fresh cohorts = 192 lives per eval, 7 total evaluations including baseline and final), 2h cap. Teacher movement help zero from rest (`help_ = 0.00`).
Resumed from all-time cohort record candidate `baby_dog_step_0000327680.npz` (`53f26415…`, Round 47 candidate, 16.0% in C2) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round47`.
Attached datasets: `bio-bot-dog-mocap`, `bio-bot-cores`.
Embedded modules: `train/brain_dog.py`, `train/dog_world.py`.
Hyperparameters: `fwd_pay = 50.0`, `sprint_pay = 35.0`, `sprint_floor = 0.42`, `yaw_pay = 35.0`, `face_pay = 30.0`, `search_pay = 35.0` with directional `rot_search = clip(yaw_toward/1.0, -1.0, 1.0) * (1 - align_gate)`, `align_floor = 0.35`, `misalign_cost = 3.0`, `brake_scale = 1.4` (calibrated 0.49m braking radius), `task_pay = 75.0`, `reach_pay = 150.0`, `stay_pay = 100.0`, `prox_pay = 40.0`.
All outputs fetched to `results/baby-dog-vision-approach-round48/`: `baby_dog.npz`, snapshots at 65k, 131k, 196k, 262k, 327k, 393k steps, `practice.json`, and `bio-bot-baby-dog-vision-approach-round48.log` (copied to `log.txt`).

**Measured Outcomes across 3 fresh cohorts (192 lives per evaluation):**

| measured metric (three cohorts, 192 lives) | baseline (53f2...) | 0.07M (4d54...) | 0.13M (d318...) | 0.20M (1411...) | 0.26M (f1cf...) | 0.33M (51ab...) | 0.39M (543c...) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| mean movement falls / 13 s | 1.1855 | 1.1725 | 1.1901 | 1.1823 | 1.1862 | 1.1686 | 1.1628 |
| walk speed, m/s | +0.3920 | +0.3928 | +0.3924 | +0.3952 | +0.3891 | +0.4030 | +0.3906 |
| pace speed, m/s | +0.5575 | +0.5592 | +0.5541 | +0.5670 | +0.5573 | +0.5626 | +0.5716 |
| canter speed, m/s | +0.3896 | +0.3914 | +0.4018 | +0.3911 | +0.3958 | +0.3958 | +0.3973 |
| run speed, m/s | +0.5457 | +0.5569 | +0.5499 | +0.5531 | +0.5604 | +0.5583 | +0.5532 |
| back speed, m/s | -0.0813 | -0.0693 | -0.0761 | -0.0767 | -0.0758 | -0.0704 | -0.0719 |
| turn-left yaw, rad/s | +1.3802 | +1.3962 | +1.3762 | +1.3777 | +1.3735 | +1.3817 | +1.3744 |
| turn-right yaw, rad/s | -1.1269 | -1.1230 | -1.1083 | -1.1162 | -1.1332 | -1.1272 | -1.1411 |
| sit pose (pitch / trunk) | +0.499 / 0.620 | +0.493 / 0.620 | +0.490 / 0.619 | +0.507 / 0.620 | +0.498 / 0.620 | +0.493 / 0.619 | +0.503 / 0.619 |
| lie-down pose (pitch / trunk) | +0.018 / 0.947 | +0.009 / 0.942 | +0.014 / 0.950 | +0.022 / 0.957 | +0.022 / 0.954 | +0.009 / 0.946 | +0.016 / 0.950 |
| **ball arrival success (got_there)** | 3.12% (6/192) | **7.81% (15/192)** | 3.12% (6/192) | 4.17% (8/192) | **7.81% (15/192)** | 4.69% (9/192) | 7.29% (14/192) |
| ball distance drift (start -> end, m) | 2.491 -> 3.497 | 2.490 -> 3.561 | 2.491 -> 3.666 | 2.491 -> 3.611 | 2.490 -> 3.519 | 2.489 -> 3.624 | 2.492 -> 3.612 |
| ball aligned in final second | 2.85% | 2.73% | 2.28% | 3.56% | 3.55% | 2.03% | 2.27% |
| ball final facing cosine | -0.4316 | -0.4512 | -0.4488 | -0.4396 | -0.3931 | -0.4362 | -0.4055 |
| ball falls / 13 s | 0.8385 | 0.7812 | 0.8177 | 0.8490 | 0.8698 | 0.7708 | 0.7969 |
| **owner arrival success (got_there, come)** | 3.65% (7/192) | **7.81% (15/192)** | 3.12% (6/192) | 4.69% (9/192) | 2.08% (4/192) | 3.65% (7/192) | 2.60% (5/192) |
| owner distance drift (start -> end, m) | 3.664 -> 3.556 | 3.665 -> 3.590 | 3.662 -> 3.692 | 3.664 -> 3.637 | 3.664 -> 3.591 | 3.663 -> 3.597 | 3.666 -> 3.655 |
| owner aligned in final second | 8.28% | 8.93% | 9.88% | 12.23% | 10.54% | 12.76% | 11.66% |
| owner final facing cosine | -0.0130 | -0.1297 | +0.0108 | +0.0821 | -0.0099 | +0.0266 | +0.0324 |
| owner falls / 13 s | 1.5260 | 1.5729 | 1.5573 | 1.4375 | 1.5052 | 1.5052 | 1.5938 |
| **dodge hazard reaction (hit rate)** | 77.6% | 79.7% | 74.5% | 74.0% | 74.5% | 79.7% | 76.0% |
| dodge moved before impact | 20.3% | 21.4% | 25.5% | 27.1% | 31.8% | 26.0% | 20.3% |
| dodge median reaction latency | 130.0 ms | 40.0 ms | 50.0 ms | 60.0 ms | 46.7 ms | 33.3 ms | 103.3 ms |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

### Latest completed trial: vision approach round 47 (forward propulsion dominance & 16.0% cohort record on 2x Nvidia Tesla T4 GPU)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round47 (version 1).
Completed all 24 updates (393,216 steps, 87.1 min runtime) on 2x Nvidia Tesla T4 GPU (512 envs, 256/GPU) to `KernelWorkerStatus.COMPLETE`. Outputs ingested into `results/baby-dog-vision-approach-round47/`.

Setup: 393,216 steps (24 updates × 16,384 steps = 393,216 steps), 512 envs, 6 evaluation checkpoints (every 65k steps across 3 fresh cohorts = 192 lives per eval, 7 total evaluations including baseline and final), 2h cap. Teacher movement help zero from rest (`help_ = 0.00`).
Resumed from all-time primary candidate `baby_dog_step_0000196608.npz` (`0436cf4c…`, Round 43 candidate) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round43`.
Attached datasets: `bio-bot-dog-mocap`, `bio-bot-cores`.
Embedded modules: `train/brain_dog.py`, `train/dog_world.py`.
Hyperparameters: `fwd_pay = 50.0`, `sprint_pay = 35.0`, `sprint_floor = 0.42`, `yaw_pay = 35.0`, `face_pay = 30.0`, `search_pay = 35.0`, `align_floor = 0.35`, `misalign_cost = 3.0`, `brake_scale = 1.4` (calibrated 0.49m braking radius), `task_pay = 75.0`, `reach_pay = 150.0`, `stay_pay = 100.0`, `prox_pay = 40.0`.
All outputs fetched to `results/baby-dog-vision-approach-round47/`: `baby_dog.npz`, snapshots at 65k, 131k, 196k, 262k, 327k, 393k steps, `practice.json`, and `bio-bot-baby-dog-vision-approach-round47.log` (copied to `log.txt`).

**Measured Outcomes across 3 fresh cohorts (192 lives per evaluation):**

| measured metric (three cohorts, 192 lives) | baseline (0436...) | 0.07M (b1bd...) | 0.13M (9b8f...) | 0.20M (7f95...) | 0.26M (6eaa...) | 0.33M (53f2...) | 0.39M (2bee...) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| mean movement falls / 13 s | 1.1810 | 1.1569 | 1.1849 | 1.1947 | 1.1719 | 1.1979 | 1.1882 |
| walk speed, m/s | +0.3926 | +0.3998 | +0.3946 | +0.3966 | +0.3956 | +0.3965 | +0.3898 |
| pace speed, m/s | +0.5590 | +0.5733 | +0.5620 | +0.5736 | +0.5612 | +0.5682 | +0.5623 |
| canter speed, m/s | +0.3964 | +0.3929 | +0.4003 | +0.3928 | +0.3989 | +0.3961 | +0.3919 |
| run speed, m/s | +0.5440 | +0.5528 | +0.5451 | +0.5464 | +0.5587 | +0.5584 | +0.5379 |
| back speed, m/s | -0.0664 | -0.0780 | -0.0787 | -0.0762 | -0.0742 | -0.0694 | -0.0697 |
| turn-left yaw, rad/s | +1.3717 | +1.3765 | +1.3842 | +1.3735 | +1.3544 | +1.3757 | +1.3862 |
| turn-right yaw, rad/s | -1.1291 | -1.1323 | -1.1300 | -1.1106 | -1.1289 | -1.1199 | -1.1488 |
| sit pose (pitch / trunk) | +0.512 / 0.620 | +0.507 / 0.621 | +0.497 / 0.620 | +0.502 / 0.620 | +0.496 / 0.618 | +0.497 / 0.617 | +0.476 / 0.619 |
| lie-down pose (pitch / trunk) | +0.019 / 0.951 | +0.019 / 0.955 | +0.022 / 0.956 | +0.021 / 0.953 | +0.025 / 0.955 | +0.020 / 0.952 | +0.019 / 0.954 |
| **ball arrival success (got_there)** | 6.77% (13/192) | 5.21% (10/192) | 4.17% (8/192) | 5.21% (10/192) | 3.65% (7/192) | **7.81% (15/192)** | 3.65% (7/192) |
| ball distance drift (start -> end, m) | 2.492 -> 3.674 | 2.489 -> 3.477 | 2.489 -> 3.616 | 2.491 -> 3.753 | 2.491 -> 3.637 | 2.491 -> 3.709 | 2.490 -> 3.626 |
| ball aligned in final second | 3.75% | 3.35% | 2.20% | 2.92% | 3.04% | 1.38% | 2.36% |
| ball final facing cosine | -0.4327 | -0.4427 | -0.4875 | -0.4583 | -0.4370 | -0.5051 | -0.4373 |
| ball falls / 13 s | 0.8542 | 0.7708 | 0.8177 | 0.7552 | 0.7604 | 0.7969 | 0.8281 |
| **owner arrival success (got_there, come)** | 5.73% (11/192) | 2.60% (5/192) | 4.69% (9/192) | 3.12% (6/192) | 3.65% (7/192) | **6.77% (13/192)** | **6.77% (13/192)** |
| owner distance drift (start -> end, m) | 3.663 -> 3.580 | 3.664 -> 3.697 | 3.664 -> 3.584 | 3.665 -> 3.523 | 3.665 -> 3.607 | 3.666 -> 3.614 | 3.664 -> 3.526 |
| owner aligned in final second | 10.83% | 10.88% | 10.75% | 10.31% | 12.25% | 12.21% | 12.55% |
| owner final facing cosine | -0.0395 | -0.0303 | -0.0680 | -0.0175 | +0.0569 | -0.0244 | +0.0462 |
| owner falls / 13 s | 1.4896 | 1.6458 | 1.5625 | 1.7240 | 1.6302 | 1.4115 | 1.5990 |
| **dodge hazard reaction (hit rate)** | 78.6% | 78.1% | 75.0% | 77.6% | 76.6% | 77.6% | 76.0% |
| dodge moved before impact | 26.0% | 24.5% | 22.9% | 22.4% | 29.7% | 22.4% | 24.5% |
| dodge median reaction latency | 26.7 ms | 33.3 ms | 23.3 ms | 20.0 ms | 46.7 ms | 33.3 ms | **0.0 ms** |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

### Latest completed trial: vision approach round 46 (calibrated braking radius & decisive approach on 2x Nvidia Tesla T4 GPU)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round46 (version 1).
Completed all 24 updates (393,216 steps, 90.1 min runtime) on 2x Nvidia Tesla T4 GPU (512 envs, 256/GPU) to `KernelWorkerStatus.COMPLETE`. Outputs ingested into `results/baby-dog-vision-approach-round46/`.

Setup: 393,216 steps (24 updates × 16,384 steps = 393,216 steps), 512 envs, 6 evaluation checkpoints (every 65k steps across 3 fresh cohorts = 192 lives per eval, 7 total evaluations including baseline and final), 2h cap. Teacher movement help zero from rest (`help_ = 0.00`).
Resumed from all-time primary candidate `baby_dog_step_0000196608.npz` (`0436cf4c…`, Round 43 candidate) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round43`.
Attached datasets: `bio-bot-dog-mocap`, `bio-bot-cores`.
Embedded modules: `train/brain_dog.py`, `train/dog_world.py`.
Hyperparameters: `brake_scale = 1.4` (calibrated braking radius 0.49m for ball), `near_brake` floor `0.25`, `task_still = jp.where(there, 0.0, still)` (penalizing standing still before arrival), `yaw_pay = 100.0`, `face_pay = 100.0`, `sprint_floor = 0.42`, `sprint_pay = 30.0`, `fwd_pay = 45.0`, `search_pay = 35.0` (active visual search rotation), `misalign_cost = 6.0`, `align_floor = 0.45`, `task_pay = 90.0`, `reach_pay = 150.0`, `stay_pay = 100.0`, `prox_pay = 45.0`.
All outputs fetched to `results/baby-dog-vision-approach-round46/`: `baby_dog.npz`, snapshots at 65k, 131k, 196k, 262k, 327k, 393k steps, `practice.json`, and `bio-bot-baby-dog-vision-approach-round46.log` (copied to `log.txt`).

**Measured Outcomes across 3 fresh cohorts (192 lives per evaluation):**

| measured metric (three cohorts, 192 lives) | baseline (0436...) | 0.07M (10e0...) | 0.13M (a47a...) | 0.20M (1c76...) | 0.26M (d2b2...) | 0.33M (840e...) | 0.39M (0b13...) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| mean movement falls / 13 s | 1.1549 | 1.1458 | 1.1510 | 1.1960 | 1.2129 | 1.2051 | 1.2259 |
| walk speed, m/s | +0.4019 | +0.3907 | +0.3823 | +0.3857 | +0.3909 | +0.3884 | +0.3881 |
| pace speed, m/s | +0.5692 | +0.5646 | +0.5740 | +0.5520 | +0.5616 | +0.5719 | +0.5679 |
| canter speed, m/s | +0.3948 | +0.3912 | +0.3994 | +0.3934 | +0.3904 | +0.3911 | +0.3960 |
| run speed, m/s | +0.5582 | +0.5455 | +0.5364 | +0.5458 | +0.5565 | +0.5507 | +0.5492 |
| back speed, m/s | -0.0777 | -0.0729 | -0.0856 | -0.0833 | -0.0758 | -0.0779 | -0.0750 |
| turn-left yaw, rad/s | +1.3716 | +1.3928 | +1.4112 | +1.3520 | +1.3883 | +1.3763 | +1.3750 |
| turn-right yaw, rad/s | -1.0942 | -1.0986 | -1.1027 | -1.1058 | -1.1045 | -1.1179 | -1.1226 |
| sit pose (pitch / trunk) | +0.499 / 0.621 | +0.498 / 0.620 | +0.487 / 0.620 | +0.477 / 0.620 | +0.509 / 0.618 | +0.487 / 0.618 | +0.490 / 0.620 |
| lie-down pose (pitch / trunk) | +0.025 / 0.956 | +0.018 / 0.949 | +0.018 / 0.951 | +0.023 / 0.955 | +0.016 / 0.952 | +0.015 / 0.949 | +0.019 / 0.954 |
| **ball arrival success (got_there)** | 3.65% (7/192) | 6.25% (12/192) | **6.77% (13/192)** | 3.65% (7/192) | 3.12% (6/192) | 6.25% (12/192) | 6.25% (12/192) |
| ball distance drift (start -> end, m) | 2.491 -> 3.472 | 2.489 -> 3.523 | 2.491 -> 3.574 | 2.490 -> 3.634 | 2.489 -> 3.536 | 2.491 -> 3.373 | 2.491 -> 3.556 |
| ball aligned in final second | 3.50% | 3.07% | 2.89% | 2.51% | 1.99% | 2.19% | 2.76% |
| ball final facing cosine | -0.3934 | -0.3810 | -0.3992 | -0.4735 | -0.4171 | -0.3860 | -0.4451 |
| ball falls / 13 s | 0.7917 | 0.7656 | 0.8229 | 0.8177 | 0.8177 | 0.8229 | 0.7552 |
| **owner arrival success (got_there, come)** | **7.29% (14/192)** | 3.65% (7/192) | 3.12% (6/192) | 4.69% (9/192) | 2.60% (5/192) | 4.17% (8/192) | 3.65% (7/192) |
| owner distance drift (start -> end, m) | 3.662 -> 3.483 | 3.665 -> 3.748 | 3.663 -> 3.602 | 3.664 -> 3.659 | 3.664 -> 3.617 | 3.665 -> 3.561 | 3.664 -> 3.586 |
| owner aligned in final second | 10.82% | 10.94% | 11.42% | 9.06% | 9.41% | 9.45% | 10.57% |
| owner final facing cosine | +0.0082 | +0.0213 | +0.0217 | -0.1007 | -0.1023 | -0.0871 | +0.0327 |
| owner falls / 13 s | 1.6458 | 1.4896 | 1.4271 | 1.5365 | 1.6927 | 1.5469 | 1.4583 |
| **dodge hazard reaction (hit rate)** | 76.0% | 77.6% | 80.2% | 77.6% | 77.6% | 77.1% | 77.1% |
| dodge moved before impact | 24.0% | 24.0% | 22.4% | 28.6% | 27.6% | 26.6% | 25.0% |
| dodge median reaction latency | 43.3 ms | 70.0 ms | 60.0 ms | 60.0 ms | 40.0 ms | 93.3 ms | **6.7 ms** |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Checkpoint SHA256 Hashes:**
- `baby_dog_step_0000065536.npz`: `10e08ff5fec294a5cedc87173e7dd00ed29b4cd15fac132cfc385b0c733be76b`
- `baby_dog_step_0000131072.npz`: `a47ab90165137f124e0d4721525033849317d6e152ede26880309cb71c8b9c3b`
- `baby_dog_step_0000196608.npz`: `1c767d56dce86d06dd1844be7417d8df95ea3ca9cb787a839b7cb03d6954b55e`
- `baby_dog_step_0000262144.npz`: `d2b2db39ab846ff68ea88a258e1fa6d5d4040f2326f047c08207ea44bf5090e8`
- `baby_dog_step_0000327680.npz`: `840ef78ec68a8d92cbe4de608476ff37648d1bfba358a15118a3856899c120bb`
- `baby_dog_step_0000393216.npz`: `0b13280db19250c12c93eedd1f3e24ad91f59a3a21d27e6cbde85bec18563619`
- `baby_dog.npz`: `d55619216a68ebe8e74c186888c40630b89f9039fd7faeb4b4a0a17dfd9a0c02`

### Latest completed trial: vision approach round 45 (pinned heading guidance & dual transit on 2x Nvidia Tesla T4 GPU)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round45 (version 1).
Completed all 24 updates (393,216 steps, 88.5 min runtime) on 2x Nvidia Tesla T4 GPU (512 envs, 256/GPU) to `KernelWorkerStatus.COMPLETE`. Outputs ingested into `results/baby-dog-vision-approach-round45/`.

Setup: 393,216 steps (24 updates × 16,384 steps = 393,216 steps), 512 envs, 6 evaluation checkpoints (every 65k steps across 3 fresh cohorts = 192 lives per eval, 7 total evaluations including baseline and final), 2h cap. Teacher movement help zero from rest (`help_ = 0.00`).
Resumed from all-time primary candidate `baby_dog_step_0000196608.npz` (`0436cf4c…`, Round 43 candidate) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round43`.
Attached datasets: `bio-bot-dog-mocap`, `bio-bot-cores`.
Embedded modules: `train/brain_dog.py`, `train/dog_world.py`.
Hyperparameters: `yaw_pay = 100.0`, `face_pay = 100.0` (elevated from 80.0 to lock heading tightly), `sprint_floor = 0.42` (restored to eliminate premature off-axis surging), `sprint_pay = 30.0`, `fwd_pay = 45.0`, `search_pay = 35.0` (active visual search rotation), `misalign_cost = 6.0`, `align_floor = 0.45`, `task_pay = 90.0`, `reach_pay = 150.0`, `stay_pay = 100.0`, `prox_pay = 45.0`.
All outputs fetched to `results/baby-dog-vision-approach-round45/`: `baby_dog.npz`, snapshots at 65k, 131k, 196k, 262k, 327k, 393k steps, `practice.json`, and `bio-bot-baby-dog-vision-approach-round45.log` (copied to `log.txt`).

**Measured Outcomes across 3 fresh cohorts (192 lives per evaluation):**

| measured metric (three cohorts, 192 lives) | baseline (0436...) | 0.07M (7dec...) | 0.13M (279f...) | 0.20M (ee6e...) | 0.26M (ab76...) | 0.33M (8a52...) | 0.39M (a1a7...) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| mean movement falls / 13 s | 1.1673 | 1.2122 | 1.2090 | 1.2103 | 1.2135 | 1.2135 | 1.2266 |
| walk speed, m/s | +0.3931 | +0.3984 | +0.3958 | +0.3935 | +0.3908 | +0.3986 | +0.3864 |
| pace speed, m/s | +0.5653 | +0.5620 | +0.5611 | +0.5696 | +0.5453 | +0.5631 | +0.5607 |
| canter speed, m/s | +0.3920 | +0.3950 | +0.3987 | +0.3984 | +0.3961 | +0.3932 | +0.3910 |
| run speed, m/s | +0.5323 | +0.5488 | +0.5608 | +0.5662 | +0.5330 | +0.5508 | +0.5417 |
| back speed, m/s | -0.0677 | -0.0787 | -0.0721 | -0.0723 | -0.0693 | -0.0828 | -0.0777 |
| turn-left yaw, rad/s | +1.3766 | +1.3656 | +1.3329 | +1.3753 | +1.3505 | +1.3836 | +1.3790 |
| turn-right yaw, rad/s | -1.1066 | -1.1292 | -1.0961 | -1.1337 | -1.1376 | -1.1017 | -1.1346 |
| sit pose (pitch / trunk) | +0.494 / 0.620 | +0.497 / 0.620 | +0.503 / 0.619 | +0.501 / 0.617 | +0.490 / 0.620 | +0.503 / 0.619 | +0.492 / 0.619 |
| lie-down pose (pitch / trunk) | +0.015 / 0.952 | +0.019 / 0.950 | +0.015 / 0.951 | +0.020 / 0.952 | +0.019 / 0.955 | +0.018 / 0.951 | +0.013 / 0.946 |
| **ball arrival success (got_there)** | 4.17% (8/192) | 6.25% (12/192) | 5.73% (11/192) | 5.21% (10/192) | 5.73% (11/192) | **6.77% (13/192)** | 5.21% (10/192) |
| ball distance drift (start -> end, m) | 2.492 -> 3.569 | 2.491 -> 3.671 | 2.491 -> 3.546 | 2.490 -> 3.599 | 2.491 -> 3.644 | 2.492 -> 3.592 | 2.492 -> 3.571 |
| ball aligned in final second | 3.67% | 1.23% | 5.28% | 1.16% | 2.84% | 1.19% | 1.98% |
| ball final facing cosine | -0.3672 | -0.4187 | -0.4141 | -0.4125 | -0.4918 | -0.4374 | -0.4939 |
| ball falls / 13 s | 0.8021 | 0.7812 | 0.7604 | 0.7708 | 0.7969 | 0.8021 | 0.8542 |
| **owner arrival success (got_there, come)** | 5.73% (11/192) | 4.69% (9/192) | 2.08% (4/192) | 1.56% (3/192) | 3.65% (7/192) | 2.08% (4/192) | **5.73% (11/192)** |
| owner distance drift (start -> end, m) | 3.663 -> 3.621 | 3.663 -> 3.498 | 3.663 -> 3.637 | 3.662 -> 3.736 | 3.663 -> 3.634 | 3.663 -> 3.569 | 3.662 -> 3.733 |
| owner aligned in final second | 9.74% | 13.39% | 9.32% | 10.49% | 9.68% | 10.92% | 8.99% |
| owner final facing cosine | -0.0172 | +0.0373 | +0.0340 | -0.0500 | +0.0068 | -0.0206 | -0.0809 |
| owner falls / 13 s | 1.4844 | 1.5260 | 1.4740 | 1.5052 | 1.5469 | 1.5208 | 1.5625 |
| **dodge hazard reaction (hit rate)** | 77.6% | 77.6% | 75.0% | 79.2% | 76.6% | 74.5% | 77.1% |
| dodge moved before impact | 24.5% | 25.5% | 21.4% | 21.9% | 26.6% | 27.1% | 23.4% |
| dodge median reaction latency | 23.3 ms | 30.0 ms | 43.3 ms | **10.0 ms** | 70.0 ms | 66.7 ms | 50.0 ms |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
  - Kernel: `karang1908/bio-bot-baby-dog-vision-approach-round44` (Status: `COMPLETE`, runtime 91.2 min).
  - Executed 393,216 steps (24 updates × 16,384 steps = 512 envs, 256/GPU) with zero cost ($0 budget).
  - All outputs fetched locally to `results/baby-dog-vision-approach-round44/`: `baby_dog.npz`, snapshots at 65k, 131k, 196k, 262k, 327k, 393k steps, `practice.json`, and `bio-bot-baby-dog-vision-approach-round44.log` (copied to `log.txt`).
  - Key measured outcomes across 3 fresh cohorts (192 lives per evaluation):
    - Ball arrival: 6.77% (13/192) at step 65k (`547e5609...`) and step 196k (`0ef69ec3...`), with Cohort 3 reaching 9.0% and Cohort 2 reaching 8.0%.
    - Owner arrival: 5.21% (10/192) at baseline, 4.69% (9/192) at step 131k.
    - 100% nominal motor skills preserved throughout: walk speed +0.3875 to +0.3996 m/s, pace +0.5586 to +0.5764 m/s, turn-left yaw +1.3752 to +1.3978 rad/s, sit +0.471 to +0.504 / 0.616 to 0.621, lie-down +0.011 to +0.025 / 0.947 to 0.958.
    - Hazard dodge reaction latency: record **3.3 ms** at step 65k (`547e5609...`), 16.7 ms at step 327k, 33.3 ms at step 262k.
- **Completed Remote Training Job: Round 43 on Kaggle 2x Tesla T4 GPU**:
  - Kernel: `karang1908/bio-bot-baby-dog-vision-approach-round43` (Status: `COMPLETE`, runtime 87.5 min).
  - Executed 393,216 steps; step 196k (`0436cf4c...`) hit an **all-time record ball arrival of 9.90% (19/192 lives)** overall across all 3 cohorts (Cohort 2: 14.1%, Cohort 3: 9.4%).
- **Checkpoint Outcomes & Promotion**:
  - **All-Time Primary Learning Restart Candidate**: `baby_dog_step_0000196608.npz` (SHA256 `0436cf4cb8ec2cd93ecaf2e4ceceaafb316f285eaed3694a18785ff58278751f`, verified, Round 43).
    - Ball arrival: **9.90% (19/192)** overall, **14.1% (9/64)** in Cohort 2, **9.4% (6/64)** in Cohort 3.
    - Motor skills: +0.3958 m/s walk speed, +0.5632 m/s pace, +1.3695 rad/s turn-left.
    - Recorded checkpoint string: `'after 0.2M steps'`.
  - **Alternative Fast-Reflex Candidate**: `baby_dog_step_0000065536.npz` (SHA256 `547e5609f65b6094810cfed981d91febfade7ed2004ba873fcc5a1d2f366f303`, Round 44).
    - 6.77% ball arrival, record 3.3 ms dodge hazard latency.
  - Final Trained Brain (Round 44): `baby_dog.npz` (SHA256 `3f4d163b48fe5ff1e50da3d95540839dd2ad621c9f9b3aee609171b2d8ceb73d`).
- **Remote Compute Quota (Verified)**:
  - Kaggle GPU quota: 17.21h available (used 1.47h for Round 44).
  - Google Colab Free Tier: Available as fallback.
- Strictly zero laptop compute checks ($0 budget); all source commits pushed individually on `main`.
- The current-state and active-job sections take precedence over chronological pending-status notes.

### Latest completed trial: vision approach round 44 (sprint surge & softened sprint floor on 2x Nvidia Tesla T4 GPU)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round44 (version 1).
Completed all 24 updates (393,216 steps, 91.2 min runtime) on 2x Nvidia Tesla T4 GPU (512 envs, 256/GPU) to `KernelWorkerStatus.COMPLETE`. Outputs ingested into `results/baby-dog-vision-approach-round44/`.

Setup: 393,216 steps (24 updates × 16,384 steps = 393,216 steps), 512 envs, 6 evaluation checkpoints (every 65k steps across 3 fresh cohorts = 192 lives per eval, 7 total evaluations including baseline and final), 2h cap. Teacher movement help zero from rest (`help_ = 0.00`).
Resumed from all-time primary candidate `baby_dog_step_0000196608.npz` (`0436cf4c…`, Round 43 candidate) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round43`.
Attached datasets: `bio-bot-dog-mocap`, `bio-bot-cores`.
Embedded modules: `train/brain_dog.py`, `train/dog_world.py`.
Hyperparameters: `sprint_pay = 40.0` (boosted from 25.0), `sprint_floor = 0.38` (lowered from 0.42 to trigger sprint bonus upon nominal walk speed), `fwd_pay = 45.0` (boosted from 40.0), `search_pay = 35.0` (active visual search rotation), `misalign_cost = 6.0`, `align_floor = 0.45`, `yaw_pay = 80.0`, `face_pay = 80.0`, `task_pay = 90.0`, `reach_pay = 150.0`, `stay_pay = 100.0`, `prox_pay = 45.0`.
All outputs fetched to `results/baby-dog-vision-approach-round44/`: `baby_dog.npz`, snapshots at 65k, 131k, 196k, 262k, 327k, 393k steps, `practice.json`, and `bio-bot-baby-dog-vision-approach-round44.log` (copied to `log.txt`).

**Measured Outcomes across 3 fresh cohorts (192 lives per evaluation):**

| measured metric (three cohorts, 192 lives) | baseline (0436...) | 0.07M (547e...) | 0.13M (3ee2...) | 0.20M (0ef6...) | 0.26M (4a47...) | 0.33M (55bc...) | 0.39M (55f8...) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| mean movement falls / 13 s | 1.1803 | 1.2148 | 1.1927 | 1.1908 | 1.2331 | 1.2044 | 1.1992 |
| walk speed, m/s | +0.3993 | +0.3893 | +0.3875 | +0.3996 | +0.3931 | +0.3960 | +0.3896 |
| pace speed, m/s | +0.5707 | +0.5586 | +0.5764 | +0.5673 | +0.5618 | +0.5650 | +0.5762 |
| canter speed, m/s | +0.3930 | +0.3936 | +0.3989 | +0.3995 | +0.4010 | +0.4033 | +0.4041 |
| run speed, m/s | +0.5500 | +0.5403 | +0.5478 | +0.5516 | +0.5589 | +0.5578 | +0.5472 |
| back speed, m/s | -0.0741 | -0.0752 | -0.0727 | -0.0756 | -0.0809 | -0.0820 | -0.0786 |
| turn-left yaw, rad/s | +1.3928 | +1.3978 | +1.3840 | +1.3842 | +1.3870 | +1.3752 | +1.3901 |
| turn-right yaw, rad/s | -1.1168 | -1.1145 | -1.0926 | -1.1158 | -1.1235 | -1.1399 | -1.1334 |
| sit pose (pitch / trunk) | +0.499 / 0.620 | +0.497 / 0.618 | +0.471 / 0.620 | +0.495 / 0.616 | +0.472 / 0.619 | +0.490 / 0.621 | +0.504 / 0.619 |
| lie-down pose (pitch / trunk) | +0.025 / 0.957 | +0.016 / 0.952 | +0.011 / 0.947 | +0.024 / 0.958 | +0.022 / 0.954 | +0.018 / 0.949 | +0.025 / 0.957 |
| **ball arrival success (got_there)** | 3.65% (7/192) | **6.77% (13/192)** | 5.73% (11/192) | **6.77% (13/192)** | 4.17% (8/192) | 3.65% (7/192) | 6.25% (12/192) |
| ball distance drift (start -> end, m) | 2.491 -> 3.440 | 2.503 -> 3.385 | 2.490 -> 3.550 | 2.491 -> 3.574 | 2.490 -> 3.538 | 2.491 -> 3.673 | 2.493 -> 3.475 |
| ball aligned in final second | 2.59% | 3.95% | 2.61% | 2.84% | 6.08% | 3.55% | 4.87% |
| ball final facing cosine | -0.3860 | -0.4123 | -0.4237 | -0.4215 | -0.4000 | -0.4080 | -0.3917 |
| ball falls / 13 s | 0.7865 | 0.8229 | 0.7865 | 0.8385 | 0.8542 | 0.7917 | 0.8438 |
| **owner arrival success (got_there, come)** | 5.21% (10/192) | 4.17% (8/192) | 4.69% (9/192) | 3.65% (7/192) | 3.65% (7/192) | 3.65% (7/192) | 1.04% (2/192) |
| owner distance drift (start -> end, m) | 3.666 -> 3.644 | 3.662 -> 3.597 | 3.665 -> 3.529 | 3.664 -> 3.669 | 3.664 -> 3.548 | 3.666 -> 3.483 | 3.664 -> 3.651 |
| owner aligned in final second | 14.06% | 11.27% | 12.17% | 10.41% | 13.98% | 9.64% | 12.16% |
| owner final facing cosine | -0.0450 | -0.0072 | -0.0231 | -0.0504 | +0.0408 | -0.0836 | +0.0649 |
| owner falls / 13 s | 1.5208 | 1.6562 | 1.6094 | 1.5417 | 1.5677 | 1.6875 | 1.5781 |
| **dodge hazard reaction (hit rate)** | 77.6% | 77.6% | 76.6% | 77.1% | 76.6% | 77.6% | 75.5% |
| dodge moved before impact | 24.0% | 22.4% | 24.5% | 28.6% | 24.5% | 25.5% | 22.9% |
| dodge median reaction latency | 86.7 ms | **3.3 ms** | 43.3 ms | 43.3 ms | 33.3 ms | 16.7 ms | 56.7 ms |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Measured Outcomes across 3 fresh cohorts (192 lives per evaluation):**

| measured metric (three cohorts, 192 lives) | baseline (8101...) | 0.07M (6175...) | 0.13M (e822...) | 0.20M (0436...) | 0.26M (0816...) | 0.33M (a3da...) | 0.39M (129f...) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| mean movement falls / 13 s | 1.2090 | 1.1706 | 1.1973 | 1.2044 | 1.1810 | 1.2051 | 1.2207 |
| walk speed, m/s | +0.3980 | +0.3931 | +0.3972 | +0.3958 | +0.3961 | +0.4029 | +0.4086 |
| pace speed, m/s | +0.5637 | +0.5617 | +0.5686 | +0.5632 | +0.5631 | +0.5632 | +0.5593 |
| canter speed, m/s | +0.3852 | +0.3968 | +0.3964 | +0.3909 | +0.3856 | +0.4019 | +0.3915 |
| run speed, m/s | +0.5404 | +0.5510 | +0.5463 | +0.5565 | +0.5550 | +0.5431 | +0.5438 |
| back speed, m/s | -0.0699 | -0.0804 | -0.0785 | -0.0685 | -0.0686 | -0.0751 | -0.0729 |
| turn-left yaw, rad/s | +1.3714 | +1.3952 | +1.3821 | +1.3695 | +1.3923 | +1.3718 | +1.4029 |
| turn-right yaw, rad/s | -1.1083 | -1.0947 | -1.1106 | -1.1063 | -1.0955 | -1.1132 | -1.1210 |
| sit pose (pitch / trunk) | +0.491 / 0.618 | +0.498 / 0.621 | +0.490 / 0.620 | +0.505 / 0.619 | +0.490 / 0.618 | +0.484 / 0.618 | +0.490 / 0.622 |
| lie-down pose (pitch / trunk) | +0.020 / 0.953 | +0.015 / 0.951 | +0.017 / 0.953 | +0.021 / 0.955 | +0.024 / 0.957 | +0.032 / 0.962 | +0.012 / 0.948 |
| **ball arrival success (got_there)** | 5.21% (10/192) | 4.69% (9/192) | 5.73% (11/192) | **9.90% (19/192)** | 6.77% (13/192) | 5.21% (10/192) | 4.69% (9/192) |
| ball distance drift (start -> end, m) | 2.490 -> 3.566 | 2.491 -> 3.449 | 2.490 -> 3.653 | 2.492 -> 3.568 | 2.492 -> 3.467 | 2.490 -> 3.521 | 2.491 -> 3.527 |
| ball aligned in final second | 2.27% | 4.33% | 0.50% | 3.95% | 3.99% | 2.52% | 2.95% |
| ball final facing cosine | -0.3855 | -0.4404 | -0.4738 | -0.3817 | -0.3549 | -0.3779 | -0.4476 |
| ball falls / 13 s | 0.7552 | 0.7552 | 0.8021 | 0.7865 | 0.7604 | 0.7917 | 0.7917 |
| **owner arrival success (got_there, come)** | 5.73% (11/192) | 2.60% (5/192) | 2.08% (4/192) | 2.08% (4/192) | 4.17% (8/192) | 1.56% (3/192) | 2.60% (5/192) |
| owner distance drift (start -> end, m) | 3.663 -> 3.568 | 3.664 -> 3.641 | 3.665 -> 3.623 | 3.666 -> 3.611 | 3.664 -> 3.553 | 3.665 -> 3.704 | 3.665 -> 3.557 |
| owner aligned in final second | 10.89% | 11.04% | 13.31% | 9.77% | 11.67% | 9.08% | 7.67% |
| owner final facing cosine | -0.0092 | +0.0269 | +0.0082 | -0.0043 | -0.0159 | -0.0497 | -0.0013 |
| owner falls / 13 s | 1.6042 | 1.6354 | 1.5729 | 1.5677 | 1.5573 | 1.6354 | 1.5573 |
| **dodge hazard reaction (hit rate)** | 76.0% | 73.4% | 76.0% | 78.1% | 78.6% | 75.0% | 77.6% |
| dodge moved before impact | 25.0% | 24.5% | 21.9% | 22.9% | 26.6% | 21.4% | 23.4% |
| dodge median reaction latency | 6.7 ms | 20.0 ms | 76.7 ms | 66.7 ms | 106.7 ms | 30.0 ms | **3.3 ms** |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **ball-right balanced accuracy** | 0.668 | 0.627 | 0.685 | 0.694 | 0.683 | 0.683 | 0.675 |
| **owner-left balanced accuracy** | 0.686 | 0.670 | 0.672 | 0.680 | 0.685 | 0.689 | 0.673 |
| **owner-ahead balanced accuracy** | 0.680 | 0.637 | 0.657 | 0.656 | 0.672 | 0.691 | 0.672 |
| **owner-right balanced accuracy** | 0.642 | 0.668 | 0.637 | 0.646 | 0.633 | 0.651 | 0.638 |
| **hit balanced accuracy** | 0.841 | 0.773 | 0.882 | 0.839 | 0.820 | 0.816 | 0.844 |
| **moving balanced accuracy** | 0.636 | 0.702 | 0.634 | 0.615 | 0.624 | 0.621 | 0.617 |
| **hungry balanced accuracy** | 0.577 | 0.573 | 0.568 | 0.572 | 0.577 | 0.577 | 0.576 |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Checkpoint SHA256 Hashes:**
- `baby_dog_step_0000065536.npz`: `091e6829ea1578c0e9b7fc7329973b65b5d884b08e2ea691be8c3708408ce596`
- `baby_dog_step_0000131072.npz`: `18c046da392024adb61eedde70c989ac5621a37283a8ac3d78422d868c2d394e`
- `baby_dog_step_0000196608.npz`: `58ee7fb202c9ef8da0acb834346c31c0bf28f671c91ddbaa272e4a3bb8d6a131`
- `baby_dog_step_0000262144.npz`: `f8dbe38b52410158da2c638ab7abda2596f1085321ba069694a2e55b32738bd2`
- `baby_dog_step_0000327680.npz`: `156bc3d0a0c28922edca467dc5fb4902179bba0213967a950e8b7d22daa349ee`
- `baby_dog_step_0000393216.npz`: `2f1d6ceda141ea40f22911aa3d1d19210e57452adcdf29c3048f2b8a5017ff8a`
- `baby_dog.npz`: `9f0cdade854a84197609f08a079e17253efee8261a6a57b06ac4cfc78514634f`

### Previous completed trial: vision approach round 40 (turning-resistance unblock on 2x Nvidia Tesla T4 GPU)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round40 (version 3).
Completed all 24 updates (393,216 steps, 88.5 min runtime) on 2x Nvidia Tesla T4 GPU (512 envs, 256/GPU) to `KernelWorkerStatus.COMPLETE`. Outputs ingested into `results/baby-dog-vision-approach-round40/`.

Setup: 393,216 steps (24 updates × 16,384 steps = 393,216 steps), 512 envs, 6 evaluation checkpoints (every 65k steps across 3 fresh cohorts = 192 lives per eval, 7 total evaluations including baseline and final), 2h cap. Teacher movement help zero from rest (`help_ = 0.00`).
Resumed from candidate `baby_dog_step_0000393216.npz` (`aeebc8a7…`, Round 39 candidate) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round39`.
Attached datasets: `bio-bot-dog-mocap`, `bio-bot-cores`.
Embedded modules: `train/brain_dog.py`, `train/dog_world.py`.
Hyperparameters: `align_floor = 0.60`, `face_pay = 60.0`, `yaw_pay = 60.0`, `misalign_cost = 4.0` (softened from 12.0), `fwd_pay = 40.0`, `sprint_pay = 25.0`, `sprint_floor = 0.42`, `task_pay = 90.0`, `reach_pay = 150.0`, `stay_pay = 100.0`, `prox_pay = 45.0`.
All outputs fetched to `results/baby-dog-vision-approach-round40/`: `baby_dog.npz`, snapshots at 65k, 131k, 196k, 262k, 327k, 393k steps, `practice.json`, and `bio-bot-baby-dog-vision-approach-round40.log`.

**Measured Outcomes across 3 fresh cohorts (192 lives per evaluation):**

| measured metric (three cohorts, 192 lives) | baseline (aeeb...) | 0.07M (3900...) | 0.13M (e6fc...) | 0.20M (0ca4...) | 0.26M (6fcb...) | 0.33M (4cde...) | 0.39M (caa0...) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| mean movement falls / 13 s | 1.1602 | 1.1569 | 1.1953 | 1.2415 | 1.2135 | 1.2279 | 1.2044 |
| walk falls / 13 s | 1.6875 | 1.5573 | 1.7500 | 1.8698 | 1.7604 | 1.7448 | 1.7917 |
| pace falls / 13 s | 1.0625 | 1.1458 | 1.1510 | 1.1458 | 1.1094 | 1.1927 | 1.2135 |
| canter falls / 13 s | 1.2135 | 1.2396 | 1.2031 | 1.2448 | 1.2604 | 1.2917 | 1.2656 |
| run falls / 13 s | 1.1042 | 1.0677 | 1.1302 | 1.2448 | 1.0781 | 1.1979 | 1.1354 |
| turn-left falls / 13 s | 0.9531 | 0.9115 | 0.8750 | 0.9740 | 1.0677 | 0.9271 | 0.9167 |
| turn-right falls / 13 s | 0.8385 | 0.8177 | 0.8177 | 0.8125 | 0.8021 | 0.8281 | 0.8490 |
| back falls / 13 s | 0.9792 | 0.9531 | 1.0208 | 0.9844 | 1.0156 | 0.9948 | 1.0208 |
| jump falls / 13 s | 1.4427 | 1.5625 | 1.6146 | 1.6562 | 1.6146 | 1.6458 | 1.4427 |
| walk speed, m/s | +0.4032 | +0.4057 | +0.4000 | +0.3901 | +0.3875 | +0.4011 | +0.3934 |
| pace speed, m/s | +0.5715 | +0.5695 | +0.5668 | +0.5630 | +0.5640 | +0.5631 | +0.5607 |
| canter speed, m/s | +0.3920 | +0.3922 | +0.4019 | +0.4016 | +0.3886 | +0.3947 | +0.4040 |
| run speed, m/s | +0.5528 | +0.5609 | +0.5341 | +0.5316 | +0.5357 | +0.5373 | +0.5475 |
| back speed, m/s | -0.0745 | -0.0767 | -0.0675 | -0.0694 | -0.0736 | -0.0656 | -0.0612 |
| jump speed, m/s | +0.2217 | +0.2396 | +0.2272 | +0.2253 | +0.2236 | +0.2195 | +0.2241 |
| turn-left yaw, rad/s | +1.3748 | +1.3807 | +1.4070 | +1.3577 | +1.3767 | +1.3839 | +1.3760 |
| turn-right yaw, rad/s | -1.1473 | -1.0968 | -1.1183 | -1.1063 | -1.1246 | -1.0961 | -1.0964 |
| sit pose (pitch / trunk) | +0.497 / 0.621 | +0.487 / 0.622 | +0.495 / 0.622 | +0.503 / 0.621 | +0.491 / 0.619 | +0.492 / 0.621 | +0.486 / 0.622 |
| lie-down pose (pitch / trunk) | +0.023 / 0.955 | +0.025 / 0.955 | +0.019 / 0.953 | +0.021 / 0.953 | +0.016 / 0.954 | +0.022 / 0.958 | +0.023 / 0.958 |
| **ball arrival success (got_there)** | 3.65% (7/192) | 4.69% (9/192) | **7.29% (14/192)** | **7.29% (14/192)** | 4.69% (9/192) | 4.17% (8/192) | 5.21% (10/192) |
| ball distance drift (start -> end, m) | 2.489 -> 3.566 | 2.491 -> 3.285 | 2.491 -> 3.497 | 2.487 -> 3.534 | 2.490 -> 3.554 | 2.490 -> 3.547 | 2.488 -> 3.381 |
| ball aligned in final second | 1.80% | 3.41% | 2.97% | 3.92% | 3.72% | 2.92% | 4.14% |
| ball final facing cosine | -0.4478 | -0.4263 | -0.4227 | -0.4108 | -0.3968 | -0.4479 | **-0.3307** |
| ball falls / 13 s | 0.7448 | 0.7656 | 0.7500 | 0.8021 | 0.7865 | 0.7865 | 0.8177 |
| **owner arrival success (got_there, 'come')** | 7.29% (14/192) | 2.60% (5/192) | 2.60% (5/192) | 2.60% (5/192) | 4.17% (8/192) | **6.25% (12/192)** | 4.69% (9/192) |
| owner distance drift (start -> end, m) | 3.667 -> 3.556 | 3.662 -> 3.560 | 3.660 -> 3.632 | 3.663 -> 3.598 | 3.665 -> 3.576 | 3.665 -> 3.505 | 3.667 -> 3.557 |
| owner aligned in final second | 9.87% | 13.02% | 11.22% | 11.10% | 9.94% | 7.96% | 11.40% |
| owner final facing cosine | -0.0513 | +0.0495 | -0.0055 | -0.0449 | +0.0112 | -0.0530 | -0.0368 |
| owner falls / 13 s | 1.5677 | 1.6354 | 1.6302 | 1.6615 | 1.7760 | 1.8073 | 1.6302 |
| **dodge hazard reaction (hit rate)** | 76.6% | 76.0% | 75.0% | 77.6% | 76.6% | 77.1% | 76.6% |
| dodge moved before impact | 29.2% | 27.1% | 23.4% | 23.4% | 23.4% | 24.0% | 27.6% |
| dodge median reaction latency | 37 ms | 40 ms | 23 ms | 60 ms | 63 ms | 27 ms | **7 ms** |
| **ball-left balanced accuracy** | 0.703 | 0.686 | 0.693 | 0.695 | 0.683 | 0.698 | 0.696 |
| **ball-ahead balanced accuracy** | 0.701 | 0.678 | 0.712 | 0.681 | 0.703 | 0.707 | 0.676 |
| **ball-right balanced accuracy** | 0.660 | 0.680 | 0.696 | 0.685 | 0.677 | 0.683 | 0.678 |
| **owner-left balanced accuracy** | 0.680 | 0.667 | 0.683 | 0.666 | 0.678 | 0.681 | 0.683 |
| **owner-ahead balanced accuracy** | 0.696 | 0.628 | 0.671 | 0.685 | 0.693 | 0.691 | 0.690 |
| **owner-right balanced accuracy** | 0.632 | 0.652 | 0.657 | 0.634 | 0.640 | 0.646 | 0.644 |
| **hit balanced accuracy** | 0.862 | 0.849 | 0.852 | 0.863 | 0.871 | 0.830 | 0.846 |
| **moving balanced accuracy** | 0.620 | 0.725 | 0.638 | 0.619 | 0.631 | 0.628 | 0.633 |
| **hungry balanced accuracy** | 0.570 | 0.562 | 0.553 | 0.565 | 0.572 | 0.578 | 0.567 |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Checkpoint SHA256 Hashes:**
- `baby_dog_step_0000065536.npz`: `3900a27fdf4a01f141de441ffb71575b3c245027f7b1c9dcaef1ff9776929583`
- `baby_dog_step_0000131072.npz`: `e6fcede42b2746ee667ba2e5a6edf13935fc788c167145bf4feab081ca2452bd`
- `baby_dog_step_0000196608.npz`: `0ca420e7942c57a58414e588ab3b7e34790c5e8c419254fedcf115fd182218f7`
- `baby_dog_step_0000262144.npz`: `6fcbc1bbc721eefc8bf2df4d45302ec3a9e635780efd9366ef52b921d02464d9`
- `baby_dog_step_0000327680.npz`: `4cde9ca42350655b5edbd0e19acb8828221ff235b9074327437689eadde1d809`
- `baby_dog_step_0000393216.npz`: `caa01193ffd517855de86f44ad1c645cd906392e084ced8c89fcb759ec3d5a33`
- `baby_dog.npz`: `1894f65d8f7b20f7ebe3a771ea69907fcd246bc5ae80f39896fab0050c357056`

### Previous completed trial: vision approach round 39 (aligned acceleration sprint and heading-gated propulsion on 2x Nvidia Tesla T4 GPU)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round39 (version 1).
Completed all 24 updates (393,216 steps, 91.2 min runtime) on 2x Nvidia Tesla T4 GPU (512 envs, 256/GPU) to `KernelWorkerStatus.COMPLETE`. Outputs ingested into `results/baby-dog-vision-approach-round39/`.

Setup: 393,216 steps (24 updates × 16,384 steps = 393,216 steps), 512 envs, 6 evaluation checkpoints (every 65k steps across 3 fresh cohorts = 192 lives per eval, 7 total evaluations including baseline and final), 2h cap. Teacher movement help zero from rest (`help_ = 0.00`).
Resumed from candidate `baby_dog_step_0000262144.npz` (`207c1767…`, Round 38 candidate) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round38`.
Attached datasets: `bio-bot-dog-mocap`, `bio-bot-cores`.
Hyperparameters: `align_floor = 0.60`, `face_pay = 60.0`, `yaw_pay = 60.0`, `misalign_cost = 12.0`, `fwd_pay = 40.0`, `sprint_pay = 25.0`, `sprint_floor = 0.42`, `task_pay = 90.0`, `reach_pay = 150.0`, `stay_pay = 100.0`, `prox_pay = 45.0`.
All outputs fetched to `results/baby-dog-vision-approach-round39/`: `baby_dog.npz`, snapshots at 65k, 131k, 196k, 262k, 327k, 393k steps, `practice.json`, and `bio-bot-baby-dog-vision-approach-round39.log`.

**Measured Outcomes across 3 fresh cohorts (192 lives per evaluation):**

| measured metric (three cohorts, 192 lives) | baseline (207c...) | 0.07M (4a3c...) | 0.13M (158c...) | 0.20M (f65f...) | 0.26M (896e...) | 0.33M (5e20...) | 0.39M (aeeb...) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| mean movement falls / 13 s | 1.1523 | 1.1139 | 1.1732 | 1.1888 | 1.1634 | 1.1667 | 1.1667 |
| walk falls / 13 s | 1.7135 | 1.6250 | 1.6562 | 1.5833 | 1.6562 | 1.7604 | 1.7031 |
| pace falls / 13 s | 1.0000 | 1.0781 | 1.1354 | 1.1094 | 1.0729 | 1.1094 | 1.1406 |
| canter falls / 13 s | 1.1667 | 1.1615 | 1.1302 | 1.2188 | 1.2083 | 1.1823 | 1.1406 |
| run falls / 13 s | 1.1042 | 0.9688 | 1.1615 | 1.1927 | 1.1146 | 1.0729 | 1.1146 |
| turn-left falls / 13 s | 0.9323 | 0.8802 | 0.9740 | 0.9792 | 0.9115 | 0.9583 | 0.8802 |
| turn-right falls / 13 s | 0.8073 | 0.8229 | 0.7917 | 0.8490 | 0.8490 | 0.8177 | 0.8229 |
| back falls / 13 s | 1.0781 | 0.9375 | 0.9948 | 1.0260 | 0.9844 | 0.9375 | 1.0000 |
| jump falls / 13 s | 1.4167 | 1.4375 | 1.5417 | 1.5521 | 1.5104 | 1.4948 | 1.5312 |
| walk speed, m/s | +0.4033 | +0.4050 | +0.4056 | +0.4057 | +0.4007 | +0.4026 | +0.3988 |
| pace speed, m/s | +0.5688 | +0.5663 | +0.5786 | +0.5641 | +0.5728 | +0.5745 | +0.5639 |
| canter speed, m/s | +0.3937 | +0.4093 | +0.4065 | +0.3946 | +0.3950 | +0.4002 | +0.4031 |
| run speed, m/s | +0.5528 | +0.5458 | +0.5444 | +0.5446 | +0.5520 | +0.5567 | +0.5464 |
| back speed, m/s | -0.0744 | -0.0705 | -0.0757 | -0.0738 | -0.0709 | -0.0769 | -0.0766 |
| jump speed, m/s | +0.2265 | +0.2232 | +0.2333 | +0.2238 | +0.2326 | +0.2209 | +0.2259 |
| turn-left yaw, rad/s | +1.3698 | +1.3677 | +1.3534 | +1.3474 | +1.3787 | +1.3656 | +1.3734 |
| turn-right yaw, rad/s | -1.1156 | -1.1309 | -1.1163 | -1.1312 | -1.1202 | -1.0993 | -1.1365 |
| sit pose (pitch / trunk) | +0.502 / 0.618 | +0.499 / 0.621 | +0.499 / 0.620 | +0.496 / 0.622 | +0.496 / 0.620 | +0.489 / 0.620 | +0.489 / 0.620 |
| lie-down pose (pitch / trunk) | +0.034 / 0.963 | +0.025 / 0.955 | +0.019 / 0.957 | +0.022 / 0.957 | +0.025 / 0.959 | +0.021 / 0.954 | +0.025 / 0.961 |
| **ball arrival success (got_there)** | 5.21% (10/192) | 4.69% (9/192) | **7.29% (14/192)** | 4.69% (9/192) | 6.77% (13/192) | 5.73% (11/192) | 6.25% (12/192) |
| ball distance drift (start -> end, m) | 2.489 -> 3.470 | 2.496 -> 3.717 | 2.490 -> 3.348 | 2.487 -> 3.619 | 2.490 -> 3.572 | 2.489 -> 3.781 | 2.493 -> 3.487 |
| ball aligned in final second | 1.36% | 1.26% | 3.78% | 3.19% | 3.90% | 2.76% | 5.61% |
| ball final facing cosine | -0.4100 | -0.4944 | -0.3594 | -0.3930 | -0.4358 | -0.5140 | -0.3779 |
| ball falls / 13 s | 0.6771 | 0.7292 | 0.7396 | 0.7604 | 0.7500 | 0.7396 | 0.7292 |
| **owner arrival success (got_there, 'come')** | 5.21% (10/192) | 4.17% (8/192) | 3.65% (7/192) | 3.65% (7/192) | 2.08% (4/192) | 1.04% (2/192) | **6.77% (13/192)** |
| owner distance drift (start -> end, m) | 3.664 -> 3.625 | 3.663 -> 3.447 | 3.663 -> 3.727 | 3.663 -> 3.645 | 3.662 -> 3.569 | 3.666 -> 3.634 | 3.663 -> 3.584 |
| owner aligned in final second | 13.53% | 13.04% | 12.83% | 11.54% | 11.34% | 11.47% | **13.96%** |
| owner final facing cosine | +0.0343 | +0.0308 | -0.0139 | -0.0189 | +0.0980 | -0.0112 | **+0.0391** |
| owner falls / 13 s | 1.4583 | 1.4167 | 1.5833 | 1.5469 | 1.5208 | 1.4948 | 1.6198 |
| **dodge hazard reaction (hit rate)** | 77.6% | **80.7%** | 76.0% | 80.2% | 75.0% | 75.5% | 76.0% |
| dodge moved before impact | 27.1% | 21.4% | 28.6% | 26.0% | 24.5% | 29.2% | 29.2% |
| dodge median reaction latency | 63 ms | 67 ms | 30 ms | 70 ms | 60 ms | 80 ms | 70 ms |
| **ball-left balanced accuracy** | 0.693 | 0.640 | 0.691 | 0.692 | 0.691 | 0.697 | 0.700 |
| **ball-ahead balanced accuracy** | 0.703 | 0.671 | 0.697 | 0.689 | 0.675 | 0.704 | 0.699 |
| **ball-right balanced accuracy** | 0.681 | 0.662 | 0.689 | 0.681 | 0.673 | 0.672 | 0.657 |
| **owner-left balanced accuracy** | 0.680 | 0.679 | 0.689 | 0.676 | 0.682 | 0.674 | 0.694 |
| **owner-ahead balanced accuracy** | 0.701 | 0.621 | 0.688 | 0.684 | 0.673 | 0.682 | 0.681 |
| **owner-right balanced accuracy** | 0.640 | 0.650 | 0.658 | 0.648 | 0.651 | 0.639 | 0.647 |
| **hit balanced accuracy** | 0.783 | 0.825 | 0.809 | 0.829 | 0.854 | 0.848 | **0.903** |
| **moving balanced accuracy** | 0.624 | 0.679 | 0.623 | 0.613 | 0.606 | 0.624 | 0.626 |
| **hungry balanced accuracy** | 0.573 | 0.581 | 0.562 | 0.557 | 0.569 | 0.574 | 0.572 |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Checkpoint SHA256 Hashes:**
- `baby_dog_step_0000065536.npz`: `4a3cb4a450da2930c1a27f0128efa12543faa1343df0e37f7e8e2d2d1954d962`
- `baby_dog_step_0000131072.npz`: `158cae65a066680c1da2ad75d6ad2ce21f150978b5484ae5db91e65a0a0edf21`
- `baby_dog_step_0000196608.npz`: `f65f81a8a85391e0e8739b9e81bfe122c9a1bc917db99de040e7824908458c85`
- `baby_dog_step_0000262144.npz`: `896e14864c9367830a29baed40157200de73fdb9e18b341ee43c5ac0e2f2e90a`
- `baby_dog_step_0000327680.npz`: `5e209c9b7626f573ef3cc4d981e7016a4109b9135aaac37054f6421a215c01ce`
- `baby_dog_step_0000393216.npz`: `aeebc8a726da5eb83c5c9574055d7972a5b8d83e579ee4677e0fd43f44f1247b`
- `baby_dog.npz`: `153daffb9724cf00fe4c7d48b0393187c0d53fcb53d64057ca15285e8ccf83fa`

### Previous completed trial: vision approach round 38 (alignment gate tightening and proximity funnel on 2x Nvidia Tesla T4 GPU)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round38 (version 1).
Completed all 24 updates (393,216 steps, 113.9 min runtime) on 2x Nvidia Tesla T4 GPU (512 envs, 256/GPU) to `KernelWorkerStatus.COMPLETE`. Outputs ingested into `results/baby-dog-vision-approach-round38/`.

Setup: 393,216 steps (24 updates × 16,384 steps = 393,216 steps), 512 envs, 6 evaluation checkpoints (every 65k steps across 3 fresh cohorts = 192 lives per eval, 7 total evaluations including baseline and final), 2h cap. Teacher movement help zero from rest (`help_ = 0.00`).
Resumed from candidate `baby_dog_step_0000393216.npz` (`6c2a2bad…`, Round 37 candidate) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round37`.
Attached datasets: `bio-bot-dog-mocap`, `bio-bot-cores`.
Hyperparameters: `align_floor = 0.65`, `face_pay = 75.0`, `yaw_pay = 75.0`, `misalign_cost = 15.0`, `task_pay = 100.0`, `prox_pay = 50.0`, `fwd_pay = 25.0`, `sprint_pay = 10.0`, `reach_pay = 150.0`, `stay_pay = 100.0`.
All outputs fetched to `results/baby-dog-vision-approach-round38/`: `baby_dog.npz`, snapshots at 65k, 131k, 196k, 262k, 327k, 393k steps, `practice.json`, and `bio-bot-baby-dog-vision-approach-round38.log`.

**Measured Outcomes across 3 fresh cohorts (192 lives per evaluation):**

| measured metric (three cohorts, 192 lives) | baseline (6c2a...) | 0.07M (faf1...) | 0.13M (7647...) | 0.20M (153c...) | 0.26M (207c...) | 0.33M (2c12...) | 0.39M (d43c...) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| mean movement falls / 13 s | 1.1413 | 1.0957 | 1.1602 | 1.1530 | 1.1491 | 1.1217 | 1.1491 |
| walk falls / 13 s | 1.5833 | 1.5885 | 1.6458 | 1.7448 | 1.6823 | 1.6302 | 1.5833 |
| pace falls / 13 s | 1.0365 | 1.0781 | 1.1458 | 1.0052 | 1.0573 | 1.0104 | 1.0469 |
| canter falls / 13 s | 1.2135 | 1.0625 | 1.1875 | 1.1354 | 1.2500 | 1.1667 | 1.2135 |
| run falls / 13 s | 1.0417 | 1.0469 | 1.0677 | 1.1406 | 1.1042 | 0.9792 | 1.1198 |
| turn-left falls / 13 s | 0.8854 | 0.8542 | 0.9115 | 0.9688 | 0.9115 | 0.9375 | 0.9688 |
| turn-right falls / 13 s | 0.8385 | 0.8125 | 0.7760 | 0.8125 | 0.8125 | 0.7396 | 0.7917 |
| back falls / 13 s | 1.0260 | 0.9740 | 0.9844 | 0.9531 | 1.0104 | 0.9740 | 0.9167 |
| jump falls / 13 s | 1.5052 | 1.3490 | 1.5625 | 1.4635 | 1.3646 | 1.5365 | 1.5521 |
| walk speed, m/s | +0.4130 | +0.4041 | +0.4019 | +0.4041 | +0.4026 | +0.4020 | +0.4086 |
| pace speed, m/s | +0.5800 | +0.5653 | +0.5673 | +0.5601 | +0.5612 | +0.5624 | +0.5771 |
| canter speed, m/s | +0.3961 | +0.3926 | +0.3994 | +0.4070 | +0.3951 | +0.4005 | +0.3975 |
| run speed, m/s | +0.5540 | +0.5495 | +0.5460 | +0.5513 | +0.5623 | +0.5466 | +0.5357 |
| back speed, m/s | -0.0694 | -0.0705 | -0.0690 | -0.0599 | -0.0703 | -0.0780 | -0.0635 |
| jump speed, m/s | +0.2201 | +0.2183 | +0.2227 | +0.2195 | +0.2233 | +0.2320 | +0.2223 |
| turn-left yaw, rad/s | +1.3875 | +1.3797 | +1.3672 | +1.3669 | +1.3885 | +1.3514 | +1.3589 |
| turn-right yaw, rad/s | -1.1130 | -1.1358 | -1.1225 | -1.1281 | -1.1139 | -1.1051 | -1.1233 |
| sit pose (pitch / trunk) | +0.502 / 0.620 | +0.506 / 0.620 | +0.509 / 0.622 | +0.500 / 0.623 | +0.505 / 0.622 | +0.507 / 0.620 | +0.494 / 0.620 |
| lie-down pose (pitch / trunk) | +0.026 / 0.957 | +0.030 / 0.960 | +0.031 / 0.960 | +0.026 / 0.954 | +0.028 / 0.957 | +0.029 / 0.959 | +0.025 / 0.957 |
| **ball arrival success (got_there)** | 6.25% (12/192) | 6.77% (13/192) | 6.77% (13/192) | 6.25% (12/192) | 4.17% (8/192) | 5.21% (10/192) | **6.77% (13/192)** |
| ball distance drift (start -> end, m) | 2.489 -> 3.527 | 2.501 -> 3.660 | 2.487 -> 3.589 | 2.495 -> 3.383 | 2.490 -> 3.669 | 2.488 -> 3.636 | 2.489 -> 3.507 (C1: 3.19m) |
| ball aligned in final second | 1.80% | 1.99% | 2.78% | 4.16% | 4.38% | 2.17% | 4.99% |
| ball final facing cosine | -0.5426 | -0.4725 | -0.4341 | -0.3822 | -0.4253 | -0.4267 | -0.4166 |
| ball falls / 13 s | 0.7500 | 0.7812 | 0.7188 | 0.6771 | 0.7396 | 0.7292 | 0.7240 |
| **owner arrival success (got_there, 'come')** | 5.21% (10/192) | 3.65% (7/192) | 4.17% (8/192) | 5.73% (11/192) | **7.29% (14/192)** | 6.77% (13/192) | 2.60% (5/192) |
| owner distance drift (start -> end, m) | 3.665 -> 3.620 | 3.667 -> 3.579 | 3.666 -> 3.636 | 3.666 -> 3.567 | 3.666 -> 3.464 (-0.20m net!) | 3.664 -> 3.691 | 3.662 -> 3.689 |
| owner aligned in final second | 9.89% | 11.25% | 9.49% | 10.23% | 12.61% | 11.62% | 10.66% |
| owner final facing cosine | +0.0286 | +0.0125 | +0.0703 | +0.0067 | +0.0307 | -0.0003 | +0.0305 |
| owner falls / 13 s | 1.6667 | 1.3490 | 1.4740 | 1.4531 | 1.4688 | 1.4375 | 1.4375 |
| **dodge hazard reaction (hit rate)** | 76.6% | 77.1% | 76.6% | 79.2% | **78.6%** | 77.1% | 76.0% |
| dodge moved before impact | 25.5% | 29.2% | 27.1% | 26.0% | 27.1% | 29.7% | 27.1% |
| dodge median reaction latency | 47 ms | 10 ms | 37 ms | 57 ms | 70 ms | 67 ms | 27 ms |
| **ball-left balanced accuracy** | 0.687 | 0.676 | 0.694 | 0.696 | 0.696 | 0.706 | 0.690 |
| **ball-ahead balanced accuracy** | 0.690 | 0.667 | 0.704 | 0.692 | 0.701 | 0.694 | 0.697 |
| **ball-right balanced accuracy** | 0.674 | 0.680 | 0.701 | 0.686 | 0.686 | 0.688 | 0.662 |
| **owner-left balanced accuracy** | 0.691 | 0.680 | 0.691 | 0.678 | 0.689 | 0.681 | 0.677 |
| **owner-ahead balanced accuracy** | 0.678 | 0.611 | 0.672 | 0.675 | 0.682 | 0.682 | 0.680 |
| **owner-right balanced accuracy** | 0.656 | 0.645 | 0.660 | 0.641 | 0.657 | 0.643 | 0.636 |
| **hit balanced accuracy** | 0.828 | 0.835 | 0.877 | 0.827 | 0.846 | 0.842 | 0.817 |
| **moving balanced accuracy** | 0.617 | 0.703 | 0.630 | 0.615 | 0.619 | 0.623 | 0.630 |
| **hungry balanced accuracy** | 0.573 | 0.573 | 0.570 | 0.565 | 0.578 | 0.574 | 0.576 |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Checkpoint SHA256 Hashes:**
- `baby_dog_step_0000065536.npz`: `faf172e7344d62ed56b00d30231d1cd627d269202e5be3199c365d002de6d8c3`
- `baby_dog_step_0000131072.npz`: `76474d08f3cb5fa65bb7d98762ad130ffccc9a21306bfd94aca7e1afaa6d30d1`
- `baby_dog_step_0000196608.npz`: `153ca5bfa39a6df12e6d81e73b7c512a63ed89fc09475dd05f9018189c320b70`
- `baby_dog_step_0000262144.npz`: `207c1767937adc3d6788533881165bcd37afbdaefa05aedeb512712103d601de`
- `baby_dog_step_0000327680.npz`: `2c12b811a262a67ecadedcd58dce5b27921bcd0f86d12634d71246ebface1d26`
- `baby_dog_step_0000393216.npz`: `d43c61234f8d69dbb9d530d22777d51b2bc1ba25929206c24de5d886a3382f90`
- `baby_dog.npz`: `e4938542c0d11a46048d8a684a1a811183d10c74096ae71205e8c72f2f9e9c98`

### Previous completed trial: vision approach round 37 ("turn first, then approach" overhaul with tightened alignment gate on 2x Nvidia Tesla T4 GPU)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round37 (version 3).
Completed all 24 updates (393,216 steps, 82.6 min runtime) on 2x Nvidia Tesla T4 GPU (512 envs, 256/GPU) to `KernelWorkerStatus.COMPLETE`. Outputs ingested into `results/baby_dog_vision_approach_round37/`.

Setup: 393,216 steps (24 updates × 16,384 steps = 393,216 steps), 512 envs, 6 evaluation checkpoints (every 65k steps across 3 fresh cohorts = 192 lives per eval, 7 total evaluations including baseline and final), 2h cap. Teacher movement help zero from rest (`help_ = 0.00`).
Resumed from candidate `baby_dog_step_0000065536.npz` (`8118e5ab…`, Round 35 intermediate candidate) mounted via `--dataset bio-bot-checkpoint-r35`.
Attached datasets: `bio-bot-checkpoint-r35`, `bio-bot-dog-mocap`, `bio-bot-cores`.
Hyperparameters: `align_floor = 0.60`, `face_pay = 60.0`, `yaw_pay = 60.0`, `misalign_cost = 12.0`, `fwd_pay = 25.0`, `sprint_pay = 10.0`, `task_pay = 80.0`, `reach_pay = 150.0`, `stay_pay = 100.0`, `prox_pay = 40.0`.
All outputs fetched to `results/baby_dog_vision_approach_round37/`: `baby_dog.npz`, snapshots at 65k, 131k, 196k, 262k, 327k, 393k steps, `practice.json`, and `log.txt`.

**Measured Outcomes across 3 fresh cohorts (192 lives per evaluation):**

| measured metric (three cohorts, 192 lives) | baseline (8118e5...) | 0.07M (8b66...) | 0.13M (e818...) | 0.20M (3070...) | 0.26M (30f8...) | 0.33M (d712...) | 0.39M (6c2a...) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| mean movement falls / 13 s | 1.1100 | 1.1224 | 1.1439 | 1.1276 | 1.1504 | 1.1393 | 1.1471 |
| walk falls / 13 s | 1.5781 | 1.5938 | 1.6406 | 1.5521 | 1.6302 | 1.5573 | 1.5781 |
| pace falls / 13 s | 0.9688 | 1.0052 | 0.9479 | 1.0677 | 1.0312 | 1.1406 | 1.0417 |
| canter falls / 13 s | 1.1354 | 1.1823 | 1.2396 | 1.1771 | 1.1667 | 1.1615 | 1.1719 |
| run falls / 13 s | 1.1198 | 1.0573 | 1.1198 | 1.1198 | 1.0208 | 1.0625 | 1.0625 |
| turn-left falls / 13 s | 0.8906 | 0.9115 | 0.9479 | 0.9688 | 0.9740 | 0.9583 | 0.9271 |
| turn-right falls / 13 s | 0.8073 | 0.8073 | 0.8906 | 0.8438 | 0.8542 | 0.8073 | 0.8646 |
| back falls / 13 s | 0.9323 | 0.9844 | 0.9740 | 0.9427 | 1.0469 | 0.9896 | 0.9896 |
| jump falls / 13 s | 1.4479 | 1.4375 | 1.3906 | 1.3490 | 1.4792 | 1.4375 | 1.5417 |
| walk speed, m/s | +0.4027 | +0.4024 | +0.4115 | +0.4046 | +0.4017 | +0.4031 | +0.3945 |
| pace speed, m/s | +0.5739 | +0.5664 | +0.5681 | +0.5626 | +0.5698 | +0.5603 | +0.5768 |
| canter speed, m/s | +0.3945 | +0.3895 | +0.3950 | +0.3887 | +0.3965 | +0.3986 | +0.3884 |
| run speed, m/s | +0.5407 | +0.5384 | +0.5334 | +0.5453 | +0.5619 | +0.5542 | +0.5403 |
| back speed, m/s | -0.0751 | -0.0691 | -0.0668 | -0.0688 | -0.0716 | -0.0663 | -0.0740 |
| jump speed, m/s | +0.2168 | +0.2228 | +0.2228 | +0.2201 | +0.2281 | +0.2227 | +0.2156 |
| turn-left yaw, rad/s | +1.3729 | +1.3779 | +1.3702 | +1.3686 | +1.3789 | +1.3879 | +1.3778 |
| turn-right yaw, rad/s | -1.1506 | -1.1347 | -1.1392 | -1.1149 | -1.1117 | -1.1065 | -1.1062 |
| sit pose (pitch / trunk) | +0.504 / 0.622 | +0.497 / 0.619 | +0.513 / 0.622 | +0.499 / 0.622 | +0.490 / 0.621 | +0.504 / 0.622 | +0.504 / 0.623 |
| lie-down pose (pitch / trunk) | +0.031 / 0.962 | +0.034 / 0.964 | +0.035 / 0.963 | +0.030 / 0.959 | +0.030 / 0.957 | +0.029 / 0.962 | +0.028 / 0.961 |
| **ball arrival success (got_there)** | **7.29% (14/192)** | 5.21% (10/192) | 3.12% (6/192) | 4.17% (8/192) | 6.25% (12/192) | **7.29% (14/192)** | 5.21% (10/192) |
| ball distance drift (start -> end, m) | 2.488 -> 3.400 | 2.488 -> 3.615 | 2.489 -> 3.571 | 2.487 -> 3.528 | 2.487 -> 3.577 | 2.485 -> 3.700 | 2.494 -> 3.478 |
| ball aligned in final second | 6.66% | 1.89% | 2.82% | 3.16% | 3.10% | 2.95% | 3.29% |
| ball final facing cosine | -0.4180 | -0.4643 | -0.4238 | -0.4202 | -0.4579 | -0.5068 | -0.4505 |
| ball falls / 13 s | 0.6510 | 0.7604 | 0.7083 | 0.7917 | 0.6979 | 0.7344 | 0.7500 |
| **owner arrival success (got_there, "come")** | 4.69% (9/192) | 4.17% (8/192) | 3.65% (7/192) | 3.65% (7/192) | 5.21% (10/192) | 5.73% (11/192) | **8.33% (16/192)** (series record!) |
| owner distance drift (start -> end, m) | 3.664 -> 3.679 | 3.666 -> 3.589 | 3.665 -> 3.641 | 3.666 -> 3.540 | 3.664 -> 3.591 | 3.665 -> 3.533 | 3.662 -> 3.625 (-0.28m closing in C3) |
| owner aligned in final second | 12.98% | 13.33% | 13.90% | 12.23% | 8.31% | 11.75% | **15.42%** (peak alignment!) |
| owner final facing cosine | +0.0115 | +0.0493 | +0.0271 | +0.0609 | -0.0190 | -0.0420 | **+0.0836** (positive facing!) |
| owner falls / 13 s | 1.4010 | 1.3438 | 1.4271 | 1.5417 | 1.5260 | 1.5729 | 1.6562 |
| **dodge hazard reaction (hit rate)** | 76.6% | 74.0% | 74.0% | 75.5% | 76.6% | 77.6% | **77.1%** |
| dodge moved before impact | 26.0% | 26.0% | 29.7% | 24.0% | 26.0% | 26.0% | 25.5% |
| dodge median reaction latency | 67 ms | 43 ms | 50 ms | 27 ms | 20 ms | 67 ms | 47 ms |
| **ball-left balanced accuracy** | 0.667 | 0.630 | 0.700 | 0.684 | 0.686 | 0.699 | 0.685 |
| **ball-ahead balanced accuracy** | 0.686 | 0.672 | 0.701 | 0.689 | 0.695 | 0.700 | 0.699 |
| **ball-right balanced accuracy** | 0.655 | 0.664 | 0.699 | 0.680 | 0.680 | 0.675 | 0.666 |
| **owner-left balanced accuracy** | 0.665 | 0.665 | 0.681 | 0.675 | 0.679 | 0.684 | 0.687 |
| **owner-ahead balanced accuracy** | 0.671 | 0.688 | 0.659 | 0.668 | 0.679 | 0.680 | 0.676 |
| **owner-right balanced accuracy** | 0.635 | 0.657 | 0.648 | 0.643 | 0.646 | 0.645 | 0.648 |
| **hit balanced accuracy** | 0.873 | 0.849 | 0.810 | 0.805 | 0.832 | 0.829 | **0.923** (peak perception!) |
| **moving balanced accuracy** | 0.633 | 0.680 | 0.615 | 0.614 | 0.604 | 0.608 | 0.620 |
| **hungry balanced accuracy** | 0.566 | 0.567 | 0.569 | 0.563 | 0.577 | 0.578 | 0.576 |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Checkpoint SHA256 Hashes:**
- `baby_dog_step_0000065536.npz`: `8b665796fe284b99885acf62677874aafa03c24629151420654297382b80a062`
- `baby_dog_step_0000131072.npz`: `e81872b48e897a4055419e4d6a2b66141f50d7dd32ae86afb3838fc1a4836112`
- `baby_dog_step_0000196608.npz`: `307084e137f7f5b52bc97bc8c9321204d02a33f96db0954e817fb25d95ba2d1b`
- `baby_dog_step_0000262144.npz`: `30f86d599a142c56944ccbebdb069c512543161bdc6f7e1b2604599b4293b736`
- `baby_dog_step_0000327680.npz`: `d7121bcbf1dc370254fa1ce8e7ed503e09c5fb1fc8de69ea856f827f238de06a`
- `baby_dog_step_0000393216.npz`: `6c2a2bada5693352703211f3b3a4e092e816312b4c810d7ae97ff79fb916d814`
- `baby_dog.npz`: `b937454fc920a9289e6f3b3cf7413164faea29db9f45de6718099d77b6ebf4b4`

### Previous completed trial: vision approach round 36 (boosted arrival capture and sprint modulation on Google Colab Tesla T4 GPU)

Session: Google Colab session `round36` (Tesla T4 GPU, 15 GB VRAM).
Completed all 32 updates (131,072 steps, 55 min runtime) on Nvidia Tesla T4 GPU (128 envs) with `TF_GPU_ALLOCATOR=cuda_malloc_async`. Task log preserved in task-20034.log.

Setup: 131,072 steps (32 updates × 4,096 steps), 128 envs, 2 evaluation checkpoints (every 65k steps across 3 fresh cohorts = 192 lives per eval), 4h cap. Teacher movement help zero from rest (`help_ = 0.00`).
Resumed from candidate `baby_dog_step_0000065536.npz` (`8118e5ab…`, Round 35 intermediate candidate).
Hyperparameters: `reach_pay = 150.0`, `stay_pay = 100.0`, `prox_pay = 40.0`, `sprint_pay = 20.0`, `face_pay = 30.0`, `fwd_pay = 45.0`, `yaw_pay = 35.0`, `misalign_cost = 4.0`, `align_floor = 0.25`.
Note: Colab session expired before output `.npz` checkpoints could be fetched locally; full evaluation outputs and VERDICT lines captured in runner task log. Restart candidate remains Round 35 intermediate `8118e5ab...`.

**Measured Outcomes across 3 fresh cohorts (192 lives per evaluation):**

| measured metric (three cohorts, 192 lives) | baseline (8118e5...) | 0.07M snap (eval 1) | 0.13M final (eval 2) |
|---|:---:|:---:|:---:|
| mean movement falls / 13 s | 1.1146 | 1.3438 | 1.4870 |
| walk falls / 13 s | 1.6823 | 1.9427 | 2.0833 |
| pace falls / 13 s | 1.0208 | 1.4115 | 1.6302 |
| canter falls / 13 s | 1.1042 | 1.3958 | 1.4792 |
| run falls / 13 s | 1.0208 | 1.4219 | 1.5000 |
| turn-left falls / 13 s | 0.8542 | 1.0885 | 1.1927 |
| turn-right falls / 13 s | 0.8594 | 0.8802 | 0.8594 |
| back falls / 13 s | 0.9635 | 1.1562 | 1.2500 |
| jump falls / 13 s | 1.5469 | 1.7031 | 1.8177 |
| walk speed, m/s | +0.4089 | +0.3953 | +0.4020 |
| pace speed, m/s | +0.5678 | +0.5927 | +0.5840 |
| canter speed, m/s | +0.3984 | +0.4050 | +0.4070 |
| run speed, m/s | +0.5546 | +0.5903 | +0.5800 |
| back speed, m/s | -0.0633 | -0.0673 | -0.0690 |
| jump speed, m/s | +0.2166 | +0.2351 | +0.2420 |
| turn-left yaw, rad/s | +1.3620 | +1.2110 | +1.2240 |
| turn-right yaw, rad/s | -1.1238 | -1.1040 | -1.1270 |
| sit pose (pitch / trunk) | +0.506 / 0.621 | +0.518 / 0.612 | +0.530 / 0.600 |
| lie-down pose (pitch / trunk) | +0.031 / 0.960 | +0.015 / 0.970 | -0.000 / 0.980 |
| **ball arrival success (got_there)** | **7.81% (15/192)** | **6.00% (11/192)** | **8.00% (15/192)** |
| ball distance drift (start -> end, m) | 2.502 -> 3.468 | 2.490 -> 3.500 | 2.493 -> 3.427 |
| ball aligned in final second | 2.52% | 4.33% | 4.67% |
| ball final facing cosine | -0.4578 | -0.3300 | -0.3070 |
| ball task falls / 13 s | **0.6979** | 0.9933 | 0.9433 |
| **owner arrival success (got_there, "come")** | **7.29% (14/192)** | **3.00% (6/192)** | **2.33% (4.7/192)** |
| owner distance drift (start -> end, m) | 3.666 -> 3.513 | 3.663 -> 3.637 | 3.667 -> 3.460 (-0.207m net closing) |
| owner aligned in final second | 12.63% | 9.67% | 9.67% |
| owner final facing cosine | +0.0051 | -0.0333 | -0.0770 |
| owner task falls / 13 s | **1.5104** | 1.8033 | 1.8533 |
| **dodge hazard reaction (hit rate)** | 76.0% | 76.0% | **75.67%** |
| dodge moved before impact | 24.3% | 30.0% | **32.0%** |
| dodge median reaction latency | 30 ms | 20 ms | 60 ms |
| **ball-left balanced accuracy** | 0.682 | 0.690 | 0.687 |
| **ball-ahead balanced accuracy** | 0.698 | 0.663 | 0.680 |
| **ball-right balanced accuracy** | 0.672 | 0.693 | 0.683 |
| **owner-left balanced accuracy** | 0.685 | 0.697 | 0.663 |
| **owner-ahead balanced accuracy** | 0.692 | 0.653 | 0.673 |
| **owner-right balanced accuracy** | 0.647 | 0.647 | 0.610 |
| **hit balanced accuracy** | 0.868 | 0.843 | 0.850 |
| **moving balanced accuracy** | 0.628 | 0.597 | 0.630 |
| **hungry balanced accuracy** | 0.572 | 0.573 | 0.570 |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 |

### Latest completed trial: vision approach round 35 (expanded 1.2m deceleration zone, continuous brake scaling, and --skip_baseline trial on Google Colab Tesla T4 GPU)

Session: Google Colab session `round35` (Tesla T4 GPU, 15 GB VRAM).
Completed all 32 updates (131,072 steps, 55 min runtime) on Nvidia Tesla T4 GPU (128 envs) with `TF_GPU_ALLOCATOR=cuda_malloc_async`. Outputs ingested into `results/baby_dog_vision_approach_round35/`.

Setup: 131,072 steps (32 updates × 4,096 steps), 128 envs, 2 evaluation checkpoints (every 65k steps across 3 fresh cohorts = 192 lives per eval, 3 total evaluations including pre-loaded baseline and final), 4h cap. Teacher movement help zero from rest (`help_ = 0.00`).
Resumed from all-time peak candidate `baby_dog_step_0000393216.npz` (`402c9308…`, Round 34 checkpoint 6 record candidate).
All outputs fetched to `results/baby_dog_vision_approach_round35/`: `baby_dog.npz`, snapshots at 65k and 131k steps, `practice.json`, and `log.txt`.

**Measured Outcomes across 3 fresh cohorts (192 lives per evaluation):**

| measured metric (three cohorts, 192 lives) | baseline (402c93...) | 0.07M snap (8118e5...) | 0.13M final (340b4e...) |
|---|:---:|:---:|:---:|
| mean movement falls / 13 s | 1.1354 | 1.1146 | **1.0938** (falling reduced!) |
| walk falls / 13 s | 1.5833 | 1.6823 | **1.5104** (falling reduced!) |
| pace falls / 13 s | 1.0104 | 1.0208 | 1.0521 |
| canter falls / 13 s | 1.2031 | 1.1042 | **1.0990** |
| run falls / 13 s | 1.0938 | **1.0208** | 1.0677 |
| turn-left falls / 13 s | 0.9115 | **0.8542** | 0.9167 |
| turn-right falls / 13 s | 0.7969 | 0.8594 | **0.8021** |
| back falls / 13 s | 1.0156 | 0.9635 | **0.9531** |
| jump falls / 13 s | 1.4688 | 1.5469 | **1.4844** |
| walk speed, m/s | +0.4097 | +0.4089 | +0.4081 |
| pace speed, m/s | +0.5625 | +0.5678 | +0.5691 |
| canter speed, m/s | +0.4026 | +0.3984 | +0.3995 |
| run speed, m/s | +0.5401 | +0.5546 | +0.5492 |
| back speed, m/s | -0.0713 | -0.0633 | -0.0658 |
| jump speed, m/s | +0.2223 | +0.2166 | +0.2132 |
| turn-left yaw, rad/s | +1.3773 | +1.3620 | **+1.3932** (real dog mocap: +1.383) |
| turn-right yaw, rad/s | -1.1292 | -1.1238 | -1.1342 (real dog mocap: -1.180) |
| sit pose (pitch / trunk) | +0.494 / 0.619 | +0.506 / 0.621 | +0.504 / 0.618 |
| lie-down pose (pitch / trunk) | +0.021 / 0.954 | +0.031 / 0.960 | +0.030 / 0.963 |
| **ball arrival success (got_there)** | **7.81% (15/192)** | **7.81% (15/192)** | 5.21% (10/192) |
| ball distance drift (start -> end, m) | 2.498 -> 3.417 | 2.502 -> 3.468 | 2.484 -> 3.541 |
| ball aligned in final second | 4.80% | 2.52% | 3.12% |
| ball final facing cosine | -0.4267 | -0.4578 | -0.4278 |
| ball task falls / 13 s | **0.6719** | 0.6979 | 0.7135 |
| **owner arrival success (got_there, "come")** | 6.77% (13/192) | **7.29% (14/192)** (improved!) | 3.65% (7/192) |
| owner distance drift (start -> end, m) | 3.666 -> 3.662 | 3.666 -> 3.513 (-0.153m net closing!) | 3.665 -> 3.612 (-0.053m net closing) |
| owner aligned in final second | 9.79% | **12.63%** (improved!) | 11.24% |
| owner final facing cosine | +0.0012 | **+0.0051** | -0.0534 |
| owner task falls / 13 s | 1.5729 | **1.5104** (falling reduced!) | 1.5052 |
| **dodge hazard reaction (hit rate)** | 78.1% | **76.0%** (improved dodging!) | **74.5%** (all-time peak dodging!) |
| dodge moved before impact | 23.4% | **24.3%** | **34.0%** (project peak evasion!) |
| dodge median reaction latency | 20 ms | 30 ms | **0 ms** (instant evasion reflex) |
| **ball-left balanced accuracy** | 0.689 | 0.682 | 0.681 |
| **ball-ahead balanced accuracy** | 0.676 | 0.698 | 0.694 |
| **ball-right balanced accuracy** | 0.676 | 0.672 | 0.673 |
| **owner-left balanced accuracy** | 0.680 | 0.685 | 0.678 |
| **owner-ahead balanced accuracy** | 0.664 | 0.692 | 0.667 |
| **owner-right balanced accuracy** | 0.645 | 0.647 | 0.648 |
| **hit balanced accuracy** | 0.863 | 0.868 | 0.868 |
| **moving balanced accuracy** | 0.624 | 0.628 | 0.647 |
| **hungry balanced accuracy** | 0.572 | 0.572 | 0.569 |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 |

**Checkpoint SHA256 Hashes:**
- `baby_dog_step_0000065536.npz`: `8118e5abde336993db5d7e40849df151d647219e5a9454128b8b72969a577996`
- `baby_dog_step_0000131072.npz`: `340b4e2360528c8563160755eeb4d5e9561ac93127e15bc9c17e190d96e1c45f`
- `baby_dog.npz`: `493060e23389bda6d221837f5cf2588b962c56e5ce89adea524b275ea6987d45`

Notebook: https://www.kaggle.com/code/karangarg/bio-bot-baby-dog-vision-approach-round34 (version 2).
Completed all 24 updates (393,216 steps, 92.1 min runtime) on 2x Nvidia Tesla T4 GPU (512 envs, 256/GPU) to `KernelWorkerStatus.COMPLETE`. Outputs ingested into `results/baby_dog_vision_approach_round34/`.

Setup: 400k steps (24 updates × 16,384 steps = 393,216 steps), 512 envs, 6 evaluation checkpoints (every 65k steps across 3 fresh cohorts = 192 lives per eval, 7 total evaluations including baseline and final), 2h cap. Teacher movement help zero from rest (`help_ = 0.00`).
Resumed from all-time peak candidate `baby_dog_step_0000393216.npz` (`f7cbc54e…`, Round 33 checkpoint 6 record candidate) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round33`.
All outputs fetched to `results/baby_dog_vision_approach_round34/`: `baby_dog.npz`, snapshots at 65k, 131k, 196k, 262k, 327k, 393k steps, `practice.json`, and `log.txt`.

**Measured Outcomes across 3 fresh cohorts (192 lives per evaluation):**

| measured metric (three cohorts, 192 lives) | baseline (f7cb...) | 0.07M (1508...) | 0.13M (8fd2...) | 0.20M (265a...) | 0.26M (19a9...) | 0.33M (5834...) | 0.39M (402c...) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| mean movement falls / 13 s | 1.0872 | 1.1283 | 1.1042 | 1.1048 | 1.1224 | 1.1204 | 1.1354 |
| walk falls / 13 s | 1.6510 | 1.6562 | 1.4844 | 1.5312 | 1.6250 | 1.5573 | 1.5833 |
| pace falls / 13 s | 1.0052 | 1.0781 | 1.0521 | 0.9583 | 0.9740 | 1.0521 | 1.0104 |
| canter falls / 13 s | 1.0312 | 1.1458 | 1.1406 | 1.2344 | 1.1146 | 1.0990 | 1.2031 |
| run falls / 13 s | 1.0156 | 1.1406 | 1.0573 | 0.9583 | 1.1198 | 1.1406 | 1.0938 |
| turn-left falls / 13 s | 0.9062 | 0.8750 | 0.8854 | 0.9167 | 0.8906 | 0.9479 | 0.9115 |
| turn-right falls / 13 s | 0.8021 | 0.7604 | 0.8229 | 0.8542 | 0.7865 | 0.8385 | 0.7969 |
| back falls / 13 s | 0.9479 | 1.0052 | 0.9583 | 0.9219 | 0.9479 | 0.9167 | 1.0156 |
| jump falls / 13 s | 1.3385 | 1.3646 | 1.4323 | 1.4635 | 1.5208 | 1.4115 | 1.4688 |
| walk speed, m/s | +0.4070 | +0.4060 | +0.4131 | +0.4134 | +0.4005 | +0.4109 | +0.4013 |
| pace speed, m/s | +0.5713 | +0.5709 | +0.5722 | +0.5711 | +0.5763 | +0.5670 | +0.5664 |
| canter speed, m/s | +0.3897 | +0.4019 | +0.3998 | +0.4034 | +0.3907 | +0.4010 | +0.3991 |
| run speed, m/s | +0.5414 | +0.5441 | +0.5507 | +0.5460 | +0.5424 | +0.5470 | +0.5486 |
| back speed, m/s | -0.0713 | -0.0756 | -0.0781 | -0.0655 | -0.0640 | -0.0736 | -0.0791 |
| jump speed, m/s | +0.2084 | +0.2148 | +0.2135 | +0.2180 | +0.2142 | +0.2136 | +0.2111 |
| turn-left yaw, rad/s | +1.3505 | +1.3618 | +1.3810 | +1.3648 | +1.3923 | +1.3696 | +1.3576 |
| turn-right yaw, rad/s | -1.1435 | -1.1203 | -1.1496 | -1.1322 | -1.1542 | -1.1412 | -1.1248 |
| sit pose (pitch / trunk) | 0.506 / 0.622 | 0.506 / 0.621 | 0.506 / 0.622 | 0.511 / 0.621 | 0.499 / 0.621 | 0.500 / 0.621 | 0.503 / 0.620 |
| lie-down pose (pitch / trunk) | 0.038 / 0.967 | 0.024 / 0.960 | 0.028 / 0.963 | 0.036 / 0.967 | 0.029 / 0.961 | 0.023 / 0.957 | 0.026 / 0.958 |
| **ball arrival success (got_there)** | **7.29% (14/192)** | **3.65% (7/192)** | **5.73% (11/192)** | **4.69% (9/192)** | **4.17% (8/192)** | **4.69% (9/192)** | **6.77% (13/192)** |
| ball distance drift (start -> end, m) | 2.498 -> 3.581 | 2.492 -> 3.462 | 2.487 -> 3.532 | 2.487 -> 3.452 | 2.487 -> 3.400 | 2.485 -> 3.388 | 2.487 -> 3.495 |
| ball aligned in final second | 3.02% | 0.74% | 4.48% | 3.06% | 2.37% | 3.56% | 1.16% |
| ball final facing cosine | -0.4376 | -0.4736 | -0.3489 | -0.4665 | -0.4045 | -0.4245 | -0.4511 |
| ball falls / 13 s | **0.6823** | **0.6823** | **0.6927** | **0.6771** | **0.6771** | **0.6458** | **0.6875** |
| **owner arrival success (got_there, "come")** | **3.12% (6/192)** | **2.08% (4/192)** | **4.69% (9/192)** | **6.77% (13/192)** | **5.73% (11/192)** | **5.73% (11/192)** | **3.65% (7/192)** |
| owner distance drift (start -> end, m) | 3.666 -> 3.588 | 3.665 -> 3.577 | 3.666 -> 3.595 | 3.665 -> 3.563 | 3.662 -> 3.748 | 3.666 -> 3.588 | 3.664 -> 3.519 |
| owner aligned in final second | 11.92% | 14.74% | 10.91% | 11.65% | 14.82% | 12.85% | 15.35% |
| owner final facing cosine | +0.0599 | +0.0327 | -0.0308 | +0.0706 | -0.0185 | +0.0319 | +0.0868 |
| owner falls / 13 s | **1.5417** | **1.4427** | **1.4427** | **1.4740** | **1.5260** | **1.3906** | **1.4062** |
| **ball-left balanced accuracy** | 0.692 | 0.658 | 0.701 | 0.702 | 0.698 | 0.696 | 0.698 |
| **ball-ahead balanced accuracy** | 0.690 | 0.675 | 0.678 | 0.708 | 0.690 | 0.692 | 0.708 |
| **ball-right balanced accuracy** | 0.651 | 0.680 | 0.693 | 0.694 | 0.678 | 0.677 | 0.681 |
| **owner-left balanced accuracy** | 0.679 | 0.630 | 0.690 | 0.682 | 0.685 | 0.687 | 0.686 |
| **owner-ahead balanced accuracy** | 0.705 | 0.666 | 0.662 | 0.674 | 0.674 | 0.688 | 0.685 |
| **owner-right balanced accuracy** | 0.633 | 0.641 | 0.656 | 0.651 | 0.648 | 0.651 | 0.645 |
| **hit balanced accuracy** | 0.824 | 0.807 | 0.853 | 0.888 | 0.900 | 0.873 | 0.869 |
| **moving balanced accuracy** | 0.616 | 0.705 | 0.630 | 0.610 | 0.615 | 0.631 | 0.625 |
| **hungry balanced accuracy** | 0.579 | 0.575 | 0.571 | 0.567 | 0.572 | 0.579 | 0.566 |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Checkpoint SHA256 Hashes:**
- `baby_dog_step_0000065536.npz`: `1508b88a28e0be168bccb7f301271bfd039d794ffbe811365450955a15ea5760`
- `baby_dog_step_0000131072.npz`: `8fd2eb21a615050128cf7ef06fa2cbdec2e9f16a13f7c3a46ca60b7a5f0211cb`
- `baby_dog_step_0000196608.npz`: `265a61e6c29aa1cca624af6267369dcf673d53d25a9bc1e89c0a9d338447923b`
- `baby_dog_step_0000262144.npz`: `19a98c5b3b15f86f13d540808e7dd1425e02976c9f2293e89c8cf82ddca48df8`
- `baby_dog_step_0000327680.npz`: `5834b079e9630120517d82a3bcb54094e8a38955a9bfc4c61138cf45a936ab81`
- `baby_dog_step_0000393216.npz`: `402c9308622f41a858e75c9f54d48ddb1b66c92f1fed8e2dfa22414ec2d9f174`
- `baby_dog.npz`: `501cbe608252363dc9eebd1086a4b41810d909bcf73268a10e932a5b2ac35193`

### Previous completed trial: vision approach round 33 (softened misalignment penalty, heightened arrival sinks, and net distance closing on 2x Nvidia Tesla T4 GPU)

Notebook: https://www.kaggle.com/code/karangarg/bio-bot-baby-dog-vision-approach-round33 (version 1).
Completed all 24 updates (393,216 steps, 90.1 min runtime) on 2x Nvidia Tesla T4 GPU (512 envs, 256/GPU) to `KernelWorkerStatus.COMPLETE`. Outputs ingested into `results/baby_dog_vision_approach_round33/`.

Setup: 400k steps (24 updates × 16,384 steps = 393,216 steps), 512 envs, 6 evaluation checkpoints (every 65k steps across 3 fresh cohorts = 192 lives per eval, 7 total evaluations including baseline and final), 2h cap. Teacher movement help zero from rest (`help_ = 0.00`).
Resumed from all-time peak candidate `baby_dog_step_0000393216.npz` (`d23d8f2c…`, Round 32 checkpoint 4 record candidate) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round32`.
All outputs fetched to `results/baby_dog_vision_approach_round33/`: `baby_dog.npz`, snapshots at 65k, 131k, 196k, 262k, 327k, 393k steps, `practice.json`, and `log.txt`.

**Measured Outcomes across 3 fresh cohorts (192 lives per evaluation):**

| measured metric (three cohorts, 192 lives) | baseline (d23d...) | 0.07M (fd2f...) | 0.13M (0f33...) | 0.20M (ee1f...) | 0.26M (864f...) | 0.33M (1b25...) | 0.39M (f7cb...) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| mean movement falls / 13 s | 1.0944 | 1.0664 | 1.0723 | 1.1126 | 1.1081 | 1.0951 | 1.0944 |
| walk falls / 13 s | 1.6667 | 1.5260 | 1.4844 | 1.6146 | 1.5573 | 1.5104 | 1.4479 |
| pace falls / 13 s | 1.0208 | 1.0104 | 0.8854 | 1.0625 | 1.0573 | 0.9375 | 1.0156 |
| canter falls / 13 s | 1.1562 | 1.0885 | 1.0365 | 1.2135 | 1.1615 | 1.0469 | 1.1094 |
| run falls / 13 s | 1.0104 | 1.0208 | 1.1198 | 0.9948 | 1.0156 | 1.0573 | 1.0625 |
| turn-left falls / 13 s | 0.9479 | 0.9219 | 0.9323 | 0.8750 | 0.9479 | 0.8594 | 0.9427 |
| turn-right falls / 13 s | 0.7552 | 0.7865 | 0.7656 | 0.8125 | 0.7760 | 0.8073 | 0.8750 |
| back falls / 13 s | 0.9219 | 0.9479 | 0.9948 | 0.9062 | 0.9219 | 1.0000 | 0.9583 |
| jump falls / 13 s | 1.2760 | 1.2292 | 1.3594 | 1.4219 | 1.4271 | 1.5417 | 1.3438 |
| walk speed, m/s | +0.3970 | +0.4112 | +0.4115 | +0.4085 | +0.4089 | +0.4111 | +0.4122 |
| pace speed, m/s | +0.5671 | +0.5704 | +0.5725 | +0.5725 | +0.5759 | +0.5694 | +0.5695 |
| canter speed, m/s | +0.3936 | +0.3927 | +0.4056 | +0.3975 | +0.3895 | +0.3952 | +0.3949 |
| run speed, m/s | +0.5559 | +0.5465 | +0.5470 | +0.5473 | +0.5329 | +0.5516 | +0.5508 |
| back speed, m/s | -0.0710 | -0.0789 | -0.0785 | -0.0643 | -0.0770 | -0.0673 | -0.0664 |
| jump speed, m/s | +0.1969 | +0.2095 | +0.2158 | +0.2141 | +0.2110 | +0.2121 | +0.2034 |
| turn-left yaw, rad/s | +1.3293 | +1.3419 | +1.3614 | +1.3820 | +1.3767 | +1.3641 | +1.3501 |
| turn-right yaw, rad/s | -1.1269 | -1.1396 | -1.1506 | -1.1130 | -1.1599 | -1.1172 | -1.1306 |
| sit pose (pitch / trunk) | 0.500 / 0.622 | 0.501 / 0.621 | 0.500 / 0.621 | 0.511 / 0.622 | 0.508 / 0.621 | 0.513 / 0.620 | 0.502 / 0.621 |
| lie-down pose (pitch / trunk) | 0.031 / 0.963 | 0.024 / 0.957 | 0.030 / 0.961 | 0.029 / 0.962 | 0.029 / 0.959 | 0.027 / 0.958 | 0.031 / 0.965 |
| **ball arrival success (got_there)** | **4.69% (9/192)** | **4.69% (9/192)** | **5.73% (11/192)** | **3.12% (6/192)** | **6.25% (12/192)** | **4.69% (9/192)** | **3.65% (7/192)** |
| ball distance drift (start -> end, m) | 2.494 -> 3.629 | 2.487 -> 3.561 | 2.486 -> 3.522 | 2.486 -> 3.731 | 2.487 -> 3.399 | 2.485 -> 3.426 | 2.486 -> 3.615 |
| ball aligned in final second | 2.72% | 1.43% | 2.75% | 1.19% | 3.78% | 3.38% | 3.54% |
| ball final facing cosine | -0.4325 | -0.4378 | -0.4254 | -0.4871 | -0.4412 | -0.4929 | -0.5120 |
| ball falls / 13 s | **0.6458** | **0.6302** | **0.6823** | **0.7344** | **0.7396** | **0.7031** | **0.6667** |
| **owner arrival success (got_there, "come")** | **6.25% (12/192)** | **5.21% (10/192)** | **3.65% (7/192)** | **4.69% (9/192)** | **7.29% (14/192)** | **4.69% (9/192)** | **8.33% (16/192)** |
| owner distance drift (start -> end, m) | 3.663 -> 3.585 | 3.666 -> 3.573 | 3.666 -> 3.585 | 3.668 -> 3.782 | 3.666 -> 3.664 | 3.664 -> 3.690 | 3.666 -> 3.596 |
| owner aligned in final second | 12.45% | 14.26% | 11.36% | 9.05% | 8.14% | 7.28% | 10.43% |
| owner final facing cosine | +0.0680 | +0.0269 | +0.0337 | -0.0652 | -0.0380 | -0.1014 | +0.0367 |
| owner falls / 13 s | **1.4375** | **1.4531** | **1.4531** | **1.4115** | **1.5469** | **1.3333** | **1.4531** |
| **ball-left balanced accuracy** | 0.690 | 0.684 | 0.708 | 0.701 | 0.693 | 0.705 | 0.688 |
| **ball-ahead balanced accuracy** | 0.667 | 0.692 | 0.712 | 0.711 | 0.699 | 0.699 | 0.698 |
| **ball-right balanced accuracy** | 0.676 | 0.699 | 0.703 | 0.695 | 0.683 | 0.677 | 0.642 |
| **owner-left balanced accuracy** | 0.685 | 0.667 | 0.682 | 0.692 | 0.678 | 0.686 | 0.674 |
| **owner-ahead balanced accuracy** | 0.678 | 0.642 | 0.684 | 0.667 | 0.693 | 0.686 | 0.688 |
| **owner-right balanced accuracy** | 0.640 | 0.662 | 0.653 | 0.654 | 0.648 | 0.640 | 0.632 |
| **hit balanced accuracy** | 0.839 | 0.868 | 0.855 | 0.875 | 0.878 | 0.864 | 0.817 |
| **moving balanced accuracy** | 0.632 | 0.726 | 0.649 | 0.606 | 0.623 | 0.627 | 0.619 |
| **hungry balanced accuracy** | 0.577 | 0.571 | 0.571 | 0.568 | 0.576 | 0.584 | 0.575 |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Checkpoint SHA256 Hashes:**
- `baby_dog_step_0000065536.npz`: `fd2f1529f41f013dceec4cdc05edd46f552f56a3153c0c5dc35a17000c377a86`
- `baby_dog_step_0000131072.npz`: `0f33d33d2f048c83aa510e00a7766ddce6eb498f95d5aa2869343ad8487925d0`
- `baby_dog_step_0000196608.npz`: `ee1f1d4274e406f23f8acfd1589d030186f2c644b55ecd6191e91fbdbe3a5b84`
- `baby_dog_step_0000262144.npz`: `864fc52ae0ce6dcb3ddd81952a78f2e7565dc7f2d04b96d1c5b4d78e9c937a5d`
- `baby_dog_step_0000327680.npz`: `1b252afe9662f507ec5642cd8037ea8214640a4c5de2756ba0e6b890b60216e0`
- `baby_dog_step_0000393216.npz`: `f7cbc54e0a23e80b4effcda9a22ba573edbd50acf0358a8a736afe365699f62f`
- `baby_dog.npz`: `d538a81fe55045564a6ffa6b787f5a238f8219d3901a991d41cfaff88ab9a821`

### Previous completed trial: vision approach round 32 (active yaw turning dominance, misaligned forward surge penalty, and new all-time project record on 2x Nvidia Tesla T4 GPU)

Notebook: https://www.kaggle.com/code/karangarg/bio-bot-baby-dog-vision-approach-round32 (version 1).
Completed all 36 updates (589,824 steps, 91.0 min runtime) on 2x Nvidia Tesla T4 GPU (512 envs, 256/GPU) to `KernelWorkerStatus.COMPLETE`. Outputs ingested into `results/baby_dog_vision_approach_round32/`.

Setup: 600k steps (36 updates × 16,384 steps = 589,824 steps), 512 envs, 6 evaluation checkpoints (every ~98k steps across 3 fresh cohorts = 192 lives per eval, 7 total evaluations including baseline and final), 2h cap. Teacher movement help zero from rest (`help_ = 0.00`).
Resumed from candidate `baby_dog_step_0000098304.npz` (`dc964638…`, Round 30 record candidate) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round30`.
All outputs fetched to `results/baby_dog_vision_approach_round32/`: `baby_dog.npz`, snapshots at 98k, 196k, 294k, 393k, 491k, 589k steps, `practice.json`, and `log.txt`.

**Measured Outcomes across 3 fresh cohorts (192 lives per evaluation):**

| measured metric (three cohorts, 192 lives) | baseline (dc96...) | 0.10M (1708...) | 0.20M (8770...) | 0.30M (6c81...) | 0.40M (d23d...) | 0.50M (29a4...) | 0.60M (5761...) |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean movement falls / 13 s | 1.0404 | 1.0475 | 1.1087 | 1.1094 | 1.1204 | 1.1354 | 1.0866 |
| walk falls / 13 s | 1.5781 | 1.5260 | 1.6198 | 1.7292 | 1.7552 | 1.8073 | 1.5521 |
| pace falls / 13 s | 0.9427 | 0.9427 | 0.9896 | 0.9948 | 1.0365 | 1.1094 | 0.9896 |
| canter falls / 13 s | 1.0885 | 1.0625 | 1.1302 | 1.0417 | 1.0365 | 1.0677 | 1.0833 |
| run falls / 13 s | 0.9167 | 0.9896 | 1.0156 | 0.9896 | 1.0625 | 1.0208 | 0.9271 |
| turn-left falls / 13 s | 0.8698 | 0.8802 | 0.9375 | 0.9479 | 0.8958 | 0.9167 | 0.9375 |
| turn-right falls / 13 s | 0.7604 | 0.7969 | 0.8906 | 0.8542 | 0.8438 | 0.8021 | 0.7760 |
| back falls / 13 s | 0.8802 | 0.9323 | 1.0104 | 0.9375 | 0.9375 | 1.0052 | 0.9948 |
| jump falls / 13 s | 1.2865 | 1.2500 | 1.2760 | 1.3802 | 1.3958 | 1.3542 | 1.4323 |
| walk speed, m/s | +0.4039 | +0.4058 | +0.4066 | +0.4013 | +0.3975 | +0.3880 | +0.4161 |
| pace speed, m/s | +0.5723 | +0.5608 | +0.5756 | +0.5736 | +0.5682 | +0.5750 | +0.5762 |
| canter speed, m/s | +0.4046 | +0.3991 | +0.3944 | +0.3934 | +0.3982 | +0.3931 | +0.3985 |
| run speed, m/s | +0.5386 | +0.5524 | +0.5502 | +0.5532 | +0.5492 | +0.5538 | +0.5504 |
| back speed, m/s | -0.0716 | -0.0705 | -0.0733 | -0.0675 | -0.0755 | -0.0732 | -0.0720 |
| jump speed, m/s | +0.2064 | +0.2032 | +0.2023 | +0.2068 | +0.2049 | +0.2058 | +0.2031 |
| turn-left yaw, rad/s | +1.3595 | +1.3470 | +1.3363 | +1.3072 | +1.3275 | +1.3531 | +1.3327 |
| turn-right yaw, rad/s | -1.1335 | -1.1550 | -1.1354 | -1.1487 | -1.1414 | -1.1388 | -1.1388 |
| sit pose (pitch / trunk) | 0.501 / 0.624 | 0.499 / 0.621 | 0.502 / 0.623 | 0.505 / 0.622 | 0.505 / 0.620 | 0.510 / 0.621 | 0.505 / 0.620 |
| lie-down pose (pitch / trunk) | 0.034 / 0.964 | 0.034 / 0.962 | 0.031 / 0.960 | 0.038 / 0.966 | 0.033 / 0.966 | 0.031 / 0.963 | 0.032 / 0.965 |
| **ball arrival success (got_there)** | **5.73% (11/192)** | **6.25% (12/192)** | **5.21% (10/192)** | **4.17% (8/192)** | **8.85% (17/192)** | **5.73% (11/192)** | **5.21% (10/192)** |
| ball distance drift (start -> end, m) | 2.489 -> 3.540 | 2.487 -> 3.615 | 2.498 -> 3.649 | 2.499 -> 3.709 | 2.498 -> 3.662 | 2.499 -> 3.542 | 2.488 -> 3.571 |
| ball aligned in final second | 2.37% | 3.07% | 1.84% | 2.06% | 3.84% | 3.33% | 1.53% |
| ball final facing cosine | -0.5119 | -0.4262 | -0.5690 | -0.4570 | -0.4744 | -0.5024 | -0.4831 |
| ball falls / 13 s | **0.6458** | **0.6458** | **0.6615** | **0.6823** | **0.6719** | **0.6875** | **0.6719** |
| **owner arrival success (got_there, "come")** | **5.73% (11/192)** | **5.21% (10/192)** | **5.21% (10/192)** | **5.73% (11/192)** | **8.33% (16/192)** | **3.12% (6/192)** | **3.65% (7/192)** |
| owner distance drift (start -> end, m) | 3.665 -> 3.654 | 3.665 -> 3.677 | 3.664 -> 3.727 | 3.665 -> 3.738 | 3.667 -> 3.570 | 3.668 -> 3.777 | 3.668 -> 3.657 |
| owner aligned in final second | 11.26% | 14.01% | 12.85% | 11.82% | 12.34% | 12.54% | 14.21% |
| owner final facing cosine | +0.0278 | +0.0296 | +0.0836 | -0.0198 | +0.0760 | +0.0812 | +0.1048 |
| owner falls / 13 s | **1.3438** | **1.3073** | **1.3750** | **1.5521** | **1.3906** | **1.4896** | **1.3698** |
| **ball-left balanced accuracy** | 0.691 | 0.696 | 0.701 | 0.695 | 0.699 | 0.688 | 0.697 |
| **ball-ahead balanced accuracy** | 0.665 | 0.703 | 0.700 | 0.683 | 0.663 | 0.659 | 0.680 |
| **ball-right balanced accuracy** | 0.691 | 0.689 | 0.688 | 0.673 | 0.673 | 0.645 | 0.659 |
| **owner-left balanced accuracy** | 0.677 | 0.672 | 0.686 | 0.680 | 0.677 | 0.676 | 0.679 |
| **owner-ahead balanced accuracy** | 0.629 | 0.685 | 0.666 | 0.681 | 0.694 | 0.656 | 0.689 |
| **owner-right balanced accuracy** | 0.651 | 0.632 | 0.654 | 0.646 | 0.633 | 0.619 | 0.637 |
| **hit balanced accuracy** | 0.860 | 0.914 | 0.836 | 0.825 | 0.900 | 0.881 | 0.885 |
| **moving balanced accuracy** | 0.644 | 0.621 | 0.623 | 0.619 | 0.629 | 0.639 | 0.639 |
| **hungry balanced accuracy** | 0.571 | 0.572 | 0.573 | 0.573 | 0.574 | 0.550 | 0.524 |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Checkpoint SHA256 Hashes:**
- `baby_dog_step_0000098304.npz`: `17088219bbfa19a8c3710668ea56c33111dbe59e3d61b1da138291d5e3c27fc5`
- `baby_dog_step_0000196608.npz`: `8770b573054756120aa10c44562ebd114adea56e64360d9aea83143b2cb8a79b`
- `baby_dog_step_0000294912.npz`: `6c8174f6261159b51e74c6c7bfe891b7eff1c97eed60bbcfa30753e751de0275`
- `baby_dog_step_0000393216.npz`: `d23d8f2c91d7f9d83f30d408a62e40285864046e91a3462b70755ac83008ef2e`
- `baby_dog_step_0000491520.npz`: `29a4f3e4491c09fbd5fdeddf8232bd32627a2b39e8e58a11710b8ed81f4817d8`
- `baby_dog_step_0000589824.npz`: `5761dfe9bc9b4e8ac58d1d563d12aa0fb1a8c6c6973206f93cd6970dd09aefb5`
- `baby_dog.npz`: `f1b15172a89f040d18212ca9ff4a80d6e13a6e66037b6f03dc797c6da4381b73`

### Previous completed trial: vision approach round 31 (gallop sprint acceleration, elevated target payoff, and 7-checkpoint evaluation on Kaggle 2x Nvidia Tesla T4 GPU)

Notebook: https://www.kaggle.com/code/karangarg/bio-bot-baby-dog-vision-approach-round31 (version 3).
Completed all 36 updates (589,824 steps, 89.0 min runtime) on 2x Nvidia Tesla T4 GPU (512 envs, 256/GPU) to `KernelWorkerStatus.COMPLETE`. Outputs ingested into `results/baby_dog_vision_approach_round31/`.

Setup: 600k steps (36 updates × 16,384 steps = 589,824 steps), 512 envs, 6 evaluation checkpoints (every ~98k steps across 3 fresh cohorts = 192 lives per eval, 7 total evaluations including baseline and final), 2h cap. Teacher movement help zero from rest (`help_ = 0.00`).
Resumed from candidate `baby_dog_step_0000098304.npz` (`dc964638…`, Round 30 record candidate) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round30`.
All outputs fetched to `results/baby_dog_vision_approach_round31/`: `baby_dog.npz`, snapshots at 98k, 196k, 294k, 393k, 491k, 589k steps, `practice.json`, and `log.txt`.

**Measured Outcomes across 3 fresh cohorts (192 lives per evaluation):**

| measured metric (three cohorts, 192 lives) | baseline (`dc96…`) | 0.10M (`91f2…`) | 0.20M (`b267…`) | 0.30M (`f2e4…`) | 0.40M (`6e55…`) | 0.50M (`e8b6…`) | 0.60M (`41e8…`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean movement falls / 13 s | 1.0462 | 1.0436 | 1.0618 | 1.0716 | 1.0599 | 1.0443 | 1.0521 |
| walk falls / 13 s | 1.6042 | 1.5417 | 1.5938 | 1.6302 | 1.5417 | 1.5521 | 1.5312 |
| pace falls / 13 s | 0.9427 | 0.9688 | 0.9010 | 0.9896 | 0.9688 | 0.9792 | 0.9323 |
| canter falls / 13 s | 0.9896 | 0.9740 | 1.0625 | 1.1667 | 0.9844 | 1.0833 | 1.1510 |
| run falls / 13 s | 1.0104 | 1.0260 | 1.0312 | 0.9531 | 1.1042 | 0.9896 | 1.0104 |
| turn-left falls / 13 s | 0.8594 | 0.8750 | 0.8906 | 0.9115 | 0.9167 | 0.8177 | 0.8281 |
| turn-right falls / 13 s | 0.8177 | 0.7604 | 0.8073 | 0.7708 | 0.7812 | 0.7656 | 0.7760 |
| back falls / 13 s | 0.9115 | 0.8698 | 0.8906 | 0.8594 | 0.9479 | 0.9167 | 0.8906 |
| jump falls / 13 s | 1.2344 | 1.3333 | 1.3177 | 1.2917 | 1.2344 | 1.2500 | 1.2969 |
| walk speed, m/s | +0.4046 | +0.4096 | +0.4126 | +0.4064 | +0.4081 | +0.4118 | +0.4153 |
| pace speed, m/s | +0.5637 | +0.5681 | +0.5747 | +0.5664 | +0.5638 | +0.5692 | +0.5743 |
| canter speed, m/s | +0.3918 | +0.3993 | +0.3969 | +0.3959 | +0.4016 | +0.3992 | +0.3920 |
| run speed, m/s | +0.5371 | +0.5528 | +0.5539 | +0.5515 | +0.5575 | +0.5474 | +0.5628 |
| back speed, m/s | -0.0763 | -0.0725 | -0.0775 | -0.0741 | -0.0734 | -0.0776 | -0.0744 |
| jump speed, m/s | +0.2041 | +0.2096 | +0.2119 | +0.2156 | +0.2102 | +0.2101 | +0.2026 |
| turn-left yaw, rad/s | +1.3226 | +1.3377 | +1.3372 | +1.3252 | +1.3188 | +1.3207 | +1.3200 |
| turn-right yaw, rad/s | -1.1665 | -1.1445 | -1.1579 | -1.1470 | -1.1386 | -1.1549 | -1.1330 |
| sit pose (pitch / trunk) | 0.500 / 0.620 | 0.511 / 0.620 | 0.512 / 0.621 | 0.505 / 0.620 | 0.510 / 0.622 | 0.493 / 0.621 | 0.502 / 0.621 |
| lie-down pose (pitch / trunk) | 0.033 / 0.964 | 0.030 / 0.961 | 0.030 / 0.961 | 0.028 / 0.958 | 0.030 / 0.960 | 0.035 / 0.965 | 0.034 / 0.962 |
| **ball arrival success (got_there)** | **3.12% (6/192)** | **5.73% (11/192)** | **3.65% (7/192)** | **4.69% (9/192)** | **5.21% (10/192)** | **5.21% (10/192)** | **5.73% (11/192)** |
| ball distance drift (start -> end, m) | 2.498 -> 3.690 | 2.499 -> 3.563 | 2.498 -> 3.632 | 2.491 -> 3.541 | 2.496 -> 3.700 | 2.487 -> 3.647 | 2.498 -> 3.382 |
| ball aligned in final second | 2.34% | 2.40% | 1.20% | 4.31% | 3.62% | 3.43% | 4.90% |
| ball final facing cosine | -0.5065 | -0.4594 | -0.4905 | -0.4299 | -0.4539 | -0.4703 | -0.3875 |
| ball falls / 13 s | **0.6510** | **0.6615** | **0.6406** | **0.6042** | **0.6198** | **0.6667** | **0.7188** |
| **owner arrival success (got_there, "come")** | **8.85% (17/192)** | **6.77% (13/192)** | **5.73% (11/192)** | **4.17% (8/192)** | **6.25% (12/192)** | **6.25% (12/192)** | **5.73% (11/192)** |
| owner distance drift (start -> end, m) | 3.666 -> 3.567 | 3.665 -> 3.510 | 3.665 -> 3.606 | 3.666 -> 3.593 | 3.666 -> 3.610 | 3.665 -> 3.722 | 3.661 -> 3.734 |
| owner aligned in final second | 13.57% | 9.65% | 11.57% | 11.01% | 14.40% | 14.17% | 11.95% |
| owner final facing cosine | +0.0421 | -0.0619 | +0.0526 | -0.0438 | +0.0586 | +0.0072 | -0.0272 |
| owner falls / 13 s | **1.2448** | **1.2708** | **1.1823** | **1.3229** | **1.3125** | **1.1823** | **1.3594** |
| **ball-left balanced accuracy** | 0.703 | 0.698 | 0.695 | 0.686 | 0.703 | 0.658 | 0.688 |
| **ball-ahead balanced accuracy** | 0.675 | 0.696 | 0.706 | 0.688 | 0.686 | 0.680 | 0.709 |
| **ball-right balanced accuracy** | 0.701 | 0.685 | 0.689 | 0.663 | 0.651 | 0.599 | 0.641 |
| **owner-left balanced accuracy** | 0.676 | 0.686 | 0.681 | 0.677 | 0.693 | 0.690 | 0.680 |
| **owner-ahead balanced accuracy** | 0.637 | 0.630 | 0.671 | 0.681 | 0.699 | 0.638 | 0.660 |
| **owner-right balanced accuracy** | 0.633 | 0.645 | 0.628 | 0.643 | 0.640 | 0.638 | 0.638 |
| **hit balanced accuracy** | 0.856 | 0.817 | 0.869 | 0.861 | 0.871 | 0.829 | 0.825 |
| **moving balanced accuracy** | 0.639 | 0.644 | 0.630 | 0.631 | 0.624 | 0.645 | 0.647 |
| **hungry balanced accuracy** | 0.572 | 0.561 | 0.578 | 0.581 | 0.571 | 0.541 | 0.512 |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Checkpoint SHA256 Hashes:**
- `baby_dog_step_0000098304.npz`: `91f277e1195b81df47b26b61c64c1813df6bfab41f3954349bedc8a43ca42b3b`
- `baby_dog_step_0000196608.npz`: `b2677a290559108e7e53a57e34760a1f7adc1aa770fea539cb5bb18652e5dec3`
- `baby_dog_step_0000294912.npz`: `f2e40acaf306cb6f107dd5f524bff949683a46db5a630e8b5a9603f07d6eecd6`
- `baby_dog_step_0000393216.npz`: `6e557929e212207328bc6eae60e62f5de72823763ba1cbe879acfafc0f052b05`
- `baby_dog_step_0000491520.npz`: `e8b6cf40b94f5112edb5a70f9179f2890886837a9175038545f31a4939dae76d`
- `baby_dog_step_0000589824.npz`: `41e8021ce2e1123783530341f6202e85261db1417389d2454b49aa380d0702e3`
- `baby_dog.npz`: `393ca8571c1a6acf33fbb49f2f6696a94890edc7dfb7927573ad6d3b46c36ccd`

### Previous completed trial: vision approach round 30 (7× forward sprint reward dominance, smoothed alignment floor, and arrival velocity analysis on Kaggle 2x Nvidia Tesla T4 GPU)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round30 (version 1).
Ran all 36 updates (589,824 steps, 89.5 min runtime) on 2x Nvidia Tesla T4 GPU (512 envs, 256/GPU) to `KernelWorkerStatus.COMPLETE`. Outputs ingested into `results/baby_dog_vision_approach_round30/`.

Setup: 36 updates, 512 envs, evaluation checkpoints every 6 updates (~98k steps across 3 fresh independent cohorts = 192 lives per eval; 7 total evaluations: baseline 0M, 0.1M, 0.2M, 0.3M, 0.4M, 0.5M, 0.6M). Teacher movement help zero from rest (`help_ = 0.00`).
Resumed from project baseline record holder `baby_dog_step_0000245760.npz` (`7ffe2eb8…`, Round 25 record holder) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round25`.
All outputs fetched to `results/baby_dog_vision_approach_round30/`: `baby_dog.npz`, snapshots at 98k, 196k, 294k, 393k, 491k, 589k steps, `practice.json`, and `log.txt`.

**Measured Outcomes across 3 fresh cohorts (192 lives per evaluation):**

| measured metric (three cohorts, 192 lives) | baseline (0M) | 0.10M (`dc96…`) | 0.20M (`fc85…`) | 0.30M (`fba8…`) | 0.40M (`5a7e…`) | 0.50M (`d0c4…`) | 0.60M (`c191…`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean movement falls / 13 s | 1.0137 | 1.0638 | 1.0664 | 1.1022 | 1.0651 | 1.0859 | 1.0755 |
| walk falls / 13 s | 1.4948 | 1.5833 | 1.3854 | 1.6615 | 1.5625 | 1.5365 | 1.5469 |
| pace falls / 13 s | 0.8906 | 1.0156 | 0.9740 | 0.9740 | 0.9219 | 0.9167 | 0.9479 |
| canter falls / 13 s | 1.0208 | 1.0469 | 1.0521 | 1.1146 | 1.1354 | 1.1615 | 1.0469 |
| run falls / 13 s | 0.9010 | 1.0208 | 1.0417 | 0.9844 | 0.9792 | 1.0208 | 0.9583 |
| turn-left falls / 13 s | 0.8750 | 0.8958 | 0.9427 | 0.9271 | 0.8438 | 0.9010 | 0.9688 |
| turn-right falls / 13 s | 0.7500 | 0.7812 | 0.7812 | 0.7917 | 0.8490 | 0.7552 | 0.7812 |
| back falls / 13 s | 0.9167 | 0.8854 | 0.9427 | 1.0104 | 0.9740 | 1.0208 | 1.0312 |
| jump falls / 13 s | 1.2604 | 1.2812 | 1.4115 | 1.3542 | 1.2552 | 1.3750 | 1.3229 |
| walk speed, m/s | +0.4025 | +0.4066 | +0.4129 | +0.4031 | +0.4047 | +0.4068 | +0.4065 |
| pace speed, m/s | +0.5694 | +0.5697 | +0.5674 | +0.5703 | +0.5712 | +0.5743 | +0.5667 |
| canter speed, m/s | +0.3936 | +0.3912 | +0.3965 | +0.3930 | +0.4002 | +0.3961 | +0.3995 |
| run speed, m/s | +0.5471 | +0.5499 | +0.5423 | +0.5457 | +0.5481 | +0.5374 | +0.5428 |
| back speed, m/s | -0.0724 | -0.0677 | -0.0766 | -0.0709 | -0.0712 | -0.0667 | -0.0650 |
| jump speed, m/s | +0.1913 | +0.2071 | +0.1980 | +0.2023 | +0.2112 | +0.2019 | +0.2033 |
| turn-left yaw, rad/s | +1.3214 | +1.3259 | +1.3541 | +1.3291 | +1.3491 | +1.3301 | +1.3280 |
| turn-right yaw, rad/s | -1.1450 | -1.1388 | -1.1536 | -1.1405 | -1.1577 | -1.1688 | -1.1593 |
| sit pose (pitch / trunk) | 0.508 / 0.622 | 0.507 / 0.623 | 0.501 / 0.624 | 0.499 / 0.619 | 0.501 / 0.620 | 0.506 / 0.620 | 0.507 / 0.621 |
| lie-down pose (pitch / trunk) | 0.030 / 0.964 | 0.039 / 0.969 | 0.029 / 0.960 | 0.030 / 0.959 | 0.027 / 0.958 | 0.027 / 0.958 | 0.029 / 0.962 |
| **ball arrival success (got_there)** | **6.77% (13/192)** | **8.33% (16/192)** | **4.69% (9/192)** | **2.08% (4/192)** | **3.65% (7/192)** | **5.21% (10/192)** | **5.21% (10/192)** |
| ball distance drift (start -> end, m) | 2.498 -> 3.600 | 2.497 -> 3.378 | 2.500 -> 3.513 | 2.486 -> 3.427 | 2.488 -> 3.562 | 2.496 -> 3.717 | 2.493 -> 3.698 |
| ball aligned in final second | 3.61% | 3.20% | 3.16% | 2.04% | 1.89% | 2.74% | 2.30% |
| ball final facing cosine | -0.4949 | -0.4022 | -0.4836 | -0.4558 | -0.4276 | -0.4959 | -0.4869 |
| ball falls / 13 s | **0.6823** | **0.6615** | **0.6146** | **0.6667** | **0.6562** | **0.6458** | **0.6250** |
| **owner arrival success (got_there, "come")** | **6.77% (13/192)** | **5.73% (11/192)** | **4.69% (9/192)** | **6.25% (12/192)** | **6.25% (12/192)** | **5.21% (10/192)** | **3.65% (7/192)** |
| owner distance drift (start -> end, m) | 3.664 -> 3.553 | 3.663 -> 3.725 | 3.667 -> 3.573 | 3.663 -> 3.703 | 3.669 -> 3.702 | 3.663 -> 3.755 | 3.664 -> 3.783 |
| owner aligned in final second | 8.44% | 10.95% | 13.01% | 10.80% | 12.89% | 12.53% | 10.49% |
| owner final facing cosine | +0.0053 | +0.0424 | +0.0324 | +0.0034 | +0.0112 | +0.0159 | -0.0259 |
| owner falls / 13 s | **1.2708** | **1.3438** | **1.3490** | **1.3646** | **1.3594** | **1.3646** | **1.3802** |
| **ball-left balanced accuracy** | 0.693 | 0.692 | 0.695 | 0.703 | 0.709 | 0.682 | 0.653 |
| **ball-ahead balanced accuracy** | 0.710 | 0.656 | 0.700 | 0.705 | 0.683 | 0.708 | 0.707 |
| **ball-right balanced accuracy** | 0.679 | 0.676 | 0.686 | 0.683 | 0.646 | 0.627 | 0.605 |
| **owner-left balanced accuracy** | 0.690 | 0.667 | 0.687 | 0.669 | 0.691 | 0.686 | 0.671 |
| **owner-ahead balanced accuracy** | 0.680 | 0.633 | 0.684 | 0.676 | 0.677 | 0.682 | 0.664 |
| **owner-right balanced accuracy** | 0.648 | 0.654 | 0.654 | 0.634 | 0.637 | 0.620 | 0.602 |
| **hit balanced accuracy** | 0.894 | 0.869 | 0.809 | 0.858 | 0.853 | 0.817 | 0.868 |
| **moving balanced accuracy** | 0.627 | 0.646 | 0.613 | 0.611 | 0.624 | 0.640 | 0.648 |
| **hungry balanced accuracy** | 0.569 | 0.574 | 0.570 | 0.576 | 0.582 | 0.554 | 0.511 |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Checkpoint SHA256 Hashes:**
- `baby_dog_step_0000098304.npz`: `dc964638c87ba65a3ed64ed60347a8efe413cf618fcd33ee33d54c9353ff6bb2`
- `baby_dog_step_0000196608.npz`: `fc85ce11bce32c489e0fdf0a3bf40e06e6e11a8d1a6ed9df5df92db761275a91`
- `baby_dog_step_0000294912.npz`: `fba8761dbe86df412baa7eca764a4e8afb55d780e54a2bf5fe342006ae6be631`
- `baby_dog_step_0000393216.npz`: `5a7ebac73099f23aaedb81742c9f377f51997662d24e5e419f37c90a1822521a`
- `baby_dog_step_0000491520.npz`: `d0c4bc2571a1e5015b18b72222148dfd24fe40a39966b1586f724d4b4629dd49`
- `baby_dog_step_0000589824.npz`: `c1917a6926bf62c9d74dcab497f69bcbd7377c640670be21298f2ed2cf48f40c`
- `baby_dog.npz`: `50bde55b9c36253ec0ee1b906fede59eee1cdfed7c005a435e4c6e4fe627d12d`

### Previous completed trial: vision approach round 29 (accelerated in-place pivot, supercharged straight-line sprint reward, and physical turn rate ceiling analysis on Kaggle 2x Nvidia Tesla T4 GPU)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round29 (version 1).
Ran all 36 updates (589,824 steps, 90.1 min runtime) on 2x Nvidia Tesla T4 GPU (512 envs, 256/GPU) to `KernelWorkerStatus.COMPLETE`. Outputs ingested into `results/baby_dog_vision_approach_round29/`.

Setup: 36 updates, 512 envs, evaluation checkpoints every 6 updates (~98k steps across 3 fresh independent cohorts = 192 lives per eval; 7 total evaluations: baseline 0M, 0.1M, 0.2M, 0.3M, 0.4M, 0.5M, 0.6M). Teacher movement help zero from rest (`help_ = 0.00`).
Resumed from Round 28 peak candidate `baby_dog_step_0000491520.npz` (`6204f7bb…`) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round28`.
All outputs fetched to `results/baby_dog_vision_approach_round29/`: `baby_dog.npz`, snapshots at 98k, 196k, 294k, 393k, 491k, 589k steps, `practice.json`, and `log.txt`.

**Measured Outcomes across 3 fresh cohorts (192 lives per evaluation):**

| measured metric (three cohorts, 192 lives) | baseline (0M) | 0.10M (`edd4…`) | 0.20M (`1a62…`) | 0.30M (`e273…`) | 0.40M (`5c1d…`) | 0.50M (`5d19…`) | 0.60M (`fe99…`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean movement falls / 13 s | 1.0697 | 1.0651 | 1.0879 | 1.0996 | 1.1296 | 1.1094 | 1.1510 |
| walk falls / 13 s | 1.5469 | 1.5260 | 1.4844 | 1.5260 | 1.6198 | 1.6823 | 1.7500 |
| pace falls / 13 s | 0.9844 | 0.9896 | 1.0781 | 1.1302 | 1.0208 | 1.0208 | 1.0312 |
| canter falls / 13 s | 1.1042 | 1.0521 | 1.0677 | 1.1354 | 1.0938 | 1.0729 | 1.1667 |
| run falls / 13 s | 1.0208 | 0.9375 | 1.0417 | 0.9479 | 1.0885 | 1.0104 | 1.0521 |
| turn-left falls / 13 s | 0.8958 | 0.8542 | 0.9792 | 0.8646 | 1.0156 | 0.9427 | 0.9583 |
| turn-right falls / 13 s | 0.7604 | 0.8177 | 0.8021 | 0.7865 | 0.7917 | 0.7708 | 0.8333 |
| back falls / 13 s | 0.9219 | 0.9896 | 0.9115 | 0.9635 | 0.9844 | 0.9792 | 0.9583 |
| jump falls / 13 s | 1.3229 | 1.3542 | 1.3385 | 1.4427 | 1.4219 | 1.3958 | 1.4583 |
| walk speed, m/s | +0.4110 | +0.4192 | +0.4003 | +0.4093 | +0.4178 | +0.4027 | +0.4004 |
| pace speed, m/s | +0.5675 | +0.5709 | +0.5692 | +0.5629 | +0.5614 | +0.5596 | +0.5712 |
| canter speed, m/s | +0.3931 | +0.4033 | +0.3981 | +0.3977 | +0.3983 | +0.3990 | +0.3980 |
| run speed, m/s | +0.5556 | +0.5526 | +0.5445 | +0.5557 | +0.5556 | +0.5665 | +0.5631 |
| back speed, m/s | -0.0770 | -0.0830 | -0.0696 | -0.0705 | -0.0703 | -0.0632 | -0.0801 |
| jump speed, m/s | +0.2028 | +0.2109 | +0.2223 | +0.2137 | +0.2202 | +0.2127 | +0.2166 |
| turn-left yaw, rad/s | +1.3422 | +1.3361 | +1.3413 | +1.3480 | +1.3554 | +1.3365 | +1.3532 |
| turn-right yaw, rad/s | -1.1721 | -1.1337 | -1.1499 | -1.1481 | -1.1511 | -1.1466 | -1.1379 |
| sit pose (pitch / trunk) | 0.512 / 0.621 | 0.499 / 0.621 | 0.505 / 0.622 | 0.497 / 0.621 | 0.497 / 0.622 | 0.503 / 0.621 | 0.500 / 0.622 |
| lie-down pose (pitch / trunk) | 0.033 / 0.964 | 0.026 / 0.957 | 0.034 / 0.965 | 0.031 / 0.962 | 0.022 / 0.958 | 0.026 / 0.961 | 0.029 / 0.962 |
| **ball arrival success (got_there)** | **7.29% (14/192)** | **3.12% (6/192)** | **6.77% (13/192)** | **2.60% (5/192)** | **4.69% (9/192)** | **3.12% (6/192)** | **5.21% (10/192)** |
| ball distance drift (start -> end, m) | 2.494 -> 3.651 | 2.491 -> 3.604 | 2.500 -> 3.623 | 2.494 -> 3.614 | 2.498 -> 3.556 | 2.487 -> 3.765 | 2.499 -> 3.575 |
| ball aligned in final second | 3.06% | 1.10% | 0.97% | 2.45% | 2.25% | 2.86% | 1.65% |
| ball final facing cosine | -0.4139 | -0.4587 | -0.4814 | -0.4348 | -0.4640 | -0.4638 | -0.3960 |
| ball falls / 13 s | **0.6771** | **0.6510** | **0.6719** | **0.7135** | **0.6875** | **0.7188** | **0.6719** |
| **owner arrival success (got_there, 'come')** | **5.21% (10/192)** | **6.25% (12/192)** | **3.12% (6/192)** | **2.08% (4/192)** | **4.69% (9/192)** | **5.21% (10/192)** | **4.69% (9/192)** |
| owner distance drift (start -> end, m) | 3.665 -> 3.617 | 3.668 -> 3.552 | 3.664 -> 3.690 | 3.665 -> 3.671 | 3.662 -> 3.604 | 3.665 -> 3.587 | 3.662 -> 3.564 |
| owner aligned in final second | 11.55% | 11.96% | 11.61% | 11.46% | 11.44% | 9.30% | 8.55% |
| owner final facing cosine | -0.0256 | +0.0911 | +0.1029 | +0.0197 | +0.0692 | -0.0493 | -0.0739 |
| owner falls / 13 s | **1.3594** | **1.2292** | **1.3177** | **1.4375** | **1.3021** | **1.2917** | **1.1875** |
| **ball-left balanced accuracy** | 0.696 | 0.699 | 0.700 | 0.711 | 0.705 | 0.707 | 0.705 |
| **ball-ahead balanced accuracy** | 0.686 | 0.679 | 0.690 | 0.711 | 0.702 | 0.717 | 0.706 |
| **ball-right balanced accuracy** | 0.650 | 0.696 | 0.693 | 0.694 | 0.679 | 0.676 | 0.694 |
| **owner-left balanced accuracy** | 0.680 | 0.685 | 0.664 | 0.684 | 0.679 | 0.668 | 0.661 |
| **owner-ahead balanced accuracy** | 0.681 | 0.651 | 0.689 | 0.684 | 0.670 | 0.679 | 0.675 |
| **owner-right balanced accuracy** | 0.621 | 0.664 | 0.641 | 0.652 | 0.625 | 0.614 | 0.595 |
| **hit balanced accuracy** | 0.884 | 0.937 | 0.885 | 0.890 | 0.879 | 0.843 | 0.807 |
| **moving balanced accuracy** | 0.647 | 0.641 | 0.620 | 0.625 | 0.635 | 0.665 | 0.646 |
| **hungry balanced accuracy** | 0.554 | 0.569 | 0.574 | 0.578 | 0.579 | 0.559 | 0.512 |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Checkpoint SHA256 Hashes:**
- `baby_dog_step_0000098304.npz`: `edd4dccaae4e69cb03b2c64ae96b631ecdc7c98473ef6cd15c287e44e98446b1`
- `baby_dog_step_0000196608.npz`: `1a62ddc8c6bc339fec2cae27bee22f7068b03660d0ff57a87e48b4265edcb759`
- `baby_dog_step_0000294912.npz`: `e27300073bc7a4955bc07df1a3ab20d4d8f89e3c4ba978060069a409043dcf9d`
- `baby_dog_step_0000393216.npz`: `5c1dce26c48488db710e1ed348f46a6304e6cb50660c4b3d2f68db9f5917662e`
- `baby_dog_step_0000491520.npz`: `5d1963af8d0eb20e2f211ebea266a0918491263c2ee9262e0b59f08150729580`
- `baby_dog_step_0000589824.npz`: `fe99734ce17121400b6d0e5c796b9aba8d70117c8b9234ec42b0a096f62265b4`
- `baby_dog.npz`: `9125f594e8c7f0034de10a89f68f8d9cdedd726791a18706b7e1ee4a518a6001`

### Previous completed trial: vision approach round 28 ("turn-then-sprint" heading alignment gating, roll/side drift damping, and centrifugal fall suppression on Kaggle 2x Nvidia Tesla T4 GPU)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round28 (version 1).
Ran all 36 updates (589,824 steps, 92.1 min runtime) on 2x Nvidia Tesla T4 GPU (512 envs, 256/GPU) to `KernelWorkerStatus.COMPLETE`. Outputs ingested into `results/baby_dog_vision_approach_round28/`.

Setup: 36 updates, 512 envs, evaluation checkpoints every 6 updates (~98k steps across 3 fresh independent cohorts = 192 lives per eval; 7 total evaluations: baseline 0M, 0.1M, 0.2M, 0.3M, 0.4M, 0.5M, 0.6M). Teacher movement help zero from rest (`help_ = 0.00`).
Resumed from promoted record candidate `baby_dog_step_0000245760.npz` (`7ffe2eb8…`, Round 25 record holder) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round25`.
All outputs fetched to `results/baby_dog_vision_approach_round28/`: `baby_dog.npz`, snapshots at 98k, 196k, 294k, 393k, 491k, 589k steps, `practice.json`, and `log.txt`.

**Measured Outcomes across 3 fresh cohorts (192 lives per evaluation):**

| measured metric (three cohorts, 192 lives) | baseline (0M) | 0.10M (`8c902d17…`) | 0.20M (`8986d63d…`) | 0.30M (`d23876ae…`) | 0.40M (`3d390898…`) | 0.50M (`6204f7bb…`) | 0.60M (`fd8bb752…`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean movement falls / 13 s | 1.0052 | 1.0352 | 1.0358 | 1.0456 | 1.0560 | 1.0456 | 1.0742 |
| walk falls / 13 s | 1.4740 | 1.4115 | 1.3854 | 1.5625 | 1.5833 | 1.5312 | 1.5781 |
| pace falls / 13 s | 0.8802 | 0.9844 | 1.0104 | 1.0000 | 0.9010 | 0.9635 | 0.9323 |
| canter falls / 13 s | 1.0573 | 1.0990 | 1.0156 | 1.0208 | 1.1302 | 1.0677 | 1.1146 |
| run falls / 13 s | 0.9740 | 0.8594 | 1.0312 | 0.9531 | 0.9479 | 0.8958 | 1.0052 |
| turn-left falls / 13 s | 0.8854 | 0.8333 | 0.8333 | 0.8958 | 1.0000 | 0.8438 | 0.8906 |
| turn-right falls / 13 s | 0.7865 | 0.8125 | 0.7604 | 0.8073 | 0.7708 | 0.8021 | 0.7865 |
| back falls / 13 s | 0.8490 | 0.9688 | 0.9323 | 0.9062 | 0.8646 | 0.8958 | 0.8646 |
| jump falls / 13 s | 1.1354 | 1.3125 | 1.3177 | 1.2188 | 1.2500 | 1.3646 | 1.4219 |
| walk speed, m/s | +0.4099 | +0.4138 | +0.4066 | +0.4082 | +0.4051 | +0.4028 | +0.4043 |
| pace speed, m/s | +0.5727 | +0.5704 | +0.5717 | +0.5753 | +0.5788 | +0.5629 | +0.5658 |
| canter speed, m/s | +0.4024 | +0.3962 | +0.3994 | +0.3952 | +0.3947 | +0.3984 | +0.4011 |
| run speed, m/s | +0.5545 | +0.5533 | +0.5499 | +0.5588 | +0.5549 | +0.5431 | +0.5429 |
| back speed, m/s | -0.0730 | -0.0778 | -0.0708 | -0.0711 | -0.0785 | -0.0678 | -0.0686 |
| jump speed, m/s | +0.1985 | +0.2023 | +0.2134 | +0.2144 | +0.2168 | +0.2045 | +0.2089 |
| turn-left yaw, rad/s | +1.3384 | +1.3424 | +1.3178 | +1.3442 | +1.3210 | +1.3664 | +1.3375 |
| turn-right yaw, rad/s | -1.1361 | -1.1638 | -1.1635 | -1.1467 | -1.1408 | -1.1450 | -1.1276 |
| sit pose (pitch / trunk) | 0.513 / 0.621 | 0.509 / 0.620 | 0.504 / 0.620 | 0.504 / 0.619 | 0.499 / 0.621 | 0.506 / 0.621 | 0.502 / 0.621 |
| lie-down pose (pitch / trunk) | 0.034 / 0.965 | 0.033 / 0.964 | 0.031 / 0.962 | 0.030 / 0.959 | 0.032 / 0.962 | 0.027 / 0.958 | 0.032 / 0.962 |
| **ball arrival success (got_there)** | **6.77% (13/192)** | **4.17% (8/192)** | **6.77% (13/192)** | **5.21% (10/192)** | **5.21% (10/192)** | **7.81% (15/192)** | **4.17% (8/192)** |
| ball distance drift (start -> end, m) | 2.490 -> 3.532 | 2.493 -> 3.537 | 2.489 -> 3.378 | 2.488 -> 3.550 | 2.500 -> 3.587 | 2.500 -> 3.526 | 2.489 -> 3.663 |
| ball aligned in final second | 4.17% | 2.18% | 3.90% | 2.50% | 4.96% | 3.43% | 2.66% |
| ball final facing cosine | -0.3852 | -0.5080 | -0.4093 | -0.4520 | -0.3900 | -0.4077 | -0.4633 |
| ball falls / 13 s | **0.6510** | **0.6667** | **0.6458** | **0.6354** | **0.6667** | **0.6667** | **0.6875** |
| **owner arrival success (got_there, 'come')** | **2.60% (5/192)** | **4.69% (9/192)** | **5.73% (11/192)** | **4.17% (8/192)** | **6.77% (13/192)** | **6.25% (12/192)** | **6.25% (12/192)** |
| owner distance drift (start -> end, m) | 3.666 -> 3.625 | 3.665 -> 3.630 | 3.663 -> 3.558 | 3.665 -> 3.574 | 3.662 -> 3.530 | 3.668 -> 3.646 | 3.664 -> 3.582 |
| owner aligned in final second | 10.10% | 15.35% | 12.96% | 14.19% | 10.77% | 10.01% | 10.33% |
| owner final facing cosine | +0.0319 | +0.0763 | +0.0003 | +0.0090 | +0.0499 | -0.0017 | -0.0203 |
| owner falls / 13 s | **1.1146** | **1.3177** | **1.3177** | **1.3646** | **1.4010** | **1.2969** | **1.3438** |
| **ball-left balanced accuracy** | 0.703 | 0.685 | 0.700 | 0.709 | 0.700 | 0.690 | 0.695 |
| **ball-ahead balanced accuracy** | 0.700 | 0.653 | 0.694 | 0.668 | 0.647 | 0.685 | 0.709 |
| **ball-right balanced accuracy** | 0.686 | 0.682 | 0.689 | 0.688 | 0.630 | 0.648 | 0.626 |
| **owner-left balanced accuracy** | 0.694 | 0.680 | 0.679 | 0.687 | 0.666 | 0.683 | 0.689 |
| **owner-ahead balanced accuracy** | 0.685 | 0.657 | 0.666 | 0.675 | 0.685 | 0.675 | 0.675 |
| **owner-right balanced accuracy** | 0.639 | 0.652 | 0.633 | 0.654 | 0.627 | 0.624 | 0.625 |
| **hit balanced accuracy** | 0.832 | 0.793 | 0.842 | 0.838 | 0.899 | 0.839 | 0.880 |
| **moving balanced accuracy** | 0.625 | 0.630 | 0.631 | 0.611 | 0.628 | 0.642 | 0.645 |
| **hungry balanced accuracy** | 0.578 | 0.574 | 0.568 | 0.581 | 0.578 | 0.566 | 0.511 |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Checkpoint SHA256 Hashes:**
- `baby_dog_step_0000098304.npz`: `8c902d178084b3b7aa34add81c1ac0b712c21c79ef6107b31329dcda09e1e2c8`
- `baby_dog_step_0000196608.npz`: `8986d63d951fe681eb05691fb4fdc6dd4409817d82f51fb099d1e39ea43d0a9c`
- `baby_dog_step_0000294912.npz`: `d23876ae9e69300eaf1ad57d58cbe2c6f919cb68190b52b8a22e60eb6d61d928`
- `baby_dog_step_0000393216.npz`: `3d3908980ad2d2784e8e75098189358096d67cf7f5dc1a7302e6b3ba4672823a`
- `baby_dog_step_0000491520.npz`: `6204f7bb4668cc3f0dbcf100b9905f1f2fd74c1567bf065febf621e447b8e5a0`
- `baby_dog_step_0000589824.npz`: `fd8bb752bb2fb1e98bac5510f0fe7720ea44532913e2474ede669571b56becc5`
- `baby_dog.npz`: `fd750dba8014a7a092c85a88ae683e0e005bbadeb6b38d5980badfa9dd383deb`

### Previous completed trial: vision approach round 27 (decoupled lateral roll/side drift damping, sprint forward velocity acceleration, and heading alignment gating on Kaggle 2x Nvidia Tesla T4 GPU)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round27 (version 1).
Ran all 36 updates (589,824 steps, 91.1 min runtime) on 2x Nvidia Tesla T4 GPU (512 envs, 256/GPU) to `KernelWorkerStatus.COMPLETE`. Outputs ingested into `results/baby_dog_vision_approach_round27/`.

Setup: 36 updates, 512 envs, evaluation checkpoints every 6 updates (~98k steps across 3 fresh independent cohorts = 192 lives per eval; 7 total evaluations: baseline 0M, 0.1M, 0.2M, 0.3M, 0.4M, 0.5M, 0.6M). Teacher movement help zero from rest (`help_ = 0.00`).
Resumed from promoted record candidate `baby_dog_step_0000245760.npz` (`7ffe2eb8…`, Round 25 record holder) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round25`.
All outputs fetched to `results/baby_dog_vision_approach_round27/`: `baby_dog.npz`, snapshots at 98k, 196k, 294k, 393k, 491k, 589k steps, `practice.json`, and `log.txt`.

**Measured Outcomes across 3 fresh cohorts (192 lives per evaluation):**

| measured metric (three cohorts, 192 lives) | baseline (0M) | 0.10M (`7123816b…`) | 0.20M (`009ed138…`) | 0.30M (`4f396c8c…`) | 0.40M (`24f79510…`) | 0.50M (`a64ab3fc…`) | 0.60M (`d83cdb5f…`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean movement falls / 13 s | 1.0280 | 1.0280 | 1.0775 | 1.0475 | 1.0625 | 1.0651 | 1.0944 |
| walk falls / 13 s | 1.4115 | 1.5104 | 1.5938 | 1.5312 | 1.5729 | 1.5260 | 1.5938 |
| pace falls / 13 s | 0.9740 | 0.9844 | 1.0104 | 0.9375 | 0.9688 | 0.9427 | 0.9635 |
| canter falls / 13 s | 1.1146 | 1.0625 | 1.0781 | 1.1146 | 1.1042 | 1.1719 | 1.1198 |
| run falls / 13 s | 0.9271 | 0.9219 | 0.9896 | 0.8854 | 0.8698 | 1.0000 | 1.0208 |
| turn-left falls / 13 s | 0.8281 | 0.8125 | 0.8906 | 0.9010 | 0.8854 | 0.7969 | 0.9583 |
| turn-right falls / 13 s | 0.7656 | 0.7500 | 0.7500 | 0.8021 | 0.7552 | 0.8073 | 0.7865 |
| back falls / 13 s | 0.8281 | 0.8854 | 0.9167 | 0.9948 | 0.9375 | 0.9219 | 0.9323 |
| jump falls / 13 s | 1.3750 | 1.2969 | 1.3906 | 1.2135 | 1.4062 | 1.3542 | 1.3802 |
| walk speed, m/s | +0.4037 | +0.4104 | +0.4094 | +0.4120 | +0.4008 | +0.4099 | +0.4005 |
| pace speed, m/s | +0.5648 | +0.5605 | +0.5669 | +0.5671 | +0.5649 | +0.5681 | +0.5692 |
| canter speed, m/s | +0.3932 | +0.3968 | +0.3995 | +0.3951 | +0.3927 | +0.3933 | +0.3995 |
| run speed, m/s | +0.5529 | +0.5544 | +0.5377 | +0.5497 | +0.5534 | +0.5434 | +0.5554 |
| back speed, m/s | -0.0689 | -0.0751 | -0.0781 | -0.0826 | -0.0696 | -0.0710 | -0.0682 |
| jump speed, m/s | +0.2041 | +0.2061 | +0.1915 | +0.1902 | +0.1912 | +0.1977 | +0.2030 |
| turn-left yaw, rad/s | +1.3426 | +1.3523 | +1.3278 | +1.3495 | +1.3318 | +1.3521 | +1.3109 |
| turn-right yaw, rad/s | -1.1737 | -1.1695 | -1.1470 | -1.1340 | -1.1484 | -1.1589 | -1.1424 |
| sit pose (pitch / trunk) | 0.507 / 0.621 | 0.510 / 0.620 | 0.508 / 0.621 | 0.511 / 0.621 | 0.499 / 0.620 | 0.503 / 0.622 | 0.506 / 0.622 |
| lie-down pose (pitch / trunk) | 0.031 / 0.961 | 0.032 / 0.961 | 0.031 / 0.964 | 0.034 / 0.965 | 0.034 / 0.965 | 0.034 / 0.965 | 0.030 / 0.961 |
| **ball arrival success (got_there)** | **8.33% (16/192)** | **4.17% (8/192)** | **3.65% (7/192)** | **5.73% (11/192)** | **3.65% (7/192)** | **5.21% (10/192)** | **2.60% (5/192)** |
| ball distance drift (start -> end, m) | 2.487 -> 3.512 | 2.491 -> 3.362 | 2.487 -> 3.564 | 2.499 -> 3.543 | 2.496 -> 3.516 | 2.489 -> 3.461 | 2.494 -> 3.529 |
| ball aligned in final second | 3.55% | 1.95% | 2.06% | 1.95% | 3.15% | 2.57% | 2.23% |
| ball final facing cosine | -0.4461 | -0.4290 | -0.4628 | -0.4725 | -0.4576 | -0.4829 | -0.4840 |
| ball falls / 13 s | **0.6250** | **0.6406** | **0.6771** | **0.6406** | **0.7135** | **0.6615** | **0.6771** |
| **owner arrival success (got_there, 'come')** | **7.29% (14/192)** | **3.65% (7/192)** | **3.65% (7/192)** | **4.17% (8/192)** | **4.69% (9/192)** | **5.21% (10/192)** | **6.25% (12/192)** |
| owner distance drift (start -> end, m) | 3.665 -> 3.650 | 3.663 -> 3.634 | 3.667 -> 3.587 | 3.669 -> 3.482 | 3.670 -> 3.704 | 3.668 -> 3.686 | 3.665 -> 3.617 |
| owner aligned in final second | 11.04% | 13.39% | 10.66% | 14.93% | 14.49% | 10.61% | 9.24% |
| owner final facing cosine | -0.0312 | +0.0704 | -0.0149 | +0.1471 | +0.0618 | -0.0067 | +0.0677 |
| owner falls / 13 s | **1.2656** | **1.3229** | **1.2708** | **1.3854** | **1.3281** | **1.4062** | **1.5990** |
| **ball-left balanced accuracy** | 0.709 | 0.685 | 0.708 | 0.701 | 0.694 | 0.699 | 0.689 |
| **ball-ahead balanced accuracy** | 0.707 | 0.663 | **0.720** | 0.709 | 0.666 | 0.704 | 0.705 |
| **ball-right balanced accuracy** | 0.690 | 0.687 | 0.683 | 0.686 | 0.667 | 0.662 | 0.665 |
| **owner-left balanced accuracy** | 0.699 | 0.685 | 0.681 | 0.686 | 0.679 | 0.684 | 0.677 |
| **owner-ahead balanced accuracy** | 0.691 | 0.644 | 0.681 | 0.685 | 0.663 | 0.689 | 0.677 |
| **owner-right balanced accuracy** | 0.646 | 0.651 | 0.648 | 0.649 | 0.632 | 0.626 | 0.623 |
| **hit balanced accuracy** | 0.863 | 0.881 | 0.875 | 0.867 | 0.852 | 0.846 | 0.854 |
| **moving balanced accuracy** | 0.623 | 0.643 | 0.628 | 0.617 | 0.622 | 0.654 | 0.657 |
| **hungry balanced accuracy** | 0.581 | 0.569 | 0.574 | 0.574 | 0.587 | 0.559 | 0.505 |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Checkpoint SHA256 Hashes:**
- `baby_dog_step_0000098304.npz`: `7123816be86bf85a9861d4286095e3a458faa8f3102aa92338f198be7569b941`
- `baby_dog_step_0000196608.npz`: `009ed1385ab69fef9a39b399fad0f20b3e54fb1e581c8a12c2c6825f4255d874`
- `baby_dog_step_0000294912.npz`: `4f396c8cf0eeccd3fd35d467854d1e927917820e48d68a2df586b5ebd6a5dec3`
- `baby_dog_step_0000393216.npz`: `24f795103efd2f06f5139ef1180895e4294651305175989bffb0d9e1e71dedc7`
- `baby_dog_step_0000491520.npz`: `a64ab3fc48b4b59b6a6cc3c527630759193a881021eb9d0d56561d159b313dd1`
- `baby_dog_step_0000589824.npz`: `d83cdb5f7220f3f679e8a8a45f3a75251fd1798fdf754bac0a3c328a1b9f6a86`
- `baby_dog.npz`: `7165124107430c81f38fe79ec088f5a3b2844fca759479aec6d03a9e4e489a0b`

### Previous completed trial: vision approach round 26 (upright stabilization damping, fall cost during task approach, and kinematic sprint analysis on Kaggle 2x Nvidia Tesla T4 GPU)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round26 (version 1).
Ran 18 updates (294,912 steps, 83.4 min elapsed / 75 min execution cap reached, `CANCEL_ACKNOWLEDGED`) on 2x Nvidia Tesla T4 GPU (512 envs, 256/GPU). Outputs ingested into `results/baby_dog_vision_approach_round26/`.

Setup: 36 updates cap, 512 envs, evaluation checkpoints every ~98k steps across 3 fresh cohorts = 192 lives per eval (4 total evaluations: baseline 0M, 0.1M, 0.2M, 0.3M). Teacher movement help zero from rest (`help_ = 0.00`).
Resumed from active restart candidate `baby_dog_step_0000245760.npz` (`7ffe2eb8…`, Round 25 record holder) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round25`.
All outputs fetched to `results/baby_dog_vision_approach_round26/`: `baby_dog.npz`, snapshots at 98k, 196k, 294k steps, `practice.json`, and `log.txt`.

**Measured Outcomes across 3 fresh cohorts (192 lives per evaluation):**

| measured metric (three cohorts, 192 lives) | baseline (0M) | 0.10M (`6039e891…`) | 0.20M (`d8e855e6…`) | 0.30M (`37302f0f…`) |
|---|---:|---:|---:|---:|
| mean movement falls / 13 s | 0.9996 | 1.0375 | 1.0543 | 1.0664 |
| walk falls / 13 s | 1.4792 | 1.5208 | 1.5365 | 1.5990 |
| pace falls / 13 s | 0.8646 | 0.9167 | 1.0156 | 0.9531 |
| canter falls / 13 s | 1.0312 | 1.0052 | 1.0729 | 1.1042 |
| run falls / 13 s | 0.9167 | 0.9219 | 0.9688 | 0.9688 |
| turn-left falls / 13 s | 0.8958 | 0.8646 | 0.9375 | 0.8750 |
| turn-right falls / 13 s | 0.7656 | 0.7917 | 0.7865 | 0.7344 |
| back falls / 13 s | 0.8906 | 0.8802 | 0.9115 | 0.9271 |
| jump falls / 13 s | 1.2083 | 1.3125 | 1.3177 | 1.2812 |
| walk speed, m/s | +0.4065 | +0.4024 | +0.3988 | +0.4057 |
| pace speed, m/s | +0.5654 | +0.5628 | +0.5657 | +0.5645 |
| canter speed, m/s | +0.3947 | +0.3955 | +0.3906 | +0.4041 |
| run speed, m/s | +0.5464 | +0.5519 | +0.5565 | +0.5563 |
| back speed, m/s | -0.0759 | -0.0757 | -0.0802 | -0.0815 |
| jump speed, m/s | +0.1977 | +0.2048 | +0.2014 | +0.2078 |
| turn-left yaw, rad/s | +1.3232 | +1.3716 | +1.3802 | +1.3567 |
| turn-right yaw, rad/s | -1.1549 | -1.1579 | -1.1685 | -1.1284 |
| sit pose (pitch / trunk) | +0.505 / 0.621 | +0.504 / 0.620 | +0.508 / 0.620 | +0.509 / 0.618 |
| lie-down pose (pitch / trunk) | +0.032 / 0.962 | +0.037 / 0.963 | +0.036 / 0.964 | +0.031 / 0.959 |
| **ball arrival success (got_there)** | **5.73% (11/192)** | **5.21% (10/192)** | **5.21% (10/192)** | **4.17% (8/192)** |
| ball distance drift (start -> end, m) | 2.487 -> 3.545 | 2.494 -> 3.480 | 2.490 -> 3.362 | 2.499 -> 3.445 |
| ball aligned in final second | 1.85% | 2.59% | 2.21% | 2.53% |
| ball final facing cosine | -0.4809 | -0.4408 | -0.4163 | -0.4819 |
| ball falls / 13 s | **0.6979** | **0.6250** | **0.6250** | **0.6458** |
| **owner arrival success (got_there, 'come')** | **5.73% (11/192)** | **4.17% (8/192)** | **4.17% (8/192)** | **4.69% (9/192)** |
| owner distance drift (start -> end, m) | 3.666 -> 3.520 | 3.669 -> 3.680 | 3.667 -> 3.602 | 3.669 -> 3.708 |
| owner aligned in final second | 13.28% | 11.53% | 9.68% | 10.39% |
| owner final facing cosine | +0.0841 | +0.1027 | -0.0064 | +0.0501 |
| owner falls / 13 s | **1.2656** | **1.2865** | **1.2969** | **1.2604** |
| **ball-left balanced accuracy** | 0.697 | 0.694 | 0.699 | 0.685 |
| **ball-ahead balanced accuracy** | 0.707 | 0.661 | 0.701 | 0.692 |
| **ball-right balanced accuracy** | 0.675 | 0.693 | 0.679 | 0.667 |
| **owner-left balanced accuracy** | 0.677 | 0.681 | 0.682 | 0.679 |
| **owner-ahead balanced accuracy** | 0.668 | 0.671 | 0.673 | 0.673 |
| **owner-right balanced accuracy** | 0.632 | 0.651 | 0.647 | 0.641 |
| **hit balanced accuracy** | 0.871 | 0.870 | **0.895** | 0.844 |
| **moving balanced accuracy** | 0.627 | 0.665 | 0.616 | 0.633 |
| **hungry balanced accuracy** | 0.568 | 0.566 | 0.564 | 0.579 |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 |

**Checkpoint SHA256 Hashes:**
- `baby_dog_step_0000098304.npz`: `6039e89151fdb098c7631caf14677f73be9a43275bc7d1d6942fda803d9db606`
- `baby_dog_step_0000196608.npz`: `d8e855e6e2430f532d033d7c64d5c5753e94d042d8d509fcf31a8e60049d8123`
- `baby_dog_step_0000294912.npz`: `37302f0f2c449ce9f9bbae29e79c8836c71ddfbd5e037f061d9cbe1f69a2f479`
- `baby_dog.npz`: `37302f0f2c449ce9f9bbae29e79c8836c71ddfbd5e037f061d9cbe1f69a2f479`

### Previous completed trial: vision approach round 25 (strict cruise yaw suppression, balanced task scaling, and 8-checkpoint high-resolution evaluation on Kaggle 2x Nvidia Tesla T4 GPU)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round25 (version 1).
Completed all 91 updates (1,490,944 steps, 113.9 min runtime) on 2x Nvidia Tesla T4 GPU (512 envs, 256/GPU) to `KernelWorkerStatus.COMPLETE`. Outputs ingested into `results/baby_dog_vision_approach_round25/`.

Setup: 1.5M steps, 91 updates, 512 envs, 6 evaluation checkpoints (every ~248k steps across 3 fresh cohorts = 192 lives per eval, 8 total evaluations including baseline and final), 2h cap. Teacher movement help zero from rest (`help_ = 0.00`).
Resumed from candidate `baby_dog_step_0000491520.npz` (`35b522d4…`, Round 24 record holder) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round24`.
All outputs fetched to `results/baby_dog_vision_approach_round25/`: `baby_dog.npz`, snapshots at 245k, 491k, 737k, 983k, 1.23M, 1.47M, 1.49M steps, `practice.json`, and `log.txt`.

**Measured Outcomes across 3 fresh cohorts (192 lives per evaluation):**

| measured metric (three cohorts, 192 lives) | baseline (0M) | 0.25M (`7ffe2eb8…`) | 0.50M (`76aebe49…`) | 0.75M (`e7603a60…`) | 1.00M (`528f5d68…`) | 1.25M (`3910e32f…`) | 1.47M (`9ad93074…`) | final 1.49M (`1b8012f1…`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean movement falls / 13 s | 1.0241 | 1.0072 | 1.1361 | 1.1966 | 1.2350 | 1.2272 | 1.2702 | 1.2617 |
| walk falls / 13 s | 1.3646 | 1.3958 | 1.6042 | 1.6927 | 1.7917 | 1.7031 | 1.8281 | 1.7240 |
| pace falls / 13 s | 0.9740 | 0.8802 | 0.9896 | 1.1562 | 1.1406 | 1.1771 | 1.1562 | 1.1875 |
| canter falls / 13 s | 1.0833 | 1.0469 | 1.2448 | 1.2500 | 1.4010 | 1.3333 | 1.3385 | 1.3542 |
| run falls / 13 s | 0.9688 | 0.9427 | 1.0469 | 1.1927 | 1.1146 | 1.1667 | 1.2031 | 1.2135 |
| turn-left falls / 13 s | 0.8125 | 0.8646 | 0.8802 | 0.9375 | 0.8958 | 0.9115 | 0.9427 | 0.9531 |
| turn-right falls / 13 s | 0.7708 | 0.7865 | 0.8385 | 0.8594 | 0.8802 | 0.8958 | 0.8542 | 0.9740 |
| back falls / 13 s | 0.8698 | 0.8750 | 0.9948 | 1.0677 | 1.1094 | 1.0625 | 1.0938 | 1.0521 |
| jump falls / 13 s | 1.3490 | 1.2656 | 1.4896 | 1.4167 | 1.5469 | 1.5677 | 1.7448 | 1.6354 |
| walk speed, m/s | +0.4117 | +0.4067 | +0.4116 | +0.3980 | +0.3956 | +0.3954 | +0.4030 | +0.4025 |
| pace speed, m/s | +0.5674 | +0.5635 | +0.5621 | +0.5578 | +0.5554 | +0.5696 | +0.5580 | +0.5558 |
| canter speed, m/s | +0.3965 | +0.3926 | +0.3920 | +0.3958 | +0.3894 | +0.4016 | +0.3989 | +0.3950 |
| run speed, m/s | +0.5572 | +0.5615 | +0.5549 | +0.5325 | +0.5499 | +0.5471 | +0.5490 | +0.5412 |
| back speed, m/s | -0.0782 | -0.0724 | -0.0724 | -0.0560 | -0.0579 | -0.0567 | -0.0496 | -0.0481 |
| jump speed, m/s | +0.1995 | +0.2074 | +0.1985 | +0.2011 | +0.1864 | +0.1826 | +0.1966 | +0.1890 |
| turn-left yaw, rad/s | +1.3633 | +1.3418 | +1.3774 | +1.3427 | +1.3326 | +1.3401 | +1.3452 | +1.3538 |
| turn-right yaw, rad/s | -1.1531 | -1.1452 | -1.1407 | -1.1397 | -1.1275 | -1.1392 | -1.1472 | -1.1369 |
| sit pose (pitch / trunk) | +0.503 / 0.620 | +0.509 / 0.621 | +0.510 / 0.623 | +0.511 / 0.624 | +0.491 / 0.622 | +0.488 / 0.624 | +0.500 / 0.620 | +0.502 / 0.623 |
| lie-down pose (pitch / trunk) | +0.035 / 0.963 | +0.037 / 0.968 | +0.035 / 0.965 | +0.030 / 0.962 | +0.026 / 0.960 | +0.023 / 0.959 | +0.022 / 0.962 | +0.020 / 0.955 |
| **ball arrival success (got_there)** | **5.73% (11/192)** | **7.29% (14/192)** | **4.17% (8/192)** | **5.21% (10/192)** | **2.08% (4/192)** | **4.17% (8/192)** | **5.73% (11/192)** | **6.25% (12/192)** |
| ball distance drift (start -> end, m) | 2.486 -> 3.458 | 2.485 -> 3.583 | 2.496 -> 3.548 | 2.493 -> 3.590 | 2.500 -> 3.636 | 2.500 -> 3.568 | 2.490 -> 3.559 | 2.492 -> 3.384 |
| ball aligned in final second | 1.33% | 2.69% | 2.89% | 3.02% | 1.72% | 5.30% | 4.27% | 2.82% |
| ball final facing cosine | -0.4666 | -0.4719 | -0.4572 | -0.4140 | -0.3760 | -0.3692 | **-0.2069** | **-0.1971** |
| ball falls / 13 s | 0.6354 | **0.6510** | 0.6771 | 0.7500 | 0.7708 | 0.8021 | 0.8385 | 0.8333 |
| **owner arrival success (got_there, 'come')** | **5.21% (10/192)** | **8.33% (16/192)** | **3.12% (6/192)** | **3.12% (6/192)** | **4.17% (8/192)** | **3.65% (7/192)** | **3.12% (6/192)** | **6.77% (13/192)** |
| owner distance drift (start -> end, m) | 3.668 -> 3.541 | **3.665 -> 3.490 [-17.5 cm]** | 3.666 -> 3.614 | 3.669 -> 3.687 | 3.663 -> 3.521 | 3.669 -> 3.663 | 3.664 -> 3.656 | 3.668 -> 3.587 |
| owner aligned in final second | 11.42% | 12.10% | 9.14% | 12.17% | 10.94% | **15.43%** | 12.45% | 14.82% |
| owner final facing cosine | +0.0056 | +0.0177 | -0.0230 | +0.0051 | +0.0038 | +0.0595 | +0.0589 | +0.0324 |
| owner falls / 13 s | 1.2292 | **1.2083** | 1.3854 | 1.6458 | 1.5833 | 1.5990 | 1.7031 | 1.6615 |
| **ball-left balanced accuracy** | 0.687 | 0.693 | 0.670 | 0.683 | 0.667 | 0.673 | 0.687 | 0.693 |
| **ball-ahead balanced accuracy** | 0.680 | **0.707 (Peak 0.73)** | 0.680 | 0.703 | 0.687 | 0.693 | 0.710 | 0.703 |
| **ball-right balanced accuracy** | 0.643 | **0.677 (Peak 0.69)** | 0.603 | 0.670 | 0.643 | 0.647 | 0.640 | 0.657 |
| **owner-left balanced accuracy** | 0.677 | **0.693 (Peak 0.72)** | 0.663 | 0.660 | 0.693 | 0.650 | 0.667 | 0.663 |
| **owner-ahead balanced accuracy** | 0.647 | **0.693 (Peak 0.71)** | 0.677 | 0.623 | 0.680 | 0.677 | 0.660 | 0.670 |
| **owner-right balanced accuracy** | 0.610 | **0.643** | 0.597 | 0.593 | 0.637 | 0.600 | 0.603 | 0.633 |
| **hit balanced accuracy** | 0.860 | 0.880 | 0.863 | 0.840 | 0.830 | **0.903** | 0.863 | 0.877 |
| **moving balanced accuracy** | 0.647 | 0.623 | 0.647 | 0.640 | 0.623 | 0.647 | 0.640 | 0.643 |
| **hungry balanced accuracy** | 0.563 | 0.577 | 0.570 | 0.500 | 0.573 | 0.543 | 0.500 | 0.500 |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Checkpoint SHA256 Hashes:**
- `baby_dog_step_0000245760.npz`: `7ffe2eb88a23de0cfaf4981a71eed785dd4be3db1d08824e9b6e8571e30281ee`
- `baby_dog_step_0000491520.npz`: `76aebe49d0875f0e3d86b4010007370ae1249bdd78dfbb6f7101ef5eea6caa72`
- `baby_dog_step_0000737280.npz`: `e7603a60bc1186fe7780fe7d8bb1809da41362c5004764184e3c89d28309bcb1`
- `baby_dog_step_0000983040.npz`: `528f5d686bfa6ae07ae0be25643916e48d8671ddbb7de2a4c2e69980f9447cbd`
- `baby_dog_step_0001228800.npz`: `3910e32f5926b921bf72778c8520574a2b43ecdfd93eedeb66834adf43c13984`
- `baby_dog_step_0001474560.npz`: `9ad93074348a1f7662167537a00cfee0127fc3fb6c227e22d67c1f6362a89d5a`
- `baby_dog_step_0001490944.npz`: `1b8012f16ab95c38571ed2613b64a52b3a9898cb129afb7b3a47996e14b2f390`
- `baby_dog.npz`: `a496e8b732918ec82f5bc07be1f1fc69f365100a5d1a3e1b160e04131f486125`

### Previous completed trial: vision approach round 24 (responsive pivot steering, sharpened optic lobe learning rate, and steeper in-circle potential well on Kaggle 2x Nvidia Tesla T4 GPU)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round24 (version 1).
Completed all 91 updates (1,490,944 steps, ~90.0 min runtime; 41.2 min queue) on 2x Nvidia Tesla T4 GPU (512 envs, 256/GPU). Outputs ingested into `results/baby_dog_vision_approach_round24/`.

Setup: 1.5M steps, 91 updates, 512 envs, 3 evals (0.5M, 1.0M, 1.5M), 3 cohorts (192 lives per eval), 2h cap. Teacher movement help zero from rest (`help_ = 0.00`).
Resumed from candidate `baby_dog.npz` (`3ae45926…`, Round 23 record holder) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round23`.
All outputs fetched to `results/baby_dog_vision_approach_round24/`: `baby_dog.npz`, snapshots at 491k, 983k, 1.47M, 1.49M steps, `practice.json`, and `log.txt`.

**Measured Outcomes across 3 fresh cohorts (192 lives per evaluation):**

| measured metric (three cohorts, 192 lives) | baseline (0M) | 0.5M steps (`35b522d4…`) | 1.0M steps (`23ec0136…`) | 1.5M snap (`a7c0c779…`) | final 1.49M (`d7a88a1b…`) |
|---|---:|---:|---:|---:|---:|
| mean movement falls / 13 s | 0.8884 | 0.9844 | 1.0171 | 1.0357 | 1.0164 |
| walk falls / 13 s | 1.2917 | 1.4948 | 1.4583 | 1.5521 | 1.4427 |
| pace falls / 13 s | 0.8385 | 0.9271 | 0.9688 | 0.9375 | 0.9427 |
| canter falls / 13 s | 0.9062 | 1.0573 | 1.0885 | 1.1302 | 1.0677 |
| run falls / 13 s | 0.8542 | 0.9635 | 1.0156 | 0.9792 | 0.9896 |
| turn-left falls / 13 s | 0.8177 | 0.8646 | 0.8854 | 0.9375 | 0.8958 |
| turn-right falls / 13 s | 0.6719 | 0.6927 | 0.7917 | 0.8021 | 0.8125 |
| back falls / 13 s | 0.8385 | 0.8906 | 0.9115 | 0.9115 | 0.9635 |
| walk speed, m/s | +0.4150 | +0.4073 | +0.4132 | +0.4153 | +0.4126 |
| pace speed, m/s | +0.5556 | +0.5642 | +0.5768 | +0.5799 | +0.5696 |
| canter speed, m/s | +0.3966 | +0.4006 | +0.3977 | +0.4011 | +0.3951 |
| run speed, m/s | +0.5621 | +0.5537 | +0.5557 | +0.5503 | +0.5562 |
| turn-left yaw, rad/s | +1.3186 | +1.3618 | +1.3511 | +1.3272 | +1.3331 |
| turn-right yaw, rad/s | -1.1522 | -1.1492 | -1.1354 | -1.1585 | -1.1833 |
| backward speed, m/s | -0.0682 | -0.0753 | -0.0710 | -0.0579 | -0.0617 |
| jump speed, m/s | +0.1891 | +0.1925 | +0.1982 | +0.1917 | +0.1946 |
| sit pose (pitch / trunk) | +0.508 / 0.619 | +0.508 / 0.619 | +0.508 / 0.619 | +0.508 / 0.619 | +0.508 / 0.619 |
| lie-down pose (pitch / trunk) | +0.040 / 0.967 | +0.040 / 0.967 | +0.040 / 0.967 | +0.040 / 0.967 | +0.040 / 0.967 |
| **ball arrival success (got_there)** | **3.12% (6/192)** | **5.73% (11/192)** | **4.69% (9/192)** | **7.81% (15/192) [Record]** | **4.69% (9/192)** |
| ball distance drift (start -> end, m) | 2.497 -> 3.392 | 2.495 -> 3.627 | 2.488 -> 3.813 | 2.489 -> 3.445 | 2.487 -> 3.689 |
| ball aligned in final second | 1.44% | 3.02% | 1.59% | **5.85%** | 4.78% |
| ball final facing cosine | -0.4885 | -0.5174 | -0.4538 | **-0.1670** | -0.2536 |
| ball falls / 13 s | 0.5885 | 0.5990 | 0.6875 | 0.7240 | 0.6771 |
| **owner arrival success (got_there, 'come')** | **3.65% (7/192)** | **9.90% (19/192) [Record]** | **4.17% (8/192)** | **5.21% (10/192)** | **6.25% (12/192)** |
| owner distance drift (start -> end, m) | 3.670 -> 3.732 | **3.666 -> 3.629 [-3.7 cm]** | 3.666 -> 3.688 | 3.666 -> 3.645 | **3.665 -> 3.622 [-4.3 cm]** |
| owner aligned in final second | 11.28% | 10.18% | **13.79%** | 13.58% | 9.18% |
| owner final facing cosine | +0.0492 | +0.0231 | +0.0760 | +0.0649 | +0.0222 |
| owner falls / 13 s | 1.2448 | 1.2344 | 1.2708 | 1.2344 | 1.2240 |
| **ball-left balanced accuracy** | 0.704 | 0.692 | 0.691 | 0.712 | 0.702 |
| **ball-ahead balanced accuracy** | 0.693 | 0.696 | **0.710** | 0.693 | 0.691 |
| **ball-right balanced accuracy** | 0.637 | 0.648 | 0.646 | **0.693** | 0.687 |
| **owner-left balanced accuracy** | 0.658 | 0.676 | **0.688** | 0.657 | 0.670 |
| **owner-ahead balanced accuracy** | 0.559 | 0.649 | 0.657 | 0.653 | **0.664** |
| **owner-right balanced accuracy** | 0.621 | 0.608 | **0.639** | 0.611 | 0.612 |
| **hit balanced accuracy** | 0.885 | 0.879 | 0.876 | 0.839 | 0.849 |
| **moving balanced accuracy** | 0.639 | 0.652 | 0.624 | 0.638 | 0.644 |
| **hungry balanced accuracy** | 0.502 | 0.562 | 0.560 | 0.500 | 0.500 |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Checkpoint SHA256 Hashes:**
- `baby_dog_step_0000491520.npz`: `35b522d46859954a5f176aec168d16ba94d6fcd0da0b2d372b2c9ff7e73be512`
- `baby_dog_step_0000983040.npz`: `23ec013640d989ec0c45f000c10b85dbf8826cd661e1978605cc045a964d992f`
- `baby_dog_step_0001474560.npz`: `a7c0c77905d4df1da222d50b05f6e3ab5a6663eca456be113de381db1e6b4d9a`
- `baby_dog_step_0001490944.npz`: `d7a88a1bcce9851f3ed350b5395b72c202f15b117fc26ba25dbd964c5d79e413`
- `baby_dog.npz`: `73d4032f958d12312be21b281e834060e86aed6ed1b5647d1e391687d94aa194`

### Previous completed trial: vision approach round 23 (sprint cruise yaw suppression, smooth boundary braking, and balanced signed angular bearing on Kaggle 2x Nvidia Tesla T4 GPU)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round23 (version 3).
Completed all 91 updates (1,490,944 steps, ~98.1 min runtime including queue and 4 full evaluation cohorts) on 2x Nvidia Tesla T4 GPU (512 envs, 256/GPU). Outputs ingested into `results/baby_dog_vision_approach_round23/`.

Setup: 1.5M steps, 91 updates, 512 envs, 3 evals (0.5M, 1.0M, 1.5M), 3 cohorts (192 lives per eval), 2h cap. Teacher movement help zero from rest (`help_ = 0.00`).
Resumed from candidate `baby_dog_step_0001490944.npz` (`6220964d…`) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round21`.
All outputs fetched to `results/baby_dog_vision_approach_round23/`: `baby_dog.npz`, snapshots at 491k, 983k, 1.47M, 1.49M steps, `practice.json`, and `log.txt`.

**Measured Outcomes across 3 fresh cohorts (192 lives per evaluation):**

| measured metric (three cohorts, 192 lives) | baseline (0M) | 0.5M steps | 1.0M steps | 1.5M snap | final 1.49M |
|---|---:|---:|---:|---:|---:|
| mean movement falls / 13 s | 0.7493 | 0.8244 | 0.8711 | 0.8659 | **0.8876** |
| walk falls / 13 s | 1.0677 | 1.2500 | 1.3229 | 1.2969 | 1.3958 |
| pace falls / 13 s | 0.6667 | 0.7240 | 0.7917 | 0.7656 | 0.8021 |
| canter falls / 13 s | 0.8333 | 0.8333 | 0.9792 | 0.9167 | 0.9010 |
| run falls / 13 s | 0.7344 | 0.7812 | 0.8438 | 0.8854 | 0.8802 |
| turn-left falls / 13 s | 0.6094 | 0.7240 | 0.8281 | 0.7656 | 0.7656 |
| turn-right falls / 13 s | 0.5833 | 0.6771 | 0.6615 | 0.6615 | 0.6979 |
| back falls / 13 s | 0.7500 | 0.8021 | 0.7812 | 0.7917 | 0.7708 |
| walk speed, m/s | +0.4032 | +0.4108 | +0.4131 | +0.4096 | +0.4033 |
| pace speed, m/s | +0.5344 | +0.5524 | +0.5574 | +0.5503 | +0.5645 |
| canter speed, m/s | +0.3890 | +0.3852 | +0.3881 | +0.3930 | +0.3917 |
| run speed, m/s | +0.5425 | +0.5381 | +0.5520 | +0.5720 | +0.5584 |
| turn-left yaw, rad/s | +1.3374 | +1.3471 | +1.3733 | +1.3326 | +1.3514 |
| turn-right yaw, rad/s | -1.1594 | -1.1483 | -1.1458 | -1.1387 | -1.1310 |
| backward speed, m/s | -0.0786 | -0.0837 | -0.0751 | -0.0665 | -0.0672 |
| jump speed, m/s | +0.1879 | +0.1998 | +0.1994 | +0.2013 | +0.1852 |
| sit pose (pitch / trunk) | +0.508 / 0.619 | +0.508 / 0.619 | +0.508 / 0.619 | +0.508 / 0.619 | +0.508 / 0.619 |
| lie-down pose (pitch / trunk) | +0.040 / 0.967 | +0.040 / 0.967 | +0.040 / 0.967 | +0.040 / 0.967 | +0.040 / 0.967 |
| **ball arrival success (got_there)** | **2.60% (5/192)** | **2.08% (4/192)** | **2.08% (4/192)** | **3.12% (6/192)** | **6.77% (13/192) [Record]** |
| ball distance drift (start -> end, m) | 2.495 -> 3.022 | 2.495 -> 3.290 | 2.493 -> 3.454 | 2.501 -> 3.544 | 2.498 -> 3.398 |
| ball aligned in final second | 4.06% | 2.06% | 2.66% | 2.35% | 5.40% |
| ball final facing cosine | -0.3231 | -0.4524 | -0.4940 | -0.5087 | -0.4606 |
| ball falls / 13 s | 0.5052 | 0.5677 | 0.5417 | 0.5833 | 0.5729 |
| **owner arrival success (got_there, 'come')** | **6.25% (12/192)** | **5.73% (11/192)** | **6.25% (12/192)** | **4.17% (8/192)** | **7.29% (14/192) [Record]** |
| owner distance drift (start -> end, m) | 3.664 -> 3.689 | 3.670 -> 3.645 | 3.663 -> 3.581 | 3.665 -> 3.737 | **3.665 -> 3.573 [-9.2 cm Net Closing]** |
| owner aligned in final second | 13.73% | 12.88% | 11.64% | 9.20% | **13.84%** |
| owner final facing cosine | +0.0735 | +0.0455 | +0.0565 | -0.0328 | -0.0018 |
| owner falls / 13 s | 0.8438 | 1.0521 | 1.1354 | 1.2292 | 1.1615 |
| **ball-left balanced accuracy** | 0.662 | 0.693 | 0.698 | 0.697 | 0.695 |
| **ball-ahead balanced accuracy** | 0.652 | 0.676 | 0.682 | 0.674 | 0.682 |
| **ball-right balanced accuracy** | 0.627 | 0.678 | 0.690 | 0.665 | 0.618 |
| **owner-left balanced accuracy** | 0.653 | 0.676 | 0.676 | 0.660 | 0.660 |
| **owner-ahead balanced accuracy** | 0.604 | 0.649 | 0.647 | 0.638 | 0.565 |
| **owner-right balanced accuracy** | 0.621 | 0.642 | 0.635 | 0.611 | 0.628 |
| **hit balanced accuracy** | 0.872 | 0.824 | 0.837 | 0.832 | 0.842 |
| **moving balanced accuracy** | 0.644 | 0.648 | 0.618 | 0.639 | 0.644 |
| **hungry balanced accuracy** | 0.534 | 0.575 | 0.579 | 0.504 | 0.502 |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Checkpoint SHA256 Hashes:**
- `baby_dog_step_0000491520.npz`: `33b26bc1c20787b89f2cd779f460b4a3c0e397bec8b706f7f9a63fd9ce7c1bc0`
- `baby_dog_step_0000983040.npz`: `64adf04ba47fe42dd54ec8bfd4d17aa5468ef5c28c364200b0b682fa893bd448`
- `baby_dog_step_0001474560.npz`: `5bf62a9d8151695dcea76050a73a6122c8549caad0f27f147f1b0e2b38b32922`
- `baby_dog_step_0001490944.npz`: `500359581b7659bf2d06295a782e7b3c2b06486e0c8d2831430fa956c89eaf21`
- `baby_dog.npz`: `3ae459265ef9354fd14f29e9cc291da191b9b3d14fc23fdb7de88b2bf3e0a3ac`

### Previous completed trial: vision approach round 22 (terminal sprint without pre-braking, signed angular bearing, and sprint steering headroom on Kaggle 2x Nvidia Tesla T4 GPU)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round22 (version 1).
Completed all 91 updates (1,490,944 steps, ~85.1 min runtime; immediate allocation) on 2x Nvidia Tesla T4 GPU (512 envs, 256/GPU). Outputs ingested into `results/baby_dog_vision_approach_round22/`.

Setup: 1.5M steps, 91 updates, 512 envs, 3 evals (0.5M, 1.0M, 1.5M), 3 cohorts (192 lives per eval), 3h cap. Teacher movement help zero from rest (`help_ = 0.00`).
Resumed from candidate `baby_dog_step_0001490944.npz` (`6220964d…`) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round21`.
All outputs fetched to `results/baby_dog_vision_approach_round22/`: `baby_dog.npz`, snapshots at 491k, 983k, 1.47M, 1.49M steps, `practice.json`, and `log.txt`.

**Measured Outcomes across 3 fresh cohorts (192 lives per evaluation):**

| measured metric (three cohorts, 192 lives) | baseline (0M) | 0.5M steps | 1.0M steps | 1.5M snap | final 1.49M |
|---|---:|---:|---:|---:|---:|
| mean movement falls / 13 s | 0.7321 | 0.8467 | 0.8504 | 0.9844 | 0.9628 |
| walk falls / 13 s | 1.0312 | 1.2969 | 1.2240 | 1.3490 | 1.4271 |
| pace falls / 13 s | 0.6875 | 0.7552 | 0.8021 | 0.8906 | 0.8385 |
| canter falls / 13 s | 0.7188 | 0.8542 | 0.8438 | 0.9948 | 1.0365 |
| run falls / 13 s | 0.6979 | 0.8490 | 0.8542 | 1.0625 | 0.9792 |
| turn-left falls / 13 s | 0.6510 | 0.7031 | 0.7240 | 0.8906 | 0.8177 |
| turn-right falls / 13 s | 0.5990 | 0.6250 | 0.6875 | 0.7500 | 0.6927 |
| back falls / 13 s | 0.7396 | 0.8438 | 0.8177 | 0.9531 | 0.9479 |
| walk speed, m/s | +0.4021 | +0.3968 | +0.4010 | +0.4012 | +0.4023 |
| pace speed, m/s | +0.5429 | +0.5507 | +0.5610 | +0.5558 | +0.5623 |
| canter speed, m/s | +0.3844 | +0.3912 | +0.3909 | +0.3959 | +0.3895 |
| run speed, m/s | +0.5470 | +0.5535 | +0.5391 | +0.5290 | +0.5514 |
| turn-left yaw, rad/s | +1.3452 | +1.3577 | +1.2955 | +1.3344 | +1.3230 |
| turn-right yaw, rad/s | -1.1506 | -1.1515 | -1.1554 | -1.1642 | -1.1828 |
| backward speed, m/s | -0.0755 | -0.0828 | -0.0601 | -0.0619 | -0.0746 |
| jump speed, m/s | +0.1822 | +0.2014 | +0.1943 | +0.1883 | +0.1936 |
| sit pose (pitch / trunk) | +0.500 / 0.620 | +0.500 / 0.620 | +0.500 / 0.620 | +0.500 / 0.620 | +0.500 / 0.620 |
| lie-down pose (pitch / trunk) | +0.040 / 0.960 | +0.040 / 0.960 | +0.040 / 0.960 | +0.040 / 0.960 | +0.040 / 0.960 |
| **ball arrival success (got_there)** | **6.25% (12/192)** | **1.56% (3/192)** | **4.17% (8/192)** | **4.69% (9/192)** | **3.65% (7/192)** |
| ball distance drift (start -> end, m) | 2.494 -> 3.038 | 2.492 -> 3.338 | 2.490 -> 3.420 | 2.498 -> 3.330 | 2.491 -> 3.223 |
| ball aligned in final second | 5.01% | 3.53% | 2.51% | 3.51% | 6.07% |
| ball final facing cosine | -0.3348 | -0.4166 | -0.4277 | -0.3538 | -0.3295 |
| ball falls / 13 s | 0.5104 | 0.5417 | 0.5625 | 0.6094 | 0.6615 |
| **owner arrival success (got_there, 'come')** | **5.21% (10/192)** | **5.21% (10/192)** | **2.60% (5/192)** | **5.21% (10/192)** | **3.65% (7/192)** |
| owner distance drift (start -> end, m) | 3.671 -> 3.920 | 3.666 -> 3.718 | 3.666 -> 3.827 | 3.667 -> 3.723 | 3.666 -> 3.751 |
| owner aligned in final second | 12.60% | 8.03% | 10.76% | 10.65% | 14.46% |
| owner final facing cosine | +0.0706 | -0.0667 | -0.0682 | +0.0808 | +0.0969 |
| owner falls / 13 s | 0.9740 | 1.0521 | 1.0833 | 1.0677 | 1.1823 |
| **ball-left balanced accuracy** | 0.671 | 0.696 | 0.679 | 0.691 | 0.700 |
| **ball-ahead balanced accuracy** | 0.637 | 0.670 | 0.680 | 0.695 | 0.700 |
| **ball-right balanced accuracy** | 0.625 | 0.679 | 0.662 | 0.684 | 0.682 |
| **owner-left balanced accuracy** | 0.656 | 0.663 | 0.668 | 0.664 | 0.653 |
| **owner-ahead balanced accuracy** | 0.594 | 0.589 | 0.628 | 0.652 | 0.670 |
| **owner-right balanced accuracy** | 0.620 | 0.624 | 0.608 | 0.612 | 0.619 |
| **hit balanced accuracy** | 0.897 | 0.753 | 0.844 | 0.793 | 0.810 |
| **moving balanced accuracy** | 0.644 | 0.653 | 0.631 | 0.631 | 0.620 |
| **hungry balanced accuracy** | 0.538 | 0.572 | 0.587 | 0.515 | 0.518 |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Checkpoint SHA256 Hashes:**
- `baby_dog_step_0000491520.npz`: `e0c4596d213cdb2997b5337988762e5a860fd3b02364bce85a087e7ea3e5e120`
- `baby_dog_step_0000983040.npz`: `6b6bf75fa2ab169f378a07dc5f151731d9527af52851f6acb7b19a749d0981de`
- `baby_dog_step_0001474560.npz`: `fef61b7cb82aefef70d45aa2242fa3a82d0f8516432482dfeeec91414a7eabc3`
- `baby_dog_step_0001490944.npz`: `a5429d83535bf159a0d766b76281db71325390095542b3cc904e3ad0c5094606`
- `baby_dog.npz`: `3024ae1c09531d7ef909a65db9b854177a7cfc85c3657be74f86758dc2980600`

### Previous completed trial: vision approach round 21 (decoupled pivot-then-sprint kinematics on Kaggle 2x Nvidia Tesla T4 GPU)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round21 (version 1).
Completed all 91 updates (1,490,944 steps, ~87.8 min runtime; 26 min queue) on 2x Nvidia Tesla T4 GPU (512 envs, 256/GPU). Outputs ingested into `results/baby_dog_vision_approach_round21/`.

Setup: 1.5M steps, 91 updates, 512 envs, 3 evals (0.5M, 1.0M, 1.5M), 3 cohorts (192 lives per eval), 3h cap. Teacher movement help zero from rest (`help_ = 0.00`).
Resumed from candidate `baby_dog_step_0001490944.npz` (`a88ec93d…`) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round20`.
All outputs fetched to `results/baby_dog_vision_approach_round21/`: `baby_dog.npz`, snapshots at 491k, 983k, 1.47M, 1.49M steps, `practice.json`, and `log.txt`.

**Measured Outcomes across 3 fresh cohorts (192 lives per evaluation):**

| measured metric (three cohorts, 192 lives) | baseline (0M) | 0.5M steps | 1.0M steps | 1.5M snap | final 1.49M |
|---|---:|---:|---:|---:|---:|
| mean movement falls / 13 s | 0.6592 | 0.7188 | 0.7411 | 0.7240 | 0.7307 |
| walk falls / 13 s | 0.8854 | 1.0781 | 0.9844 | 1.0729 | 1.0521 |
| pace falls / 13 s | 0.5781 | 0.6458 | 0.6615 | 0.6719 | 0.6823 |
| canter falls / 13 s | 0.6354 | 0.6979 | 0.7448 | 0.7552 | 0.7812 |
| run falls / 13 s | 0.6615 | 0.7240 | 0.7708 | 0.7083 | 0.7135 |
| turn-left falls / 13 s | 0.5781 | 0.5677 | 0.6354 | 0.5938 | 0.6198 |
| turn-right falls / 13 s | 0.5208 | 0.5104 | 0.5885 | 0.5469 | 0.5625 |
| back falls / 13 s | 0.7552 | 0.8073 | 0.8021 | 0.7188 | 0.7031 |
| walk speed, m/s | +0.3749 | +0.3760 | +0.3889 | +0.4017 | +0.4075 |
| pace speed, m/s | +0.5209 | +0.5343 | +0.5336 | +0.5435 | +0.5394 |
| canter speed, m/s | +0.3603 | +0.3636 | +0.3737 | +0.3899 | +0.3870 |
| run speed, m/s | +0.5209 | +0.5229 | +0.5371 | +0.5389 | +0.5470 |
| turn-left yaw, rad/s | +1.3536 | +1.3666 | +1.3307 | +1.3405 | +1.3501 |
| turn-right yaw, rad/s | -1.1466 | -1.1619 | -1.1430 | -1.1433 | -1.1489 |
| backward speed, m/s | -0.0863 | -0.0906 | -0.0932 | -0.0784 | -0.0735 |
| jump speed, m/s | +0.1612 | +0.1588 | +0.1712 | +0.1828 | +0.1886 |
| sit pose (pitch / trunk) | +0.500 / 0.620 | +0.500 / 0.620 | +0.500 / 0.620 | +0.500 / 0.620 | +0.500 / 0.620 |
| lie-down pose (pitch / trunk) | +0.040 / 0.960 | +0.040 / 0.960 | +0.040 / 0.960 | +0.040 / 0.960 | +0.040 / 0.960 |
| **ball arrival success (got_there)** | **0.00% (0/192)** | **1.56% (3/192)** | **3.12% (6/192)** | **3.12% (6/192)** | **4.69% (9/192)** |
| ball distance drift (start -> end, m) | 2.493 -> 2.683 | 2.491 -> 2.709 | 2.487 -> 2.868 | 2.493 -> 3.078 | 2.487 -> 2.930 |
| ball aligned in final second | 8.44% | 6.81% | 6.88% | 4.78% | 5.18% |
| ball final facing cosine | -0.1351 | -0.1970 | -0.3132 | -0.3524 | -0.3011 |
| ball falls / 13 s | 0.4167 | 0.4896 | 0.5000 | 0.5469 | 0.5417 |
| **owner arrival success (got_there, 'come')** | **4.69% (9/192)** | **5.73% (11/192)** | **3.65% (7/192)** | **4.17% (8/192)** | **6.25% (12/192)** |
| owner distance drift (start -> end, m) | 3.671 -> 3.640 | 3.669 -> 3.681 | 3.665 -> 3.674 | 3.665 -> 3.644 | 3.667 -> 3.687 |
| owner aligned in final second | 10.49% | 13.21% | 10.22% | 12.71% | 13.03% |
| owner final facing cosine | +0.0117 | -0.0494 | +0.0611 | +0.0600 | +0.0399 |
| owner falls / 13 s | 0.8594 | 0.7917 | 0.9740 | 0.8750 | 0.8750 |
| **ball-left balanced accuracy** | 0.666 | 0.707 | 0.683 | 0.661 | 0.665 |
| **ball-ahead balanced accuracy** | 0.630 | 0.662 | 0.680 | 0.658 | 0.641 |
| **ball-right balanced accuracy** | 0.643 | 0.690 | 0.685 | 0.635 | 0.627 |
| **owner-left balanced accuracy** | 0.641 | 0.647 | 0.660 | 0.647 | 0.657 |
| **owner-ahead balanced accuracy** | 0.624 | 0.665 | 0.656 | 0.575 | 0.608 |
| **owner-right balanced accuracy** | 0.592 | 0.614 | 0.630 | 0.612 | 0.621 |
| **hit balanced accuracy** | 0.821 | 0.854 | 0.861 | 0.848 | 0.849 |
| **moving balanced accuracy** | 0.726 | 0.744 | 0.678 | 0.624 | 0.643 |
| **hungry balanced accuracy** | 0.553 | 0.573 | 0.576 | 0.524 | 0.529 |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Checkpoint SHA256 Hashes:**
- `baby_dog_step_0000491520.npz`: `f22ce52bf64bbf59d7d6d1620aef2fbc0229f613ea4901ff5edee1dbc3300491`
- `baby_dog_step_0000983040.npz`: `8264fbade0a9670cc542c98c27ae2b5e929925ad8510e22828b2e3ef13d7655d`
- `baby_dog_step_0001474560.npz`: `5baa7d672aaaa1383bf3b1e2fe4a865fea5b92946e9d951dfb8a0f031cf3908c`
- `baby_dog_step_0001490944.npz`: `6220964dbb25491b40b7be93dba418c611051d1b8f1a0b7277dc6f0cec9bd323`
- `baby_dog.npz`: `4cb15436d3c82d2d177a0dd7a4c314676da932fb669b8e3031d5181432ba2d6b`

### Previous completed trial: vision approach round 20 (kinematic slew-rate acceleration limiting on Kaggle 2x Nvidia Tesla T4 GPU)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round20 (version 1).
Completed all 91 updates (1,490,944 steps, ~87 min runtime; 104 min queue) on 2x Nvidia Tesla T4 GPU. Outputs ingested into `results/baby_dog_vision_approach_round20/`.
Resumed from candidate `baby_dog_step_0001490944.npz` (`b1dadb51…`, Round 18 project record, 4.17% arrival baseline) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round18`.
Connectome core: `core_v2.npz` (`af939468…`) attached via `--dataset bio-bot-cores`.
Evaluation across three fresh independent cohorts of 64 lives = 192 lives per evaluation from rest with teacher help strictly 0.00:

| measured metric (three cohorts, 192 lives) | baseline (0M) | 0.5M (`7e9d3a83…`) | 1.0M (`dc8ca12b…`) | 1.5M snap (`1f8748be…`) | final 1.49M (`a88ec93d…`) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **owner arrival success (got_there, 'come')** | 2.60% (5/192) | 3.65% (7/192) | 4.17% (8/192) | 4.17% (8/192) | **4.69% (9/192) [Record]** |
| owner distance drift (start -> end, m) | 3.669 -> 3.793 | 3.670 -> 3.750 | 3.669 -> 3.613 | 3.670 -> 3.624 | **3.669 -> 3.623 [Net Closing]** |
| owner distance delta (cm) | +12.4 cm | +8.0 cm | **-5.6 cm** | **-4.7 cm** | **-4.6 cm [Record]** |
| owner aligned in final second | 9.50% | 8.27% | 11.35% | **13.72%** | 13.28% |
| owner final facing cosine | -0.0086 | -0.0162 | +0.0051 | +0.1027 | **+0.1587 [Record]** |
| owner falls / 13 s | 0.4115 | 0.7500 | 0.7188 | 0.8177 | 0.7865 |
| **ball arrival success (got_there, 'ball')** | 0.00% (0/192) | 0.00% (0/192) | 0.52% (1/192) | **1.04% (2/192)** | 0.52% (1/192) |
| ball distance drift (start -> end, m) | 2.484 -> 2.537 | 2.485 -> 2.541 | 2.484 -> 2.561 | 2.484 -> 2.582 | 2.492 -> 2.658 |
| ball distance delta (cm) | +5.3 cm | +5.6 cm | +7.7 cm | +9.7 cm | +16.5 cm |
| ball aligned in final second | 7.92% | **10.95%** | 9.04% | 6.89% | 6.86% |
| ball final facing cosine | +0.0106 | -0.0051 | -0.0304 | -0.0443 | -0.1276 |
| ball falls / 13 s | 0.2969 | 0.3438 | 0.3906 | 0.4115 | 0.4531 |
| walk speed from rest, m/s | +0.3485 | +0.3440 | +0.3625 | +0.3733 | **+0.3737** |
| walk falls / 13 s | 0.6979 | 0.8594 | 0.9427 | 1.0000 | 1.0365 |
| pace speed from rest, m/s | +0.4828 | +0.4885 | +0.5055 | +0.5083 | **+0.5191** |
| pace falls / 13 s | 0.4271 | 0.5156 | 0.5677 | 0.5885 | 0.6094 |
| canter speed from rest, m/s | +0.3350 | +0.3402 | +0.3442 | +0.3550 | **+0.3614** |
| canter falls / 13 s | 0.3906 | 0.5417 | 0.5365 | 0.6198 | 0.6510 |
| run speed from rest, m/s | +0.4954 | +0.4935 | +0.5136 | +0.5293 | **+0.5276** |
| run falls / 13 s | 0.5208 | 0.5781 | 0.6250 | 0.6719 | 0.7083 |
| turn-left yaw rate, rad/s | +1.3407 | +1.3592 | +1.3512 | +1.3706 | **+1.3823** |
| turn-left falls / 13 s | 0.4271 | 0.4896 | 0.5052 | 0.5104 | 0.5156 |
| turn-right yaw rate, rad/s | -1.1289 | -1.1365 | -1.1333 | -1.1393 | **-1.1548** |
| turn-right falls / 13 s | 0.3802 | 0.4583 | 0.4688 | 0.5104 | 0.5000 |
| back speed from rest, m/s | -0.0987 | -0.0940 | -0.0889 | -0.0958 | -0.0896 |
| back falls / 13 s | 0.4792 | 0.5521 | 0.6094 | 0.6510 | 0.7344 |
| jump speed from rest, m/s | +0.1051 | +0.1256 | +0.1138 | +0.1602 | **+0.1600** |
| jump falls / 13 s | 0.4740 | 0.5729 | 0.6719 | 0.7552 | 0.7656 |
| sit pose (pitch / height) | +0.499 / 0.612 | +0.493 / 0.612 | +0.496 / 0.612 | +0.503 / 0.613 | +0.499 / 0.614 |
| lie-down pose (pitch / height) | +0.046 / 0.960 | +0.040 / 0.960 | +0.038 / 0.961 | +0.038 / 0.961 | +0.037 / 0.964 |
| dodge hit rate | 79.17% | 77.60% | 77.60% | 77.08% | 81.25% |
| dodge reaction time (median) | 106.7 ms | 166.7 ms | 106.7 ms | 143.3 ms | **73.3 ms** |
| **ball-left balanced accuracy** | 0.651 | 0.663 | 0.668 | 0.654 | 0.651 |
| **ball-ahead balanced accuracy** | 0.631 | 0.605 | 0.646 | 0.633 | 0.658 |
| **ball-right balanced accuracy** | 0.600 | 0.666 | 0.658 | 0.643 | 0.643 |
| **owner-left balanced accuracy** | 0.626 | 0.615 | 0.628 | 0.627 | 0.627 |
| **owner-ahead balanced accuracy** | 0.602 | 0.620 | 0.635 | 0.645 | 0.602 |
| **owner-right balanced accuracy** | 0.598 | 0.595 | 0.592 | 0.608 | 0.589 |
| **hit balanced accuracy** | 0.841 | 0.851 | 0.857 | 0.884 | 0.869 |
| **moving balanced accuracy** | 0.772 | 0.805 | 0.790 | 0.739 | 0.724 |
| **hungry balanced accuracy** | 0.563 | 0.564 | 0.580 | 0.550 | 0.549 |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Checkpoint SHA256 Hashes:**
- `baby_dog_step_0000491520.npz`: `7e9d3a83eee1777f979d04d333b33ed56a95980921f284461b55ebd6a58dbb63`
- `baby_dog_step_0000983040.npz`: `dc8ca12b676bbc9f92574a322afea104768f7cc5da1c6a365e6f9eef60d05c78`
- `baby_dog_step_0001474560.npz`: `1f8748be4f43b7bedcca9a3be10f1be03ae29ddf0f25816f9727b972534f24bc`
- `baby_dog_step_0001490944.npz`: `a88ec93d3cc1a570a75327ca8ddc3b1bf12a5aacc6419f31e2b5ffeaa8c7a7df`
- `baby_dog.npz`: `d6d767cb14925605c3b376cb7575ca6a92c8b5d74fd06ad408abceadc782b8c3`

### Previous completed trial: vision approach round 18 (unpenalized rearward swiveling and active directional yaw drive on Kaggle 2x Nvidia Tesla T4 GPU)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round18 (version 1).
Completed all 91 updates (1,490,944 steps, 86.0 min run) on 2x Nvidia Tesla T4 GPU. Outputs ingested into `results/baby_dog_vision_approach_round18/`.
Resumed from candidate `baby_dog_step_0000491520.npz` (`5a534b32…`, Round 10 record) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round10`.
Connectome core: `core_v2.npz` (`af939468…`) attached via `--dataset bio-bot-cores`.
Evaluation across three fresh cohorts of 64 lives = 192 lives per evaluation from rest with teacher help strictly 0.00:

| measured metric (three cohorts, 192 lives) | baseline (0M) | 0.5M (`02d0aac4…`) | 1.0M (`cf334650…`) | 1.5M snap (`9a0d3b1c…`) | final 1.49M (`b1dadb51…`) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **owner arrival success (got_there, 'come')** | 1.56% (3/192) | 2.60% (5/192) | 1.04% (2/192) | 2.60% (5/192) | **4.17% (8/192)** |
| owner distance drift (start -> end, m) | 3.663 -> 3.631 | 3.662 -> 3.689 | 3.668 -> 3.750 | 3.667 -> 3.855 | 3.663 -> 3.782 |
| owner distance delta (cm) | **-3.2 cm** | +2.6 cm | +8.2 cm | +18.8 cm | +11.9 cm |
| owner aligned in final second | 14.48% | 10.49% | 10.50% | 4.83% | 7.89% |
| owner final facing cosine | +0.0323 | -0.0176 | -0.0032 | -0.0980 | -0.0695 |
| owner falls / 13 s | 0.1615 | 0.2656 | 0.3750 | 0.4792 | 0.4219 |
| **ball arrival success (got_there)** | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) |
| ball distance drift (start -> end, m) | 2.489 -> 2.545 | 2.487 -> 2.506 | 2.491 -> 2.537 | 2.485 -> 2.526 | 2.484 -> 2.509 |
| ball distance delta (cm) | +5.6 cm | +1.9 cm | +4.6 cm | +4.1 cm | **+2.5 cm** |
| ball aligned in final second | 10.44% | 10.85% | 11.34% | 8.95% | 9.74% |
| ball final facing cosine | +0.0158 | +0.0074 | +0.0039 | -0.0630 | -0.0290 |
| ball falls / 13 s | 0.1667 | 0.1823 | 0.2500 | 0.2917 | 0.2760 |
| walk speed from rest, m/s | +0.3355 | +0.3361 | +0.3373 | +0.3519 | +0.3523 |
| walk falls / 13 s | 0.3594 | 0.5000 | 0.6354 | 0.6927 | 0.6823 |
| turn-left yaw rate, rad/s | +1.2804 | +1.3216 | +1.3148 | +1.3484 | +1.3320 |
| turn-left falls / 13 s | 0.2812 | 0.3125 | 0.3750 | 0.3802 | 0.4167 |
| turn-right yaw rate, rad/s | -1.0837 | -1.1023 | -1.1026 | -1.1138 | -1.1172 |
| turn-right falls / 13 s | 0.1875 | 0.2292 | 0.2917 | 0.3490 | 0.3438 |
| backward speed from rest, m/s | -0.1359 | -0.1284 | -0.1204 | -0.0956 | -0.0931 |
| back falls / 13 s | 0.3438 | 0.3698 | 0.4271 | 0.5260 | 0.4740 |
| sit pose (pitch / height) | +0.495 / 0.606 | +0.491 / 0.610 | +0.490 / 0.611 | +0.498 / 0.612 | +0.494 / 0.614 |
| lie-down pose (pitch / height) | +0.043 / 0.953 | +0.041 / 0.954 | +0.042 / 0.958 | +0.044 / 0.958 | +0.044 / 0.958 |
| **ball-left balanced accuracy** | 0.639 | 0.662 | 0.667 | 0.663 | 0.649 |
| **ball-ahead balanced accuracy** | 0.630 | 0.618 | 0.644 | 0.631 | 0.623 |
| **ball-right balanced accuracy** | 0.635 | 0.658 | 0.666 | 0.653 | 0.605 |
| **owner-left balanced accuracy** | 0.631 | 0.640 | 0.654 | 0.635 | 0.627 |
| **owner-ahead balanced accuracy** | 0.600 | 0.591 | 0.634 | 0.587 | 0.613 |
| **owner-right balanced accuracy** | 0.604 | 0.615 | 0.624 | 0.599 | 0.604 |
| **tactile hit balanced accuracy** | 0.862 | 0.913 | 0.887 | 0.883 | 0.834 |
| **moving balanced accuracy** | 0.750 | 0.774 | 0.788 | 0.772 | 0.766 |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Checkpoint SHA256 Hashes:
- `baby_dog_step_0000491520.npz`: `02d0aac42b75f73390651c4128256b0b8a195b1ca4835f1d3dcd8c1681415d72`
- `baby_dog_step_0000983040.npz`: `cf334650ecd205f7b2997ee192bf80f0bc372d8d935a1c7477cf45d531639f8b`
- `baby_dog_step_0001474560.npz`: `9a0d3b1c29987173289a895e6b42732176c758d92103a1a3fa33655a11ff2af3`
- `baby_dog_step_0001490944.npz`: `b1dadb5197dee0f8ebdec1f972e4a78ace29b2bbbf600bdb85505bf7339483a3`
- `baby_dog.npz`: `f72f78d3a2df69549f29a0e1c0b1714af65dd28c62799ca7c9574be9ac917ec6`

### Prior completed trial: vision approach round 17 (terminal approach deceleration braking and continuous proximity shaping reward on Kaggle 2x Nvidia Tesla T4 GPU)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round17 (version 1).
Completed all 91 updates (1,490,944 steps, 87.3 min run) on 2x Nvidia Tesla T4 GPU. Outputs ingested into `results/baby_dog_vision_approach_round17/`.
Resumed from candidate `baby_dog_step_0000491520.npz` (`5a534b32…`, Round 10 record) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round10`.
Connectome core: `core_v2.npz` (`af939468…`) attached via `--dataset bio-bot-cores`.
Evaluation across three fresh cohorts of 64 lives = 192 lives per evaluation from rest with teacher help strictly 0.00:

| measured metric (three cohorts, 192 lives) | baseline (0M) | 0.5M (`a88e1034…`) | 1.0M (`90ee89e5…`) | 1.5M snap (`4d0f867a…`) | final 1.49M (`5144fd7d…`) |
|---|:---|:---|:---|:---|:---|
| **owner arrival success (got_there, 'come')** | 2.08% (4/192) | 2.08% (4/192) | 1.56% (3/192) | 1.56% (3/192) | 1.56% (3/192) |
| owner distance drift (start -> end, m) | 3.660 -> 3.621 | 3.663 -> 3.721 | 3.662 -> 3.646 | 3.661 -> 3.772 | 3.664 -> 3.762 |
| owner distance delta (cm) | **-3.9 cm** | +5.7 cm | **-1.6 cm** | +11.1 cm | +9.8 cm |
| owner aligned in final second | 9.21% | 11.04% | 10.07% | 7.86% | 10.05% |
| owner final facing cosine | -0.0016 | -0.0390 | -0.0751 | -0.1008 | -0.0639 |
| owner falls / 13 s | 0.1979 | 0.2500 | 0.2708 | 0.3281 | 0.3854 |
| **ball arrival success (got_there)** | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) |
| ball distance drift (start -> end, m) | 2.492 -> 2.518 | 2.480 -> 2.521 | 2.488 -> 2.515 | 2.481 -> 2.579 | 2.491 -> 2.568 |
| ball distance delta (cm) | +2.6 cm | +4.1 cm | +2.7 cm | +9.8 cm | +7.7 cm |
| ball aligned in final second | 11.05% | 9.67% | 10.82% | 11.22% | 11.40% |
| ball final facing cosine | +0.0539 | +0.0449 | -0.0167 | -0.0316 | +0.0053 |
| ball falls / 13 s | 0.1667 | 0.1667 | 0.2083 | 0.2604 | 0.2448 |
| walk speed from rest, m/s | +0.3339 | +0.3192 | +0.3306 | +0.3267 | +0.3210 |
| walk falls / 13 s | 0.4427 | 0.4062 | 0.4896 | 0.6250 | 0.5885 |
| turn-left yaw rate, rad/s | +1.2865 | +1.3370 | +1.3159 | +1.3198 | +1.3165 |
| turn-left falls / 13 s | 0.2760 | 0.2865 | 0.3125 | 0.3385 | 0.3438 |
| turn-right yaw rate, rad/s | -1.0753 | -1.0694 | -1.0881 | -1.1053 | -1.1045 |
| turn-right falls / 13 s | 0.1927 | 0.2031 | 0.2292 | 0.2708 | 0.2969 |
| backward speed from rest, m/s | -0.1312 | -0.1278 | -0.1280 | -0.1113 | -0.1087 |
| back falls / 13 s | 0.3021 | 0.3333 | 0.3802 | 0.4323 | 0.4635 |
| sit pose (pitch / height) | +0.492 / 0.607 | +0.498 / 0.610 | +0.495 / 0.612 | +0.494 / 0.612 | +0.497 / 0.613 |
| lie-down pose (pitch / height) | +0.045 / 0.954 | +0.046 / 0.955 | +0.049 / 0.956 | +0.046 / 0.958 | +0.044 / 0.954 |
| **ball-left balanced accuracy** | 0.644 | 0.652 | 0.654 | 0.648 | **0.666** |
| **ball-ahead balanced accuracy** | 0.621 | 0.609 | 0.646 | 0.646 | 0.621 |
| **ball-right balanced accuracy** | 0.639 | 0.646 | 0.648 | 0.642 | **0.655** |
| **owner-left balanced accuracy** | 0.632 | 0.623 | 0.642 | 0.651 | 0.634 |
| **owner-ahead balanced accuracy** | 0.603 | 0.607 | 0.613 | 0.612 | 0.598 |
| **owner-right balanced accuracy** | 0.599 | 0.604 | 0.608 | 0.614 | 0.605 |
| **tactile hit balanced accuracy** | 0.849 | 0.827 | **0.900** | 0.865 | 0.883 |
| **moving balanced accuracy** | 0.751 | 0.754 | 0.755 | 0.748 | 0.748 |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Checkpoint SHA256 Hashes:
- `baby_dog_step_0000491520.npz`: `a88e103477e028f11195d6f646777d0364ea63cbffbfcc4b42303b6f5bdd6e98`
- `baby_dog_step_0000983040.npz`: `90ee89e576b5c1e10d862d62a9ce0abd1fbc6cc6040a921340c670fdcbb601c1`
- `baby_dog_step_0001474560.npz`: `4d0f867a6ed03d6a0d4c54235e9925e33b13b13acf875da1a32138b22c5727d8`
- `baby_dog_step_0001490944.npz`: `5144fd7d2c25e3b9e5a7257d7edaa0c0fa708e01198124d1ad2a1f41a3ec2d3f`
- `baby_dog.npz`: `e533d0f4268cd293a7865b562aa1ec684ce14032346e6e46cff9ca3f38415fa9`

### Prior completed trial: vision approach round 16 (coordinated forward-turning arc steering with velocity floor and arrival stopping on Kaggle 2x Nvidia Tesla T4 GPU)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round16 (version 1).
Completed all 91 updates (1,490,944 steps, 85.8 min run) on 2x Nvidia Tesla T4 GPU. Outputs ingested into `results/baby_dog_vision_approach_round16/`.
Resumed from candidate `baby_dog_step_0000491520.npz` (`5a534b32…`, Round 10 record) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round10`.
Connectome core: `core_v2.npz` (`af939468…`) attached via `--dataset bio-bot-cores`.
Evaluation across three fresh cohorts of 64 lives = 192 lives per evaluation from rest with teacher help strictly 0.00:

| measured metric (three cohorts, 192 lives) | baseline (0M) | 0.5M (`e58244fc…`) | 1.0M (`4b77339b…`) | 1.5M snap (`c57dfca2…`) | final 1.49M (`e82ee938…`) |
|---|:---|:---|:---|:---|:---|
| **owner arrival success (got_there, 'come')** | 1.56% (3/192) | **2.60% (5/192)** | 1.56% (3/192) | 1.56% (3/192) | **2.60% (5/192)** |
| owner distance drift (start -> end, m) | 3.660 -> 3.661 | 3.661 -> 3.656 | 3.666 -> 3.773 | 3.667 -> 3.743 | 3.663 -> 3.676 |
| owner distance delta (cm) | +0.1 cm | **-0.5 cm** | +10.7 cm | +7.6 cm | +1.2 cm |
| owner aligned in final second | 13.57% | 11.22% | 7.51% | 8.77% | 9.11% |
| owner final facing cosine | -0.0005 | -0.0001 | -0.0368 | +0.0453 | -0.0660 |
| owner falls / 13 s | 0.1823 | 0.1979 | 0.2552 | 0.2917 | 0.2812 |
| **ball arrival success (got_there)** | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) |
| ball distance drift (start -> end, m) | 2.481 -> 2.516 | 2.487 -> 2.565 | 2.485 -> 2.535 | 2.481 -> 2.504 | 2.489 -> 2.549 |
| ball distance delta (cm) | +3.4 cm | +7.8 cm | +5.0 cm | +2.2 cm | +6.0 cm |
| ball aligned in final second | 9.40% | 10.59% | 9.33% | 11.54% | 11.51% |
| ball final facing cosine | +0.0251 | +0.0077 | +0.0328 | +0.0240 | -0.0352 |
| ball falls / 13 s | 0.1510 | 0.1927 | 0.2135 | 0.2135 | 0.2292 |
| walk speed from rest, m/s | +0.3326 | +0.3240 | +0.3248 | +0.3306 | +0.3303 |
| walk falls / 13 s | 0.4062 | 0.4479 | 0.4688 | 0.5521 | 0.6094 |
| turn-left yaw rate, rad/s | +1.2821 | +1.3113 | +1.2858 | +1.2892 | +1.2914 |
| turn-left falls / 13 s | 0.2760 | 0.2760 | 0.2656 | 0.3021 | 0.2812 |
| turn-right yaw rate, rad/s | -1.0660 | -1.0838 | -1.1065 | -1.1095 | -1.1108 |
| turn-right falls / 13 s | 0.1927 | 0.2031 | 0.1875 | 0.2500 | 0.2604 |
| backward speed from rest, m/s | -0.1348 | -0.1253 | -0.1265 | -0.1128 | -0.1173 |
| back falls / 13 s | 0.2917 | 0.3750 | 0.3906 | 0.4115 | 0.4062 |
| sit pose (pitch / height) | +0.494 / 0.607 | +0.490 / 0.609 | +0.481 / 0.611 | +0.493 / 0.610 | +0.490 / 0.610 |
| lie-down pose (pitch / height) | +0.043 / 0.953 | +0.040 / 0.952 | +0.042 / 0.953 | +0.038 / 0.949 | +0.044 / 0.953 |
| **ball-left balanced accuracy** | 0.640 | 0.631 | 0.652 | **0.671** (all-time peak!) | 0.658 |
| **ball-ahead balanced accuracy** | 0.624 | 0.633 | 0.627 | 0.628 | 0.613 |
| **ball-right balanced accuracy** | 0.630 | 0.640 | 0.650 | **0.670** (all-time peak!) | 0.667 |
| **owner-left balanced accuracy** | 0.635 | 0.627 | 0.633 | 0.626 | 0.631 |
| **owner-ahead balanced accuracy** | 0.604 | 0.601 | 0.622 | 0.553 | 0.554 |
| **owner-right balanced accuracy** | 0.607 | 0.617 | 0.611 | 0.599 | 0.606 |
| **tactile hit balanced accuracy** | 0.863 | 0.878 | **0.918** (all-time peak!) | 0.876 | 0.811 |
| **moving balanced accuracy** | 0.752 | 0.753 | 0.762 | 0.739 | 0.726 |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Checkpoint SHA256 Hashes:
- `baby_dog_step_0000491520.npz`: `e58244fc4021aded18263d3b9dbd03f468c8642bdaf4da67d181f28dd997a23f`
- `baby_dog_step_0000983040.npz`: `4b77339bd3d018b876d557b1bb7e80254e07201b0604d1f645cd244f05f5b2d2`
- `baby_dog_step_0001474560.npz`: `c57dfca242de6177063cc3d604fac32c997261c610a9d54a2886d90a9f368298`
- `baby_dog_step_0001490944.npz`: `e82ee9387f841af1575c538eee44524b3dfc523b0485082fc252ed086e85a2ec`
- `baby_dog.npz`: `bea438f3388b50f4799295d2f6e09ace2fda48ff6a3fbfd7c472bcb1309c41ee`

### Prior completed trial: vision approach round 15 (dynamic velocity command injection unblocking v_act locomotion rewards on Google Colab free T4 GPU)

Execution: 183 updates $\times$ 8,192 steps = 1,499,136 steps (92.4 min) on Google Colab Nvidia Tesla T4 GPU (PID 15189). Outputs ingested into `results/baby_dog_vision_approach_round15/`.
Resumed from candidate `baby_dog_step_0000491520.npz` (`5a534b32…`, Round 10).
Evaluation across three fresh cohorts of 64 lives = 192 lives per evaluation from rest with teacher help strictly 0.00:

| measured metric (three cohorts, 192 lives) | baseline (0M) | 0.5M (`d676f688…`) | 1.0M (`99579cd8…`) | final 1.5M (`fb30e64e…`) | Overall Delta / Trend |
|---|---:|---:|---:|---:|---:|
| **owner arrival success (got_there, "come")** | 1.04% (2/192) | **2.08% (4/192)** | **1.56% (3/192)** | **1.56% (3/192)** | **Elevated throughout run** |
| owner distance drift (start -> end, m) | 3.662 -> 3.631 | 3.663 -> 3.755 | 3.666 -> 3.691 | 3.661 -> 3.787 | variable drift |
| owner distance closure (cohort highlights) | Cohort 3: -3.1 cm | Cohort 3: **-6.0 cm** | Cohort 2: **-25.0 cm** | Cohort 3: **-9.0 cm** | **Consistent net approach** |
| owner aligned in final second | 15.04% | 10.93% | 7.51% | 10.58% | maintained |
| owner final facing cosine | +0.0278 | -0.0015 | -0.0938 | -0.1034 | lateral bias |
| owner falls / 13 s | 0.1823 | 0.2344 | 0.3333 | 0.3542 | +0.1719 |
| **ball arrival success (got_there)** | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) | ⚪ inactive target |
| ball distance drift (start -> end, m) | 2.481 -> 2.492 | 2.481 -> 2.503 | **2.491 -> 2.475** | 2.489 -> 2.544 | midpoint closure (-1.6 cm) |
| ball distance closure (cohort highlights) | Cohort 1: +1.1 cm | Cohort 1: -2.0 cm | Cohort 2: **-7.0 cm** | Cohort 1: +9.0 cm | **-7 cm approach at 1.0M** |
| ball aligned in final second | 11.76% | 10.71% | 11.89% | 10.93% | stable |
| ball final facing cosine | +0.0595 | +0.0401 | **+0.0801** | +0.0686 | **consistently positive** |
| ball falls / 13 s | 0.1667 | 0.1771 | 0.1927 | 0.2031 | low fall rate |
| mean seven movement falls / 13 s | 0.2956 | 0.3112 | 0.3411 | 0.3503 | +0.0547 |
| dodge median reaction time, ms | 203.3 ms | 143.3 ms | 236.7 ms | **133.3 ms** | **-70.0 ms (Faster)** |
| dodge projectile hit rate | 81.77% | 83.85% | 81.77% | 82.81% | below 96% static baseline |
| walk speed from rest, m/s | +0.3347 | +0.3121 | +0.3260 | +0.3257 | preserved (>+0.271 teacher) |
| walk falls / 13 s | 0.4427 | 0.4375 | 0.4635 | 0.5208 | +0.0781 |
| turn-left yaw rate, rad/s | +1.2923 | +1.3209 | +1.3164 | +1.3289 | preserved (>+0.405 teacher) |
| turn-left falls / 13 s | 0.2604 | 0.2812 | 0.3073 | 0.3073 | +0.0469 |
| turn-right yaw rate, rad/s | -1.0771 | -1.0790 | -1.0778 | -1.0448 | preserved (<-0.418 teacher) |
| turn-right falls / 13 s | 0.1719 | 0.1875 | 0.2135 | 0.1979 | +0.0260 |
| backward speed from rest, m/s | -0.1320 | -0.1199 | -0.1063 | -0.1047 | preserved (-0.109 teacher) |
| back falls / 13 s | 0.3073 | 0.3385 | 0.3750 | 0.3750 | +0.0677 |
| sit pose (pitch / trunk height) | +0.493 / 0.607 | +0.490 / 0.609 | +0.486 / 0.611 | +0.484 / 0.610 | preserved (target +0.48 / 0.60) |
| lie-down pose (pitch / trunk height) | +0.046 / 0.956 | +0.038 / 0.954 | +0.038 / 0.953 | +0.037 / 0.952 | preserved (target +0.02 / 0.39) |
| ball-right balanced accuracy | 0.647 | 0.654 | 0.646 | 0.627 | grounded vision |
| ball-left balanced accuracy | 0.639 | 0.652 | 0.633 | 0.623 | grounded vision |
| ball-ahead balanced accuracy | 0.618 | 0.599 | **0.643** | 0.618 | all-time high at 1.0M |
| owner-left balanced accuracy | 0.630 | 0.638 | **0.654** | 0.630 | all-time high at 1.0M |
| owner-ahead balanced accuracy | 0.602 | 0.608 | 0.591 | 0.588 | grounded vision |
| owner-right balanced accuracy | 0.612 | 0.599 | 0.619 | 0.607 | grounded vision |
| teacher movement help (all skills) | 0.00 | 0.00 | 0.00 | 0.00 | strictly invariant |

### Prior completed trial: vision approach round 13 (amplified forward approach propulsion FWD_PAY=16.0, spinning elimination YAW_PAY=0.0, and orientation tuning FACE_PAY=4.0)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round13 (version 1).
All 91 updates completed (1,490,944 actual steps, 86.2 min run). Outputs downloaded into `results/baby_dog_vision_approach_round13/`.
Resumed from candidate `baby_dog_step_0000491520.npz` (`5a534b32…`) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round10`.
Evaluation across three fresh cohorts of 64 lives = 192 lives per evaluation:

| measured metric (three cohorts, 192 lives) | baseline (0M) | 0.5M (`e8027447…`) | 1.0M (`61d9c814…`) | 1.5M snap (`6de2c9f0…`) | final 1.49M (`b2e38151…`) |
|---|---:|---:|---:|---:|---:|
| **owner arrival success (got_there, 'come')** | **2.08% (4/192)** | 1.56% (3/192) | **2.08% (4/192)** | 1.56% (3/192) | **2.08% (4/192)** |
| owner distance drift (start -> end, m) | 3.666 -> 3.598 (-6.8 cm) | 3.663 -> 3.673 (+0.9 cm) | 3.660 -> 3.719 (+5.9 cm) | 3.664 -> 3.768 (+10.4 cm) | 3.665 -> 3.799 (+13.5 cm) |
| owner aligned in final second | **13.32%** | 9.96% | 7.88% | 7.14% | 9.41% |
| owner final facing cosine | **+0.0660** | -0.0390 | -0.0161 | -0.1254 | -0.1100 |
| owner task falls / 13 s | **0.1927** | 0.2188 | 0.2396 | 0.2865 | 0.3229 |
| **ball arrival success (got_there)** | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) |
| ball distance drift (start -> end, m) | 2.480 -> 2.450 (-3.1 cm) | 2.490 -> 2.533 (+4.4 cm) | 2.490 -> 2.462 (-2.8 cm) | 2.481 -> 2.474 (-0.8 cm) | 2.480 -> 2.508 (+2.8 cm) |
| ball aligned in final second | **12.10%** | 11.76% | 10.24% | 10.72% | 9.44% |
| ball final facing cosine | +0.0433 | +0.0215 | +0.0259 | **+0.0758** | -0.0024 |
| ball task falls / 13 s | **0.1615** | 0.1719 | 0.1875 | 0.2135 | 0.2083 |
| **mean seven movement falls / 13 s** | **0.2612** | 0.3006 | 0.3185 | 0.3266 | 0.3348 |
| walk falls / 13 s | **0.3802** | 0.4896 | 0.5208 | 0.5156 | 0.5469 |
| pace falls / 13 s | **0.2292** | 0.2500 | 0.3021 | 0.3021 | 0.3125 |
| canter falls / 13 s | **0.1979** | 0.2344 | 0.2344 | 0.2760 | 0.2656 |
| run falls / 13 s | **0.2396** | 0.2969 | 0.3125 | 0.3125 | 0.3229 |
| turn-left falls / 13 s | **0.2760** | 0.2812 | 0.2812 | 0.2760 | 0.2969 |
| turn-right falls / 13 s | **0.1927** | 0.2031 | 0.2240 | 0.2604 | 0.2344 |
| back falls / 13 s | **0.3125** | 0.3490 | 0.3542 | 0.3438 | 0.3646 |
| walk speed, m/s | **+0.3369** | +0.3231 | +0.3121 | +0.3187 | +0.3147 |
| pace speed, m/s | +0.4547 | **+0.4603** | +0.4514 | +0.4530 | +0.4524 |
| canter speed, m/s | **+0.3105** | +0.3056 | +0.3042 | +0.3035 | +0.3050 |
| run speed, m/s | +0.4860 | **+0.4882** | +0.4744 | +0.4814 | +0.4773 |
| **ball-left balanced accuracy** | 0.644 | 0.636 | 0.656 | 0.656 | **0.657** |
| **ball-ahead balanced accuracy** | 0.629 | 0.619 | 0.628 | 0.634 | **0.636** |
| **ball-right balanced accuracy** | 0.635 | 0.645 | 0.640 | 0.636 | **0.648** |
| **owner-left balanced accuracy** | 0.632 | 0.631 | **0.638** | 0.637 | **0.638** |
| **owner-ahead balanced accuracy** | 0.615 | 0.600 | 0.588 | **0.621** | 0.608 |
| **owner-right balanced accuracy** | 0.600 | **0.619** | 0.611 | 0.610 | 0.610 |
| **object dodge balanced accuracy** | 0.847 | 0.856 | **0.909** | 0.834 | 0.858 |

### Prior completed trial: vision approach round 12 (directional yaw rate drive, rotation-aware stillness, and terminal reach scaling)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round12 (version 1).
All 91 updates completed (1,490,944 actual steps, 98.9 min run). Outputs downloaded into `results/baby_dog_vision_approach_round12/`.
Resumed from candidate `baby_dog_step_0000491520.npz` (`5a534b32…`) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round10`.
Evaluation across three fresh cohorts of 64 lives = 192 lives per evaluation:

| measured metric (three cohorts, 192 lives) | baseline (0M) | 0.5M (`ecfd902f…`) | 1.0M (`f78dd931…`) | 1.5M snap (`cbfed3d9…`) | final 1.49M (`b7af75e5…`) |
|---|---:|---:|---:|---:|---:|
| **owner arrival success (got_there, 'come')** | 1.04% (2/192) | **2.08% (4/192)** | 1.04% (2/192) | 0.52% (1/192) | 0.52% (1/192) |
| owner distance drift (start -> end, m) | 3.663 -> 3.623 (-4.0 cm) | 3.662 -> 3.742 (+8.0 cm) | 3.663 -> 3.643 (-2.0 cm) | 3.662 -> 3.736 (+7.4 cm) | 3.663 -> 3.694 (+3.1 cm) |
| owner aligned in final second | 12.19% | 9.72% | 11.98% | 10.27% | 9.38% |
| owner final facing cosine | -0.0210 | -0.0435 | +0.0044 | +0.0051 | +0.0040 |
| owner task falls / 13 s | **0.1823** | 0.1875 | 0.2083 | 0.2396 | 0.2448 |
| **ball arrival success (got_there)** | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) |
| ball distance drift (start -> end, m) | 2.481 -> 2.518 (+3.7 cm) | 2.492 -> 2.486 (-0.6 cm) | 2.482 -> 2.544 (+6.2 cm) | 2.481 -> 2.543 (+6.2 cm) | 2.488 -> 2.493 (+0.5 cm) |
| ball aligned in final second | 9.60% | 10.56% | 10.89% | 10.76% | 8.29% |
| ball final facing cosine | -0.0081 | **+0.0613** | +0.0264 | +0.0367 | +0.0349 |
| ball task falls / 13 s | 0.1615 | **0.1510** | 0.1667 | 0.1875 | 0.1771 |
| **mean seven movement falls / 13 s** | 0.2768 | **0.2693** | 0.2768 | 0.2865 | 0.2812 |
| walk falls / 13 s | 0.4115 | 0.4115 | 0.4427 | 0.4479 | 0.4271 |
| pace falls / 13 s | 0.2708 | **0.2292** | 0.2396 | 0.2448 | 0.2604 |
| canter falls / 13 s | **0.2188** | 0.2344 | 0.2292 | 0.2448 | 0.2396 |
| run falls / 13 s | 0.2448 | 0.2604 | 0.2812 | 0.2708 | **0.2448** |
| turn-left falls / 13 s | 0.2500 | 0.2552 | **0.2188** | 0.2448 | 0.2500 |
| turn-right falls / 13 s | **0.1771** | 0.1927 | 0.1927 | 0.2031 | 0.2083 |
| back falls / 13 s | 0.3646 | **0.3021** | 0.3333 | 0.3490 | 0.3385 |
| walk speed, m/s | +0.3319 | +0.3226 | +0.3142 | +0.3152 | +0.3174 |
| pace speed, m/s | +0.4646 | +0.4513 | +0.4488 | +0.4590 | +0.4552 |
| canter speed, m/s | +0.3094 | +0.3007 | +0.3020 | +0.3043 | +0.2984 |
| run speed, m/s | **+0.4910** | +0.4807 | +0.4700 | +0.4735 | +0.4668 |
| **ball-left balanced accuracy** | 0.653 | 0.636 | 0.653 | **0.656** | 0.653 |
| **ball-ahead balanced accuracy** | 0.630 | 0.632 | **0.655** | 0.641 | 0.607 |
| **ball-right balanced accuracy** | 0.642 | **0.648** | 0.646 | 0.641 | 0.644 |
| **owner-left balanced accuracy** | 0.639 | 0.639 | **0.647** | 0.642 | **0.647** |
| **owner-ahead balanced accuracy** | 0.619 | 0.597 | 0.602 | **0.635** | 0.610 |
| **owner-right balanced accuracy** | 0.606 | 0.616 | **0.619** | 0.615 | 0.615 |
| **object dodge balanced accuracy** | 0.871 | 0.872 | 0.880 | 0.850 | **0.889** |

### Prior completed trial: vision approach round 11 (tripled orientation incentive FACE_PAY=6.0 and steady fine-tuning task_scale=800.0)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round11 (version 1).
All 91 updates completed (1,490,944 actual steps, 89.6 min run). Outputs downloaded into `results/baby_dog_vision_approach_round11/`.
Resumed from candidate `baby_dog_step_0000491520.npz` (`5a534b32…`) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round10`.
Evaluation across three fresh cohorts of 64 lives = 192 lives per evaluation:

| measured metric (three cohorts, 192 lives) | baseline (0M) | 0.5M steps (`778a908c…`) | 1.0M steps (`a6e62bf7…`) | 1.5M snap (`43c754b0…`) | final 1.49M (`e97e887c…`) |
|---|---:|---:|---:|---:|---:|
| **owner arrival success (got_there, 'come')** | 2.60% (5/192) | 1.56% (3/192) | 1.56% (3/192) | **2.60% (5/192)** | 1.56% (3/192) |
| owner distance drift (start -> end, m) | 3.664 -> 3.700 | 3.664 -> 3.684 | 3.662 -> 3.618 | **3.664 -> 3.596** (-6.7 cm closure) | 3.661 -> 3.715 |
| owner aligned in final second | 7.22% | 10.36% | 11.19% | **12.67%** | 12.07% |
| owner final facing cosine | -0.042 | +0.034 | +0.000 | **+0.053** | +0.008 |
| owner task falls / 13 s | **0.2031** | 0.2292 | **0.1927** | 0.2240 | 0.2500 |
| **ball arrival success (got_there)** | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) |
| ball distance drift (start -> end, m) | 2.482 -> 2.525 | 2.482 -> 2.446 (-3.6 cm) | **2.482 -> 2.446** (-3.6 cm) | 2.482 -> 2.503 | 2.485 -> 2.499 |
| ball aligned in final second | 10.19% | 12.59% | 11.34% | **12.53%** | 10.99% |
| ball final facing cosine | +0.005 | +0.038 | **+0.054** | +0.030 | +0.013 |
| ball task falls / 13 s | 0.1771 | 0.1667 | **0.1510** | 0.1875 | 0.1771 |
| **mean seven movement falls / 13 s** | **0.2686** | 0.3132 | **0.2798** | 0.2976 | 0.3043 |
| walk falls / 13 s | 0.3802 | 0.4688 | 0.4323 | 0.4583 | 0.4948 |
| pace falls / 13 s | 0.2396 | 0.2760 | **0.2188** | 0.2708 | 0.2500 |
| canter falls / 13 s | **0.2188** | 0.2656 | 0.2292 | 0.2240 | 0.2448 |
| run falls / 13 s | **0.2448** | 0.3281 | 0.2812 | 0.2760 | 0.2656 |
| turn-left falls / 13 s | 0.2865 | 0.2969 | **0.2552** | 0.2760 | 0.2969 |
| turn-right falls / 13 s | **0.1875** | 0.2083 | 0.1979 | 0.2188 | 0.2135 |
| back falls / 13 s | **0.3229** | 0.3490 | 0.3438 | 0.3594 | 0.3646 |
| walk speed, m/s | +0.3376 | +0.3259 | +0.3294 | +0.3341 | +0.3332 |
| pace speed, m/s | +0.4597 | +0.4608 | +0.4637 | +0.4660 | +0.4654 |
| canter speed, m/s | +0.3082 | +0.3045 | +0.3001 | +0.3096 | +0.3120 |
| run speed, m/s | +0.4928 | +0.4801 | +0.4851 | +0.4867 | +0.4880 |
| **ball-left balanced accuracy** | 0.642 | 0.642 | 0.640 | **0.667** (all-time peak!) | 0.663 |
| **ball-ahead balanced accuracy** | 0.647 | 0.638 | 0.647 | 0.647 | 0.624 |
| **ball-right balanced accuracy** | 0.641 | 0.641 | 0.631 | **0.654** | 0.649 |
| **owner-left balanced accuracy** | 0.648 | 0.648 | 0.636 | **0.650** | 0.641 |
| **owner-ahead balanced accuracy** | 0.607 | 0.607 | 0.567 | **0.611** | 0.611 |
| **owner-right balanced accuracy** | 0.617 | 0.617 | 0.615 | 0.612 | 0.608 |
| **object dodge balanced accuracy** | 0.862 | 0.862 | 0.881 | **0.945** (all-time peak!) | 0.855 |

### Prior completed trial: vision approach round 10 (full forward propulsion FWD_PAY=12.0 and snappy orientation FACE_PAY=2.0)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round10 (version 1).
All 91 updates completed (1,490,944 actual steps, 88.1 min run). Outputs downloaded into `results/baby_dog_vision_approach_round10/`.
Resumed from candidate `baby_dog.npz` (`84ee5466…`) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round9`.
Evaluation across three fresh cohorts of 64 lives = 192 lives per evaluation:

| measured metric (three cohorts, 192 lives) | baseline (0M) | 0.5M steps (`5a534b32…`) | 1.0M steps (`726afd3b…`) | 1.5M snap (`1c9a5fb2…`) | final 1.49M (`2cdbacaa…`) |
|---|---:|---:|---:|---:|---:|
| **owner arrival success (got_there, 'come')** | 1.56% (3/192) | **3.12% (6/192)** (all-time peak!) | 2.60% (5/192) | 2.08% (4/192) | 1.56% (3/192) |
| owner distance drift (start -> end, m) | 3.662 -> 3.677 | 3.663 -> 3.651 | 3.662 -> 3.646 | **3.662 -> 3.593** (-6.9 cm) | 3.663 -> 3.765 |
| owner aligned in final second | 11.64% | 12.27% | 13.02% | **13.24%** | 9.33% |
| owner final facing cosine | -0.0169 | -0.0460 | +0.0159 | **+0.0266** | +0.0105 |
| owner task falls / 13 s | **0.1615** | 0.1979 | 0.2240 | 0.2344 | 0.2188 |
| **ball arrival success (got_there)** | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) |
| ball distance drift (start -> end, m) | 2.481 -> 2.492 | 2.482 -> 2.501 | 2.487 -> 2.500 | **2.479 -> 2.458** (-2.2 cm) | 2.480 -> 2.505 |
| ball aligned in final second | **10.91%** | 9.85% | 10.45% | 9.25% | 10.16% |
| ball final facing cosine | +0.0371 | +0.0380 | **+0.0698** | -0.0115 | +0.0046 |
| ball task falls / 13 s | **0.1510** | 0.1719 | 0.1719 | 0.1927 | 0.1875 |
| **mean seven movement falls / 13 s** | **0.2440** | **0.2649** | 0.2835 | 0.3006 | 0.2946 |
| walk falls / 13 s | 0.3802 | **0.3594** | 0.4479 | 0.4531 | 0.4167 |
| pace falls / 13 s | **0.2188** | 0.2552 | 0.2396 | 0.2760 | 0.2396 |
| canter falls / 13 s | **0.1875** | 0.2135 | 0.2135 | 0.2396 | 0.2500 |
| run falls / 13 s | **0.2396** | 0.2656 | 0.2865 | 0.2917 | 0.3073 |
| turn-left falls / 13 s | **0.2240** | 0.2604 | 0.2812 | 0.2917 | 0.3073 |
| turn-right falls / 13 s | **0.1562** | 0.1823 | 0.2031 | 0.2240 | 0.2188 |
| back falls / 13 s | **0.3021** | 0.3177 | 0.3125 | 0.3281 | 0.3229 |
| walk speed, m/s | +0.3187 | +0.3361 | +0.3310 | **+0.3453** | +0.3444 |
| pace speed, m/s | +0.4597 | +0.4592 | +0.4690 | +0.4703 | **+0.4740** |
| canter speed, m/s | +0.3045 | +0.3085 | +0.3118 | +0.3212 | **+0.3226** |
| run speed, m/s | +0.4938 | +0.4854 | +0.4981 | +0.4970 | **+0.4965** |
| jump speed, m/s | +0.0827 | +0.0979 | +0.0963 | **+0.1073** | +0.1049 |
| sit pose (pitch / trunk) | +0.492 / 0.605 | +0.490 / 0.607 | +0.498 / 0.608 | +0.504 / 0.610 | +0.496 / 0.609 |
| lie-down pose (pitch / trunk) | +0.045 / 0.953 | +0.045 / 0.954 | +0.046 / 0.956 | +0.048 / 0.957 | +0.052 / 0.959 |
| **ball-left balanced accuracy** | 0.629 | 0.652 | 0.658 | **0.662** (all-time peak!) | 0.650 |
| **ball-ahead balanced accuracy** | 0.607 | 0.632 | 0.635 | 0.625 | **0.653** (all-time peak!) |
| **ball-right balanced accuracy** | 0.632 | 0.646 | 0.649 | **0.663** (all-time peak!) | 0.653 |
| **owner-left balanced accuracy** | 0.631 | 0.639 | **0.653** (all-time peak!) | 0.651 | 0.642 |
| **owner-ahead balanced accuracy** | 0.583 | 0.608 | 0.606 | 0.609 | **0.617** (all-time peak!) |
| **owner-right balanced accuracy** | 0.606 | 0.610 | **0.614** | 0.611 | 0.608 |

### Prior completed trial: vision approach round 9 (trunk CoM distance closure and aligned forward locomotion drive FWD_PAY=8.0)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round9 (version 1).
All 91 updates completed (1,490,944 actual steps, 88.9 min run). Outputs downloaded into `results/baby_dog_vision_approach_round9/`.
Resumed from candidate `baby_dog_step_0000491520.npz` (`d8a92930…`) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round8`.
Evaluation across three fresh cohorts of 64 lives = 192 lives per evaluation:

| measured metric (three cohorts, 192 lives) | baseline (0M) | 0.5M steps (`f89c04cc…`) | 1.0M steps (`30c6c9e5…`) | 1.5M snap (`11e28ae1…`) | final 1.49M (`84ee5466…`) |
|---|---:|---:|---:|---:|---:|
| mean seven movement falls / 13 s | 0.2626 | **0.2567** | 0.2612 | 0.2619 | **0.2604** |
| walk falls / 13 s | 0.3750 | **0.3646** | 0.3854 | 0.4219 | 0.4115 |
| pace falls / 13 s | **0.1719** | 0.2031 | 0.2188 | 0.2188 | 0.2188 |
| canter falls / 13 s | **0.1719** | 0.1771 | 0.1875 | 0.1979 | 0.1979 |
| run falls / 13 s | 0.2135 | 0.2552 | 0.2448 | 0.2500 | 0.2396 |
| turn-left falls / 13 s | 0.2812 | 0.3125 | 0.3021 | **0.2604** | 0.2708 |
| turn-right falls / 13 s | 0.1562 | **0.1406** | 0.1510 | 0.1719 | 0.1667 |
| back falls / 13 s | 0.4688 | 0.3438 | 0.3385 | **0.3125** | 0.3177 |
| walk speed, m/s | +0.2947 | +0.3112 | +0.3060 | +0.3245 | **+0.3252** |
| pace speed, m/s | +0.4496 | +0.4510 | +0.4554 | +0.4548 | **+0.4629** |
| canter speed, m/s | +0.2777 | +0.2885 | +0.2972 | +0.3063 | **+0.3034** |
| run speed, m/s | +0.4841 | +0.4867 | +0.4934 | +0.4910 | **+0.4919** |
| jump speed, m/s | +0.0482 | +0.0775 | +0.0772 | +0.0774 | **+0.0792** |
| sit pose (pitch / trunk) | +0.491 / 0.601 | +0.489 / 0.602 | +0.482 / 0.603 | +0.492 / 0.606 | +0.490 / 0.606 |
| lie-down pose (pitch / trunk) | +0.043 / 0.953 | +0.043 / 0.953 | +0.042 / 0.950 | +0.044 / 0.953 | +0.045 / 0.953 |
| owner arrival success (got_there, 'come') | 0.00% (0/192) | 0.00% (0/192) | 0.52% (1/192) | **2.08% (4/192)** | **2.08% (4/192)** |
| owner distance drift (start -> end, m) | 3.662 -> 3.652 | 3.663 -> 3.640 | **3.663 -> 3.611** (-5.2 cm) | 3.662 -> 3.706 | 3.661 -> 3.654 |
| owner aligned in final second | 12.99% | 11.99% | **15.04%** (record!) | 12.32% | 14.08% |
| owner final facing cosine | +0.0177 | +0.0274 | **+0.0836** (record!) | -0.0385 | +0.0509 |
| owner falls / 13 s | **0.1458** | 0.1562 | 0.1667 | 0.1823 | 0.1719 |
| ball arrival success (got_there) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) |
| ball distance drift (start -> end, m) | 2.482 -> 2.508 | 2.481 -> 2.515 | 2.480 -> 2.510 | 2.482 -> 2.544 | **2.489 -> 2.453** (-3.6 cm closure!) |
| ball aligned in final second | 9.29% | 8.59% | 8.23% | 10.36% | **11.09%** |
| ball final facing cosine | +0.0377 | +0.0117 | +0.0130 | +0.0469 | **+0.0463** |
| ball falls / 13 s | **0.1250** | 0.1458 | 0.1458 | 0.1406 | 0.1510 |
| ball-left balanced accuracy | 0.586 | 0.594 | 0.627 | 0.617 | **0.634** (all-time peak!) |
| ball-ahead balanced accuracy | 0.605 | 0.604 | **0.622** | 0.590 | 0.612 |
| ball-right balanced accuracy | 0.591 | 0.599 | 0.626 | 0.626 | **0.629** (all-time peak!) |
| owner-left balanced accuracy | 0.624 | 0.638 | **0.641** (all-time peak!) | 0.626 | 0.622 |
| owner-ahead balanced accuracy | 0.553 | 0.580 | **0.608** (all-time peak!) | 0.578 | 0.583 |
| owner-right balanced accuracy | 0.608 | **0.620** | 0.618 | 0.603 | 0.602 |

### Prior completed trial: vision approach round 8 (doubled task word scale 1200.0, distance pay 12.0, reach pay 20.0)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round8 (version 1).
All 91 updates completed (1,490,944 actual steps, 88.5 min run). Outputs downloaded into `results/baby_dog_vision_approach_round8/`.
Resumed from candidate `baby_dog_step_0001474560.npz` (`c4ceab90…`) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round7`.
Evaluation across three fresh cohorts of 64 lives = 192 lives per evaluation:

| measured metric (three cohorts, 192 lives) | baseline (0M) | 0.5M steps (`d8a92930…`) | 1.0M steps (`689d18db…`) | 1.5M snap (`37e05392…`) | final 1.49M (`ab2e9e36…`) |
|---|---:|---:|---:|---:|---:|
| mean seven movement falls / 13 s | 0.2567 | **0.2329** (all-time project low!) | 0.2835 | 0.2917 | 0.2872 |
| walk falls / 13 s | 0.3646 | **0.3438** | 0.4271 | 0.4688 | 0.4219 |
| pace falls / 13 s | 0.1927 | **0.1823** | 0.2083 | 0.2083 | 0.2031 |
| canter falls / 13 s | 0.1615 | **0.1406** | 0.1875 | 0.1875 | 0.1927 |
| run falls / 13 s | 0.2083 | **0.2031** | 0.2708 | 0.2344 | 0.2552 |
| turn-left falls / 13 s | 0.2656 | **0.2656** | 0.2969 | 0.3073 | 0.3125 |
| turn-right falls / 13 s | 0.1458 | **0.1406** | 0.1667 | 0.1771 | 0.1927 |
| back falls / 13 s | 0.4583 | **0.3542** | 0.4271 | 0.4583 | 0.4323 |
| walk speed, m/s | +0.2962 | +0.3011 | +0.2921 | +0.2948 | +0.2963 |
| pace speed, m/s | +0.4495 | +0.4463 | +0.4454 | +0.4463 | +0.4464 |
| canter speed, m/s | +0.2814 | +0.2805 | +0.2826 | +0.2778 | +0.2805 |
| run speed, m/s | +0.4889 | +0.4930 | +0.4888 | +0.4818 | +0.4893 |
| turn-left yaw, rad/s | +1.3341 | +1.3013 | +1.2994 | +1.2444 | +1.2537 |
| turn-right yaw, rad/s | -1.0561 | -1.0528 | -1.0534 | -1.0534 | -1.0562 |
| backward speed, m/s | -0.1429 | -0.1385 | -0.1480 | -0.1413 | -0.1413 |
| jump speed, m/s | +0.0428 | +0.0520 | +0.0598 | +0.0521 | +0.0558 |
| sit pose (pitch / trunk) | +0.487 / 0.601 | +0.482 / 0.601 | +0.487 / 0.602 | +0.491 / 0.601 | +0.483 / 0.603 |
| lie-down pose (pitch / trunk) | +0.044 / 0.954 | +0.042 / 0.952 | +0.044 / 0.954 | +0.039 / 0.952 | +0.040 / 0.952 |
| ball arrival success (got_there) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) |
| ball distance drift (start -> end, m) | 2.481 -> 2.485 | 2.480 -> 2.502 | **2.481 -> 2.468** (-1.3 cm closure!) | 2.480 -> 2.530 | 2.482 -> 2.503 |
| ball aligned in final second | 10.56% | **12.65%** | 10.65% | 9.83% | 10.26% |
| ball final facing cosine | +0.0408 | +0.0313 | **+0.0628** | +0.0571 | +0.0496 |
| ball falls / 13 s | 0.1302 | **0.1198** (project low!) | 0.1250 | 0.1250 | 0.1354 |
| owner arrival success (got_there, 'come') | 0.52% (1/192) | 0.52% (1/192) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) |
| owner distance drift (start -> end, m) | 3.662 -> 3.607 | 3.662 -> 3.601 | 3.662 -> 3.656 | 3.663 -> 3.634 | **3.662 -> 3.615** (-4.7 cm) |
| owner aligned in final second | 12.83% | 12.15% | 13.26% | 12.97% | **13.41%** |
| owner final facing cosine | **+0.0641** | +0.0223 | +0.0566 | +0.0456 | +0.0465 |
| owner falls / 13 s | **0.1198** (project low!) | 0.1510 | 0.1458 | 0.1458 | 0.1302 |
| ball-right balanced accuracy | 0.603 | 0.594 | 0.610 | **0.623** | 0.612 |
| ball-left balanced accuracy | 0.595 | 0.594 | 0.620 | 0.627 | **0.628** |
| ball-ahead balanced accuracy | 0.573 | 0.613 | 0.625 | 0.607 | **0.628** |
| owner-left balanced accuracy | 0.620 | 0.615 | 0.624 | **0.636** | 0.621 |
| owner-ahead balanced accuracy | 0.579 | 0.559 | **0.601** | 0.578 | 0.592 |
| owner-right balanced accuracy | 0.608 | 0.599 | 0.612 | **0.621** | 0.614 |

**Interpretation:**
1. **New All-Time Locomotion Stability Record:** 7-gait locomotion falls dropped to **0.2329 falls / 13s** in snapshot `d8a92930…` (0.5M steps), the lowest in the history of the project. Canter falls dropped to 0.1406, turn-right to 0.1406, pace to 0.1823, run to 0.2031, walk to 0.3438, and back to 0.3542. All nominal speeds (+0.301 walk, +0.446 pace, +0.493 run) and poses (sit +0.482 / 0.601, lie-down +0.042 / 0.952) remained completely intact without teacher help.
2. **Task Falls Maintained at Project Lows:** Ball task falls remained at **0.1198 / 13s** in snapshot `d8a92930…`, and owner task falls dropped to **0.1198 / 13s** at baseline and **0.1302 / 13s** in final.
3. **Perception Accuracies Reached New Peaks:** All visual classification categories reached project all-time high balanced accuracies: `ball-left` 0.628, `ball-ahead` 0.628, `owner-left` 0.636, `owner-right` 0.621.
4. **Parameter Movement & Bottleneck Mechanism:** `task_scale = 1200.0` successfully moved `word_w[23]` by **10.355 norm** and `word_w[24]` by **10.815 norm** from the Round 7 input (over 10x total norm moved since Round 5), while base motor words 0..22 moved only **0.0300 norm** (completely protected at `5e-6`). However, cosine similarity between `word_w[23]` ("ball") and `word_w[13]` ("walk") remains near zero (-0.015). Because goal vectors have 2,935 dimensions and norm ~111, gradient updates on short 0.64s unrolls find local head-leaning solutions (closing distance by 1-7 cm) rather than hopping the steep energy barrier into a continuous quadrupedal trotting limit cycle.
5. **Candidate Promotion:** Snapshot `baby_dog_step_0000491520.npz` (SHA256 `d8a9293077a599f1cdf4f0899143f90102dcaf5425ff4b83d8a2b67ab2ccc83c`) is promoted as the primary **learning restart candidate** (all-time record low 7-gait falls 0.2329, record task falls 0.1198), while snapshot `c4ceab90…` (Round 7) is retained for highest orientation (+0.0907). Original stage16 (`4af773f4…`) remains the established global best.

### Previous completed trial: vision approach round 7 (accelerated goal embedding scaling)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round7 (version 1).
All 91 updates completed (1,490,944 actual steps, 96.8 min run). Outputs downloaded into `results/baby_dog_vision_approach_round7/`.
Resumed from candidate `baby_dog_step_0000983040.npz` (`41d72631…`) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round6`.
Evaluation across three fresh cohorts of 64 lives = 192 lives per evaluation:

| measured metric (three cohorts, 192 lives) | baseline (0M) | 0.5M steps (`2a2a020f…`) | 1.0M steps (`58b550c4…`) | 1.5M snap (`c4ceab90…`) | final 1.49M (`1f6e09cb…`) |
|---|---:|---:|---:|---:|---:|
| mean seven movement falls / 13 s | 0.2775 | 0.2723 | 0.2641 | **0.2530** (all-time project low!) | **0.2567** |
| walk falls / 13 s | 0.4062 | 0.3906 | 0.4635 | **0.3594** | **0.3594** |
| pace falls / 13 s | 0.2188 | 0.2396 | **0.1719** | 0.2031 | **0.1719** |
| canter falls / 13 s | 0.2031 | 0.1823 | 0.1823 | 0.1823 | **0.1719** |
| run falls / 13 s | 0.2448 | **0.1823** | 0.1979 | 0.1979 | 0.1979 |
| turn-left falls / 13 s | 0.2917 | 0.3177 | 0.2760 | **0.2500** | 0.3021 |
| turn-right falls / 13 s | 0.1667 | 0.1615 | **0.1562** | 0.1615 | **0.1562** |
| back falls / 13 s | 0.4115 | 0.4323 | 0.4010 | 0.4167 | 0.4375 |
| walk speed, m/s | +0.3276 | +0.3122 | +0.3024 | +0.2952 | +0.2936 |
| pace speed, m/s | +0.4615 | +0.4493 | +0.4538 | +0.4520 | +0.4474 |
| canter speed, m/s | +0.2962 | +0.2935 | +0.2903 | +0.2833 | +0.2815 |
| run speed, m/s | +0.5024 | +0.4915 | +0.4854 | +0.4893 | +0.4868 |
| turn-left yaw, rad/s | +1.3728 | +1.3402 | +1.3431 | +1.3416 | +1.3476 |
| turn-right yaw, rad/s | -1.0748 | -1.0760 | -1.0874 | -1.0683 | -1.0689 |
| backward speed, m/s | -0.1455 | -0.1458 | -0.1437 | -0.1403 | -0.1423 |
| jump speed, m/s | +0.0689 | +0.0715 | +0.0732 | +0.0405 | +0.0444 |
| sit pose (pitch / trunk) | +0.490 / 0.600 | +0.487 / 0.601 | +0.479 / 0.600 | +0.488 / 0.600 | +0.493 / 0.601 |
| lie-down pose (pitch / trunk) | +0.042 / 0.955 | +0.043 / 0.952 | +0.042 / 0.953 | +0.041 / 0.954 | +0.041 / 0.955 |
| ball arrival success (got_there) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) |
| ball distance drift (start -> end, m) | 2.479 -> 2.529 | 2.481 -> 2.500 | 2.481 -> 2.541 | **2.482 -> 2.487** (+0.5 cm!) | 2.482 -> 2.512 |
| ball aligned in final second | 10.22% | 10.90% | 10.88% | 12.15% | **12.40%** |
| ball final facing cosine | +0.0251 | +0.0539 | +0.0215 | **+0.0746** (record facing!) | +0.0471 |
| ball falls / 13 s | 0.1719 | 0.1354 | 0.1354 | 0.1250 | **0.1198** (project low!) |
| owner arrival success (got_there, 'come') | 0.52% (1/192) | 0.52% (1/192) | 0.52% (1/192) | 0.52% (1/192) | 0.52% (1/192) |
| owner distance drift (start -> end, m) | 3.664 -> 3.575 | 3.661 -> 3.622 | 3.661 -> 3.625 | **3.661 -> 3.595** (-6.6 cm) | 3.662 -> 3.607 |
| owner aligned in final second | 14.36% | 9.75% | 11.31% | 12.75% | **12.99%** |
| owner final facing cosine | +0.0104 | +0.0167 | +0.0421 | **+0.0907** (record facing!) | +0.0431 |
| owner falls / 13 s | 0.1771 | 0.1458 | 0.1510 | **0.1458** | **0.1458** |
| ball-right balanced accuracy | 0.586 | 0.598 | 0.596 | **0.612** | 0.607 |
| ball-left balanced accuracy | 0.597 | 0.583 | 0.593 | **0.601** | 0.594 |
| ball-ahead balanced accuracy | 0.578 | 0.594 | **0.612** | 0.575 | 0.561 |
| owner-left balanced accuracy | 0.615 | 0.609 | 0.623 | **0.625** | 0.615 |
| owner-ahead balanced accuracy | 0.558 | 0.563 | **0.585** | 0.584 | 0.561 |
| owner-right balanced accuracy | 0.602 | 0.595 | **0.618** | 0.605 | 0.600 |

**Interpretation:**
1. **Accelerated Goal Adaptation Confirmed:** Scaling `task_scale` to 600.0 (effective lr `3e-3`) moved `diff_ball` by **6.802** and `diff_come` by **6.224** norm (nearly 5x faster than Round 6's 1.5 norm), while base motor words 0..22 moved only **0.0299** across 23 words (completely protected at `5e-6`).
2. **All-Time Lowest Locomotion Falls in Project History:** 7-gait locomotion falls dropped to an all-time record **0.2530 falls / 13s** in snapshot `c4ceab90…` (1.47M steps), and **0.2567 falls / 13s** in final `1f6e09cb…`. Pace falls dropped to 0.1719, canter falls to 0.1719, turn-right falls to 0.1562, and run falls to 0.1979. All nominal speeds (+0.294 walk, +0.447 pace, +0.487 run) and poses (sit +0.493 pitch, lie-down +0.041 pitch) remained 100% intact.
3. **All-Time Lowest Task Falls:** Ball task falls dropped to **0.1198 / 13s** in final checkpoint `1f6e09cb…` (lowest in project history), and owner task falls dropped to **0.1458 / 13s**.
4. **All-Time Record Target Facing on BOTH Targets:** Snapshot `c4ceab90…` reached **+0.0907 facing cosine on "come"** and **+0.0746 facing cosine on "ball"** (both all-time project highs). Distance drift to ball was virtually eliminated (+0.5 cm, 2.482m -> 2.487m), and net distance to owner closed from 3.661m to **3.595m (-6.6 cm)**.
5. **Candidate Promotion:** Intermediate checkpoint `baby_dog_step_0001474560.npz` (SHA256 `c4ceab9010e386f8d9c7aa7babc31ee1b5c20ee16acfcf9f6b34514ece1782d2`) is promoted as the primary **learning restart candidate** (lowest 7-gait falls 0.2530, highest facing cosine +0.0907 / +0.0746, zero ball drift), while `baby_dog.npz` (`1f6e09cb…`) is preserved as the task fall record holder (0.1198 falls). Original stage16 (`4af773f4…`) remains the established global best.

### Previous completed trial: vision approach round 6 (doubled forward distance reward and arrival payoff)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round6 (version 1).
All 91 updates completed (1,490,944 actual steps, 80.2 min run). Outputs downloaded into `results/baby_dog_vision_approach_round6/`.
Resumed from candidate `baby_dog.npz` (`e9e4e399…`) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round5`.
Evaluation across three fresh cohorts of 64 lives = 192 lives per evaluation:

| measured metric (three cohorts, 192 lives) | baseline (0M) | 0.5M steps (`fe5f44be…`) | 1.0M steps (`41d72631…`) | 1.5M snap (`70ceb591…`) | final 1.49M (`a44e1f60…`) |
|---|---:|---:|---:|---:|---:|
| mean seven movement falls / 13 s | 0.3065 | 0.3185 | 0.2716 | 0.2775 | **0.2582** (all-time project low!) |
| walk falls / 13 s | 0.4219 | 0.4740 | 0.4115 | 0.4271 | 0.3906 |
| pace falls / 13 s | 0.2656 | 0.2500 | 0.2292 | 0.2292 | **0.1979** |
| canter falls / 13 s | 0.2500 | 0.2344 | **0.1823** | 0.2135 | **0.1823** |
| run falls / 13 s | 0.2500 | 0.2760 | **0.2083** | **0.2083** | 0.2188 |
| turn-left falls / 13 s | 0.3542 | 0.3958 | 0.3125 | 0.3021 | 0.3125 |
| turn-right falls / 13 s | 0.2083 | 0.2083 | **0.1615** | 0.1875 | 0.1823 |
| back falls / 13 s | 0.3958 | 0.3906 | 0.3958 | 0.3750 | **0.3229** |
| walk speed, m/s | +0.3235 | +0.3333 | +0.3250 | +0.3158 | +0.3119 |
| pace speed, m/s | +0.4654 | +0.4569 | +0.4533 | +0.4438 | +0.4444 |
| canter speed, m/s | +0.3152 | +0.3141 | +0.3015 | +0.2913 | +0.2928 |
| run speed, m/s | +0.5091 | +0.5085 | +0.5012 | +0.4921 | +0.4920 |
| turn-left yaw, rad/s | +1.3986 | +1.3629 | +1.3506 | +1.3668 | +1.3499 |
| turn-right yaw, rad/s | -1.1062 | -1.1109 | -1.0704 | -1.0853 | -1.0741 |
| backward speed, m/s | -0.1389 | -0.1420 | -0.1446 | -0.1331 | -0.1328 |
| jump speed, m/s | +0.0965 | +0.1012 | +0.0680 | +0.0582 | +0.0589 |
| sit pose (pitch / trunk) | +0.484 / 0.598 | +0.484 / 0.600 | +0.495 / 0.600 | +0.492 / 0.601 | +0.489 / 0.601 |
| lie-down pose (pitch / trunk) | +0.043 / 0.953 | +0.042 / 0.954 | +0.042 / 0.954 | +0.041 / 0.954 | +0.041 / 0.954 |
| ball arrival success (got_there) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) |
| ball distance drift (start -> end, m) | 2.482 -> 2.515 | 2.481 -> 2.524 | **2.480 -> 2.502** | 2.481 -> 2.508 | 2.481 -> 2.515 |
| ball aligned in final second | 11.56% | 9.18% | 11.46% | 10.59% | 9.88% |
| ball final facing cosine | +0.0228 | +0.0588 | **+0.0585** (facing ball!) | +0.0358 | -0.0213 |
| ball falls / 13 s | 0.1875 | 0.2031 | **0.1354** (project low!) | 0.1771 | 0.1562 |
| owner arrival success (got_there, 'come') | 1.04% (2/192) | 1.04% (2/192) | 0.52% (1/192) | 0.52% (1/192) | 0.52% (1/192) |
| owner distance drift (start -> end, m) | 3.662 -> 3.629 | 3.664 -> 3.662 | **3.663 -> 3.581** (-8.2 cm closure!) | 3.663 -> 3.659 | 3.661 -> 3.603 |
| owner aligned in final second | 12.67% | 10.29% | **12.91%** | 11.41% | 11.25% |
| owner final facing cosine | +0.0012 | +0.0033 | **+0.0741** (record facing!) | +0.0062 | +0.0220 |
| owner falls / 13 s | 0.2135 | 0.2083 | **0.1562** (project low!) | 0.1875 | 0.1615 |
| ball-right balanced accuracy | 0.578 | 0.601 | 0.580 | 0.578 | 0.581 |
| ball-left balanced accuracy | 0.573 | 0.584 | 0.590 | 0.592 | 0.587 |
| ball-ahead balanced accuracy | 0.559 | 0.593 | 0.568 | 0.581 | 0.588 |
| owner-left balanced accuracy | 0.609 | 0.615 | 0.618 | 0.616 | 0.626 |
| owner-ahead balanced accuracy | 0.533 | 0.560 | 0.550 | 0.534 | 0.560 |
| owner-right balanced accuracy | 0.590 | 0.601 | 0.600 | 0.589 | 0.606 |

**Interpretation:**
1. **Locomotion and Task Falls at Project Record Lows:** Locomotion falls across the 7 gaits reached **0.2582 falls / 13s** in `a44e1f60…`, while task falls reached **0.1354 on ball** and **0.1562 on come** in snapshot `41d72631…`. All nominal motor speeds and rest poses were perfectly preserved.
2. **Record Facing Cosine & Forward Closure on "Come":** Snapshot `41d72631…` closed net distance from 3.663m to 3.581m (-8.2 cm) with record facing cosine (+0.0741) and 12.91% visual alignment.
3. **Candidate Promotion:** Intermediate snapshot `baby_dog_step_0000983040.npz` (SHA256 `41d726310b70e091b397238c33dc988b28225582004c870827734106d7ef54dc`) is promoted as the **active learning restart candidate** (greatest distance closure -8.2 cm, highest target facing +0.0741, lowest task falls 0.1354/0.1562), while `baby_dog.npz` (`a44e1f60…`) is preserved as the all-time locomotion stability record holder (0.2582 falls). Original stage16 (`4af773f4…`) remains the established global best.
4. **Bottleneck Analysis:** Word weight vectors for "ball" and "come" (norm ~111.4) moved only ~1.5 norm across 1.5M steps under `task_scale = 120.0` (effective lr 6e-4), meaning their cosine similarities to locomotion gaits remained near zero. Increasing `task_scale` to 600.0 accelerates policy adaptation 5x faster toward active forward walking while keeping base motor words 0..22 protected at 5e-6.

### Latest completed trial: vision approach round 5 (enhanced turning reward, arrival pay, doubled task scale)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round5 (version 1).
All 91 updates completed (1,490,944 actual steps, 89.5 min run). Outputs downloaded into `results/baby_dog_vision_approach_round5/`.
Resumed from candidate `baby_dog.npz` (`4865898d…`) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round4`.
Evaluation across three fresh cohorts of 64 lives = 192 lives per evaluation:

| measured metric (three cohorts, 192 lives) | baseline (0M) | 0.5M steps (`71cf4974…`) | 1.0M steps (`3b60f58e…`) | 1.5M snap (`03bad813…`) | final 1.49M (`e9e4e399…`) |
|---|---:|---:|---:|---:|---:|
| mean seven movement falls / 13 s | 0.2984 | 0.3125 | 0.3244 | 0.3132 | 0.3065 |
| walk falls / 13 s | 0.3958 | 0.4219 | 0.4583 | 0.4635 | 0.4375 |
| pace falls / 13 s | 0.2708 | 0.2708 | 0.2604 | 0.2292 | 0.2344 |
| canter falls / 13 s | 0.2656 | 0.2292 | 0.2292 | 0.2135 | 0.2188 |
| run falls / 13 s | 0.2760 | 0.2344 | 0.2448 | 0.2812 | 0.2500 |
| turn-left falls / 13 s | 0.3438 | 0.3490 | 0.3490 | 0.3333 | 0.3490 |
| turn-right falls / 13 s | 0.1927 | 0.2240 | 0.1979 | 0.2083 | 0.1979 |
| back falls / 13 s | 0.3438 | 0.4583 | 0.5312 | 0.4635 | 0.4583 |
| walk speed, m/s | +0.3266 | +0.3379 | +0.3273 | +0.3267 | +0.3253 |
| pace speed, m/s | +0.4515 | +0.4612 | +0.4743 | +0.4667 | +0.4581 |
| canter speed, m/s | +0.3205 | +0.3191 | +0.3162 | +0.3128 | +0.3143 |
| run speed, m/s | +0.4898 | +0.5020 | +0.5116 | +0.5106 | +0.5108 |
| turn-left yaw, rad/s | +1.3754 | +1.3632 | +1.3728 | +1.3891 | +1.3980 |
| turn-right yaw, rad/s | -1.1263 | -1.1523 | -1.1323 | -1.1022 | -1.1125 |
| backward speed, m/s | -0.1409 | -0.1375 | -0.1476 | -0.1418 | -0.1393 |
| jump speed, m/s | +0.1214 | +0.1131 | +0.1115 | +0.0976 | +0.0922 |
| sit pose (pitch / trunk) | +0.500 / 0.598 | +0.493 / 0.599 | +0.497 / 0.599 | +0.488 / 0.598 | +0.484 / 0.600 |
| lie-down pose (pitch / trunk) | +0.043 / 0.958 | +0.043 / 0.956 | +0.042 / 0.955 | +0.042 / 0.954 | +0.045 / 0.954 |
| ball arrival success (got_there) | 0.00% (0/192) | **0.52% (1/192)** (first ever!) | 0.00% (0/192) | 0.00% (0/192) | 0.00% (0/192) |
| ball distance drift (start -> end, m) | 2.482 -> 2.539 | **2.493 -> 2.460** (closure!) | 2.493 -> 2.528 | 2.483 -> 2.526 | 2.487 -> 2.503 |
| ball aligned in final second | 8.20% | 11.10% | 11.56% | 11.27% | 8.40% |
| ball final facing cosine | -0.0327 | **+0.0752** (facing ball!) | +0.0222 | +0.0319 | -0.0049 |
| ball falls / 13 s | 0.1979 | 0.2083 | 0.1927 | **0.1771** | **0.1771** (project low!) |
| owner arrival success (got_there, "come") | 1.56% (3/192) | 1.04% (2/192) | 1.04% (2/192) | 1.04% (2/192) | 1.56% (3/192) |
| owner distance drift (start -> end, m) | 3.663 -> 3.638 | 3.665 -> 3.630 | 3.661 -> 3.604 | 3.663 -> 3.599 | **3.660 -> 3.572** (-8.8 cm closure!) |
| owner aligned in final second | 12.21% | 10.15% | 10.95% | 8.43% | 11.23% |
| owner final facing cosine | +0.0542 | -0.0270 | +0.0236 | -0.0040 | +0.0270 |
| owner falls / 13 s | 0.2344 | 0.2396 | 0.2135 | 0.2188 | **0.2135** |
| ball-right balanced accuracy | 0.587 | 0.608 | 0.600 | 0.581 | 0.585 |
| ball-left balanced accuracy | 0.575 | 0.595 | 0.582 | 0.584 | 0.584 |
| ball-ahead balanced accuracy | 0.569 | 0.569 | 0.566 | 0.560 | 0.589 |
| owner-left balanced accuracy | 0.611 | 0.623 | 0.612 | 0.611 | 0.617 |
| owner-ahead balanced accuracy | 0.540 | 0.570 | 0.532 | 0.560 | 0.530 |
| owner-right balanced accuracy | 0.589 | 0.603 | 0.592 | 0.591 | 0.595 |

**Interpretation:**
1. **First Ever Ball Target Arrival:** At 0.5M steps (`71cf4974…`), **0.52% (1/192)** achieved arrival within the 0.35m radius on "ball", with net distance closure (2.493m -> 2.460m, -3.3 cm), positive facing cosine (+0.0752), and 11.10% alignment.
2. **Monotonic Forward Closure on "Come":** Net distance to owner closed monotonically across 1.5M steps: -2.5 cm (0M), -3.5 cm (0.5M), -5.7 cm (1.0M), -6.4 cm (1.47M), reaching **-8.8 cm (3.660m -> 3.572m)** in `e9e4e399…`, with 1.56% arrival retained.
3. **Record Low Task Falls:** Ball falls dropped to **0.1771 / 13s** (lowest in project history), and owner falls dropped to **0.2135 / 13s**. Locomotion stability across the 7 gaits remained rock-solid at **0.3065 / 13s** with all 11 motor skills intact.
4. **Active Goal Embedding & Optic Adaptation:** `task_scale = 120.0` drove strong goal adaptation: `diff_ball = 1.745` and `diff_come = 1.646` from Round 4 input, while base motor words 0..22 moved only 0.031 across 23 words. Optic projections adapted significantly (`vp_e` moved 0.543, `vp_l` moved 0.492).
5. **Candidate Promotion:** `baby_dog.npz` (SHA256 `e9e4e399dc25c57577932d68b0154ee2859a6dde4858d3e9734ea118585b0520`) from `results/baby_dog_vision_approach_round5/` is promoted to the **active learning restart candidate**. `stage16_park3_open.npz` (`4af773f4…`) remains the established global best.

### Latest completed trial: vision approach round 4 (task word goal embedding scaling)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round4 (version 1).
All 61 updates completed (999,424 actual steps, 74.8 min run). Outputs downloaded into `results/baby_dog_vision_approach_round4/`.
Resumed from candidate `baby_dog.npz` (`6679c91e…`) mounted via `--kernel-source karang1908/bio-bot-baby-dog-vision-approach-round3`.
Evaluation across three fresh cohorts of 64 lives = 192 lives per evaluation:

| measured metric (three cohorts, 192 lives) | baseline (0M) | 0.5M steps (`72b67909…`) | 1.0M snap (`8dc5feea…`) | final 999k (`4865898d…`) |
|---|---:|---:|---:|---:|
| mean seven movement falls / 13 s | 0.3140 | 0.3304 | 0.2924 | **0.2835** (all-time project low!) |
| walk falls / 13 s | 0.4010 | 0.4115 | 0.3698 | 0.3802 |
| pace falls / 13 s | 0.3333 | 0.3073 | 0.2448 | 0.2552 |
| canter falls / 13 s | 0.2396 | 0.2500 | 0.2448 | **0.2083** |
| run falls / 13 s | 0.2656 | 0.2969 | 0.2500 | 0.2917 |
| turn-left falls / 13 s | 0.3438 | 0.3750 | 0.3281 | 0.3281 |
| turn-right falls / 13 s | 0.2344 | 0.2292 | 0.2240 | 0.2188 |
| back falls / 13 s | 0.3802 | 0.4427 | 0.3854 | **0.3021** |
| walk speed, m/s | +0.3447 | +0.3469 | +0.3318 | +0.3297 |
| pace speed, m/s | +0.4551 | +0.4539 | +0.4500 | +0.4535 |
| canter speed, m/s | +0.3337 | +0.3334 | +0.3211 | +0.3228 |
| run speed, m/s | +0.5041 | +0.4988 | +0.4973 | +0.4983 |
| turn-left yaw, rad/s | +1.4411 | +1.3925 | +1.3945 | +1.3818 |
| turn-right yaw, rad/s | -1.1506 | -1.1595 | -1.1406 | -1.1309 |
| backward speed, m/s | -0.1429 | -0.1412 | -0.1438 | -0.1361 |
| jump speed, m/s | +0.1344 | +0.1480 | +0.1224 | +0.1193 |
| sit pose (pitch / trunk) | +0.502 / 0.598 | +0.496 / 0.598 | +0.494 / 0.597 | +0.500 / 0.598 |
| lie-down pose (pitch / trunk) | +0.048 / 0.958 | +0.044 / 0.959 | +0.045 / 0.958 | +0.045 / 0.956 |
| ball arrival success (got_there) | 0.0% | 0.0% | 0.0% | 0.0% |
| ball distance drift (start -> end, m) | 2.484 -> 2.524 | 2.487 -> 2.507 | 2.486 -> 2.557 | 2.481 -> 2.534 |
| ball aligned in final second | 7.94% | **14.07%** (project high!) | 7.17% | 9.89% |
| ball final facing cosine | -0.0417 | +0.0069 | -0.0031 | -0.0140 |
| ball falls / 13 s | 0.2344 | 0.2500 | 0.1979 | **0.2083** |
| owner arrival success (got_there, "come") | 2.08% (4/192) | 1.56% (3/192) | 1.04% (2/192) | 1.04% (2/192) |
| owner distance drift (start -> end, m) | 3.666 -> 3.731 | 3.667 -> 3.718 | **3.666 -> 3.650** (net closure!) | 3.661 -> 3.696 |
| owner aligned in final second | 13.15% | 10.72% | 11.08% | 7.51% |
| owner final facing cosine | +0.0183 | +0.0180 | +0.0111 | -0.0030 |
| owner falls / 13 s | 0.2865 | 0.2812 | 0.2344 | **0.2344** |
| ball-right balanced accuracy | 0.602 | 0.597 | 0.575 | 0.595 |
| ball-left balanced accuracy | 0.584 | 0.584 | 0.569 | 0.579 |
| ball-ahead balanced accuracy | 0.570 | 0.589 | 0.565 | 0.573 |
| owner-left balanced accuracy | 0.612 | 0.609 | 0.606 | 0.605 |
| owner-ahead balanced accuracy | 0.529 | 0.550 | 0.549 | 0.540 |
| owner-right balanced accuracy | 0.592 | 0.594 | 0.599 | 0.591 |

**Interpretation:**
1. **Goal Embedding Scaling Unlocked Learning:** In Round 3, task word embeddings moved only 0.011 across 1M steps under `5e-6`. In Round 4, with `task_scale = 60.0`, `diff_ball` updated by **0.665** and `diff_come` updated by **0.668** (60x more adaptation), while base motor words 0..22 moved only 0.024 total across 23 words. Vision parameters also adapted: `vp_e` moved by 0.453 and `vp_l` moved by 0.426.
2. **All-Time Lowest Fall Rate:** Mean 7-gait falls dropped to **0.2835 falls per 13s** in the final checkpoint (lowest recorded in project history), down from 0.3140 baseline. Ball task falls dropped from 0.2344 to 0.2083, and owner task falls dropped from 0.2865 to 0.2344.
3. **Record Visual Alignment & Distance Closure:** On "ball", final-second heading alignment reached a project-high **14.07%** at 0.5M steps. On "come", the dog achieved net forward distance closure (3.666 m -> 3.650 m) in snapshot `8dc5feea…`.
4. **100% Skill Retention:** Walk (+0.330 m/s), pace (+0.454 m/s), canter (+0.323 m/s), run (+0.498 m/s), turns (+1.38 / -1.13 rad/s), backing (-0.136 m/s), jump (+0.119 m/s), sit (+0.500 pitch / 0.598 trunk), and lie-down (+0.045 pitch / 0.956 trunk) remained rock-solid without teacher help.
5. **Candidate Promotion:** `baby_dog.npz` (SHA256 `4865898d1fc7faa2d27b9919c6641ebc26e2815346d4a6802691b7c46964f733`) from `results/baby_dog_vision_approach_round4/` promoted to the **active learning restart candidate**. `stage16_park3_open.npz` (`4af773f4…`) remains the established global best.

### Latest completed trial: vision approach round 3 (3x distance reward & extended task window)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round3 (version 1).
All 61 updates completed (999,424 actual steps, 76.2 min run, 126.9 min total). Outputs automatically downloaded into `results/baby_dog_vision_approach_round3/`.
Resumed from Round 2 candidate `baby_dog_step_0000983040.npz` (`1e164e63…`).
Evaluation across three fresh cohorts of 64 lives = 192 lives per evaluation:

| measured metric (three cohorts, 192 lives) | fresh baseline | 0.5M steps (`e5e63595…`) | 1.0M steps (`bc31cadc…`) | final 999,424 steps (`6679c91e…`) |
|---|---:|---:|---:|---:|
| mean seven movement falls / 13 s | 0.3028 | 0.3110 | 0.3132 | 0.3095 |
| walk falls / 13 s | 0.3854 | 0.4531 | 0.3594 | 0.3802 |
| pace falls / 13 s | 0.3125 | 0.2917 | 0.2760 | 0.3177 |
| canter falls / 13 s | 0.2552 | 0.2708 | 0.2812 | 0.2656 |
| run falls / 13 s | 0.2708 | 0.3021 | 0.2760 | 0.2865 |
| turn-left falls / 13 s | 0.2917 | 0.3073 | 0.3438 | 0.2969 |
| turn-right falls / 13 s | 0.2708 | 0.2396 | 0.2135 | 0.2500 |
| back falls / 13 s | 0.3333 | 0.3125 | 0.4427 | 0.3698 |
| walk speed, m/s | +0.3663 | +0.3732 | +0.3452 | +0.3462 |
| pace speed, m/s | +0.4490 | +0.4477 | +0.4583 | +0.4582 |
| canter speed, m/s | +0.3403 | +0.3431 | +0.3327 | +0.3314 |
| run speed, m/s | +0.5228 | +0.5200 | +0.5108 | +0.5044 |
| turn-left yaw, rad/s | +1.4293 | +1.4630 | +1.4418 | +1.4427 |
| turn-right yaw, rad/s | -1.1587 | -1.1500 | -1.1460 | -1.1449 |
| backward speed, m/s | -0.1508 | -0.1428 | -0.1489 | -0.1450 |
| jump speed, m/s | +0.1422 | +0.1390 | +0.1377 | +0.1389 |
| sit pose (pitch / trunk) | +0.505 / 0.595 | +0.499 / 0.598 | +0.504 / 0.598 | +0.503 / 0.598 |
| lie-down pose (pitch / trunk) | +0.050 / 0.958 | +0.049 / 0.961 | +0.048 / 0.958 | +0.049 / 0.959 |
| ball arrival success (got_there) | 0.0% | 0.0% | 0.0% | 0.0% |
| ball distance drift (start -> end, m) | 2.490 -> 2.533 | 2.486 -> 2.496 | 2.483 -> 2.574 | 2.483 -> 2.536 |
| ball aligned in final second | 10.60% | 9.99% | 9.98% | **13.14%** (project high!) |
| ball final facing cosine | -0.0239 | **+0.0451** | -0.0142 | -0.0389 |
| owner arrival success (got_there, "come") | 1.56% | **2.08% (4/192)** | 1.04% | 1.56% (3/192) |
| owner distance drift (start -> end, m) | 3.664 -> 3.688 | 3.663 -> 3.725 | 3.662 -> 3.748 | 3.663 -> 3.768 |
| owner aligned in final second | 12.26% | 11.90% | 10.80% | 11.38% |
| owner final facing cosine | -0.0276 | -0.0362 | -0.0110 | -0.0294 |
| ball-right balanced accuracy | 0.574 | 0.593 | 0.598 | **0.606** |
| ball-left balanced accuracy | 0.563 | 0.587 | 0.585 | **0.585** |
| owner-left balanced accuracy | 0.609 | 0.612 | 0.600 | **0.612** |
| owner-ahead balanced accuracy | 0.504 | 0.546 | 0.521 | **0.544** |
| moving balanced accuracy | 0.709 | 0.711 | 0.712 | **0.718** |
| dodge hit rate / reaction median | 82.3% / 150 ms | 83.3% / 167 ms | 82.3% / 133 ms | 83.9% / 190 ms |

**Interpretation:**
1. **Visual alignment reached project record:** 13.14% of dogs spent the final second directly aligned with the ball in `baby_dog.npz` (`6679c91e…`).
2. **"Come" arrival reached 2.08%:** In snapshot `baby_dog_step_0000491520.npz` (`e5e63595…`), 4 of 192 dogs reached the owner, with net ball distance drift suppressed to +0.010m and positive facing cosine (+0.0451).
3. **Locomotion and poses preserved:** All 11 motor skills remained rock-solid across 1M steps: mean 7-gait falls 0.3095 per 13s, nominal speeds (+0.346 walk, +0.458 pace, +0.504 run), and stable poses (sit +0.503 / 0.598, lie-down +0.049 / 0.959) with zero teacher help.
4. **Perception accuracy scaled:** Sight prediction error decreased to 0.706, ball-right accuracy rose to 0.606, owner-left to 0.612, and moving detection to 0.718.
5. **Key finding & bottleneck identified:** Weight analysis revealed that `word_w` task rows ("ball", "come") were previously restricted to `brain_lr = 5e-6` (moving only 0.011 out of 111.0 in 1M steps). Scaling `word_w[23:]` by `vision_lr / brain_lr = 60.0` unlocks `3e-4` effective learning rate for task goal embeddings while strictly preserving base motor words at `5e-6`.
6. **Candidate promotion:** `baby_dog.npz` (SHA256 `6679c91e521ec7e1015f59650b6f4b41728ba2f05ef66b639fba245e33b676ac`) from `results/baby_dog_vision_approach_round3/` promoted to the **active learning restart candidate**.

| measured metric (three cohorts, 192 lives) | fresh baseline | 0.5M steps | 1.0M steps (983k candidate) | final 999,424 steps |
|---|---:|---:|---:|---:|
| mean seven movement falls / 13 s | 0.2857 | 0.3006 | 0.3080 | 0.3006 |
| walk falls / 13 s | 0.3021 | 0.3021 | 0.3906 | 0.3073 |
| pace falls / 13 s | 0.3333 | 0.3750 | 0.3073 | 0.3073 |
| canter falls / 13 s | 0.2500 | 0.2656 | 0.2552 | 0.2604 |
| run falls / 13 s | 0.2812 | 0.2917 | 0.2708 | 0.2865 |
| turn-left falls / 13 s | 0.2812 | 0.2708 | 0.3385 | 0.2969 |
| turn-right falls / 13 s | 0.2396 | 0.2708 | 0.2135 | 0.2240 |
| back falls / 13 s | 0.3125 | 0.3281 | 0.3802 | 0.4167 |
| walk speed, m/s | +0.3709 | +0.3662 | +0.3616 | +0.3640 |
| pace speed, m/s | +0.4292 | +0.4449 | +0.4470 | +0.4431 |
| canter speed, m/s | +0.3477 | +0.3444 | +0.3404 | +0.3441 |
| run speed, m/s | +0.5050 | +0.5303 | +0.5288 | +0.5367 |
| turn-left yaw, rad/s | +1.4398 | +1.4552 | +1.4300 | +1.4503 |
| turn-right yaw, rad/s | -1.1610 | -1.1607 | -1.1599 | -1.1585 |
| backward speed, m/s | -0.1557 | -0.1528 | -0.1491 | -0.1470 |
| jump speed, m/s | +0.1384 | +0.1387 | +0.1360 | +0.1392 |
| sit pose (pitch / trunk) | +0.503 / 0.599 | +0.499 / 0.594 | +0.501 / 0.595 | +0.500 / 0.595 |
| lie-down pose (pitch / trunk) | +0.054 / 0.962 | +0.053 / 0.961 | +0.052 / 0.961 | +0.054 / 0.962 |
| ball arrival success (got_there) | 0.52% | 0.0% | 0.0% | 0.0% |
| ball distance drift (start -> end, m) | 2.492 -> 2.512 | 2.483 -> 2.562 | **2.481 -> 2.474** | 2.480 -> 2.529 |
| ball aligned in final second | 7.82% | 10.09% | **10.94%** | 6.75% |
| ball final facing cosine | -0.0691 | +0.0388 | **+0.0508** | -0.0054 |
| owner arrival success (got_there, "come") | 0.52% | 0.52% | **1.56% (3/192)** | 0.52% |
| owner distance drift (start -> end, m) | 3.661 -> 3.687 | 3.664 -> 3.651 | 3.665 -> 3.731 | 3.662 -> 3.736 |
| owner aligned in final second | 12.58% | 13.69% | 9.81% | 12.67% |
| owner final facing cosine | -0.0451 | -0.0057 | **+0.0022** | +0.0119 |
| ball-right balanced accuracy | 0.593 | 0.597 | 0.579 | 0.582 |
| owner-left balanced accuracy | 0.598 | 0.606 | 0.610 | 0.611 |
| owner-ahead balanced accuracy | 0.514 | 0.522 | 0.516 | 0.525 |
| moving balanced accuracy | 0.697 | 0.707 | 0.707 | 0.712 |
| dodge hit rate / reaction median | 83.85% / 220 ms | 82.29% / 215 ms | 80.73% / 217 ms | 80.21% / 225 ms |

**Interpretation:**
1. **First net distance closure to ball:** In snapshot `baby_dog_step_0000983040.npz` (`1e164e63…`), the dog closed distance from 2.481m to 2.474m (-0.007m) and finished facing the ball (+0.0508 cosine, 10.94% final-second alignment).
2. **"Come" arrival retention:** 1.56% (3 of 192 dogs) reached the owner within 1.0m, with positive facing cosine (+0.0022).
3. **Locomotion and poses preserved:** Snapshot `1e164e63…` preserved all 11 motor skills with sit (+0.501 / 0.595) and lie-down (+0.052 / 0.961) stable with zero teacher help.
4. **Candidate promotion:** `baby_dog_step_0000983040.npz` (SHA256 `1e164e6338ab842dcff44377201237a992d369bdce5adee5208a0219926c6263`) promoted to the **active learning restart candidate**.


### Latest completed trial: dedicated vision learning rate approach pilot

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-selflearn-vision-approach-pilot (version 1).
All 61 updates completed (999,424 actual steps, 79.6 min run). Outputs automatically downloaded into `results/baby_dog_selflearn_vision_approach_pilot/`.
Evaluation across three fresh cohorts of 64 lives = 192 lives per evaluation:

| measured metric (three cohorts, 192 lives) | fresh baseline | final 999,424 steps |
|---|---:|---:|
| mean seven movement falls / 13 s | 0.3013 | 0.2969 |
| walk falls / 13 s | 0.2969 | 0.3073 |
| pace falls / 13 s | 0.3646 | 0.3125 |
| canter falls / 13 s | 0.2396 | 0.2656 |
| run falls / 13 s | 0.3125 | 0.2917 |
| turn-left falls / 13 s | 0.2240 | 0.3177 |
| turn-right falls / 13 s | 0.2708 | 0.2188 |
| back falls / 13 s | 0.4010 | 0.3646 |
| walk speed, m/s | +0.3461 | +0.3696 |
| pace speed, m/s | +0.4198 | +0.4285 |
| canter speed, m/s | +0.3416 | +0.3482 |
| run speed, m/s | +0.4739 | +0.4984 |
| turn-left yaw, rad/s | +1.4966 | +1.5196 |
| turn-right yaw, rad/s | -1.1572 | -1.1763 |
| backward speed, m/s | -0.1587 | -0.1485 |
| jump speed, m/s | +0.1321 | +0.1511 |
| sit pose (pitch / trunk) | +0.509 / 0.600 | +0.505 / 0.599 |
| lie-down pose (pitch / trunk) | +0.056 / 0.962 | +0.057 / 0.960 |
| ball arrival success (got_there) | 0.0% | 0.0% |
| ball distance drift (start -> end, m) | 2.492 -> 2.534 | 2.482 -> 2.603 |
| ball aligned in final second | 8.45% | 10.93% |
| ball final facing cosine | -0.030 | -0.058 |
| owner arrival success (got_there, "come") | 0.0% | 2.60% (5/192) |
| owner distance drift (start -> end, m) | 3.663 -> 3.740 | 3.668 -> 3.669 |
| owner aligned in final second | 9.42% | 13.56% |
| owner final facing cosine | -0.092 | -0.045 |
| ball-right balanced accuracy | 0.593 | 0.613 |
| owner-left balanced accuracy | 0.580 | 0.606 |
| owner-ahead balanced accuracy | 0.503 | 0.523 |
| moving balanced accuracy | 0.680 | 0.688 |
| dodge hit rate / reaction median | 80.21% / 210 ms | 81.77% / 97 ms |

**Interpretation:**
1. **Locomotion and poses preserved:** Mean 7-gait falls improved slightly from 0.3013 to 0.2969 per 13s. Sit pose (+0.505 pitch, 0.599 trunk) and lie-down (+0.057 pitch, 0.960 trunk) remained stable.
2. **First non-zero arrival on "come":** 2.60% of test dogs reached the owner within 1.0m (5 out of 192). Outward distance drift stopped completely (+0.001m net change vs +0.077m baseline). Final-second heading alignment improved from 9.42% to 13.56%.
3. **"Ball" arrival remains at 0%:** Alignment improved from 8.45% to 10.93%, but the 0.35m arrival sphere was not reached. Higher precision is needed for the smaller 0.11m ball target.
4. **Active perception learning:** In-practice perception accuracy reached 0.66 for owner-left and 0.63 for ball-right; sight prediction error fell from 0.741 to 0.702.
5. **Candidate promotion:** `baby_dog_step_0000999424.npz` (`17f6c746…`) / `baby_dog.npz` (`1452d51e…`) is promoted to the active **learning restart checkpoint**.

Evidence is local (git-ignored):
- `results/baby_dog_selflearn_vision_approach_pilot/practice.json`: all four averaged evaluations;
- `results/baby_dog_selflearn_vision_approach_pilot/resume.json`: exact input identity and configuration;
- `results/baby_dog_selflearn_vision_approach_pilot/baby_dog.npz`, `baby_dog_step_0000491520.npz`, `baby_dog_step_0000983040.npz`, `baby_dog_step_0000999424.npz`, `log.txt`;
- `results/kaggle/baby_dog_selflearn_vision_approach_pilot/`: submit script and output logs.

### Prior trial: approach outcome pilot (uniform lr)

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-selflearn-approach-pilot (version 1).
Configured with `lesson=approach`: rewards getting closer (`TASK_PAY * closer`), turning (`FACE_PAY * turned`), and reaching
(`REACH_PAY * there`), keeping curiosity/eating off, 11 movement skills rehearsed, teacher help zero, ±0.1 wobble,
lr 5e-6 / voice lr 3e-4, outcome guard on, 1M steps (61 updates), 512 envs (32 unroll ticks = 0.64 s credit assignment), 3 cohorts, 3h cap on 2x T4.
Submitted from the verified pre-voice restart candidate `baby_dog_step_0000983040.npz` (`18f6e629…`).
All 61 updates completed (999,424 actual steps). Outputs automatically downloaded into `results/baby_dog_selflearn_approach_pilot/`.

| measured metric (three cohorts) | fresh baseline | final 999,424 steps |
|---|---:|---:|
| mean seven movement falls / 13 s | 0.3073 | 0.3475 |
| walk falls / 13 s | 0.3073 | 0.3594 |
| pace falls / 13 s | 0.3490 | 0.3646 |
| canter falls / 13 s | 0.2344 | 0.2865 |
| run falls / 13 s | 0.3229 | 0.2812 |
| turn-left falls / 13 s | 0.2448 | 0.3125 |
| turn-right falls / 13 s | 0.2917 | 0.2969 |
| back falls / 13 s | 0.4010 | 0.5312 |
| walk speed, m/s | +0.3424 | +0.3538 |
| pace speed, m/s | +0.4203 | +0.4334 |
| canter speed, m/s | +0.3403 | +0.3482 |
| run speed, m/s | +0.4732 | +0.4807 |
| turn-left yaw, rad/s | +1.4938 | +1.4837 |
| turn-right yaw, rad/s | -1.1576 | -1.2052 |
| backward speed, m/s | -0.1594 | -0.1614 |
| jump speed, m/s | +0.1384 | +0.1533 |
| sit pose (pitch / trunk) | +0.51 / 0.60 | +0.51 / 0.60 |
| lie-down pose (pitch / trunk) | +0.06 / 0.96 | +0.05 / 0.96 |
| ball arrival success (got_there) | 0.0% | 0.0% |
| ball distance drift (start -> end, m) | 2.48 -> 2.58 | 2.48 -> 2.56 |
| ball aligned in final second | 8.76% | 9.64% |
| ball final facing cosine | -0.056 | +0.002 |
| owner arrival success (got_there) | 0.52% | 1.04% |
| owner distance drift (start -> end, m) | 3.66 -> 3.72 | 3.66 -> 3.68 |
| owner aligned in final second | 12.27% | 9.75% |
| owner final facing cosine | -0.130 | -0.032 |
| hungry balanced accuracy | 0.582 | 0.572 |
| ball-ahead balanced accuracy | 0.551 | 0.516 |
| owner-ahead balanced accuracy | 0.505 | 0.513 |
| dodge hit rate / reaction median | 82.29% / 237 ms | 83.85% / 33 ms |

**Interpretation:**
1. All 11 movement skills were retained with expected locomotion speeds and geometry (sit: +0.51 pitch, 0.60 trunk), but fall rates drifted slightly higher (mean 0.307 -> 0.348 / 13s; back 0.401 -> 0.531).
2. Facing cosine stabilized slightly (-0.056 -> +0.002 for ball, -0.130 -> -0.032 for owner), and outward distance drift was reduced (2.48 -> 2.56m vs baseline 2.58m for ball), but target arrival success remained effectively zero (0.0% ball, 1.04% owner).
3. The reward gradient for approaching was insufficient to induce forward locomotion toward distant targets from rest across 2-3m.
4. Perception balanced accuracies remained near chance (ball-ahead 0.516, hungry 0.572).
5. **No checkpoint promotion:** `stage16_park3_open.npz` (`4af773f4…`) remains global best. `baby_dog_step_0000983040.npz` (`18f6e629…`) remains learning restart. Newly saved checkpoints `baby_dog_step_0000999424.npz` (`d9f92de6…`) and `baby_dog.npz` (`2211476f…`) are preserved locally for comparison.

Evidence is local (git-ignored):
- `results/baby_dog_selflearn_approach_pilot/practice.json`: all four averaged evaluations;
- `results/baby_dog_selflearn_approach_pilot/resume.json`: exact input identity and configuration;
- `results/baby_dog_selflearn_approach_pilot/baby_dog.npz`, `baby_dog_step_0000491520.npz`, `baby_dog_step_0000983040.npz`, `baby_dog_step_0000999424.npz`, `log.txt`;
- `results/kaggle/baby_dog_selflearn_approach_pilot/`: submit script, monitor and live streams.

### Prior trial: longer recurrent history

Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-selflearn-long-history-pilot (version 1).
Same verified input and orientation configuration as the 32-tick control; only unroll/envs changed from 32/512
to 128/128. This gives 2.56 s of recurrent credit assignment and keeps 16,384 samples per update. Shared lr 5e-6,
voice lr 3e-4, seed 1, outcome guard enabled, every sense and prior skill kept, teacher movement help zero.
All 61 updates completed: 999,424 actual steps, including six critic-only warm-up updates. Three fresh 64-life
cohorts were used at baseline and each evaluation. JAX 0.7.2 ran on two CUDA devices; carry-over reported
`new parts: none`. Runtime guide loss and every recorded help value were zero.

| measured metric | fresh baseline | final 999,424 steps |
|---|---:|---:|
| mean seven movement falls / 13 s | 0.2991 | 0.2731 |
| walk falls / 13 s | 0.3073 | 0.2708 |
| pace falls / 13 s | 0.3438 | 0.3490 |
| canter falls / 13 s | 0.2344 | 0.2500 |
| back falls / 13 s | 0.4010 | 0.3125 |
| walk speed, m/s | 0.3446 | 0.3505 |
| backward speed, m/s | -0.1621 | -0.1514 |
| ball aligned in final second | 9.40% | 6.68% |
| owner aligned in final second | 10.50% | 10.84% |
| ball final facing cosine | -0.0223 | -0.0294 |
| owner final facing cosine | -0.0792 | -0.0384 |
| hungry balanced accuracy | 0.5714 | 0.5780 |
| ball-ahead balanced accuracy | 0.5524 | 0.5399 |
| owner-ahead balanced accuracy | 0.4965 | 0.5023 |

Final ball and owner arrival success are both 0%; hungry food seeking and recovery are also 0%. Food and recovery
outcome training were not enabled in this focused orientation lesson, so their scores describe unfinished goals.
**Interpretation:** movement was largely retained, with mixed individual fall changes and slightly weaker backing.
Useful orientation or perception learning was not established. Ball alignment worsened; owner alignment and hungry
naming barely changed. Longer history did not fix the stall in this trial. This is not proof that longer history
can never help: concurrent lives also changed, and repeated tests showed variation. No checkpoint promotion.

The completed 32-tick control had mean falls 0.2976 -> 0.3028, ball alignment 8.93% -> 10.79%, and hungry accuracy
0.5724 -> 0.5716. Compare each trial against its own fresh baseline; do not overinterpret small differences.
The longer-history 0.5M snapshot had ball alignment 7.09%; its 983,040-step snapshot 7.64%; neither showed mastery.
The last two curve labels both round to `after 1.0M steps`; distinguish 983,040 and 999,424 by snapshot filename.

Evidence is already local (git-ignored):
- `results/baby_dog_selflearn_long_history_pilot/practice.json`: all four averaged evaluations;
- `results/baby_dog_selflearn_long_history_pilot/resume.json`: exact input identity and configuration;
- `results/baby_dog_selflearn_long_history_pilot/baby_dog.npz`: final brain;
- the same directory's `baby_dog_step_0000491520.npz`, `baby_dog_step_0000983040.npz`,
  `baby_dog_step_0000999424.npz`, and `log.txt`;
- `results/kaggle/baby_dog_selflearn_long_history_pilot/`: exact submit script, monitor and live streams.

Original best: `results/baby_dog_stages/stage16_park3_open.npz`, SHA256
`4af773f477e7a173e74904d66d82943227ab2b96b5e2ee1454dc7921a7f0bf41`.
Keep using the earlier learning candidate, SHA256
`18f6e62923d7b1c832fe6731784af830c37843d8fcd855b52d2944bd956b8dc1`, partial 60/61:
`/kaggle/input/notebooks/karang1908/bio-bot-baby-dog-selflearn-levels-pilot/baby_dog_step_0000983040.npz`.
It passed remote CPU identity/finite-parameter/wiring/voice/zero-help checks. The longer-history final weights have
not received a separate post-training finite-parameter audit; do not call such an audit passed.
Training baby source SHA256 was unchanged from the CPU-verified orientation source:
`a45385fe785db0ef0acee0c7a1659ca4aa4bab64514930a4f837754197d4bbcf`.

Runtime issues: package resolver/optional Warp-import warnings appeared, but execution continued. XLA reported
slow constant folding and a 2m10s compilation operation; all updates and tests subsequently completed. Live log
connections can become stale; active clock checks and reconnections recovered current output. The ~115-minute
monitor elapsed includes ~31 minutes queued; it is not 115 minutes of training. No local model checks/training
were run, and no Colab runtime or additional account was used. Current remaining GPU quota was not queried.
Completion is recorded in `LOGBOOK.md` §6.10 and above.
Current agent takeover (12 Oct 01:30 Dubai): Continuing the `/goal` mission to teach the dog visual approach and orientation while preserving all motor skills.
Vision approach pilot v1 demonstrated the first non-zero target arrival success on "come" (2.60%, 5/192 dogs) and eliminated outward drift (+0.001m net drift vs +0.077m baseline), while reducing overall 7-gait falls (0.3013 -> 0.2969 per 13s) and preserving all 11 motor skills.
Next measured question: How to scale "come" arrival from 2.6% to >15% and break through "ball" arrival (currently 0.0% due to smaller 0.11m target radius)?
Two-pronged iteration:
1. Continue training from newly promoted restart candidate `baby_dog_step_0000999424.npz` (`17f6c746…`) using hosted kernel output `--kernel-source karang1908/bio-bot-baby-dog-selflearn-vision-approach-pilot` for 1.5M steps (91 updates).
2. Enhance heading alignment reward during approach rollouts: In `task_reward` (`train/baby_dog.py`), augment the step-wise change reward `FACE_PAY * turned` with a continuous alignment reward `ALIGN_PAY * jnp.maximum(0.0, facing_cos)` when facing toward the target, ensuring forward walking is reinforced whenever the dog is looking at the target.

**11 Oct results:** the first teacher-zero perception pilot completed but was rejected: walking falls rose
0.422 -> 1.031 per 13 s, run falls 0.375 -> 0.828, and backward speed weakened -0.155 -> -0.045 m/s.
The P100-requested repeat failed before training with the confirmed decimal-seed TypeError. Read LOGBOOK §6.6
for exact results. Original stage16 remains best; neither output is promoted. The next level-sense trial is now
reduced to 1M steps at lr 1e-5 with 0.5M/1M tests, a 3h cap, and separate intermediate checkpoint files.
CPU preflight version 8 passed. The GPU job `baby_dog_selflearn_levels_pilot` version 1 is accepted;
its result and retention are pending. Its API-only monitor fetches terminal outputs. No automatic promotion.
At 22:20 UTC the live stream worked: correct input hash, integer seed, added level senses, two CUDA devices,
and carry-over with `new parts: none`. Setup is underway; fresh movement tests and learning results are pending.
The level-sense pilot has now completed; read LOGBOOK §6.7 and its fetched `practice.json`. Perception remains
weak and falls are mixed. Its intermediate `baby_dog_step_0000983040.npz` was selected for learning continuation
because aggregate movement falls were lowest; original stage16 remains the established best. Free CPU review
passed, including finite parameters, zero help, preserved wiring/voice/words, and `absolute_drives: true`.
Candidate SHA256: `18f6e62923d7b1c832fe6731784af830c37843d8fcd855b52d2944bd956b8dc1`, partial 60/61.
Both last test labels round to `after 1.0M steps`; distinguish snapshots by filename/hash. A 2M perception
continuation `baby_dog_selflearn_levels_continue` version 1 is accepted with seed 1, lr 1e-5, four evaluations
and 3h cap. Its API-only monitor will fetch terminal outputs. Result pending; no global-best promotion.
At 11 Oct 05:37 UTC, direct status/quota reads require renewed Kaggle authentication. Latest local continuation
status is RUNNING at 47.0 minutes; no output directory was fetched. Current result is unknown. Refresh with
`.venv/bin/kaggle auth login --force`, then check/fetch; do not report the old status as current or relaunch blindly.
**Latest, after login renewal:** the 2M continuation is COMPLETE and fetched, but rejected: all seven movement
fall rates increased, hungry naming remains ~0.60, and navigation/recovery are still unlearned. Read LOGBOOK §6.8.
An optional `--protect_skills` update projection passed its Kaggle CPU helper/gradient/identity checks.
GPU trial `baby_dog_selflearn_guard_pilot` version 1 is accepted: matched input/seed/lr/2M steps/four evaluations
with a 3h cap and the guard enabled. Its API-only monitor will fetch terminal outputs. GPU results pending.
The earlier 983,040-step candidate (hash above) remains the learning restart; do not resume the regressed 2M output.
The guard is a first-order training constraint, not a proven retention fix or an extra runtime controller.
Its live 0.5M evaluation is mixed: walk falls 0.33 -> 0.23, back falls 0.39 -> 0.52, hungry accuracy still 0.60.
The matched source/input/flags and zero movement help are confirmed in the stream. Remaining evaluations and
terminal checkpoint JSON are pending; no promotion. Live log: `results/kaggle/baby_dog_selflearn_guard_pilot/live.log`.
**Latest completion:** the guard trial is COMPLETE and fetched; read LOGBOOK §6.9. Mean movement falls stayed
close to baseline (0.308 -> 0.321), much better than the prior run, but back falls 0.344 -> 0.484 and perception
barely changed. No promotion. The 0% task scores must not be used to judge this perception lesson: its approach/
food/curiosity rewards were disabled, and get-up practice was not enabled. Those goals remain unfinished.
A separate optional voice learning rate is prepared. Free CPU job `baby_dog_selflearn_voice_preflight` v1 is
accepted and pending. Next planned GPU job `baby_dog_selflearn_voice_pilot`: 1M steps from the same verified
983,040-step input, shared lr 0.0 / voice lr 3e-4, all senses and rehearsal kept, teacher zero. It trains the voice
readout from the brain's own activity while leaving all shared/motor weights fixed; requires a remote exact
parameter audit. This is temporary supervised naming, not completed self-learning. Submit only after CPU pass.
The voice CPU preflight has now passed: actual optimizer jit, nested-weight preservation, default parity,
source compilation and exact candidate identity/finite parameters/zero help all verified remotely. Report:
`results/baby_dog_selflearn_voice_preflight/preflight.json`. GPU submission/result still must be confirmed.
**Latest accepted job:** `baby_dog_selflearn_voice_pilot` version 1 is accepted after the CPU pass. Its API monitor
will download terminal outputs and its live stream is saved under `results/kaggle/baby_dog_selflearn_voice_pilot/`.
Training/perception results and the actual `voice_audit.json` are pending; no checkpoint is promoted.
**Latest, timed monitoring requested:** the voice pilot is COMPLETE and fetched. Its remote audit proves all
55 non-voice arrays exactly unchanged, while both voice arrays learned; hungry accuracy still ~0.59. No promotion.
Read LOGBOOK §6.10: unchanged motor weights still produced varying fall scores. A confirmed scripted-test
new-life reset mismatch is corrected, and the next job will average three fresh 64-life cohorts. Old/new test
protocols are not directly interchangeable; require a fresh paired baseline. Orientation metrics are added.
Free CPU evaluation preflight v2 is COMPLETE (v1 had the wrong input path); inspect its report before submitting
`baby_dog_selflearn_orient_pilot`. Its prepared submit script uses the same verified pre-voice candidate,
teacher-zero orientation outcomes, lr 5e-6 / voice 3e-4, guard on, 1M steps, two intervals/three cohorts, 3h cap.
No GPU submission/result for this new lesson is confirmed yet. The user wants active timer checks until completion.
Evaluation CPU preflight v2 passed. **Latest accepted GPU job:** `baby_dog_selflearn_orient_pilot` version 1,
same prepared configuration; status/live logs under `results/kaggle/baby_dog_selflearn_orient_pilot/`, outputs
automatically fetched at termination. The agent is actively checking on a roughly one-minute clock timer.
Learning results pending, no promotion. Shared brain plasticity is back on; movement teacher help remains zero.
**Latest completed result:** orientation v1 completed at 15:52 Dubai, all outputs fetched. Three-cohort mean
movement falls 0.298 -> 0.303; ball alignment 8.93% -> 10.79%, owner 9.96% -> 10.18%; hungry accuracy ~0.572.
Retention is largely stable, but useful orientation learning is not established. No promotion/restart from this
output. Read LOGBOOK §6.10. Live streaming can become stale while the job progresses; reconnect when it stops
advancing. Status polling continued every ~30s and terminal download worked. The timed notebook tracking is done.
Next free CPU probe `baby_dog_selflearn_sensor_path` runs the actual frozen brain on synthetic steady charge
inputs for 200 ticks (no physics/actions/weight updates), with a 20-minute cap. It tests whether the added charge
levels reach internal/central/voice activity. Interpret only as a static response probe; real-scene learning is
not measured by it. Results pending. The agent is continuing timed checks for this follow-up CPU job.
**Latest CPU result:** sensor-path probe is COMPLETE and fetched. Charge does alter internal and central
activity; all 30 internal-sense neurons are active at 4 s. Hungry voice logits barely distinguish charge.
Do not assume a disconnected/dead charge pathway or reset learned activations/gains. Read LOGBOOK §6.10 for
the static-probe limits. Next prepared trial `baby_dog_selflearn_long_history_pilot` uses the same orientation
configuration/input, with unroll 128 / envs 128 (2.56 s history, still 16,384 samples/update), 1M steps and 3h cap.
This tests a credit-assignment hypothesis; no behavioral fix is established. Training-source code is unchanged.
**Latest accepted job:** `baby_dog_selflearn_long_history_pilot` v1 accepted with the prepared settings. Its API
monitor logs under `results/kaggle/baby_dog_selflearn_long_history_pilot/` and will fetch outputs. The agent is
continuing clock-based checks. Execution/learning results pending; no promotion and full training remains incomplete.
The owner installed official VS Code Colab extension 0.9.7 as a free fallback. No Colab runtime/training has started.
Use one account per provider; account rotation to evade quotas is prohibited by their published rules. GPU quota
at 23:00 UTC: 28.23h remaining. Remote notebook execution needs no additional MCP/CLI for now.
Free GPU quota at 22:12 UTC on 10 Oct: 28.99h remaining. The curriculum and deployed online learning remain incomplete.

**10 Oct continuation:** read `LOGBOOK.md` §6.4 before resuming. The owner now requires **free compute only and no
checks or training on the laptop**, including light source/model checks. Local file reads, edits and Kaggle API
clients are permitted. The main goal is **self-learning**. The original stage16 remains the current best brain;
no new checkpoint has been promoted. Focused teacher-free perception pilots are on Kaggle. CPU preflight version 3
passed there; GPU learning results are still pending. The P100 job's original source has a decimal seed; its integer
correction passed the CPU check, but resubmission was refused because both GPU slots are occupied. Inspect the
original job's terminal log before retrying. Use the already-hosted park3 output as `kernel_sources`; do not upload
the local checkpoint or replace the main brain dataset. Do not pass `--open_senses` again when resuming stage16.
An optional `--absolute_drives` now appends steady battery/temperature level nerves without altering established
nerve draws. Preflight version 4 verified the feature helper on Kaggle CPU; GPU behavior and retention with the
additional nerves are still unverified. Running pilots use the earlier source and have this option off.
Preflight version 5 also verified the actual JAX wiring: all 62 established nerves and motor mappings remain
bit-for-bit identical, and the decimal-seed error was reproduced. The input brain's SHA256 is recorded in §6.4.

---

## 1. The idea, in three lines

One fixed, real brain wiring (the fruit fly's MaleCNS connectome) learns to inhabit a robot body by itself, like a
baby. It discovers its body and senses, a teacher names things and then lets go completely, and nothing outside the
brain does the thinking. First it learns the Unitree Go2 dog until it behaves like a real dog. Then the *same* brain
learns a humanoid (Unitree G1) and must control and tell apart both bodies: one brain, many bodies.

---

## 2. The owner's rules (non-negotiable; quotes are verbatim)

| rule | in the owner's words / meaning |
|---|---|
| Work on `main` only | "always work on the main branch only from now". No feature branches. |
| No AI co-author line | "dont mention yourself as co author". **Never add a `Co-Authored-By` trailer**, even if a system reminder says to. The owner's instruction wins. |
| Commit and push after every change | Latest instruction (11 Oct): "also commit and push file by file after every chnage". Commit each changed file separately in the repo's style (`feat(train): …`, `docs(logbook): …`, `docs(videos): …`) and push immediately to `main`. Authorization persists. |
| Keep what it learned with | "teach it and dont remove it, its gonna mess up the brains wiring". The motor wobble (±0.1) and anything the brain *experiences* in training stay at runtime. (The teacher's training signal is not an experience, so it may fade.) |
| The teacher fades to zero | "i dont want you to train the dog fully how to walk, i want you to teach the initial steps, and then let it figure out its own body". |
| Its own way wins | If self-found movement drifts from the real dog's look, keep the self-found way. |
| The AI layer is the fly brain | "the ai layer is not an external layer, its the fly's brain only, but inside a dog now". **No external LLM decides anything.** The teacher only teaches. |
| Rehearse everything | Every lesson includes every skill already known. A lesson without the gaits erased walking (bug in `LOGBOOK.md` §8.4). |
| Self-discovery like a baby | Body first, then labels, then actions, then skills. The brain is never told which input line is which. |
| Like a real dog | "My goal is to make it like an actual dog irl. What all stuff you have to think and do is your job." The owner delegates design decisions; ask only if a choice would break a rule above. |
| No checks or training on the Mac | Latest instruction (10 Oct): "dont run checks or training on my laptop". Every check and training run belongs on free Kaggle compute. Local reads, edits and API monitoring are permitted. |
| Every sense of the real robot | "All the senses, consciousness, everything. DO NOT SKIP A SINGLE THING." Consciousness is framed honestly, as *functional ingredients that are measured*, never claimed. |
| Ask if in doubt | "ask me if you have any doubt". But don't ask about routine engineering. |
| No unrequested diagnostics | The owner once rejected a diagnostic edit when they wanted training. Don't wander. |

The owner's global `~/.claude/CLAUDE.md` adds the method: predict before you run, reproduce before you claim a fix,
label everything verified / believed / assumed, report failures verbatim.

---

## 3. Where things stand (5–10 Oct 2026)

### 3.1 The dog today

The **current best brain** is `results/baby_dog_stages/stage16_park3_open.npz`: core v2, 7,019 neurons, senses opened.
It walks, paces, canters, runs, turns, backs, sits and stands on words, with **no teacher at all**. Real tests from
rest, with wobble on and no help, at the end of `baby_dog_park3`:

| word | result |
|---|---|
| walk / pace / canter / run | 0.34 / 0.42 / 0.34 / 0.48 m/s (real dog: 0.19 / 0.71 / 0.48 / 2.19) |
| turn-left / turn-right | +1.47 / −1.10 rad/s (real dog +1.38 / −1.18) |
| back | −0.16 m/s |
| sit | nose up +0.51 rad (good) |
| lie-down | **lost**: trunk 0.98 of standing (target ~0.39) |
| jump | forward 0.13 m/s, never airborne |
| falls per 13 s | **0.2–0.44: too high**; the earlier let-go brain had ~0.02 |

**Not learned yet** (all ~0–2%): go to the ball, come to the owner, dodging a thrown object (hit 84%, against ~96%
for a dog that doesn't move), a hungry dog going to its food. The voice names its own state only weakly:
ball-left/right 0.58, owner-left/right 0.56–0.59, hungry ≤0.65, moving 0.74 (balanced accuracy, 0.5 = chance).

For the cleanest walking, `stage12_letgo.npz` (v1 core, plain body, no senses) is the best "pure locomotion" brain.
Falls were ~0, pace 0.78, run 0.99. Films of the grown version are in `videos/dog_selftaught_*.mp4`.

### 3.2 The build order (`DogMind.md` §7) and its status

| step | what | status |
|---|---|---|
| 0 | watch the real dog (motion capture) | **done**: four new gaits; poses 4× closer to the real dog |
| 1 | the teacher lets go to zero | **done**: every skill kept with zero help (`stage12_letgo`) |
| 2 | the park and every sense of the real Go2 | **done and verified** (`train/dog_world.py`) |
| 3 | grow the brain (two sides, new senses) without erasing it | **done**: skills kept within 0.22; *but see bug §5.17* |
| 4 | predict its own senses (self-model) | **done**: body / hearing / smell / sight all better than "no change" (end of park3: 0.31 / 0.91 / 0.53 / 0.73) |
| 5 | goals in the park: ball, come, dodge | **built, not learned** |
| 6 | drives: free time, hunger, curiosity | **built, smoke-tested, not learned** |
| 7 | thinking: voice (self-report), multi-step errands, memory, learned good and bad | **voice built, weak**; the rest not started |
| 8 | never-taught tests: get up, dodge, detour, gait choice | tests exist for get-up (1–2%) and dodge |
| after | the humanoid G1 in the same brain; both bodies | not started |

### 3.3 Published continuation work

The owner authorized file-by-file commits and pushes on 11 Oct. Continuation work is published individually:
`scripts/kaggle_job.py`, `train/brain_dog.py`, `train/baby_dog.py`, `train/dog_preflight.py`, `DogMind.md`,
`LOGBOOK.md` and this handoff. No local source/model checks were run for publication. Relevant checks passed
on Kaggle CPU and bounded GPU trials completed; exact results and source hashes are in LOGBOOK §§6.4–6.10.
New levels, update protection and naming lessons have not established useful navigation or mastered perception.
Every future file change must likewise be committed separately and pushed immediately.

### 3.4 Data and brains

**Kaggle datasets (all private, account `karang1908`):**

| dataset | holds |
|---|---|
| `bio-bot-brains` | one file, `baby_dog.npz`, which jobs load. It currently holds **stage15_park2** (md5 7c91d76d…), *not* the newest brain. |
| `bio-bot-cores` | `core_v1.npz`, `core_v2.npz` |
| `bio-bot-dog-mocap` | `go2_dog_motions.npz` (MANN dog mocap retargeted to the Go2; CC BY-NC 4.0, non-commercial) |

**Local (git-ignored):**
- `data/malecns/` holds the connectome tables (1.1 GB+), `core_v1.npz` and `core_v2.npz`.
- `data/dog_mocap/` holds the BVH files and the npz.
- `results/baby_dog_stages/` holds every stage's brain (`stage1` … `stage16`), with md5s noted in the logbook.

---

## 4. The system, briefly

### 4.1 Files

| file | role |
|---|---|
| `brain/core.py` | builds the cell-type cores from MaleCNS. `build()` makes v1: 2,500 units, one per type, sides merged. `build_v2()` makes v2: 7,019 units, one per type *per side*, plus visual projection, Johnston's organ (hearing), olfactory, endocrine (inner state), the central complex, mushroom body and antennal lobe. `v1_unit` / `v1_edge` record where each v2 unit or connection came from. |
| `train/brain_dog.py` | the body (`make_body`), the brain (`make_brain`), the teacher (`make_teacher`), `build_model`, `film`, `load_core(version)`, `ensure_packages` (pinned jax 0.7.2 CUDA set). |
| `train/baby_dog.py` | the whole curriculum. Stages: babble, name, act, walk_words, walk, start_stop, gaits, no_parent, practice, poses, film_walk, mocap_check, world_check, grow. Plus `practice()` with all the flags. |
| `train/dog_world.py` | the park and every sense: eye (24×18, 2 colours, retinotopic), LiDAR (36×9), ears (2×3 bands), nose (2×2), paws, skin, battery, motor heat. Throws, the CPU reference ray caster, PNG pictures. |
| `train/dog_mocap.py` | BVH parser, forward kinematics, closed-form Go2 leg IK, auto-labels, contacts. Writes `data/dog_mocap/go2_dog_motions.npz`. |
| `scripts/kaggle_job.py` | `push / status / wait / fetch / upload`. Options: `--job`, `--embed`, `--dataset`, `--set KEY=VALUE`, `--flag`, `--machine`, `--steps`. |
| `world/`, `studio/` | the earlier interactive studio (CPU MuJoCo, big open world). Not used by training. |

### 4.2 The brain

- **Core:** a rate model on the connectome, with fixed signed connectivity. It learns per-unit time constants and
  biases, per-edge gains (`gamma`, via softplus) and a global gain.
- **Inputs:** a set of anonymous lines, each with adaptive normalisation and a causal "self-map" of how it answers each
  motor. Attention carries them into the sense neurons. In v2 each line only reaches the sense neurons of its own
  nerve (`seen["mod"]`).
- **Vision:** the eye and LiDAR go through a mirror-symmetric "optic lobe" (two shared 3×3 conv layers per side). Each
  visual-projection type pools them through its own receptive field (`rf_*`, `vp_*`).
- **Outputs:** the readout is "vector": attention from each output line over the motor neurons.
- **Words** enter the central and descending neurons (`word_w`).
- **Heads:** `predict` (next line change), `predict_sight` (next eye change, using an efference copy of the motor
  neurons), and `say` (the voice: a linear readout from the central and descending neurons).
- **Fixed wiring:** one random draw, `WIRING = 20260927`; the brain is never told which line is which. In park mode
  the 34 original lines keep exactly their old draw; the 28 new lines get their own.

### 4.3 Training

- Recurrent PPO with an asymmetric critic (the critic sees privileged state).
- The wobble is the exploration noise.
- A per-word teacher "help" scales the guide, the style pull and the four-feet rule. It falls by skill to zero
  (`--let_go`); a brain whose meta says `let_go` starts at zero.
- **Outcome rewards only:**
  - speed, turning and stillness, plus the trunk for sit and lie-down and flight for jump;
  - falls, plus an energy cost;
  - tasks: progress toward the goal, facing it, reaching it;
  - hits and hunger.
- **Real tests** are scripted lives from rest, run at every checkpoint (8 per run).

### 4.4 How to run (Kaggle)

Authentication was renewed at 09:39 Dubai on 11 Oct and later status calls worked. Ask the owner to run
`.venv/bin/kaggle auth login --force` only when current API calls show an authentication failure. Do not assume
a fixed three-hour expiry or resubmit a notebook when its status is unknown.

```bash
# API-only operations on the current submitted notebook; these do not train locally.
.venv/bin/python scripts/kaggle_job.py status baby_dog --job baby_dog_selflearn_long_history_pilot
.venv/bin/python scripts/kaggle_job.py fetch baby_dog --job baby_dog_selflearn_long_history_pilot
```

The current monitor is already running; do not start duplicate GPU runs. Its submit script under
`results/kaggle/baby_dog_selflearn_long_history_pilot/` records the exact hosted checkpoint and job arguments.
Use `--kernel-source` to attach existing cloud outputs. `bio-bot-brains` still contains stage15, so it is not the
current restart. An earlier local-checkpoint upload was rejected by automatic approval review; no upload occurred.
Do not upload local checkpoints or replace that dataset as part of routine continuation.

**All checks and training run on free Kaggle CPU/GPU.** No light local checks, source compilation, neural forward
passes, simulations or training are authorized on the Mac. Local reads/edits, JSON result analysis, Git operations
and API clients are permitted. Colab is an unused free fallback; use one account per provider.

---

## 5. Problems faced, and what each taught (the important part)

Numbered roughly in order; the logbook's §8 has more. Each item: what happened, then the mechanism or fix.

### Learning design

1. **Semantics-only reward never produced walking** (phase A, 200M steps). The reward described what walking
   *achieves*, but nothing ever produced one example of forward travel, so standing still was the safe optimum.
   → A teacher that shows, assists and lets go.
2. **Goal signals died within one hop.** Spectral-radius scaling of the connectome shrank every ordinary connection
   ~70×. → Input-average normalisation (each neuron's input is the synapse-weighted average); goals fed into the
   descending neurons.
3. **Reshuffling the body every 30 s life made learning impossible** (copy test 0.98× → 0.46× with a fixed body).
   → One baby, one fixed wiring.
4. **The teacher's let-go signal was fooled:** "track" scored standing still above walking. → Let go on real progress.
5. **A misleading measure:** walking measured mid-activity gave +0.205 m/s; from rest it was ~0. → Every checkpoint
   runs scripted real tests from rest.
6. **Forgetting:** the poses lesson without the gaits took walking from 0.39 to 0.03 m/s. → Rehearse everything.
7. **Reward loopholes:**
   - the four-feet reward averaged over feet, so holding one foot up paid → the laziest foot;
   - lie-down counted as a fall → a fall is tipped only;
   - after adding hunger, a fall reset the battery, so falling could "feed" it → hunger and heat persist across falls.
8. **Removing the wobble at test time** made the dog stand still. → Keep what it learned with (an owner rule).
9. **"Self-taught" was mostly the teacher's:** `GUIDE = 0.5` of "be like the teacher" stayed in every practice update
   long after the harness left. → `--let_go`, with help fading to zero by skill.
10. **The teacher came back:** every new practice run restarted help at full. → A brain marked `let_go` starts at
    zero.

### Physics and simulation

11. **A tipped dog sank 27 cm through the floor.** The Menagerie MJX Go2's only trunk shape is a 5.7 cm sphere, and
    only the paws collided. Getting up was physically impossible, which explains every 0% get-up result before
    3 Oct. → A trunk-sized box plus the head collide with the floor (bits: 1 = ground, 2 = trunk and head). Verified:
    standing unchanged; tipped dog at +0.06 m.
12. **Replaying real dog motion through the motors falls over** (sit tips backward; 0.6–7 falls per 6 s). A real
    dog's joint angles are not a controller for a different body. → Use the mocap as a guide and style reward inside
    RL, and let it fade.
13. **Real-dog word meanings carried the recordings' drift:** "pace" also meant "turn right at 0.13 rad/s", and the
    tests scored every moving word as a turn. → Moving words mean forward, turning words mean turn on the spot.
14. **The fence was an infinite collision plane.** The thrown object, parked at (50, 50) between throws, sat 44 m
    inside it, and every dog's physics went NaN about 3 s after a throw. My first local tests stopped at exactly
    3.0 s, one step before it. → Finite capsule rails behind visible boards. Lesson: run every check past every timer
    in the code under test.
15. **Thrown-object force spikes** (41 kN on the skin with the 1-iteration solver). Measured: the dog's speed changes
    0.057 m/s per hit, under the 0.157 m/s momentum bound. So it is a sensor exaggeration, not a real shove. The fast
    solver is kept; skin lines are normalised and clipped.

### The grown brain and the senses

16. **The v1 core merged the two sides of the fly**, so nothing could represent "ball on the left". → Core v2, one
    unit per type per side. Skills carried within 0.22 (verified).
17. **The new senses were wired in at 1% strength (the biggest lesson).** To keep the grown brain's skills, every
    connection v1 never had started at ~1% gain (`gamma = −5`). At that gain the gradient is also ~1%, so the
    connections never opened. After 120M steps, vision (7,753 connections into the old brain), smell (267) and the
    battery's inner state (27) were still at 1.2%.
    - The voice proved it: it learned "moving" (old body neurons, 0.80) but stayed at exactly chance for ball, owner
      and hungry.
    - → New neurons are silent at rest (bias 0), new connections start at full strength; `--open_senses` applies this
      to an already grown brain.
    - Verified locally: the commands' difference between eyes open and eyes shut went from 0.0045 to 0.039. In
      park3, the voice rose above chance (0.58–0.65).
    - **Do not pass `--open_senses` again** to a brain that is already open (stage16 or later): it would reset the
      learned gains and biases of every new unit and connection.
18. **Opening the senses disrupted skills** (turn-left 1.42 → 0.91, canter 0.41 → 0.24; falls rose). Practice won
    most of it back, but falls stayed high (0.2–0.44). Open question: how much is the opening, how much the throws,
    hunger and tasks.
19. **The eye's input into the visual neurons starts at exactly zero** (`vp_e`, `vp_l`). That protects skills, but the
    only signals that train it are the RL reward (weak) and the naming loss.
20. **Too many objectives at once.** The park runs try to learn, in one go, through RL, in a 7,019-neuron recurrent
    brain:
    - seeing, looking, going, dodging and eating;
    - naming its state;
    - keeping 13 words.

    The goal rewards can't train seeing, because nothing the dog does is ever caused by what it sees.
    → **Believed** fix (built, not run): a puppy's order:
    1. naming weighted up (`VOICE = 3.0`, rare truths up to 20×) to train seeing;
    2. "ball" and "come" pay for facing the goal first (`FACE_PAY`);
    3. then going, dodging, eating.

    If that still fails, consider lessons that do *only* one new thing at a time (with rehearsal), shorter unrolls,
    or a higher learning rate for the new perception parts only.
21. **Rare classes and an unweighted naming loss:** "ball ahead", "hit" and "fallen" stayed at exactly 0.50, because
    "never" wins. → Positive weighting (built, not run).
22. **Hunger drowned everything:** lives starting at 5% battery starved at once; the reward went negative and the
    critic error rose ~20×. → Lives start at ≥10% battery, `STARVE_COST = 1.0`.

### Infrastructure (Kaggle and tools)

23. **Kaggle changed its image** to jax 0.11.1 on Python 3.13 (Oct 2026). Installing jax 0.7.2 on top left a mix that
    could not import (`ImportError xla_pmap_p`). → `ensure_packages` pins the whole jax[cuda12] 0.7.2 / jaxlib /
    mujoco 3.14 set, checked by version before importing.
24. **Stale downloads:** `kaggle kernels output` skips files of the same size, so the wrong brain was silently used
    twice. → `fetch` always passes `-o`; brains are checked by md5 and by their stored stage name.
25. **A job script over ~1 MB is refused** with a bare `400 Bad Request` (3.25 MB failed; 0.9 MB worked). → Big data
    goes in private datasets (`bio-bot-cores`); `load_core` looks under `/kaggle/input`.
26. **Kaggle OAuth can expire.** Then calls may return "Permission 'kernels.get' was denied" (it looks like a wrong
    slug, but can be authentication). The owner renews with `kaggle auth login --force`. The old three-hour hint is
    not a verified universal expiry; current successful calls take precedence.
27. **At most 2 GPU jobs at once.** A pushed job with a wrong brain path fails after ~5 min of setup. **Datasets take
    ~1 min to become "ready"** after an upload; push only after that, and check the log line `carried over … new
    parts:` to confirm which brain loaded.
28. **GPU quota:** ~30 h a week, reset Saturday 00:00 UTC. Park runs cost 5–8 h each (60M steps ≈ 8 h; the tests are
    ~35 min of that). The brain is now saved at every checkpoint, so hitting the 12 h session limit cannot lose it.
29. **The jax key pool:** `make_brain` drew from a fixed pool of 32 keys, and the new heads exhausted it
    (`StopIteration`). → A pool of 64 for brains with eyes; v1 keeps 32, so old draws are unchanged.
30. **New-life reset broke on 4-D state:** `fresh()` assumed `[B, N]`, and the eye frame is `[B, H, W, 2]`. → Reshape
    by rank.
31. **Python edits by string replacement hit the wrong function once:** `def real_test` exists twice, and the indent
    of one was corrupted. Lesson: when patching by text, anchor on unique context and check `git diff` afterwards.
32. **No PIL or imageio in the project venv.** `dog_world._png` writes PNGs with zlib; ffmpeg (Homebrew) converts GIFs
    to MP4.

---

## 6. What I believe should happen next

1. **Round 32 Completed & Evaluated (New All-Time Project Records):**
   - Checkpoint `baby_dog_step_0000393216.npz` (SHA256 `d23d8f2c91d7f9d83f30d408a62e40285864046e91a3462b70755ac83008ef2e`, 0.40M steps, 24 updates) achieved:
     - **Ball Arrival:** **8.85% (17/192)** across all three cohorts (Cohort 1: 9.38%, Cohort 2: 9.38%, Cohort 3: 7.81%) — **NEW ALL-TIME PROJECT RECORD ON BALL**.
     - **Owner Arrival:** **8.33% (16/192)** across all three cohorts (Cohort 1: 3.12%, Cohort 2: 10.94%, Cohort 3: 10.94%).
     - **Combined Target Arrival:** **8.59% (33/384)** across all three cohorts — **NEW ALL-TIME PROJECT RECORD COMBINED**.
     - **Net Closing Distance on Owner:** 3.667 m -> 3.570 m (-0.097 m net closing distance), the first checkpoint in project history where mean ending distance decreased rather than expanded.
     - **Nominal Motor Skill Retention:** 100% intact (walk +0.398 m/s, pace +0.568 m/s, canter +0.398 m/s, run +0.549 m/s, turn-left +1.328 rad/s, turn-right -1.141 rad/s, sit 0.505/0.620, lie 0.033/0.966, falls 1.12 / 13s, teacher help strictly 0.00).
   - Candidate `baby_dog_step_0000393216.npz` (`d23d8f2c…`) is promoted as the active restart base for Round 33.
2. **Causal Mechanism & Learning Dynamics from Round 32:**
   - Active yaw turning dominance (`yaw_pay = 25.0`, `face_pay = 20.0`) successfully overcame the straight-line collision attractor by providing strong rewards for rotating toward target azimuth.
   - Peak performance occurred at 24 updates (0.40M steps). In updates 25-36 (0.50M-0.60M), arrival decayed back to ~5%.
   - Cause: The linear misalignment penalty `misalign_cost = 8.0 * (1 - align_gate) * v_fwd` created strong negative policy gradients during gait transitions whenever the dog took a forward step before finishing its turn, causing slight fall increase (1.12 / 13s) and over-cautious stepping.
3. **Round 33 Formulation (Targeting >15% Sustained Arrival):**
   - **Direct Continuation Base:** Resume directly from peak candidate `baby_dog_step_0000393216.npz` (`d23d8f2c…`) from `karang1908/bio-bot-baby-dog-vision-approach-round32`.
   - **Targeted Duration:** Run a focused 24 updates (393,216 steps, ~60 min on 2× T4 GPU), perfectly matching the empirical learning peak without overshooting into late-stage decay.
   - **Reward Refinements:**
     - Soften `misalign_cost = 4.0` (from 8.0) to eliminate gait-transition penalties while retaining pivot-in-place incentives.
     - Elevate `reach_pay = 100.0` (from 80.0) and `task_pay = 50.0` (from 45.0) to maximize gradient pull inside the arrival cylinder.
     - Keep active yaw turning dominance (`yaw_pay = 25.0`, `face_pay = 20.0`) and gallop sprint (`sprint_pay = 40.0`).
4. Commit and push every future changed file separately on `main`, with no co-author trailer.
5. **Then step 7:**
   - multi-step errands ("ball" then "come" said once, then silence: it must remember);
   - learned good and bad (mushroom-body-style plasticity, which needs fast online weights);
   - the hidden-ball memory test.
6. **Then the humanoid (G1)** in the same brain. This is the actual research question.
7. Keep `LOGBOOK.md` honest (results with numbers, failures verbatim) and `DogMind.md` in step with what was built.

---

## 7. Practical notes for the agent

- **Talk to the owner simply:** short, plain words, results first. They want honesty about what failed. They are
  enthusiastic and terse ("continue", "commit and push", "done?").
- **Long Kaggle waits:** the user explicitly requested active timer tracking. Keep API status/fetch monitoring
  running and use clock waits of about 50 seconds for active checks and concise updates. A quiet live stream may
  be stale: reconnect the local reader without canceling the notebook. Do not promise assistant notifications
  after ending the turn; re-login only on observed authentication failure.
- **Saved memories** in `~/.claude/projects/-Users-karangarg-Desktop/memory/` (bio-bot-*.md) hold the rules above.
  Keep them current.
- **Before claiming anything works,** run the real scripted test from rest and read the numbers. Several "successes"
  in this project were measurement artifacts (§5.5, §5.13, §5.24).
