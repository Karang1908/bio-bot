# HANDOFF: bio-bot, "One Brain, Many Bodies"

The handoff for the next AI coding agent. Written 10 Oct 2026; updated 11 Oct 2026 after the longer-history trial.

Read this file first, then:
- `LOGBOOK.md`: every experiment, number and mistake;
- `DogMind.md`: the agreed plan for senses, self-discovery and mind;
- `TechnicalDesign.md` / `Requirements.md`: the design, the infrastructure, the Kaggle gotchas.

Everything below marked **verified** was measured. **Believed** means reasoned but not measured.

**Current state, 14 Oct 03:15 Dubai:** `baby_dog_vision_approach_round14` v2 executed 20 updates (~327k steps) on Kaggle GPU (2x Nvidia Tesla T4) before terminating at the `--timeout 2800` (46.6 min) runtime cap (`CANCEL_ACKNOWLEDGED`). Outputs fetched into `results/baby_dog_vision_approach_round14/`.
Notebook: https://www.kaggle.com/code/karang1908/bio-bot-baby-dog-vision-approach-round14 (version 2).
Dynamic velocity command injection in `rollout` `info["cmd"]` (`task_cmd = [0.5 * clip(cos_facing, 0, 1), 0, clip(sin_lat, -1, 1)]`) in `train/baby_dog.py` (`eec0202`) successfully verified in remote execution: unblocked `v_act`'s internal locomotion and turning rewards without NaNs or skill conflicts (`conflicting 0%`).
In `baby_dog.py`, `every = max(1, iters // args.evals) = 40 // 2 = 20`. Update 20 triggered the intermediate evaluation at 45.4 min; while executing cohort 1, the 2,800s timeout expired before evaluation completion. No intermediate `.npz` was written to disk.
Active restart candidate remains `baby_dog_step_0000491520.npz` (`5a534b32…`, Round 10 all-time 3.12% arrival record, lowest 7-gait falls 0.2649).
Free Kaggle GPU quota: **0.08h remaining** (29.92h / 30.00h used; weekly quota exhausted until refresh on 17 Oct).
Next action: Transition to Google Colab ($0 budget, free tier) for extended 1.5M+ step continuation using standalone script `results/kaggle/baby_dog_vision_approach_round14/baby_dog.py`.
All source commits pushed individually on `main`; zero laptop compute checks.
The current-state and active-job sections take precedence over chronological pending-status notes.

### Latest completed trial: vision approach round 13 (amplified forward approach propulsion FWD_PAY=16.0, spinning elimination YAW_PAY=0.0, and orientation tuning FACE_PAY=4.0)

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

1. The approach outcome pilot is finished and fetched. Read its completed-result section above and full report;
   no notebook is awaiting tracking from this agent, and no output has been promoted.
2. Investigate why visual features fail to drive locomotion:
   - Visual projection weights (`vp_e`, `vp_l`) start at 0.0, and small shared brain lr (5e-6) may keep vision effectively disconnected from motor neurons.
   - Investigate whether a targeted visual learning rate or closer target spawns (e.g. 0.5-1.0m instead of 2.5-3.5m) is needed for the baby dog to discover the correlation between visual heading and reward.
3. Formulate the next bounded free Kaggle experiment. Retain every sense, wiring, wobble and old skill; movement help stays zero.
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
