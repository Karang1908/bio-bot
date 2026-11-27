# CHECK-ALPHA

A frozen checkpoint of the Embodiogenesis dog brain, taken on 27 Nov 2026 at the owner's request, before the same
brain is taught the humanoid. It holds the exact code, the learned weights, the connectome they run on, and the
measured evidence of what this brain can do.

Repository state: git commit `4718f4f` (`main`).

## What is in this folder

| path | what it is |
|---|---|
| `code/train/babble.py` | the only training/evaluation script (body, brain, babbling, skills, rhythm, tests) |
| `code/scripts/kaggle_job.py` | pushes `babble.py` to Kaggle with its options, checks status, fetches outputs |
| `code/pyproject.toml` | the project's Python dependencies |
| `connectome/core_v2.npz` | the fly connectome core the brain is built from (7,019 neurons, 502,438 synapses) |
| `weights/stilltrial2/brain_dog{1,2}.npz` | **the CHECK-ALPHA brains** (established best, both dogs) |
| `weights/walk1/brain_dog{1,2}.npz` | the same brains after one walking practice: accelerometer nerves and walking rhythm synapses added, walking not yet learned |
| `evidence/*.babble_log.txt` | the logs of the runs that produced and measured these brains |

The `.npz` weight files are git-ignored (the repo keeps checkpoints out of git), so they exist in this folder on disk.
Their hashes are listed below. The same files are also hosted as Kaggle notebook outputs:
`karang1908/bio-bot-babble-stilltrial2` and `karang1908/bio-bot-babble-walk1`.

## SHA-256

```
f912cc64724d81a32bdd45c0a24ea693b9d889a03e69731c15e65c16caaa11fd  weights/stilltrial2/brain_dog1.npz
50410a90e4a68126dbfa1d9db2cf1174a0eead7166a8565619895f198e43e55e  weights/stilltrial2/brain_dog2.npz
12071f847d1057175c64381c7f5c820390a440c09babed9066dfab3b91bbb321  weights/walk1/brain_dog1.npz
f249aad49b082ced133b960c58fdc2c60ba4a93b999d5bbb78f4a2878856a806  weights/walk1/brain_dog2.npz
af939468e3fb156a74e5b5ca44b67d81d0cea2ae2c94e4543875c934ff5e2d8f  connectome/core_v2.npz
e12b6a09358c12d57b18b81a319ac80cf616b3faec2ac01679e322692213268e  code/train/babble.py
a3b359c7fe10f3aca7568a919c02699e7043e2f4a57d6352d55028cf2320478a  code/scripts/kaggle_job.py
```

## The brain

- **Core:** the fly connectome `core_v2` (MaleCNS), 7,019 neurons, run as a rate model with a 40 ms membrane. 1,219
  sensory neurons and 370 motor neurons.
- **Body wiring (genome `20261117`):** each motor neuron drives one of the Go2's 12 motors, pushing one way. Each
  nerve reaches 16 sensory neurons (half excited by a rise, half by a fall). The brain is told nothing about which
  line is which.
- **Nerves:** 58 in stilltrial2 (12 joints × angle, speed, effort, limit push-back; 4 paw touch; 3 gravity; 3
  turning). walk1 has 61 (plus 3 accelerometer nerves, appended after the 58 with the earlier wiring unchanged).
- **Learned parts (in each `.npz`):**
  - `map`, the Hebbian body map (motor → sensory), learned by babbling.
  - `Q`, the skill synapses (sensory → motor), learned by reward, at about 0.40 × the map's size.
  - `Qc`, the rhythm synapses: grown pattern generator → motors, one set of 4 per direction, learned by reward-gated
    synaptic noise. stilltrial2: turn left / right sets at 1.33–1.50 × the map. walk1: plus walk forward / back sets
    at 0.66–0.71 × the map, not yet learned.
  - The senses' adapted means and spreads, the at-rest references, and the motor pools' resting drive.
- **Experience:** 150,000 brain steps felt (stilltrial2); 157,500 (walk1).
- **Grown parts:** the pattern generator is 4 extra neurons (mutual inhibition + fatigue, 0.62 s cycle), created when a
  run uses `--cpg`. It is not stored in the `.npz`; its synapses to the motors (`Qc`) are.

## What it can do (measured, LOGBOOK §6.50–6.55)

On 30 fresh lives per condition (varied set-down, ±0.1 motor wobble, seeds 9100+):

| | dog 1 | dog 2 |
|---|---|---|
| stands, nothing asked | 0.18–0.21 m, up 87–97% | 0.19 m |
| spins when nothing is asked | +2% of a 0.5 rad/s turn (essentially still) | +9% |
| turns on request, left / right (caused by the request) | +25% / +35% | +17% / +24% |
| …same with its rhythm synapses switched off | about 0% | about 0% |
| tilts on request (roll L / roll R / nose down / nose up) | +59 / +131 / +73 / +81% | +42 / +49 / +58 / +60% |

- **Learned by itself, not hand-set:** the body map, the reflexes, tilt, turning, standing still without spinning.
  No gait, pose, phase or speed rule is written by hand.
- **Partial:** front paws lift on request (single tests); single-joint control is weak (15–30% of the way).
- **Not learned:**
  - Weight onto an asked front foot.
  - Walking (first attempt flat, §6.57).
  - Back up, trot, sit, lie down.
- **Excluded by the owner:** getting up from its side; the robot cannot physically do it.

## How to continue this brain

Upload this folder's `weights/stilltrial2` (or reuse the hosted Kaggle output) and run from the repo root:

```
.venv/bin/python scripts/kaggle_job.py push babble --job NAME --machine CPU --dataset bio-bot-cores \
  --kernel-source karang1908/bio-bot-babble-stilltrial2 --set resume=1 --set dogs=2 --set ground=1 --set tone=1 \
  --set leak=1 --set leak_s=0.5 --set set_speed=9 --set reflex=1 --set "reflex_gains=0,0.125" --set skills=180 \
  --set skill_rate=0.000012 --set skill_noise=0.05 --set skill_lift=1 --set skill_per_task=1 --set skill_level=1 \
  --set skill_curriculum=1 --set skill_max=0.5 --set cpg=1 --set cpg_wp=1 --set cpg_wp_mean=1 --set skill_vary=1 \
  --set no_q_in_turn=1 --set still_trial=1 --set "skill_mix=tilt,weight,lift,joint,still,still,turn,turn"
```

- **Measuring it:** add `--set turn_eval=1 --set eval_seed=N` (and no skills).
- **Resuming checks:** a resumed brain must match its wiring exactly, or as a prefix when new nerves are appended
  (`--accel`). Anything else stops with an error rather than silently re-initialising.
- **The Go2 model:** fetched from MuJoCo Menagerie at commit `367e3d9884401dcf6f9c27fa69f118992539039f`.
- **Compute:** Kaggle CPU only (free).

## Lessons recorded with this checkpoint

- Measure turning relative to the dog's own drift with nothing asked (§6.49).
- Judge a lesson by a trial-level felt signal, not moment-by-moment closeness (r 0.99 vs 0.2–0.6, §6.46).
- Recording the at-rest reference with the brain running makes the dogs fall (§6.52).
- A trial-level update must be scaled 1/√T, or the synapses run away (§6.53).
