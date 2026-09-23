# One Brain, Many Bodies: Technical Design

_Technical design · 23 September 2026 · Status: planning, no code yet_

Companion docs: [README](README.md) (summary) · [Overview](Overview.md) (non-technical) · [Requirements](Requirements.md) (infrastructure) · [LegacyIdea](LegacyIdea.md) (original concept)

---

## 1. Research questions

The project has **two research questions**. The fly connectome is the brain both are built on, and whether it helps is a supporting question tested across every experiment.

### RQ1: Self-learning (zero configuration)

> Given only unnamed, randomly ordered sensor and actuator channels, can one brain discover what it is connected to, learn to use it toward a goal, and keep learning after deployment with no teachers and no manual configuration?

| # | Hypothesis | Tested in |
|---|---|---|
| H1a | After a varied "childhood", the brain reaches competence on **unseen bodies** within one lifetime, with no gradient updates (in-context adaptation) | E1, E2 |
| H1b | Its **causal self-map** (which actuator affects which sensor) converges toward the simulator's true map as it explores | E1 |
| H1c | With teachers stripped, local plasticity driven by its **internal reward** still improves performance on new or damaged bodies | E3, E4 |
| H1d | The same machinery configures non-robot bodies (smart-home devices, computers) inside a safety fence | E13 |

### RQ2: Coordination (one brain, many bodies)

> When several separately learned, different bodies are connected to one brain at once, does coordination emerge, and does one shared brain beat separate brains that communicate?

| # | Hypothesis | Tested in |
|---|---|---|
| H2a | **Naive merge** (all channels into one core) performs *worse* than the best single body | E6 |
| H2b | **Shared brain + one nerve cord per body + shared world memory**, zero-shot at merge, is *faster* than the best single body (shared perception) | E6 |
| H2c | A short joint phase with **team reward only** produces division of labour | E7 |
| H2d | One brain beats three independent brains and three brains with a learned communication channel, at equal compute | E8 |
| H2e | The brain re-plans when a body is unplugged and absorbs a new body plugged in mid-task | E9 |
| H2f | *(exploratory)* The brain recognises its own body seen through another body's camera | E10 |

### Supporting question: does connectome wiring help?

Every experiment is repeated with the core swapped: **MaleCNS** vs **degree-preserving rewired MaleCNS** vs **Erdős–Rényi random graph** (same size, sign ratio) vs **GRU** (parameter-matched) vs **small Transformer**. Everything else is identical. Either outcome is reportable.

---

## 2. System overview

```
┌──────────────────────── INTENT LAYER ────────────────────────┐
│ speech → Whisper → local LLM → goal spec {verb, target}      │
│ → goal vector (vision-language text embedding + verb embed)  │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────── SHARED BRAIN (MaleCNS brain) ────────────┐
│ shared world memory: object → location belief (one frame)    │
│ goal + associative memory (mushroom body, plastic)           │
│ navigation / heading / goal direction (central complex)      │
│ dopamine neurons: internal reward;  prediction head: surprise│
│ body-slot attention: which body gets deliberation now        │
└───────┬───────────────────────┬───────────────────────┬──────┘
   DN_1 ▼ ▲ AN_1           DN_2 ▼ ▲ AN_2           DN_3 ▼ ▲ AN_3
┌───────┴──────┐        ┌───────┴──────┐        ┌───────┴──────┐
│ nerve cord 1 │        │ nerve cord 2 │        │ nerve cord 3 │  one VNC copy per body
│ (VNC copy)   │        │ (VNC copy)   │        │ (VNC copy)   │  runs at body rate
└───┬──────▲───┘        └───┬──────▲───┘        └───┬──────▲───┘
    ▼      │                ▼      │                ▼      │
 motor   sensor          motor   sensor          motor   sensor   ← per-episode shuffled,
 readout tokens          readout tokens          readout tokens     unnamed channels
    ▼      │                ▼      │                ▼      │
  BODY 1 (drone)          BODY 2 (humanoid)       BODY 3 (fly-shaped walker)

TRAINING ONLY (stripped before deployment): meaning code per word · privileged critic ·
simulator ground truth · demonstrations from free teachers · LLM
```

DN = descending neurons (brain → cord); AN = ascending neurons (cord → brain). With one body, this reduces to the ordinary brain + one nerve cord.

---

## 3. The brain: connectome core

### 3.1 Source data

Janelia FlyEM **MaleCNS v1.0**, CC-BY 4.0: about **166,700 annotated neurons** (brain + ventral nerve cord) and about **25.5M neuron-to-neuron connections**.

| File | Size | Use |
|---|---|---|
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1.1 GB | connection graph + synapse counts |
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 13 MB | cell type, class, side, region |
| `body-neurotransmitters-male-cns-v1.0.feather` | 42 MB | neurotransmitter prediction → sign |

Synapse-level tables (6.8–12.7 GB) are not needed.

### 3.2 Signs and weights

- **Signs from predicted neurotransmitter** (the convention used by Shiu et al. 2024 and Lappalainen et al. 2024):
  - acetylcholine → excitatory (+)
  - GABA and glutamate → inhibitory (−)
- **Neuromodulators** (dopamine, serotonin, octopamine) go to the modulation and plasticity pathway, not fast excitation or inhibition. This is a design choice to validate.
- **Low-confidence predictions:** sign 0, following Fly.exe's policy. Ablation: excluding those neurons instead.
- **Weight:**

  `w_ij = s_j · softplus(γ[T(i),T(j)]) · n_ij / Z_i`

  - `s_j`: the sign of neuron j
  - `n_ij`: synapse count from j to i
  - `T(·)`: cell type
  - `γ`: a learned gain per connected cell-type pair
  - `Z_i`: input normalisation (e.g. √ of total input synapses)

### 3.3 Dynamics (rate model, not spiking)

```
τ[T(i)] · dv_i/dt = −v_i + Σ_j w_ij · r_j + b[T(i)] + u_i(t)      r = relu(v)
Euler step Δt; K ticks (K≈2–4) per control step
```

- **Learned, per cell type:** time constant `τ`, bias `b`, and pair gain `γ`. The parameter count is independent of neuron count, and the wiring stays fixed.
- **Optional ablation:** a per-neuron descriptor `η_i ∈ R^D` (the FlyGM design).
- **Stability:** normalise to a target spectral radius at initialisation, clip `v`, and require 10k-step stability under random input for every graph variant.

### 3.4 Graph variants (same code, different sizes)

| Variant | Units | Built by | Role |
|---|---|---|---|
| Full | ≈166.7k | all annotated neurons | running / demos |
| Pruned | ≈70k | drop optic lobes unless the body has a camera (optic lobes ≈95k of 166.7k by one count) | fine-tuning |
| Mesoscale | ≈5–10k | keep **individual** neurons in the central complex (column structure carries heading and goal), MBONs, DANs, DNs and ANs; subsample Kenyon cells; collapse large repetitive populations to cell type | main experiments |
| Coarse | ≈2k | cell-type-level collapse | fast iteration, many seeds |

Null models are built **at each scale**:
- **Directed degree-preserving rewiring:** each neuron keeps its in/out degree and sign, and the synapse-count distribution is preserved.
- **Erdős–Rényi:** matched N, E and excitatory/inhibitory ratio.

Parameters are shared per cell type, so parameters trained at mesoscale can be loaded into the full graph. **Whether behaviour survives that transfer is itself an experiment, not an assumption.**

---

## 4. Self-learning machinery (RQ1)

### 4.1 Enforcing "unnamed": per-episode randomisation

Every episode, apply to all channels:
- a random permutation
- a random sign flip
- a log-uniform rescale and offset
- random actuator gain and sign

Without this, a network with fixed input order memorises "channel 3 = left knee" and the discovery claim is false. Channel identity must be *inferred* from how a channel behaves.

### 4.2 Sensor tokens → afferent neurons

- **Per-channel token:** `e_k,t = φ(x_k,t−H..t , ā_t−H..t−1)`
  - `φ` is a small encoder shared by all channels. It sees the channel's own recent history plus an attention-pooled summary of recent actions, so it can capture *how this channel responds to what I do*.
  - Design follows Tang & Ha 2021 (*Sensory Neuron as a Transformer*).
- **Afferent injection:** learned queries, one per afferent-neuron group, cross-attend over the tokens and inject currents into those neurons. This is permutation-invariant and works for any channel count.

### 4.3 Motor readout

- **Efferent states:** VNC motor-neuron states act as keys and values. MaleCNS includes the nerve cord, so readout uses the full DN → VNC → motor neuron path. Ablation: read out from DNs instead, as FlyGM and Eon did.
- **Actuator queries:**
  - each actuator j gets a query built from a **per-episode random ID embedding** plus its own recent command history
  - output: `a_j = tanh(MLP(attn(q_j, efferent)))`
  - the brain must discover what each ID does by trying it
- **Interface budget:** at most ~50k parameters, **identical for every core variant**, so the core, not the interface, is what gets compared.

### 4.4 Prediction head and surprise

- **Head:** predicts next-step sensor tokens from brain state + action.
- **Loss:** MSE on normalised channels.
- **Uses of the prediction error `ε_t`:**
  1. a dense self-supervised learning signal
  2. modulatory input to DANs
  3. a curiosity bonus during childhood

### 4.5 Plasticity (mushroom body, dopamine-gated)

Plasticity is restricted to **Kenyon cell → MBON synapses**, the fly's actual associative-learning site. It uses a three-factor rule (the ABCD form of Najarro & Risi 2020):

```
Δw_kj = η_c · d_c(t) · (A_c·r_k·r_j + B_c·r_k + C_c·r_j + D_c),   w clipped to [w_min, w_max]
```

- `c`: MB compartment
- `d_c(t)`: DAN activity in compartment c
- `(η, A, B, C, D)`: learned per compartment, a few hundred parameters in total, which suits evolution strategies
- **Fast weights:** reset per lifetime for the adaptation experiments; persist across bodies for the continual-learning experiments

### 4.6 Internal reward: distilling the teacher's *judgement*

- **During childhood:** DANs are trained to predict the teacher's reward and success probability. The loss is a **proper scoring rule** (log score or Brier) so the confidence is calibrated. The idea is borrowed from Laya's training recipe; Laya itself is not used.
- **After stripping:** DAN output is the only reward, and it drives plasticity.
- **Why this matters:** if only *actions* are distilled, the stripped brain can act but can never improve. Distilling the *judgement* is what makes post-deployment self-learning possible.
- **Risk:** misgeneralisation on unusual bodies. Agreement with the teacher is measured (E4c).

### 4.7 Causal self-map (the measurable "self")

- **Estimate:** `M̂[k,j] = E_states[∂x̂_k / ∂a_j]`, obtained by probing the prediction head.
- **Ground truth:** `M[k,j]`, from finite-difference actuator perturbations in the simulator, averaged over states.
- **Metrics:** Spearman correlation of |M̂| vs |M|, and AUROC for detecting non-zero entries, tracked through each lifetime.
- **Multi-body case:** M̂ should become block-diagonal, one block per body. Off-block entries appear when one body sees another, which is the basis for E10.

### 4.8 The fence (real systems)

For non-simulated bodies:
- **Allowed actions:** only whitelisted actions exist.
- **Hard limits:** rate limits, value ranges and forbidden state predicates (e.g. heater on for more than X min, door unlock) are enforced *outside* the brain by a shield that vetoes actions.
- **Exploration order:** passive observation first, then reversible probes, then goal-directed use.
- **Summary:** configure the fence, not the function.

---

## 5. Coordination machinery (RQ2)

### 5.1 One shared brain, one nerve cord per body

- **Shared part:** the central brain (all non-VNC neurons).
- **Per-body part:** each connected body b gets its own **VNC copy**, plus its own DN and AN populations.
  - `DN_b` receives: the shared brain's output + `AN_b` feedback + a **body-slot gate** `g_b(t)` from a small attention module.
  - Without the gate, all DN copies receive identical input and send identical commands. **"Which body am I commanding" is the key open design detail**, to be resolved in Phase 1.
- **Solo phases** train the shared brain and that body's cord together. At merge, the cords come with the bodies.

### 5.2 Shared world memory

- **Pragmatic first version:** an explicit key–value memory of object → location belief with confidence.
  - written by any body's sensor stream when an object is detected
  - read by every body's goal-seeking
- **Important:** solo training must already condition goal-seeking on *memory*, not only on the body's own camera. That is what makes shared perception work zero-shot at merge (H2b).
- **Later refinement:** map this memory onto central complex and mushroom body circuitry. Until then the memory is a declared non-connectome component, and its ablation is part of E6.

### 5.3 Common frame

- **MVP:** each body exposes an unlabeled world-position channel (GPS-like) in shared coordinates. The brain discovers these channels are comparable.
- **Later:** remove it and learn frame alignment from shared visual landmarks.

### 5.4 Multi-rate control

- Nerve cords run at each body's control rate.
- The shared brain runs slower (≈5–10 Hz), asynchronously.
- Fast reflexes stay local, as in animals.

### 5.5 Merge, joint phase, hot-plug

1. **Zero-shot merge:** connect solo-trained bodies and evaluate. No training.
2. **Joint phase:** team reward only (any body succeeds, or the fetch task completes). Per-body value heads with a counterfactual baseline handle credit assignment.
3. **Hot-plug and unplug:** body slots are allocated dynamically; bodies are added or removed mid-episode.

### 5.6 Task suite (coordination must be *necessary*)

"Find the apple" alone can be solved by parallel search, so racing can't be told apart from coordination. Bodies therefore get complementary abilities:

| Body | Speed | View | Can grasp | Fits in gaps |
|---|---|---|---|---|
| Flyer (drone) | fast | wide | no | no |
| Grabber (humanoid) | slow | narrow | yes | no |
| Crawler (fly-shaped) | slow | short | no | yes |

| Task | Requires |
|---|---|
| T1 Find | any body |
| T2 Find & fetch to table | grabber, plus whoever finds it |
| T3 Hidden in gap | crawler to see, grabber to fetch |
| T4 Guided traverse | flyer's view to route the grabber around obstacles |

### 5.7 Coordination ladder (what we expect, when)

| Level | Behaviour | Expected |
|---|---|---|
| L1 | parallel independent search | free at merge (with nerve cords) |
| L2 | shared perception redirects other bodies | free at merge (with shared memory) |
| L3 | division of labour | after joint phase |
| L4 | physical cooperation | after joint phase, hardest |
| L5 | self-recognition across bodies | open question |

---

## 6. Intent layer (language)

- **Runtime:**
  - Whisper (local) turns speech into text.
  - A local LLM turns the text into a goal spec, e.g. `{verb, target, constraints}`.
  - The target is embedded with a vision-language text encoder (SigLIP-class); the verb uses a learned embedding.
  - The resulting goal vector goes to goal neurons and memory.
- **Training time:** the **meaning of each verb is code**, written once and stored in the repo. Each meaning is a predicate over simulator world state, never over the brain's channels:
  - `run`: ground speed ≥ 1.5× this body's walking speed, upright
  - `find X`: X in view and within reach
  - `go to X`: distance to X near zero
  - `stop`: speed ≈ 0 for 1 s

  This code produces the task reward. Shared verbs (`find`, `go to`, `bring`) apply to every body; body-specific verbs (`run`, `fly`) are simply unachievable for some bodies, and the brain learns that too.
- **After stripping:** a learned vocabulary table maps word → goal vector. The vocabulary becomes closed, and an unknown word optionally escalates to the LLM once.

---

## 7. Training pipeline

### 7.1 Childhood (simulation, teachers on)

- **Bodies:** a procedural distribution (§8). The number of distinct training bodies matters more than data per body (Embodiment Scaling Laws, 2025).
- **Objectives:**
  - task reward from meaning code
  - prediction loss
  - DAN reward-prediction loss (proper scoring rule)
  - curiosity bonus
  - optional behaviour cloning from free teachers
- **Free teachers:**
  - classical controllers: PID/LQR hover for the drone, rhythmic gait generators for legged bodies
  - RL experts trained per body where needed
  - the imitation-then-PPO recipe is the one FlyGM used to make a connectome controller trainable
- **Optimisers:**
  - *Slow parameters* (interface, per-type τ, b, γ): PPO (JAX / Brax, as used by MuJoCo Playground) with an **asymmetric critic** that sees privileged simulator state, and truncated BPTT.
  - *Plasticity coefficients:* evolution strategies (evosax: OpenAI-ES / SNES) in an outer loop. Fitness = improvement within a lifetime, averaged over sampled bodies, which directly optimises *learning to learn*.
  - The split between PPO and ES is revisited at the Phase 2 gate.

### 7.2 Joint phase (RQ2)

Merged bodies, team reward, tasks T1–T4.

### 7.3 Strip

| Removed | Kept (ships) |
|---|---|
| meaning code / LLM | connectome core + learned gains |
| privileged critic | nerve cords + channel interface |
| simulator ground truth | plasticity rule (MB) |
| demonstrations / experts | DAN internal reward + prediction head |
| vision-language model (optional) | vocabulary table |

### 7.4 Lifetime (on device)

Learning uses local plasticity only, driven by internal reward and prediction error. There is no backpropagation, and the cost per step is roughly the cost of inference.

---

## 8. Environments and bodies

| Body | Implementation | Phase |
|---|---|---|
| 2D world with flyer / grabber / crawler | custom JAX; analytic "vision" (bearing, distance, unlabeled object features); gaps and occluders | 1 |
| Procedural 2D/3D bodies | random limb count and length, actuator count, sensor sets (incl. novel senses), dynamics randomisation | 1–4 |
| Drone | custom JAX rigid-body quadrotor, or MuJoCo Crazyflie model in MJX | 4 |
| Quadruped | Unitree Go1 (MuJoCo Playground) | 4 |
| Fly-shaped walker | hexapod built from the flybody leg morphology, scaled to robot size | 4 |
| Humanoid | Unitree G1 / H1 or Berkeley Humanoid (MuJoCo Playground) | 4 (last) |
| Real fly | flybody (MuJoCo), solo demo only | stretch |
| Homelab | Mac / Linux / Pi metrics as sensors, whitelisted jobs as actions; simulated queueing twin for training | 6 |
| Smart home | Home Assistant entities → channels, services → actions, behind the fence | 6 |

A real fruit fly is about 2.5 mm long and walks about 3 cm/s. It cannot share a room-scale world with robots, and in a shared scene it would force the whole simulation onto its fine physics timestep. Hence the scaled fly-shaped walker.

---

## 9. Experiments

| ID | RQ | Experiment | Metric | Pre-registered prediction |
|---|---|---|---|---|
| E1 | 1 | Unseen bodies, shuffled channels | return vs steps within a lifetime; causal self-map Spearman / AUROC over time | return and self-map accuracy rise within the lifetime |
| E2 | 1 | Novel sense (informative unfamiliar channel) | performance gain; attention mass on the channel | used when informative, ignored when noise |
| E3 | 1 | Damage: weaken an actuator, drop a sensor, lock a joint | drop, recovery time, final level; plasticity on vs off | plasticity recovers faster |
| E4 | 1 | Strip tests: (a) trained bodies, (b) new body using only internal reward, (c) internal vs teacher reward on out-of-distribution bodies | (a) performance delta, (b) slope, (c) agreement | (a) unchanged, (b) positive but slower than with teacher, (c) degrades on unusual bodies |
| E5 | 1 | Return to an old body after others | relearning speedup, forgetting | faster relearning |
| E6 | 2 | Zero-shot merge: naive vs nerve cords vs nerve cords + shared memory | time to goal vs best solo body | naive < solo < nerve cords + memory |
| E7 | 2 | Joint phase | role-specialisation index, redundant coverage | flyer scouts, grabber fetches |
| E8 | 2 | One brain vs 3 independent vs 3 with learned comm | time to goal, sample efficiency, coverage overlap | one brain best |
| E9 | 2 | Unplug and hot-plug mid-task | re-plan time; integration time of an unseen body | recovers; integrates without retraining |
| E10 | 2 | Self-recognition | off-block causal influence localised at own body in another body's camera; representation probes | open |
| E11 | 1+2 | Language | command success with LLM vs stripped vocabulary; unseen words via embeddings; word → goal consistency across bodies | small drop after strip; consistent across bodies |
| E12 | sup. | Core swap across E1–E9 | all of the above per core | reported either way |
| E13 | 1 | Real systems (Home Assistant, homelab) inside the fence | device-function discovery accuracy (which switch changes which sensor); job-placement quality; fence violations = 0 | discovers lamp ↔ light-sensor links by probing |

## 10. Baselines and statistics

- **Cores:** MaleCNS, rewired, Erdős–Rényi, GRU (parameter-matched), Transformer. All use the same interface, training budget and hyperparameter-search budget.
- **Coordination:** independent brains; brains with a learned message channel (standard multi-agent RL with communication).
- **Held-out data:** bodies never seen in childhood are used for every evaluation.
- **Seeds:** ≥5 per condition, ≥10 for headline plots. Report the interquartile mean with 95% stratified bootstrap CIs (rliable).
- **Pre-registration:** the predictions in §9 are written here before any runs.

---

## 11. Roadmap and gates

| Phase | Build | Gate (don't move on until) |
|---|---|---|
| P0 | Signed graph, four variants, null models, dynamics | every variant stable 10k steps; step time measured on Mac and Kaggle T4 |
| P1 | 2D world, three bodies, channel interface, **GRU core**; solo discovery + merge | GRU learns held-out bodies with in-lifetime improvement; E6 run once at toy scale (validates both RQs before the connectome is involved) |
| P2 | Swap in connectome cores + all baselines | E1 and E6 headline plots across cores |
| P3 | Plasticity, DAN reward, prediction head, self-map, strip, damage | E3 / E4: plasticity beats frozen after strip |
| P4 | 3D: drone, Go1, fly-shaped walker; 3D merge; humanoid last | E1 / E6–E9 in 3D |
| P5 | Intent layer: speech, LLM parse, grounding, vocabulary | E11 |
| P6 | Real systems: Home Assistant + homelab behind the fence | E13 |
| Stretch | humanoid in the merged team; optic-lobe vision for cameras; E10 self-recognition study; real fly solo demo | — |

## 12. Applications (technical mapping)

| Application | RQ | Channels in | Actions out | Fence examples |
|---|---|---|---|---|
| New or modified robot, no calibration | 1 | joint encoders, IMU, contact, camera | joint torques / positions | torque, speed and workspace limits |
| Robot wear / damage compensation | 1 | same | same | same |
| Smart home / IoT | 1 | Home Assistant entity states (lux, temperature, motion, power) | entity services (switch, dim, set) | no locks, no heating over X min, rate limits |
| Homelab / servers | 1 | CPU, RAM, temperature, disk, latency, container state | whitelisted jobs, start/stop approved containers, GPIO | no shell, no deletes, approved scripts only |
| Buildings / industrial sensor networks | 1 | new sensors join as channels | HVAC and lighting set-points | comfort and safety bands |
| Undocumented equipment | 1 | unknown signals | unknown controls | reversible probes only, amplitude caps |
| Search and rescue | 2 | per-robot sensors + shared frame | per-robot actions | geofence, collision limits |
| Home assistance ("where are my keys") | 2 | house cameras (sensor-only body) + home robot | robot actions | no-go zones |
| Warehouses / agriculture / inspection | 2 | heterogeneous fleets | per-robot actions | zones, speeds |

Sensor-only bodies (cameras) and actuator-only bodies (smart plugs) are ordinary cases of the same interface.

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| RL on a large recurrent core is slow or unstable | mesoscale graph; imitation warm-start; spectral normalisation; truncated BPTT |
| The interface does all the work and the connectome is decoration | small, identical interface for every core; core-swap ablations |
| The connectome shows no advantage | fair baselines at equal budget; negative result reported (YYK2007/flybrain saw shuffled wiring pass when only a decoder was trained) |
| Credit assignment for unlabeled actuators | motor-babbling phase; curriculum from few to many actuators; fixed actuator count first |
| All nerve-cord copies issue identical commands | body-slot gate + AN feedback (§5.1); resolved at the P1 gate |
| Internal reward misgeneralises after strip | E4c monitoring; optional brief teacher return |
| Sim-to-real and real-world data are slow | childhood in simulation; real systems only in-context and behind the fence |
| Scale mismatch (fly vs robots) | scaled fly-shaped walker in shared scenes |
| Kaggle 12 h session limit | checkpoint every 15–30 min; chained sessions |

## 14. Considered and rejected

| Option | Why rejected |
|---|---|
| Spiking (LIF) neurons for learning | Full MaleCNS LIF runs at ≈0.015× real time on an RTX 3060 (Fly.exe); FlyGM's spiking baseline never produced a stable gait |
| Training all 25.5M weights | discards the wiring prior; the topology comparison becomes meaningless |
| Sequential body training as *the* method | kept as a continual-learning test (E5); generalisation comes from a distribution of bodies |
| Laya as a training teacher | reads only text; the per-step judge would cost 3–8 CPU-weeks (193–464 ms per call) or 20–44 T4-hours per 10M steps; base model scores 0.362 vs a 0.461 majority baseline zero-shot. Only its calibration *idea* is borrowed (§4.6) |
| Jev (via OpenCode free tier) | free "for a limited time"; reports from Sept 2026 say the free tier now rejects non-OpenCode clients; text-only and remote |
| Naive merge (concatenate channels) | predicted to fail (H2a); kept only as a baseline |
| Structural edge growth | undermines the topology claim; per-body reorganisation is measured through learned gains and gates instead (deferred) |
| jax-metal / MLX-only stack | jax-metal is unmaintained; training runs on Kaggle CUDA; the Mac runs JAX on CPU for debugging |

## 15. Related work

| Work | Relevance |
|---|---|
| [FlyGM (arXiv 2602.17997, 2026)](https://arxiv.org/html/2602.17997v1) | FlyWire connectome as a fixed-wiring graph controller for the flybody fly; imitation then PPO; beat rewired, Erdős–Rényi and MLP baselines. Fly body only, known I/O mapping |
| [Eon Systems (2026)](https://eon.systems/updates/embodied-brain-emulation) | LIF whole-brain fly in NeuroMechFly; hand-chosen I/O; DN readout to pre-trained controllers |
| [Fly.exe](https://github.com/Ibtisam-Mohammad/Fly.exe) | full MaleCNS LIF in closed loop; no learning; 0.015× real time |
| [YYK2007/flybrain](https://github.com/YYK2007/flybrain) | MaleCNS rate model, decoder-only training; shuffled wiring also passed |
| [annel0/flybrain](https://github.com/annel0/flybrain) | GPU spiking simulator; records what could not be established (≈57% of neurons non-spiking) |
| [Embodiment Scaling Laws (CoRL 2025)](https://arxiv.org/abs/2505.05753) | ~1,000 generated robots; RL experts distilled into URMA; zero-shot to Go2 / H1; needs joint descriptions |
| Tang & Ha 2021, *Sensory Neuron as a Transformer* | permutation-invariant per-channel inputs |
| AnyMorph (Trabucco et al. 2022) | morphology-agnostic policy with learned per-body tokens |
| Najarro & Risi 2020; Miconi et al. 2018 / 2019 | learned plasticity rules; adaptation to damage |
| [Hebbian Attractor Networks (arXiv 2603.22512)](https://arxiv.org/abs/2603.22512) | plastic networks for Go1 locomotion |
| Lappalainen et al. 2024; Shiu et al. 2024 | connectome-constrained models; sign conventions |
| RMA (Kumar et al. 2021); asymmetric actor-critic (Pinto et al. 2017) | privileged teacher–student training |
| Bongard, Zykov & Lipson 2006 | robots that build self-models for resilience |
| [DADS (ICLR 2020)](https://arxiv.org/abs/1907.01657) | self-discovered skills + planning |
| Eureka (Ma et al. 2024) | LLM-written reward code |
| [Capability-aware heterogeneous teams (arXiv 2401.13127)](https://arxiv.org/html/2401.13127) | heterogeneous team policies with *given* capabilities (ours discovers them) |
| [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) | vectorised multi-robot scenarios (reference for 2D tasks) |
| [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground); [flybody](https://www.nature.com/articles/s41586-025-09029-4) | bodies |
| Salimans et al. 2017; rliable (Agarwal et al. 2021) | evolution strategies; statistics |

## 16. Claim status

- **Verified in the planning session (read directly):**
  - MaleCNS file list, sizes and licence
  - Laya's code, README latency and accuracy figures
  - Mac benchmarks (see [Requirements](Requirements.md))
- **Read through a summarising tool; check the PDFs before citing:**
  - FlyGM numbers (5.57° vs 7.77° / 11.33° / 7.18° angle error)
  - Eon, Fly.exe, YYK2007, annel0 and ESL details
  - the ≈95k optic-lobe count
- **From memory, not re-checked:** Tang & Ha, AnyMorph, Najarro & Risi, Miconi, Lappalainen, Shiu, RMA, Pinto, Bongard, Eureka, Salimans, rliable.
- **Novelty:** based on a non-exhaustive search. A gated Hugging Face repo (`hwihwalab/malecns-connectome-robotics-2026`) claims MaleCNS driving 8 robot bodies and could not be read (401). Check it before any novelty claim.
