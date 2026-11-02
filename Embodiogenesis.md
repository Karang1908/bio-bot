# Embodiogenesis: the growing brain plan

_Agreed with the owner on 2 Nov 2026. It replaces the reward-tuning "vision approach" rounds (3–68) as the way forward._
_Working name: **Embodiogenesis**. The owner may rename it. Check that the term is not already in use before publishing it._

**Embodiogenesis** is how a brain comes to have a body. It is born with no configuration: nobody tells it its motors,
joints or senses. It discovers them, wires itself to suit them and keeps learning for life, the way a newborn does.
It covers three timescales:

| timescale | in biology | in this project |
|---|---|---|
| **inheritance** | the genome's starting wiring | the fly connectome (MaleCNS core v2) is the **seed**, not a cage |
| **development** | connections form and are pruned; new neurons are added | the brain grows connections and neurons where it needs them |
| **lifetime** | dopamine-gated change at synapses | the brain keeps learning after deployment, without backprop |

Decision models (Jev, OpenJev, Laya and similar) are **out of scope** (owner, 2 Nov). The brain decides.

---

## 1. Why the plan changed: the 2 Nov review (verified unless marked)

1. **The dog never learned to go to the ball.** In every "ball" test in Rounds 67 and 68, the dog ended farther from
   the ball than it started:
   - Round 67: 2.34–2.64 m → 3.11–3.90 m.
   - Round 68: 2.34–2.64 m → 3.10–3.78 m.

   Its facing also got worse: the cosine fell from about 0 to between −0.15 and −0.52. The reported arrival rates
   (2–12% per cohort) are about what a dog wandering past the ball would give. The "end distance 3.24–3.36 m" that
   earlier notes called a closing record was larger than the starting distance.
2. **The "champions" were selected from noise.** One 192-life evaluation at about 10% success has a standard error
   of about ±2.2 points; one 64-life cohort has about ±3.7. Picking the best of 8 evaluations × 3 cohorts × several
   metrics over 60+ rounds will produce 14–25% by chance. Resuming a champion showed regression to the mean every
   time:
   - Round 66's 11.98% checkpoint measured 7.29% combined at the start of Round 67.
   - Round 58's 14.58% checkpoint measured about 5% ball and 3.7% come at the start of Round 68 (cohort means).
3. **The eye often cannot see the ball** (from the code; how often it actually happens is not yet measured).
   - The ball is 0.22 m across (`BALL_R = 0.11`). The eye casts one point ray every 5° (`EYE_STEP = 5.0`).
   - The ball is placed 1.0–4.0 m away, in a random direction (`dog_world.place`).
   - Beyond about 2.5 m the ball is narrower than the gap between rays, so it can fall between them.
   - The 120° camera covers only a third of the circle, so the ball usually starts outside the view.
4. **Only about 10–15 of each round's 88 minutes was learning.** The rest was install and compile (about 14 min,
   plus about 5 min more compiling), the starting test (about 20 min) and six tests during training (about 7 min
   each). This is from the Round 67 log timestamps.
5. **The brain step is about 98% wasted work.** `weights()` scatters the 502,438 real connections into a full
   7,019 × 7,019 matrix (49.3M entries, about 1% filled), and every step multiplies by all of it
   (`train/brain_dog.py:677-679, 757`).
6. **The reward overwhelmed learning.** Task terms of 250–350 sit beside walking rewards of about 1. The critic's
   error was 500–1,500 in Round 67. That this is a main cause of the failure is believed, not shown. The 12 tuned
   terms also amount to a hand-written "turn, then sprint, then brake" controller written into the reward.
7. **The "0.0 ms" dodge reactions are a measurement artifact.** The brain acts every 20 ms (`TICK_DT = 0.02`), and a
   signal needs several steps to reach the legs from the eye.
8. **What is real:**
   - The brain carries some information about where the ball is: its voice names ball-left/ahead/right at 0.68–0.70
     on held-out lives (0.5 = chance). The legs do not use it.
   - Gaits and turns stay intact with no teacher help.
   - Falls are high: the walk word has 1.9 falls per 13 s in the Round 68 verdict.
   - Getting up is 0%.

---

## 2. The brain: from a fixed fly wiring to a growing brain

Nothing learned so far is thrown away. Growth only adds to the brain, as the v1 → v2 growth did. The **body-to-brain
line mapping stays fixed** (one random wiring, `WIRING = 20260927`).

### 2.1 Engine first: compute only the real connections
- Replace the full matrix product with a sum over the connections (gather by `pre`, `segment_sum` by `post`).
- Expected: far fewer operations. The actual GPU speedup must be **measured** (estimate 5–20×). A full matrix
  cannot hold a grown brain: 50k neurons would need 2.5 billion entries.
- Gate: the same checkpoint gives the same commands (to float tolerance) on the old and new engine, on Kaggle CPU.

### 2.2 Lifetime learning: dopamine-gated plasticity (never built until now)
- **Fast weights** on chosen synapse groups: first the mushroom body's Kenyon cell → output neuron synapses, then
  others. They change by a three-factor rule (`TechnicalDesign.md` §4.5): eligibility trace × dopamine signal.
- **Rule parameters** (learning rate and Hebbian terms, per cell type or compartment) are learned during training.
  This is "learning to learn."
- **The brain's own reward:** the real dopamine neurons learn to predict "how likely am I to succeed," with a
  calibrated proper scoring rule (log or Brier score). After training, that prediction is the only reward, and it
  drives the fast weights.
- Gate: with backprop switched off, performance **improves within one life** on a changed body (a weak motor, a new
  task), beyond what the same brain with plasticity frozen does.

### 2.3 Development: connections form and are pruned
- A fixed connection budget. Periodically:
  - **prune:** remove the weakest or least-used connections;
  - **grow:** add connections between neurons that are often active together, or where learning shows a connection
    would help (RigL / DEEP R style).
- Biology constraints:
  - A new connection's sign comes from the sending neuron's transmitter (Dale's law).
  - Growth prefers neurons in the same or neighbouring regions.
- **New connections start learnable, not near zero.** LOGBOOK §8.17: connections started at 1% strength never opened
  in 120M steps.
- Gate: a grown brain beats the same brain without growth on the same task budget, with every earlier skill kept.

### 2.4 Development: new neurons by copying cell types and letting the copies diverge
- When a region stays overloaded (its prediction error stays high), or a new body arrives, copy existing cell types
  with their wiring pattern plus noise. The copies then specialise. This is how evolution enlarges brains (gene
  duplication).
- For the humanoid: copy the nerve cord's leg circuits for new limbs, instead of squeezing 29 actuators through a
  circuit built for six fly legs.
- The rule that only appends keeps every existing unit, connection and learned value.

### 2.5 What stays an abstraction
- **Rate neurons, not spiking.** Full-brain spiking runs at about 0.015× real time (Fly.exe) and failed to give a
  stable gait in FlyGM.
- **Backprop during childhood** stands in for evolution plus development. The lifetime learning in §2.2 is where the
  biology standard applies.

### 2.6 The research comparison
- fly seed + growth;
- random seed + growth (same size, same rules);
- fly seed, no growth.

This answers whether the fly's wiring helps as a starting genome. Either answer is reportable.

---

## 3. Fix the learning problem before scaling the brain

1. **An eye that integrates over each ommatidium's cone** (a real ommatidium collects light over about 5°; it does
   not sample a point).
   - Compute exactly what share of each ommatidium the ball, the owner and the thrown object cover (spheres and
     capsules). Rays stay for the background.
   - The brain still receives only pixels.
   - Gate: the ball is visible in at least 95% of in-view placements, measured on Kaggle CPU.
2. **A distance curriculum.** The ball starts 0.5–1 m away and in view, and moves farther out as success rises.
3. **A simple outcome reward:** getting closer, arriving, and not falling. Reward sizes stay comparable to the motor
   skills' rewards. The 12-term approach formula is retired.
4. **Every lesson still rehearses every known skill** (standing rule).

---

## 4. How results are judged from now on

- **The floor to beat:** a scripted dog that walks at random, tested on the same lives.
- **Ball and come must actually get closer:** the mean end distance must be below the start distance.
- **Promotion:** a checkpoint is promoted only if it beats its parent on **fresh lives not used to select it**, by at
  least 2 standard errors. Report counts (k/n) with intervals, not single cohort percentages.
- **Reaction time** is measured in brain steps from the first visible loom, with 20 ms as the floor.
- **Retention:** every evaluation reports gaits, poses, falls and getting up, as before.

---

## 5. Compute

| change | expected effect (to be measured) |
|---|---|
| sparse engine (§2.1) | brain step several times faster; growth becomes possible |
| one long job (8–10 h, a save every 30 min) instead of 90-min rounds | about 8 h of learning per job instead of about 12 min |
| a small test during training; the full three-cohort test once at the end | about 5 min of testing per job instead of about 50 |
| packages as a Kaggle dataset; compile cache kept between jobs | setup of a few minutes instead of about 19 |
| both free GPU slots: two seeds or conditions side by side | comparisons with a control at no extra cost |

### 5.1 The owner's RTX 5070 (access expected soon; not yet authorized for jobs)
- **Specs (vendor and spec databases):** 12 GB GDDR7, 672 GB/s, about 30.9 TFLOPS FP32. One Kaggle T4 is
  16 GB, 320 GB/s and about 8 TFLOPS FP32. So one 5070 is roughly the Kaggle 2× T4 pair in raw throughput, with less
  memory. This is believed, not measured.
- **The real gain:**
  - no 12 h session limit, no 30 h weekly quota, no queue and no reinstall;
  - so probably several times more learning hours per week;
  - and Kaggle stays as a second lane for control runs.
- **12 GB is enough** for the 7,019-unit brain with 512 environments, especially with the sparse engine (estimate).
  A much larger grown brain will need fewer environments or shorter unrolls.
- **Must verify before use:**
  1. Linux or WSL2. JAX's CUDA build does not run on native Windows.
  2. CUDA 12.8 or newer for Blackwell (sm_120). Whether the pinned jax 0.7.2 runs on sm_120 must be tested; a newer
     pin may be needed.
  3. NVIDIA's JAX notes say the random number generator is **not deterministic on consumer Blackwell GPUs**. The
     body wiring is drawn with `jax.random` from `WIRING = 20260927`. Draw it on CPU, or check its hash against the
     Kaggle draw, so the brain's one fixed wiring cannot silently change.
- **Rules:** the "no checks or training on the owner's laptop" rule still holds. The 5070 machine is separate compute
  and needs the owner's explicit go-ahead, plus remote access set up by the owner. The budget stays $0.

---

## 6. Phases and gates

| phase | work | done when (observable) |
|---|---|---|
| **0** | measure where time goes; sparse engine; ommatidium-cone eye; curriculum; simple reward; random-walk floor | engine speedup measured and commands unchanged; ball visible ≥ 95% in view; first long run where ball distance **falls** and beats the random-walk floor on fresh lives |
| **1** | dopamine-gated lifetime learning + calibrated success prediction | improvement within one life with backprop off, beyond frozen plasticity |
| **2** | connections that form and are pruned; neurons by copying and diverging | grown brain beats the ungrown one; all earlier skills kept |
| **3** | the humanoid G1 in the same growing brain, then both bodies | the project's research question |

Phase 0 takes days, and it decides everything after it. Phases 2–3 have no promised dates.
