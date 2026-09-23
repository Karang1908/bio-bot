# One Brain, Many Bodies

**A single self-learning brain, wired like the fruit fly's nervous system (MaleCNS), that configures itself for whatever it's plugged into (a robot, software, a house full of devices) and runs several bodies at once as one mind toward a shared goal.**

Nobody tells it what its sensors and motors are, and nobody writes setup rules. It works that out by experimenting, the way an animal learns its own body.

---

## The two research questions

### 1. Self-learning: no manual configuration

Today every robot, smart-home system and piece of software has to be set up by hand. Someone calibrates the robot, maps each switch to a light, and writes the automations and rules. We want a system you just connect. It discovers what each input and output does and learns to use them toward a goal, within safety limits a human sets once.

> *Can one brain, given unnamed sensor and motor channels, discover what it is connected to and learn to use it without anyone configuring it?*

### 2. Coordination: one brain, many bodies (the "weirdly agentic" part)

Teach the brain three bodies separately (a drone, a humanoid, a fly-shaped walker), then connect all three at once and say *"find the apple."* The drone searches from above. The moment it sees the apple, the humanoid is already walking there. No message passes between the bodies, because they share one mind. This is neither a single robot nor a team of robots: it is one agent spread across several bodies.

> *Does a single shared brain coordinate different bodies better than separate brains that have to communicate, and what kind of agency emerges when one mind lives in several bodies?*

---

## How it's deep learning

- **Architecture study.** The brain is a recurrent neural network whose wiring is copied from the fly connectome; only its parameters are trained. We test whether wiring borrowed from biology learns faster, transfers better and recovers better than generic networks (GRU, Transformer, randomly wired networks of the same size).
- **Training methods.**
  - Deep reinforcement learning and evolution strategies (the brain learns from reward).
  - Teacher–student distillation. Teachers exist only during training and are stripped before deployment.
  - Self-supervised prediction of what it will sense next.
- **Learning to learn.** The brain learns its own plasticity rules, so after deployment it keeps adapting on its own without backpropagation. This is the engine of self-learning.
- **Attention-based input and output.** A Transformer-style interface accepts any number of unlabeled sensor and motor channels.
- **Shared memory across bodies.** One world map and goal memory that every body writes into and reads from. This is the engine of coordination.
- **Language grounding.** Vision-language embeddings connect words like "apple" to what the brain's cameras see.
- **Continual learning.** One persistent brain moves across many bodies without forgetting the earlier ones.

## The gap it fills

**Self-learning**
1. Systems today are configured by hand. Robots are calibrated and trained per model; smart homes and IT systems run on rules people write. Nothing in common use learns its own setup from scratch.
2. Methods that work across many robot bodies are told what each joint is. Embodiment Scaling Laws and its URMA architecture give the network a description of every joint. This brain gets unnamed channels and works out what they are.
3. Connectome-driven bodies so far are flies. FlyGM (2026) and Eon drive a fly body with inputs and outputs mapped to known fly neurons. Nobody has tested whether the fly's wiring helps a brain learn bodies it never evolved for.

**Coordination**

4. Robot teams have separate brains that must message each other. One learned brain running several *different* bodies through shared memory, one that discovers those bodies itself and can take bodies added or removed mid-task, is mostly untested.

> How sure we are about the gap: a literature search found no prior work covering these points together, but the search was not exhaustive. One gated Hugging Face repo (`hwihwalab/malecns-connectome-robotics-2026`) claims something close to point 3 and could not be read. Check it before claiming novelty in a writeup.

---

## Applications

**From self-learning (plug it in, it works out the rest)**
- **Robots:** a new robot, arm or drone learns its own body with no calibration or per-model training. It re-adapts when a part wears out, breaks or is swapped.
- **Smart home / IoT:** connect devices (e.g. through Home Assistant) and it learns what each one actually does. For example, it finds out that `switch.sonoff_1003a` is the living-room lamp by testing it, and learns routines without hand-written automations.
- **Computers and software:** a homelab or small server fleet where it learns which machine suits which job and keeps services healthy, using only pre-approved actions.
- **Buildings and industry:** heating, ventilation, energy and sensor networks. New sensors join without integration work.
- **Undocumented or legacy equipment:** it works out what unknown control channels do through safe, reversible experiments.

**From coordination (one mind, many bodies)**
- **Search and rescue:** a drone scouts, a ground robot reaches the spot, and a small robot gets into gaps in rubble.
- **Home assistance:** house cameras act as eyes, and a robot acts as hands. *"Where are my keys?"*
- **Warehouses and logistics:** drones scan shelves while ground robots move stock.
- **Agriculture and inspection:** aerial bodies spot a problem and ground bodies deal with it.
- **Mixed fleets:** add or remove a body mid-task without reprogramming the rest.

Safety principle for all real-world use: **configure the fence, not the function.** A human states once what is forbidden (no touching the stove, no deleting files, speed limits); the brain learns everything inside that fence.

---

## Documents

| File | What it is | Read it if you are |
|---|---|---|
| [Overview.md](Overview.md) | The whole idea in plain words: no jargon, with analogies, applications and the demo | anyone |
| [TechnicalDesign.md](TechnicalDesign.md) | Architecture, training, experiments, baselines, risks, related work | building or reviewing it |
| [Requirements.md](Requirements.md) | Hardware, software, data, compute budget and workflow for running and training | setting it up |
| [LegacyIdea.md](LegacyIdea.md) | The original concept document (17 Sep 2026), kept for reference | curious how it started |

## Status

Planning. No code yet. Everything runs on free resources: a MacBook Air M3 for development and Kaggle's free GPUs for training.

## Data credit

Connectome data: Janelia FlyEM **MaleCNS v1.0** (male *Drosophila* central nervous system), licensed **CC-BY 4.0**. Attribution is required in any publication or demo.
