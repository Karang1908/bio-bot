# One Brain, Many Bodies
## Selected Approach and Execution Strategy

_Last updated: September 17, 2026_

---

# 1. Final Project Concept

The project uses the **MaleCNS fruit-fly connectome** as the structural basis of a trainable neural system.

The core idea is:

> **Take one persistent brain, place it into radically different bodies, and make it discover for itself what body it has, what it can sense, what it can control, and how to use that body.**

The brain is **not** told:

- “this is a humanoid,”
- “this is a drone,”
- “this is a robot dog,”
- “this output is the left leg,”
- “this sensor is a camera,”
- “this actuator is a propeller.”

Instead, it receives unnamed sensory channels and unnamed action channels and must learn their causal relationships through interaction.

The long-term idea is:

```text
ONE PERSISTENT BRAIN
        ↓
   Fruit Fly Body
        ↓
     Humanoid
        ↓
    Robot Dog
        ↓
    Quadcopter
        ↓
   Multiple Bodies
        ↓
 Digital / Software Body
```

The same brain continues learning across all of them.

---

# 2. The Main Research Question

The main research question is:

> **Can a biologically structured neural system autonomously discover and adapt to unknown embodiments, while preserving and transferring what it learns across completely different bodies?**

The stronger version is:

> **Can one persistent neural system learn the general concept of embodiment rather than memorizing one specific body?**

---

# 3. Why Use MaleCNS?

We are deliberately not starting from a standard Transformer or generic neural network.

The MaleCNS connectome gives us a biological neural topology that evolved for:

- sensory integration,
- movement,
- closed-loop interaction,
- navigation,
- adaptation,
- action selection,
- survival.

That makes it a strong structural prior for embodied intelligence research.

The project is **not claiming that the fly brain is automatically better than a Transformer**.

Instead, the scientific comparison is:

```text
MaleCNS-constrained brain
vs
Transformer
vs
RNN / GRU / LSTM
vs
random sparse recurrent network
```

We will test whether the biological topology gives advantages in:

- body discovery,
- low-data adaptation,
- continual learning,
- damage recovery,
- transfer across embodiments,
- multimodal integration.

---

# 4. What MaleCNS Gives Us

MaleCNS gives us the biological wiring structure.

It is not a pretrained AI model and it does not have a system prompt.

It is best thought of as:

```text
Neuron A → Neuron B
Neuron A → Neuron C
Neuron C → Neuron D
...
```

with additional biological metadata.

We turn this into a runnable neural system by adding:

```text
neural dynamics
+
plasticity
+
rewiring
+
sensory injection
+
action readout
+
learning
```

Our project therefore becomes:

```text
MaleCNS topology
      ↓
Sparse recurrent neural system
      ↓
Plasticity + rewiring
      ↓
Embodiment discovery
      ↓
Cross-body learning
```

---

# 5. The Brain Must Learn, Not Just Execute

The brain is not fixed.

It will be allowed to:

- change synaptic strengths,
- strengthen useful pathways,
- weaken unused pathways,
- prune connections,
- create limited new connections,
- reorganize after body changes,
- reorganize after damage,
- preserve persistent memory.

This structural plasticity is part of the main project.

The research question becomes:

> **Does the same initial connectome reorganize itself differently for different bodies?**

---

# 6. No Predefined Walking, Flying, or Running

We do **not** program:

```text
walk()
fly()
run()
jump()
```

into the brain.

The brain has to discover useful motion patterns.

For example, with a humanoid:

```text
unknown output
    ↓
some joint moves
    ↓
sensors change
    ↓
brain learns causal relationship
```

After enough exploration:

```text
groups of actions
    ↓
balance
    ↓
standing
    ↓
stepping
    ↓
forward locomotion
```

What we humans call “walking” is simply a movement pattern that achieves the goal.

The brain does not need the symbolic concept of walking in order to produce walking.

---

# 7. Self-Discovery of the Body

The key learning loop is:

```text
Action
  ↓
World changes
  ↓
Sensors change
  ↓
Prediction error
  ↓
Brain updates body model
  ↓
Better action
```

Initially, the brain may produce chaotic movement.

That is expected.

The system gradually learns:

- which outputs affect the world,
- which sensors respond to those outputs,
- which outputs belong together,
- which actions create stable movement,
- what parts of the environment are under its control.

This creates an internal **body schema**.

---

# 8. The Causal Self

We will treat “self” as something learned rather than hardcoded.

The core idea:

> **Anything whose state reliably changes as a consequence of my internal actions is probably part of me or part of my controllable embodiment.**

Example:

```text
brain sends action
      ↓
arm moves
      ↓
proprioception changes
      ↓
camera changes
      ↓
arm becomes part of learned self-model
```

This creates a concept we can describe as:

## Learned Causal Self

This is one of the most distinctive research ideas in the project.

---

# 9. Selected Bodies

The project will use several bodies.

## 9.1 Fruit Fly

Use:

```text
FlyGym / NeuroMechFly
```

This gives us the natural body corresponding most closely to the biological connectome.

Useful modalities include:

- vision,
- proprioception,
- joint state,
- touch/contact,
- olfaction,
- environmental interaction.

The fly becomes the biological baseline.

---

## 9.2 Human / Humanoid

For a complex human-like body we will use a simulation model.

The selected direction is to begin with a **humanoid robot body** because it is easier to control and debug than a full biological musculoskeletal human.

A realistic progression is:

```text
robot humanoid first
→ musculoskeletal human later
```

The humanoid body can expose:

- camera vision,
- microphones,
- proprioception,
- joint position,
- joint velocity,
- balance,
- contact / touch,
- force,
- orientation.

The brain is not told which channel represents what.

---

## 9.3 Robot Dog / Quadruped

Use a quadruped model such as a Unitree-style robot dog.

The brain receives:

- body orientation,
- joint angles,
- joint velocity,
- contact sensors,
- optional vision,
- optional audio,
- unnamed actuator channels.

The goal is for the same brain to discover:

- four-legged locomotion,
- balance,
- gait,
- turning,
- navigation.

---

## 9.4 Quadcopter

Use a simulated quadcopter.

Possible observations:

- camera,
- IMU,
- altitude,
- orientation,
- velocity,
- depth,
- optional GPS.

Actions are motor/thrust channels.

But the brain does not know:

```text
A0 = front-left motor
A1 = front-right motor
A2 = rear-left motor
A3 = rear-right motor
```

It simply sees:

```text
A0
A1
A2
A3
```

and learns what they do through consequences.

---

# 10. Multimodal Learning

The project deliberately includes multiple sensory modalities.

These include:

- images,
- video,
- audio,
- touch,
- proprioception,
- orientation,
- smell-like signals,
- telemetry,
- movement,
- environmental state.

The same brain should learn to use different modalities depending on the body.

Example:

```text
Humanoid:
vision + audio + touch + proprioception

Drone:
vision + IMU + altitude + depth

Robot dog:
vision + audio + force + proprioception

Fly:
vision + smell + touch + proprioception
```

This makes the project a genuine multimodal deep-learning system.

---

# 11. Language: Selected Approach

We decided that the project should eventually support **natural-language goals**.

For example:

> “Find the apple in this room.”

The fly-brain system itself does not naturally understand English.

So we use a staged approach.

---

# 12. Language Phase 1 — LLM as Teacher

At the beginning, use an external LLM only as a **teacher / semantic parser**.

The LLM does not control the body.

Its role is:

```text
Natural language
      ↓
LLM teacher
      ↓
structured semantic target
      ↓
grounded goal representation
      ↓
MaleCNS-based brain
      ↓
brain figures out how to act
```

Example:

```text
"Find the apple"
```

can initially be converted into something conceptually like:

```text
target_concept = apple
goal = approach
success = target reached
```

The LLM defines **what** the goal means.

The embodied brain still learns **how** to accomplish it.

---

# 13. Language Phase 2 — Distill Language Into the Embodied Brain

The long-term goal is not to depend on the LLM forever.

We will use the LLM as a teacher and gradually train a smaller internal language representation.

The selected research direction is:

> **Ground language concepts inside the embodied neural system itself.**

For example:

```text
heard word "apple"
      ↓
internal language representation
      ↓
visual / sensory apple representation
      ↓
goal state
      ↓
body-specific behavior
```

The word “apple” should eventually become associated with actual sensory experience.

This is not just text classification.

It is **grounded language learning**.

---

# 14. Audio Input

For the human-like body, we want “ears.”

Audio pipeline:

```text
microphone / simulated audio
        ↓
audio features / speech encoder
        ↓
language representation
        ↓
brain / goal system
```

Early versions can use:

```text
speech-to-text
```

Later versions can move toward:

```text
audio directly → learned representation
```

This lets the agent respond to spoken instructions.

---

# 15. Audio Output / Speech

The human-like body can also have a “mouth.”

Early version:

```text
brain / language state
      ↓
text or token output
      ↓
speech synthesizer
      ↓
audio
```

A more advanced version could later attempt lower-level vocal control, but that should not be required for the first successful project.

---

# 16. Language as a Grounded Concept, Not a Prompt

The final goal is not:

```text
LLM receives prompt
→ LLM directly controls robot
```

Instead:

```text
language
→ grounded internal concept
→ MaleCNS-based embodied learner
→ behavior
```

The external LLM is a temporary teacher.

The long-term research experiment is:

> **Can an embodied biological neural topology learn language concepts by associating words with actual sensory and behavioral experience?**

---

# 17. Cross-Embodiment Meaning

A particularly interesting language experiment is:

> **Does the meaning of an action concept change when the body changes?**

Example:

```text
"move toward the target"
```

Humanoid:

```text
walk
```

Quadcopter:

```text
fly
```

Quadruped:

```text
run
```

Fly:

```text
walk / fly depending on learned behavior
```

The semantic goal stays constant.

The motor implementation changes.

This creates a strong research question:

> **Can semantic concepts remain stable while their physical realization changes across embodiments?**

---

# 18. Persistent Brain Across Bodies

The brain is not reset after every body.

Example:

```text
Brain v0
  ↓
learns fly
  ↓
Brain v1
  ↓
learns humanoid
  ↓
Brain v2
  ↓
learns robot dog
  ↓
Brain v3
  ↓
learns quadcopter
```

Then return to an old body.

Example:

```text
quadcopter
→ humanoid
→ dog
→ quadcopter again
```

Measure whether the second quadcopter learning phase is faster.

This tests:

- memory,
- transfer,
- continual learning,
- catastrophic forgetting,
- meta-learning of embodiment.

---

# 19. Multiple Bodies at Once

The advanced stage is:

```text
                 ┌── Humanoid
                 │
One Brain ───────┼── Drone
                 │
                 └── Robot Dog
```

The brain receives several sensory streams and several action groups.

It is not initially told which channels belong together.

It must discover:

```text
these sensors/actions belong to one body
those belong to another
```

The system may eventually coordinate all of them.

Example:

```text
drone scouts
      ↓
robot dog moves equipment
      ↓
humanoid manipulates object
```

All controlled by one persistent neural system.

---

# 20. Damage and Recovery

After the agent learns a body, damage it.

Examples:

### Drone

```text
motor loses 40% thrust
camera becomes noisy
IMU fails
```

### Humanoid

```text
joint locks
actuator weakens
sensor disappears
```

### Robot dog

```text
one leg loses torque
contact sensor fails
```

The brain should detect increased prediction error and adapt.

This tests:

- body-model updating,
- rewiring,
- fault tolerance,
- robotics resilience.

---

# 21. Neural Damage

We can also damage the brain itself.

Examples:

```text
remove random edges
silence neurons
disable subnetwork
remove learned connections
```

Then test whether structural plasticity allows recovery.

This creates another research question:

> **Can a connectome-derived neural system reorganize after internal damage better than standard artificial neural architectures?**

---

# 22. New Senses

The brain can be given senses a fruit fly never evolved for.

Examples:

- depth,
- GPS,
- infrared,
- radar,
- magnetic direction,
- machine telemetry,
- software state.

The brain is not told what these signals mean.

It must discover whether they are useful.

This lets us test:

> **Can the system incorporate completely novel sensory modalities into its internal world model?**

---

# 23. Software as a Body

A later extension treats software as another embodiment.

Instead of:

```text
camera + legs
```

the brain receives:

```text
screen + state + action channels
```

Example environments:

- text editor,
- terminal sandbox,
- simple desktop,
- file manager,
- spreadsheet-like app,
- browser sandbox.

The system is not told:

```text
this button means save
this action opens a file
this command compiles code
```

It has to discover these capabilities.

This creates the concept of:

## Digital Embodiment

---

# 24. Homelab Demo

A strong demonstration is to treat the user's homelab as another “body.”

Available machines:

- Mac,
- Linux machine,
- Raspberry Pi.

The brain should **not** receive unrestricted shell access.

Instead, expose a safe, sandboxed interface.

Possible sensor channels:

```text
CPU usage
RAM usage
temperature
disk usage
container status
network latency
Pi GPIO state
```

Possible action channels:

```text
run whitelisted benchmark
start safe container
stop safe container
move workload
toggle Raspberry Pi LED
run approved script
```

The brain is not initially told:

```text
machine A is Mac
machine B is Linux
machine C is Raspberry Pi
```

It can learn which device has which capabilities.

Possible demo:

```text
Goal:
"Run this job on the best available machine."
```

The system observes resource state, experiments safely, and learns where to place the workload.

This is a strong real-world extension of the embodiment idea.

---

# 25. Why Not Give Open Shell Access?

For the demo, unrestricted shell access is unnecessary and risky.

Instead use:

```text
sandboxed actions
+
whitelisted commands
+
containers
+
limited permissions
```

This keeps the research focused on:

```text
capability discovery
resource discovery
action consequence learning
```

without turning the system into an uncontrolled automation agent.

---

# 26. Core Learning Loop

The complete loop becomes:

```text
Observation
   ↓
Sensory encoders
   ↓
MaleCNS brain
   ↓
Action
   ↓
Body / software environment
   ↓
New observation
   ↓
Prediction error
   ↓
Plasticity + rewiring
   ↓
Updated body model
```

If language is present:

```text
Speech / text
     ↓
LLM teacher initially
     ↓
grounded concept
     ↓
brain goal state
     ↓
embodied action
```

Later:

```text
Speech / text
     ↓
internal grounded language representation
     ↓
brain goal state
     ↓
embodied action
```

---

# 27. Execution Strategy

We should not attempt the entire system at once.

The project will be built in stages.

---

# 28. Stage 1 — MaleCNS Runtime

Goal:

```text
load full MaleCNS
run neural activity
inject signals
read neural state
```

Use:

```text
official MaleCNS data
+
flybrain and/or our own sparse runtime
```

Success:

```text
stable full-graph neural stepping
```

---

# 29. Stage 2 — Add Plasticity and Rewiring

Implement:

```text
weight plasticity
connection strengthening
connection weakening
pruning
limited edge growth
neuromodulation-inspired learning
```

Success:

```text
network structure can change safely over time
```

---

# 30. Stage 3 — Build Generic Body Interface

Create one body API.

Example:

```python
class UnknownBody:
    def reset(self):
        ...

    def observe(self):
        ...

    def step(self, action):
        ...
```

All body semantics remain hidden.

Success:

```text
the brain can interact with any environment through the same interface
```

---

# 31. Stage 4 — Simple Toy Body

Before using a humanoid, use a tiny environment.

Example:

```text
2D body
2–4 unknown actuators
position sensors
distance sensors
```

Goal:

```text
discover which outputs move the body
```

This proves the self-discovery mechanism.

---

# 32. Stage 5 — Fruit Fly

Connect the brain to:

```text
FlyGym / NeuroMechFly
```

Test:

- locomotion,
- sensory integration,
- simple target behavior,
- adaptation.

This is the biologically aligned baseline.

---

# 33. Stage 6 — Quadcopter

Use a simulated quadcopter.

The brain must discover:

```text
thrust
orientation
hover
turning
translation
```

without being told motor semantics.

Initial target tasks:

```text
hover
move to target
follow visual marker
```

---

# 34. Stage 7 — Robot Dog

Connect the same brain checkpoint to the quadruped.

Measure:

- time to stable gait,
- balance,
- forward motion,
- transfer from previous bodies.

---

# 35. Stage 8 — Humanoid

Only after simpler bodies work.

Progression:

```text
joint discovery
→ balance
→ standing
→ stepping
→ locomotion
→ reaching
```

Vision and audio can be added progressively.

---

# 36. Stage 9 — Language Teacher

Add the external LLM teacher.

Initial job:

```text
natural language
→ structured semantic target
```

Example:

```text
"Find the apple."
```

The LLM produces the goal representation.

The MaleCNS system still decides how to act.

---

# 37. Stage 10 — Ground Language

Train an internal language representation.

The LLM remains a teacher during this phase.

Associate:

```text
word
+
audio
+
visual target
+
reward
+
action consequence
```

Eventually test whether the LLM can be removed for a small learned vocabulary.

---

# 38. Stage 11 — Damage Recovery

Damage:

```text
body
sensors
actuators
brain connections
```

Measure:

```text
performance drop
recovery time
rewiring amount
final recovered performance
```

---

# 39. Stage 12 — Return to Old Bodies

Test:

```text
does the brain relearn an old embodiment faster?
```

This is one of the clearest cross-body transfer experiments.

---

# 40. Stage 13 — Multiple Bodies

Only after individual body learning works.

Connect:

```text
drone
+
robot dog
+
humanoid
```

to one brain.

Test whether it can discover independent embodiment groups.

---

# 41. Stage 14 — Homelab / Digital Body

Expose the Mac, Linux machine and Raspberry Pi through safe wrappers.

Tasks can include:

```text
choose machine
run approved workload
monitor result
adapt to availability
control simple GPIO
```

This becomes a real-world demonstration of capability discovery.

---

# 42. Compute Strategy

Primary development machine:

```text
MacBook Air M3
16 GB RAM
```

Use it for:

- coding,
- local tests,
- small simulations,
- visualization,
- debugging,
- final presentation.

Use cloud/free GPU resources for heavier experiments.

Possible services:

- Modal,
- Kaggle,
- Colab,
- student / academic GPU credits.

The goal is to keep the project as close to zero-cost as possible.

---

# 43. Efficiency Strategy

Because MaleCNS is large, use:

- sparse graph representation,
- no dense adjacency matrix,
- sparse neural updates,
- short recurrent training windows,
- mixed precision where stable,
- local plasticity,
- selective gradient training,
- parallel simulations,
- staged curricula,
- smaller image resolutions,
- frozen subsets where useful.

We preserve the full graph structure where possible without pretending to perform a full biological molecular simulation.

---

# 44. What We Will Compare Against

To make the project scientifically meaningful, compare against:

```text
MaleCNS-constrained brain
vs
random sparse recurrent network
vs
RNN
vs
GRU/LSTM
vs
Transformer-style temporal model
```

Possible comparisons:

- body discovery speed,
- sample efficiency,
- cross-body transfer,
- damage recovery,
- forgetting,
- multimodal learning,
- structural adaptation.

---

# 45. Evaluation Metrics

## Body discovery

- steps until useful control,
- prediction error,
- actuator/sensor causal mapping quality.

## Control

- success rate,
- target accuracy,
- stability,
- collision rate,
- energy use.

## Transfer

- adaptation speed on new body,
- relearning speed,
- zero-shot transfer,
- few-shot transfer.

## Continual learning

- forgetting,
- retained performance,
- forward transfer,
- backward transfer.

## Rewiring

- edges added,
- edges removed,
- percentage of structure changed,
- body-specific subnetworks.

## Damage

- recovery time,
- recovery quality,
- structural change required.

## Language

- command grounding accuracy,
- vocabulary retention,
- performance without LLM teacher,
- cross-body semantic consistency.

---

# 46. Final Demonstration

A strong final presentation can show:

```text
ONE BRAIN
```

with several buttons:

```text
[Fly]
[Humanoid]
[Robot Dog]
[Quadcopter]
[Homelab]
```

The audience sees the same brain checkpoint moved between embodiments.

For each body:

```text
initial confusion
→ exploration
→ body discovery
→ stable behavior
```

Then:

```text
spoken command:
"Find the apple."
```

The language system grounds the goal.

The active body decides how to accomplish it.

Then:

```text
damage one actuator
```

and show recovery.

Finally:

```text
MULTI-BODY MODE
```

as the stretch demo.

---

# 47. Final Project Identity

The project should **not** be described as:

> “A fly brain that controls a drone.”

It should be described as:

> **A persistent connectome-constrained learning system that discovers its own embodiment, rewires itself for unfamiliar bodies, grounds language through experience, transfers knowledge across embodiments, and adapts when its body or environment changes.**

The shortest version is:

> **Instead of telling an AI what tools it has, we give it a body and make it discover its own capabilities.**

And the strongest research framing is:

> **Can one biologically structured brain learn what it means to have a body?**
