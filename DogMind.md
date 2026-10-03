# The dog's mind: self-discovery, every sense, and the ingredients of consciousness

This is the plan for the next part of the work, agreed on 3 October 2026. It covers:
- turning the Go2 that walks on words into a dog that **discovers its own body and world**;
- **sensing everything the real Go2 senses**;
- **deciding and thinking with its own fly brain**;
- carrying the functional ingredients that scientific theories link to consciousness.

Results go in `LOGBOOK.md` as they come in. Built: step 1 (`--let_go`; a short Kaggle smoke run is underway). Every other step is
still a plan.

---

## 1. Decisions (in the user's words)

| date | decision |
|---|---|
| 3 Oct | "train the dog everything. Every single thing, make it mirrored to the real dog". The model is a real animal dog, taught from real dog motion capture. |
| 3 Oct | "we also have to teach them decision making for tasks ... run to the ball ... suddenly an object comes, so it dodges it ... thinking, reasoning, response to stimuli ... with a fly's brain you get some mutations, like the response time for a fly is really fast". |
| 3 Oct | "doesnt the go2 have a sensor or a camera? ... see the entire robot, we will give that robot consciouness". |
| 3 Oct | "All the senses, consciousness, everything. DO NOT SKIP A SINGLE THING." |
| 3 Oct | "i dont want you to train the dog fully how to walk, i want you to teach the initial steps, and then let it figure out its own body". |
| 3 Oct | **The teacher fades to zero.** Its hands, its corrections and the real dog's style pull all go away completely. |
| 3 Oct | **Its own way wins.** If the dog's self-found movement drifts from the real dog's look, the self-found way is kept. |
| 3 Oct | "the ai layer is not an external layer, its the fly's brain only, but inside a dog now". **Every decision, goal, plan and report comes from the fly brain itself.** The teacher (code written between runs) only teaches, and fades. Nothing outside the brain thinks for the dog. |

These are standing rules from earlier, and they still hold:
1. **One brain, one fixed wiring.** The brain is never told which line is which.
2. **Keep everything it experiences.** The wobble, the senses and the body stay at runtime exactly as in training.
   Fading the teacher doesn't break this rule. The teacher is a training signal, not something the dog senses, just as
   the harness (the parent) was allowed to leave.
3. **Every lesson rehearses everything already known.**
4. **Real tests from rest:** every checkpoint is a fresh, scripted test with nobody helping.

---

## 2. Self-discovery: what is taught, and what the dog finds itself

### 2.1 Where it stands today (honestly)

| stage | self-discovered? |
|---|---|
| babble: random twitching, learning to predict its own body | **Yes.** Real self-discovery. |
| names, words | Taught. That's right: a baby is taught words too. |
| walking, turning, backing | **Mostly taught.** The teacher found a trot by trial on the body, then its hands pulled the legs through it. The harness left, but the teacher's correction never did: every practice still adds `GUIDE = 0.5` of "be like the teacher" to the brain's update (5.0 for rest poses). |
| the real-dog run (`baby_dog_realdog`, running 3 Oct) | **Even more copying.** At every step, the guide pulls toward the real dog's next frame, and a style reward pays for real-dog poses. |

So the walk it has today is mostly the teacher's walk, polished by trial and error.

### 2.2 How a real puppy learns, and how this dog will

**A parent shows, holds the hands, then lets go completely.** After that, the puppy's own needs shape its movement:
- get to the thing;
- don't fall;
- don't get tired.

That is how it ends up with its own walk. Six mechanisms:

1. **Hands that let go based on skill, not a timer.** Each word has its own `help` level, starting at 1.
   - Every few updates, the dog's own result on that word is measured, using the body's meaning reward only, with no
     teacher terms.
   - Good enough (at least 90% of what it achieved with full help): help drops by 20%.
   - Falling apart: help steps back in (×1.25, up to 1).
   - On top of that, a hard deadline caps help at zero by 70% of the lesson, so the last 30% is always fully alone.
   - `help` scales:
     - the teacher's guide (its trot, or the real dog's next frame);
     - the real-dog style reward;
     - the four-feet stepping reward.

     These were all the teacher's form rules.
2. **Rewards for outcomes, never for form.** What stays forever is what each word *achieves*:
   - go the asked way at the asked speed;
   - stand still;
   - turn;
   - for sit and lie-down, the trunk's tilt and height (taken from the sit and lie-down the teacher found on this body:
     the real dog's sit tilts the nose up 0.82 rad, and the Go2 tipped over when it played that sit through its motors);
   - for jump, all four feet off the ground at once;
   - don't fall;
   - don't waste energy.

   Nothing says "put your hip at 0.3 rad".
3. **Energy, so gaits can emerge.** Legged robots *discover* walk, trot and gallop by themselves when they are only
   given a speed to reach and an energy cost to pay (Fu et al., "Minimizing Energy Consumption Leads to the Emergence
   of Gaits in Legged Robots", CoRL 2021).
   - The dog pays for mechanical power: Σ|torque × joint speed| per kg.
   - The speeds come from the real dog's recordings: walk 0.19 m/s, pace 0.71, canter 0.48, run 2.19; turns +1.38 and
     −1.18 rad/s.
   - The question this answers: does it find its own run, which nobody ever showed it, and does its foot pattern change
     with speed the way an animal's does?
4. **Curiosity.** The brain already predicts its own senses (the prediction head from the babble stage).
   - It will be rewarded for **getting better at predicting**: the fall in prediction error, not the error itself, so
     unlearnable noise is not attractive.
   - It then plays with what it doesn't understand yet. This is its free time.
5. **The real dog as something it watched, not a puppet string.** The real-dog motions were the first thing it watched
   (`baby_dog_realdog`). They fade with `help` like every other teacher signal. Its likeness to the real dog is still
   *measured* at every checkpoint, so we can see whether the real-dog look survives without the pull.
6. **Skills that are never taught at all.** The reward says only *what* (be on your feet, don't get hit, get to the
   ball), never *how*. These are the experiments that would actually prove self-discovery:
   - getting up after a fall;
   - dodging a thrown object;
   - walking around an obstacle;
   - choosing a gait for a speed.

---

## 3. Every sense of the real Go2

The real robot (Unitree Go2 product pages and datasheets), what the simulated Go2 has today, and how each sense gets
added to the simulation (MuJoCo MJX 3.14):

| real Go2 sensor | what it is | in the simulation today | how it is added |
|---|---|---|---|
| front camera | HD 1280×720, 120° wide angle | no | One ray per ommatidium from the head with `mjx.ray`, coloured by the surface hit and shaded by distance. MJX's batched renderer needs MuJoCo Warp, which does not import on Kaggle; rays do not. |
| 4D LiDAR L1 (Go2 Pro; the EDU has the L2) | 360° × 90°, 21,600 points/s | no | A ring of `mjx.ray` distance rays covering 360° × 90° from the LiDAR's mount on the head. |
| foot-end force sensors (Go2 EDU) | how hard each paw presses | no (the brain only feels the paws indirectly, through how its joints move and the IMU) | One `touch` sensor site per paw (MJX supports TOUCH). |
| IMU | angular velocity, acceleration, orientation | yes | unchanged |
| joint encoders | 12 joint angles and speeds | yes | unchanged |
| microphone | voice and sound | no (the words arrive directly) | Sound sources in the world: a squeaking ball, a whoosh from a thrown object, the speaker's voice. Loudness falls off with distance, with a left/right difference, so the brain can learn direction. The words stay as heard words. |
| battery | charge level | no | A battery that drains with mechanical power, and recharges on a charging pad (its food). |
| motor temperature | per-motor heat (the real Go2 limits torque when hot) | no | Each motor heats with torque² and cools slowly. A hot motor gets weaker, as on the real robot. This works like tiredness. |

**Camera resolution: fly-eye resolution.** The full 120° view goes in, sampled at about the number of ommatidia in a
fly's two eyes (about 1,500).
- Nothing in the scene is left out; it is seen at the resolution a fly brain is built for.
- The full 1280×720 feed is 921,600 pixels per frame. That is far more than the fly's visual system takes in, and far
  too heavy to train on free GPUs.
- Measured first on Kaggle: cost per step of 1,500 rays × the scene's shapes × 512 dogs. If too slow, fewer rays, with
  the measured cost in the logbook.

**Every sense stays anonymous within itself.** Pixels, rays and paws are shuffled, sign-flipped and rescaled by the
fixed wiring. The brain must discover which pixel looks where, just as it discovered which line was which joint.

---

## 4. Wiring each sense to the fly brain area built for it

A real animal is born with each sense wired to its brain area; the eye reaches the visual brain by development, not by
learning. What is learned is everything *within* a sense. So the senses enter the matching fly neurons:

| Go2 sense | fly counterpart | MaleCNS neurons it enters |
|---|---|---|
| camera, LiDAR | compound eyes, optic lobes | the optic-lobe input and visual projection types (lobula columnar LC and LPLC, including the looming detectors LC4 and LPLC2) |
| IMU | halteres (the fly's gyroscopes) and the neck's balance senses | the mechanosensory types (already in the core) |
| microphone | Johnston's organ in the antenna (the fly's ear) | the antennal mechanosensory (AMMC / JO) types |
| foot force | leg touch and load sensors | the leg mechanosensory types (already in the core) |
| battery, motor heat | internal state: hunger, energy, fatigue | the central-brain neurosecretory and internal-state types |
| words | (none; a fly has no language) | central and descending neurons, as now |

**Growing the brain without erasing it.** The current core (`data/malecns/core_v1.npz`, `brain/core.py`) deliberately
**left the optic lobes out**, because no body had a camera yet.
- A new core (`core_v2`) adds:
  - the optic-lobe types the eye and LiDAR need;
  - the Johnston's organ and internal-state types;
  - every real connection between them and the existing 2,500 types, with signs kept.
- New units and connections are appended at the end, so every learned time constant, bias and connection gain is kept
  (`carry_over` grows tables along their first axis and keeps the learned rows).
- **The test that the growth did no harm:** every known skill is checked before and after, with the new senses present
  but not yet used.

**The fly's "mutations" (fly speed in a dog's body).** A fly escapes a looming object through the giant-fiber pathway
(LC4 and LPLC2 to DNp01) in roughly tens of milliseconds; a dog reacts in roughly 150–250 ms. These are typical values
from the literature, to be confirmed against the papers before we compare with them.
- The brain acts at 50 Hz, so 20 ms is the floor.
- **Reaction time is measured**: from the moment an object starts to loom to the first motor change, and to the moment
  the body is out of its path.
- These are compared with the fly's and the dog's typical values.
- Because the looming detectors and the descending escape pathway are real wiring in the core, a fast escape may need
  little learning. That is a testable prediction.

---

## 5. Thinking and deciding, inside the fly brain

No external AI decides anything. **The fly's own circuits do the thinking.**

| capability | how it is built | fly circuit it relies on |
|---|---|---|
| goals ("go to the ball") | The word sets a goal; the brain must find the ball with its eyes, orient, approach and stop. Reward: getting closer, and touching it. | central complex (heading, navigation, steering) |
| reacting to the unexpected | Objects are thrown at random moments while it is doing something else. Reward: not being hit, and getting back to the task. | looming detectors to descending neurons |
| choosing between competing things | The ball and a threat at the same time; a tired dog far from its charger. | central complex and mushroom body (valence, action selection) |
| learning what is good and bad | Things that hurt (a hard object) and things that pay (the ball, the charger) are learned from experience, not told. | mushroom body (learned valence) |
| memory | The ball rolls behind an obstacle. Does it go to where the ball went? | recurrent central circuits (the rate model's state) |
| multi-step plans | "Get the ball, then come back": two goals in order, held by the brain itself. | central complex and recurrent state |
| reporting its own state | **A voice from the brain itself:** a learned readout from its own neurons to words (the same way the motor readout works), trained to name what it senses, does and needs ("ball left", "tired", "fallen"). Tested on states it was never trained to name. | central and descending neurons |

---

## 6. The functional ingredients of consciousness

**Nobody can currently test whether anything is conscious, and this project does not claim to make it so.** What it
builds, and **measures**, are the functional ingredients that the main scientific theories tie to consciousness:

| ingredient | theory that emphasises it | built as | measured by |
|---|---|---|---|
| one unified picture of world and body | global workspace (Baars; Dehaene) | every sense feeds one shared, recurrent brain state | **Cross-sense use:** the decision to dodge uses both sight and LiDAR, so it degrades when either is blocked. **Broadcast:** a seen object changes activity in motor, internal-state and voice neurons, not only in visual ones. |
| a model of itself | predictive processing; self-model theory | the prediction head extended to predict what it will *see and feel* when it moves | prediction error for its own movements versus for things that move by themselves (does it know what it caused?) |
| attention | attention schema theory (Graziano) | the attention that carries lines into the brain, under competition | which of two stimuli wins, and whether the voice reports the one that won |
| its own drives | homeostatic and affective theories (Damasio; Panksepp) | battery, motor heat, curiosity | behaviour changes with its internal state: a tired dog rests, a low battery seeks the charger |
| memory | (needed by all the theories) | recurrent state, learned valence | the hidden-ball test; avoiding what hurt it before |
| reporting its own state | higher-order and report-based views | the voice readout from its own neurons | accuracy of its reports on new situations (not memorised ones) |

---

## 7. Build order

**Every step rehearses everything already known** and ends with real tests from rest. Every step is a Kaggle job, with
a check every 10–15 minutes of training.

| step | what | done when (observable) |
|---|---|---|
| 0 | **The real-dog run** (`baby_dog_realdog`): watching the real dog. Running on 3 Oct. | real tests and films; distance to the real dog's poses before and after |
| 1 | **Letting go of walking** (`--let_go`): teacher help fades to zero by skill, with a hard deadline; outcome rewards plus an energy cost; jump means all feet off the ground. Starts from the step-0 brain. | With help at exactly 0 for the last 30% of the lesson: every word still works from rest. Report each gait's speed, the foot pattern by speed, falls, and distance to the real dog (measured, not rewarded). |
| 2 | **The full robot:** eye rays, LiDAR rays, paw touch, sound, battery, motor heat. | sensor values checked against the scene in a film (the ball appears where it is, rays end on the wall) |
| 3 | **Growing the brain** (`core_v2`): optic-lobe, Johnston's organ and internal-state neurons added. | every skill unchanged within its test noise, with the new senses on but unused |
| 4 | **Sensory babbling:** it moves and learns to predict what it will see, hear and feel. | prediction error on the new senses falls below the "no change" guess |
| 5 | **The world:** a ball, obstacles, thrown objects; outcome rewards only. | goes to the ball; dodges; does both at once; reaction time measured |
| 6 | **Drives, curiosity and free time.** | behaviour changes with battery and heat; it explores when nobody speaks |
| 7 | **Thinking:** multi-step goals, learned good and bad, memory, and the brain's own voice. | the two-goal test, the hidden-ball test, report accuracy on new states |
| 8 | **Never-taught tests:** get up after a fall, dodge, walk around an obstacle, choose a gait for a speed. | success rates with no demonstration ever given |

After the dog, the same brain learns the human body (Unitree G1). It must then control and tell apart both bodies,
which is the research question of the whole project.

---

## 8. Sources

- [Unitree Go2 (official)](https://www.unitree.com/go2/)
- [Go2 model variants](https://www.docs.quadruped.de/projects/go2/html/Overview_1.html)
- [Go2 brochure](https://static.generation-robots.com/media/brochure-unitree-go2-de.pdf)
- [Go2 EDU Plus](https://www.generationrobots.com/en/404127-go2-quadruped-robot-edu-plus.html)
- Fu, Z., Kumar, A., Malik, J., Pathak, D. "Minimizing Energy Consumption Leads to the Emergence of Gaits in Legged
  Robots." CoRL 2021.
- Zhang, H., Starke, S., Komura, T., Saito, J. "Mode-Adaptive Neural Networks for Quadruped Motion Control."
  SIGGRAPH 2018. This is the dog motion capture (CC BY-NC 4.0, non-commercial).
