# Logbook: teaching the fly-connectome brain to control the Go2 dog

This is the running record of every experiment in the "one brain, many bodies" work: what was tried, what came out,
what it meant, and what went wrong. The newest results are in §6, and the next steps in §7. The plan for the dog's
senses, self-discovery and mind is in `DogMind.md`.

- **Code:** `train/brain_dog.py` (body, brain, teacher, early runs); `train/baby_dog.py` (the baby curriculum);
  `train/dog_mocap.py` (real dog motion capture retargeted onto the Go2);
  `scripts/kaggle_job.py` (runs them on Kaggle).
- **Results:** `results/` (ignored by git: logs, JSON curves, every stage's brain `.npz`). The final films are in
  `videos/`.
- **Compute:** Kaggle free GPUs (2× T4, or 1× P100). All numbers below come from Kaggle job logs or films.

---

## 1. The setup (what stays the same in every experiment)

**Brain.** It is built from the MaleCNS fly connectome (`brain/core.py`): 2,500 cell-type units and 167,630 real
type-to-type connections, with neurotransmitter signs kept. It is a rate model with learned time constants, biases and
per-connection gains.

- It meets a body through a set-based interface: anonymous input and output lines, any number of them, and no
  parameter whose size depends on the body.
- Each input line gets adaptive normalisation. A *causal self-map* (how each input changes after each output changes)
  is built during each life.
- Attention carries the lines into the fly's sense neurons and out of its motor neurons. The readout is "vector": each
  motor neuron contributes a learned vector.
- Goals and words enter the central-brain and descending neurons.

**Body.** Unitree Go2 (MuJoCo Menagerie, simulated with MJX). The brain sees 34 sensor lines (joint encoders, IMU) and
drives 12 motor lines. Every line is shuffled, sign-flipped and rescaled. A command of 0 holds the motor where it woke up.
The brain acts at 50 Hz; physics runs at 250 Hz.

**Meaning reward** (the body's own scoring):
- go the asked way;
- stand still when asked;
- turn the asked way;
- keep the trunk up and level;
- small costs for effort and jitter;
- a cost for falling.

**The teacher** is the AI layer: code written between runs, which knows what "walk" or "sit" means. It never gives the
brain line names. It finds its own demonstrations by practising on the body first: a trot, a sit pose, a lie-down pose.
From 3 Oct it only teaches, and lets go completely. Every decision, goal and plan comes from the fly brain itself.

---

## 2. Decisions (in the user's words)

| date | decision |
|---|---|
| 25 Sep | "I didn't want you to teach the dog the commands, I wanted you to teach purely the semantics, and let the fly brain learn the body itself." The per-body PPO dog controller was removed. |
| 25 Sep | "we don't wanna give brain that 'these are 30 things'... Self learning." Inputs and outputs became anonymous and are reshuffled every life. |
| 27 Sep | Teach "like real biology. You tell the child a lil about walking, or you show them, assist them, teach them, and then they improve", with an AI agent on top as the teacher. |
| 27 Sep | "no check once that is the brain even learning from the teaching or not ... quick 15 min test". From here on, every step is a short job with its own check. |
| 27 Sep | "the human baby learns what an arm is by itself and then teacher gives it a label ... from scratch, like a baby". This became the staged curriculum. |
| 27 Sep | "teach it one body from scratch, then the same brain ... learns a human body ... control and distinguish both bodies, thats the entire plan". The dog comes first, with a learning check every 10–15 min. |
| 27 Sep | "teach it and dont remove it, its gonna mess up the brains wiring". Keep everything it trained with (option C): its wobble, and anything else present during training, stays at runtime. |
| 27 Sep | "experiment by removing the harness, that just means that the parent is gone". |
| 28 Sep | "make sure the dog now behaves like a proper biological dog". This added independent skills: balance, poses, getting up, free time. |
| 28 Sep | "always work on the main branch only from now". |
| 3 Oct | "train the dog everything. Every single thing, make it mirrored to the real dog". The choices were a real animal dog, and real dog motion capture. |
| 3 Oct | Decision-making, goals ("run to the ball"), dodging, "response to stimuli", with the fly's fast reactions as a "mutation"; and "see the entire robot, we will give that robot consciouness". All of the Go2's real sensors get added. |
| 3 Oct | "i dont want you to train the dog fully how to walk, i want you to teach the initial steps, and then let it figure out its own body". **The teacher fades to zero, and its own way wins** over looking like the real dog. |
| 3 Oct | "the ai layer is not an external layer, its the fly's brain only, but inside a dog now". The full plan is in `DogMind.md`. |

---

## 3. Phase A (25–26 Sep): semantics only, reward-driven PPO. **Result: it never walked.**

The brain got anonymous lines, reshuffled every 30 s life, plus a goal. It was trained by recurrent PPO on the meaning
reward alone. Kaggle `bio-bot-brain-dog`, versions 1–11.

| attempt | change | result |
|---|---|---|
| 1 | first run | The critic's value loss started at 795k. Fixed with a normaliser floor and clipping. |
| 2 | the motors' zero was mid-range | It fell every 0.7 s. Fixed so that zero means "hold still where it woke up". |
| 3 | correlated exploration noise and partial credit, 60M steps | It learned to stand still, and never moved. |
| 4 | reward restructured and a floor on the noise, 200M steps | It still stood still. |
| diagnosis | goal influence on the motor neurons measured | Exactly 0: spectral-radius scaling of the connectome killed the signal within one hop. Fixed with input-average normalisation, goals fed into the descending neurons, and a gate. |
| gate stop | motor neurons silent (rate 0.0001) | A starting bias of 0.3 wakes them. The goal gate was set to 5e-4, goal_scale to 2.0, gain to 4.0. |
| 5 (v11) | 150M steps | Gate passed (0.0029). Falls dropped to about 0, but "forward when asked" stayed at −0.01 m/s. Evaluation: −0.01 to −0.07 m/s, 0.19–0.75 falls per life. |

**Conclusion:** the reward only described what walking *achieves*. It never produced one example of forward travel to
learn from, so standing still was the safe optimum.

---

## 4. Phase B (27 Sep): an AI teacher that shows, assists and lets go. **Result: it copied, but did not walk alone.**

`train/brain_dog.py` with `make_teacher`. Kaggle `bio-bot-brain-dog`, versions 12–25.

| run | what | result |
|---|---|---|
| v12–v14 | The teacher practises its own trot on the body (a grid over rhythm, stride and lift; later "every foot must lift" and "don't crouch"). | The first walk dragged its rear legs, so the rule became every foot must lift. Chosen: a 3 Hz trot at 0.52 m/s, which also turns at 1.5 rad/s and backs at 0.3 m/s, with no falls. The trunk runs at about 72% of standing height, so "upright" was redefined as above 65%. |
| v15 | 150M steps: teacher's hands plus harness plus corrections, faded out | It stood still. The let-go signal ("track") scored standing still (0.69) above walking (0.22), so help was removed too early. The copy error never fell below the do-nothing level (0.043). |
| v17 | **Copy test:** can the brain copy the teacher at all? (3 readouts, 15M steps each) | Yes. The "rates" readout reached 0.13× the do-nothing error; "vector" and "line" reached 0.01×. The brain *can* learn from the teacher. |
| v18 | Watch first (15M), then practise, with let-go on real progress | While watching: 0.03×. Alone, the watched brain fell 35 times per life. In practice, progress rose from 0.35 to 0.52 with hands on, then dropped to 0 once the hands came off. |
| v19–v20 | The teacher's rhythm follows the legs (tested couplings 0 / 0.05 / 0.2) | Coupling 0.2 is fine for the teacher's own walk. |
| v21 | Full run with the in-step teacher | The same collapse: walking came only from the teacher's hands. |
| v23 | **Teach check (15 min):** does it learn while it is in control? | No: 1.00× the do-nothing error before, and 0.98× after. |
| v24 | The teacher follows the brain's own commands | Still 0.98×. |
| v25 | **Control: the same body presentation every life** (no reshuffle) | **0.46×.** It learns once its body stops changing. |

**Conclusion:** the teaching wasn't the blocker. Rediscovering a reshuffled body every 30 s was. This led directly to
the baby curriculum (one body, one fixed wiring, learned once).

---

## 5. Phase C (27–28 Sep): the baby curriculum. **Result: the dog walks, turns, backs, sits and lies down on words.**

`train/baby_dog.py`. One brain, one fixed random wiring (`WIRING = 20260927`). Each stage is a short job whose brain is
passed to the next stage through the private Kaggle dataset `bio-bot-brains`.

Every stage below uses the brain's motor wobble (±0.1): it learned with it, so it keeps it. Anything the brain learned
with stays on in tests.

### 5.1 Learning the body and its words

| stage | what is taught | the stage's own check (brain alone) | before → after |
|---|---|---|---|
| 1 babble (15M) | Joints twitch while the trunk is held; the brain predicts each sensor's next change. | prediction error ÷ "no change" guess | 1.04 → **0.63**. Most of the drop came in the first 2M steps. |
| 2 name (10M) | "move <leg> <joint>". The teacher's hands fade by 60%, then it only corrects. | moves the named joint (chance 8%) | 2% → **94%**. It learned mostly *after* the hands were gone. Untaught pairs: 0%, so it memorised pairs rather than words. |
| 3 act (10M + 15M) | lift / swing-forward / swing-back / put-down, for single legs and diagonal pairs | pose error ÷ doing nothing | lift **0.08**, swing-forward **0.55**, swing-back **0.62**, "move" 100%. The untaught "lift rear-left" went 1.14 → 0.57, the first sign of combining words. |
| 4a walk by words (15M) | The teacher talks it through a trot: lift, swing, other pair. Harness on. | forward speed when talked through | Already +0.142 m/s before this lesson (from stage-3 skills alone), **+0.167** after (79% of the teacher's +0.211). The teacher's spoken walk does *not* balance without the harness. |
| 4b "walk" | One word, with the teacher counting the steps aloud and then fading out | lesson measure (mid-activity) | −0.02 → +0.205 m/s. **Caveat:** the brain filmed afterwards was the wrong file (bug 8.1), so how this brain behaved from rest was never actually tested. |
| 4c start/stop (20M) | Long "stand", then "walk" from rest, on the same first beat | **Real test:** fresh start, stand 2 s, walk 8 s, stand 3 s | −0.022 → **+0.257 m/s** (teacher +0.278), 0 falls. It stops on "stand". Filmed: +0.243 m/s. |
| 4d gaits (25M) | turn-left, turn-right, back, plus walk (harness on) | real test for each gait | walk **+0.272**, turn-left **+0.60 rad/s**, turn-right **−0.72**, back **−0.063** (teacher +0.278 / +0.62 / −0.65 / −0.098). 0 falls. The new words were added without wiping the old ones. |

### 5.2 Without the parent, then practice

| stage | what | real test, no harness | before → after |
|---|---|---|---|
| 5 the parent leaves (25M) | The harness fades out by half the lesson; practice continues without it. | walk / turns / back; falls per 13 s | Right after letting go: walk +0.16 m/s, turns ±0.54–0.60, back −0.07, **1.0–2.1 falls**. After practice: walk +0.18, falls 0.7–1.5. The teacher's corrections can't teach balance. |
| 6 reward practice (40M) | PPO on the meaning reward. A fall costs 2 s of good behaviour. Shoves about every 5 s. The teacher is kept as a light guide (0.5). The wobble is its exploration noise. | same | walk 0.12 → **0.40 m/s**, turn-left **0.96**, turn-right **−0.94**, back **−0.34**. **Falls 2.1 → 0.05.** It outran the teacher's trot (0.27). |
| 7 poses (20M) | sit, lie-down, then "stand" to get back up. The teacher found the poses by practice. | nose-up (sit), trunk height (lie-down), then back up | sit nose −0.03 → **+0.47 rad** (teacher +0.48); lie-down trunk 0.85 → **0.40** (teacher 0.37); back up to 0.91; ~0 falls. **But the gaits were not rehearsed and were forgotten** (walk 0.39 → 0.03). |
| 7′ practice round 2 (40M, from the forgetful brain) | everything together, plus a four-feet stepping reward (mean over feet) | same | Walking never came back (+0.06). Back reached +0.50. It found a loophole: holding one foot up earned the stepping reward. **Discarded.** |
| **8 practice round 3 (50M, from stage 6)** | everything together from the start: stand, 4 gaits, sit, lie-down. Stepping reward = the *laziest* foot. Pose guide weight 5. | same | **walk +0.648 m/s, turn-left +1.09 rad/s, turn-right −1.06, back −0.44, sit +0.45 rad, lie-down trunk 0.37, back up 0.84; falls 0.02 per 13 s.** Share of time each foot is in the air: walk 0.18/0.31/0.12/0.04 → **0.39/0.48/0.33/0.34**. The rear legs step now. |
| 9 get up (50M, from stage 8) | Falls stay down; a quarter of lives start fallen; the teacher's guide is off while down. | fallen on its side or back, told "stand": share on its feet after 1/2/4/8 s | **0% → 0% at every checkpoint.** Practising this also cost walk 0.65 → 0.40 and back 0.44 → 0.29. **Discarded.** Brain kept in `results/baby_dog_stages/getup_try1/`. |

### 5.3 The current best brain: stage 8, practice round 3

Filmed on 28 Sep (`videos/`). No harness, from a standing start, then back to "stand":

| told | result | video |
|---|---|---|
| walk | +0.634 m/s. A diagonal trot: FL+RR alternate with FR+RL, and all four feet leave the ground. | `videos/dog_brain_walk.mp4` |
| turn-left | +1.11 rad/s, turning on the spot | `videos/dog_brain_turn-left.mp4` |
| back | −0.44 m/s | `videos/dog_brain_back.mp4` |
| sit, then stand | nose up +0.46 rad, trunk at 0.60 of standing height, back up to 0.85 | `videos/dog_brain_sit_then_stand.mp4` |
| lie-down, then stand | trunk at 0.38, level, back up to 0.85 | `videos/dog_brain_lie-down_then_stand.mp4` |
| (teacher's own trot) | +0.264 m/s; its rear legs drag | `videos/dog_teacher_walk.mp4` |

0 falls in every clip. The brain file is `results/baby_dog_stages/stage8_practice3.npz` (git-ignored; also the current
`baby_dog.npz` in the Kaggle dataset `bio-bot-brains`).

---

## 6. Phase D (3 Oct): the real dog, then letting go

### 6.1 Real dog motion capture on the Go2

**Data.** The MANN dog motion capture (Zhang, Starke, Komura, Saito, SIGGRAPH 2018; CC BY-NC 4.0, non-commercial): 51
BVH files of a real dog.
- `train/dog_mocap.py` reads them and solves the Go2's leg joints in closed form from where each real paw is.
- It labels each stretch from the motion itself (speed, turning, trunk height and tilt, feet on the ground), and writes
  `data/dog_mocap/go2_dog_motions.npz` at 50 Hz (git-ignored; a copy is in the private Kaggle dataset
  `bio-bot-dog-mocap`).
- Kine2Go, a ready-made Go2 retarget, was tried first and rejected: it had duplicate and mislabelled clips.

| behaviour | seconds recorded | real dog's mean speed / turning |
|---|---|---|
| stand | 732 | 0 |
| walk | 218 | +0.19 m/s |
| pace | 203 | +0.71 m/s |
| canter | 319 | +0.48 m/s |
| run | 83 | +2.19 m/s |
| turn-left / turn-right | 50 / 52 | +1.38 / −1.18 rad/s |
| sit | 218 | trunk at 0.73 of standing, nose up 0.82 rad |
| lie | 253 | trunk at 0.43, level |
| jump | 73 | +0.11 m/s, all four feet off the ground |

**Can the Go2 simply replay it?** (Kaggle `bio-bot-baby-dog-mocap`; 8 stretches per behaviour)
- Kinematic playback (no physics) looks like a dog in every behaviour.
- Played through the motors with physics on and no balance, it falls:
  - falls per 6 s replay (a fall puts the body back, so it can fall again): stand 0.6, walk 0.8, pace 1.3, canter 4.3,
    run 5.1;
  - sit 7.3 (it tips backward), lie 4.9, jump 3.9, turns 4.1–4.7.
- The replayed walk barely moves (+0.02 m/s against the dog's +0.17).

**Conclusion:** a real dog's joint angles are not a controller for a different body. The recordings can only be
something the brain watches and is pulled toward while it keeps its own balance. They can't be the teacher's hands.

### 6.2 Real-dog practice (`baby_dog_realdog`): four new behaviours, and it moves 4× more like a real dog

- **Setup:** from the stage-8 brain, 60M steps, learning rate 1e-4, every earlier skill rehearsed.
- **Word meanings:** each word means what the real dog does, at the dog's speed (walk, pace, canter, run, turns, sit,
  lie-down, jump; "back" keeps the teacher's meaning).
- **Guide:** the real dog's next frame from the closest-matching recorded pose, so it is motion matching.
- **Style reward:** paid for being near some real-dog pose of that word.
- **Tests:** every test also reports the distance to the real dog's poses.

**Results** (60M steps, 3 Oct; real tests from rest, no help).

The log scored the moving words as turns (bug 8.13), so the forward speeds below come from `practice.json`:

| told | before | after | real dog |
|---|---|---|---|
| walk | +0.65 m/s | +0.50 | +0.19 |
| pace (new word) | −0.04 | **+0.58** | +0.71 |
| canter (new) | +0.03 | **+0.55** | +0.48 |
| run (new) | +0.01 | **+0.68** | +2.19 |
| jump (new) | −0.03 | +0.15 forward, but not off the ground | +0.11, all four feet in the air |
| turn-left / turn-right | +1.13 / −1.08 rad/s | **+1.72 / −1.50** | +1.38 / −1.18 |
| back | −0.44 | −0.30 | (−0.25, the teacher's) |
| sit (nose up) | +0.46 rad | +0.38 | (+0.48, the teacher's) |
| lie-down (trunk) | 0.38 | **0.85: lost** | (0.37) |
| distance to the real dog's poses (walk / run) | 0.054 / 0.090 | **0.014 / 0.023** | 0 |

- **What worked:** it learned four new behaviours from the recordings, and its leg poses are 4× closer to the real dog's.
- **The cost:**
  - **lie-down was lost.** In this mode only the joint poses were rewarded, never where the trunk ends up, and the
    lie-down pull from the recordings can't be followed on this body;
  - falls rose a little (0.03–0.14 per 13 s);
  - getting up stayed at 0%. The body then sank through the floor when tipped (bug 8.14).
- The brain is kept as `results/baby_dog_stages/stage10_realdog.npz` (md5 918a366d…), and is now `baby_dog.npz` in
  `bio-bot-brains`.

### 6.3 Step 1 of `DogMind.md`: letting go (`--let_go`)

**The finding behind it.** Since stage 6, the teacher's correction never left. Every practice adds `GUIDE = 0.5` of
"be like the teacher" to the brain's update, 5.0 for rest poses. In the real-dog run, the guide is the real dog's next
frame plus a style reward. So the walk was still mostly the teacher's walk.

**What `--let_go` does:**
- **Help fades by skill.** Each word's help (its guide, the style reward, the four-feet rule) drops 20% whenever the
  dog does that word at least 90% as well as its best. It comes back (×1.25) if the dog falls below 70%.
- **A hard deadline.** Help is zero for every word from 70% of the lesson on.
- **What stays for good:**
  - what each word achieves (speed, turning, stillness, the trunk of a sit or lie-down, all four feet off the ground
    for "jump");
  - not falling;
  - an energy cost of 0.01 per W/kg of mechanical power.
- **New measures:**
  - which feet lift together (diagonal = trot, same-side = pace, front/rear pair = bound or gallop);
  - the mechanical power while practising;
  - each word's help at every checkpoint.

**Checked on the Mac (numpy only):**
- The foot measure gives diagonal 1.0 for a synthetic trot, same-side 1.0 for a pace, and pair 1.0 for a bound.
- The help schedule drops for an improving word, comes back for a collapsing word, handles negative scores, leaves an
  unused word alone, and is zero at the deadline.

**Smoke run** (`baby_dog_letgo_smoke`, 3M steps, from the stage-8 brain): it ran end to end, and the help reached zero
for every word.
- Nothing collapsed: the turns, sit and lie-down held.
- Walk slowed from 0.65 to 0.48 m/s, toward the real dog's 0.19. Nothing pays for going faster than asked, and moving
  now costs energy.

**The full run** (`baby_dog_letgo`, 60M steps, from the real-dog brain, on the corrected body (bug 8.14) and with
corrected meanings (bug 8.13)). **The teacher is gone, and the dog kept every skill.** Help was zero for every word from
update 2,563 of 3,662, so the last 30% had no teacher at all. Real tests from rest:

| told | before (real-dog brain) | after | real dog |
|---|---|---|---|
| walk | +0.50 m/s | +0.32 (slowed toward the asked speed) | +0.19 |
| pace | +0.58 | **+0.78** | +0.71 |
| canter | +0.55 | **+0.54** | +0.48 |
| run | +0.68 | **+0.99** | +2.19 |
| turn-left / turn-right | +1.74 / −1.48 rad/s | +1.89 / −1.47 | +1.38 / −1.18 |
| back / jump (forward) | −0.29 / +0.15 | −0.34 / +0.21 | −0.25 / +0.11 |
| sit nose-up | +0.39 | **+0.48** (recovered) | (teacher +0.48) |
| lie-down trunk | 0.85 | 0.72 (recovering slowly) | (teacher 0.39) |
| on its feet 8 s after a fall | 0% | **1–2%**: the first ever, never trained | |
| falls per 13 s | 0.02–0.06 | 0.00–0.02 | |
| distance to the real dog's poses (walk / run) | 0.014 / 0.023 | 0.057 / 0.083 | |

- **Its own way won**, as decided: the speeds moved toward what each word asks, and the poses drifted from the real
  dog's.
- **The feet:**
  - walk, pace and canter lift diagonal pairs together (trot-like);
  - running starts to pair the front feet (0.11) and the same-side feet (0.07), with every foot in the air 34–38% of
    the time.
- **Not yet:**
  - lie-down is only part-way back;
  - "jump" still never leaves the ground;
  - "run" reaches 45% of the real dog's speed.
- Brain: `results/baby_dog_stages/stage12_letgo.npz` (md5 5cb02e1f…), now `baby_dog.npz` in `bio-bot-brains`.
- **Films of the current dog** (the grown let-go brain, no help, wobble on, from a standing start):
  `videos/dog_selftaught_<word>.mp4` for walk, pace, canter, run, turn-left, turn-right, back, jump, sit_then_stand and
  lie-down_then_stand. Filmed:
  - walk +0.27 m/s, pace +0.60, canter +0.54, run +0.90;
  - turns +1.63 / −1.36 rad/s, back −0.29, jump +0.24 forward;
  - sit nose-up +0.47, lie-down trunk 0.72 (part-way).

**The risk, from phase B.** In v18, removing the teacher's *hands* collapsed the walk. Here, nothing moves the legs:
the brain has acted alone in every practice since stage 6, and only the extra pull in its update is removed.

---

### 6.4 Step 2: the park and every sense of the real robot (`train/dog_world.py`)

**The park.** A fenced 12 × 12 m square with:
- grass with patches, so the ground streams past the eye;
- three trees and the owner;
- a ball;
- a charging pad, the dog's food;
- an object thrown at the dog about every 10 s (when switched on).

Trees, owner, pad and ball each get their own sixth of the circle around the dog, so nothing starts inside anything
else.

**The senses:**
- **eye:** the Go2's 120° × 90° camera at a fly's acuity (5°), 24 × 18 points, in blue and yellow as a dog sees,
  retinotopic;
- **LiDAR:** 360° × 90° at 10°, 36 × 9 distances;
- **ears:** two, with three pitch bands;
- **nose:** two nostrils, two smells (the real Go2 has no nose; a dog does);
- **paws:** a touch sensor on each;
- **skin:** on the trunk and head;
- **battery;**
- **heat:** the temperature of each of the 12 motors.

Hot motors and a flat battery weaken the legs. Time is compressed so the battery lasts minutes.

**Checked** (`--set stage=world_check`; CPU pictures in `results/dog_world/`):

| check | result |
|---|---|
| the MJX eye and LiDAR against MuJoCo's own C ray caster, 4 scenes | **100% of points identical**; largest LiDAR difference 0.00002 m |
| cost, 256 dogs on one T4, one 20 ms step | plain body 11.7–12.3 ms; in the park 21.0 ms; all the senses 2.9 ms |
| paw sensors, dog standing | 149 N in total = the dog's weight (15.2 kg × 9.81) |
| throws, first try | **every dog's physics blew up (NaN)** about 3 s after a throw (bug 8.15) |
| throws, after the fix (8 dogs, local MJX) | 0% blew up; 88% felt (hardest 4,042 N with the 1-iteration solver; 105 N with 4 iterations), 100% heard coming |
| throws, after the fix, 256 dogs on a T4 | 0% blew up; 96% passed within 20 cm of the trunk; 90% felt; 100% heard coming. Skin spikes reach 41 kN with the Go2 model's 1-iteration solver, 1.2 kN with 4 iterations (37.6 ms per step instead of 22.0). Whether a spike shoves the dog or only shows on the skin is measured before throws enter training. |

### 6.5 Step 3: a bigger brain with both sides, eyes, ears, a nose and an inner state (`core_v2`)

`brain/core.py build_v2`:
- **Two sides:** one unit per fly cell type **per side**. The v1 core merged the two sides, so nothing in it could tell
  "on the left" from "on the right".
- **The senses, by the nerve they arrive on:**
  - body: 309 units;
  - vision, the visual projection neurons fed by the eye module: 681;
  - hearing, Johnston's organ: 55;
  - smell, the olfactory receptor neurons: 144;
  - inner state, endocrine cells such as IPC, LK and DH44: 30.
- **New circuits:**
  - the central complex (navigation);
  - the mushroom body (learning);
  - the antennal lobe (smell).
- **Size:** 7,019 units (v1: 2,500) and 502,438 connections.
- **Carried over from v1:** every v1 type is kept on both sides. 4,884 units map back to v1 units, and 165,553 of v1's
  167,630 connections (98.8%) are represented.

**Growing a trained brain into it** (`grow_brain`, stage `grow`):
- every learned number goes to the v2 units and connections it came from;
- an old neuron's inputs keep their v1 proportions;
- connections from new neurons start at about 1% strength;
- the optic lobe's output starts at exactly zero.

**Local check:** given exactly the sensory stream the trained v1 brain experienced, the grown brain's commands differ
by 0.032 on average, against a command size of 0.124. So it is close but not identical. The causes are the real
left/right asymmetries, the new neurons, and hearing neurons no longer receiving body lines.

**The real measure** (`baby_dog_grow`, from the real-dog brain): the closed-loop test of every word from rest, as it runs
now (wobble on, no help).

| told | v1 brain | grown, plain body | grown, in the park with every sense on |
|---|---|---|---|
| walk / pace / canter / run (m/s) | 0.51 / 0.58 / 0.54 / 0.68 | 0.45 / 0.52 / 0.50 / 0.54 | 0.43 / 0.51 / 0.49 / 0.54 |
| turn-left / turn-right (rad/s) | +1.75 / −1.49 | +1.60 / −1.30 | +1.69 / −1.26 |
| back / jump (m/s) | −0.29 / +0.15 | −0.27 / +0.15 | −0.27 / +0.13 |
| sit nose-up (rad) / lie-down trunk | +0.38 / 0.85 | +0.38 / 0.85 | +0.37 / 0.85 |
| falls per 13 s | 0.02–0.12 | 0.02–0.12 | 0.02–0.19 |

**The grown brain keeps every skill.**
- The largest changes: run −0.14 m/s, turn-right 0.19 rad/s slower. Everything else is within 0.07.
- Living in the park with eyes, ears and a nose (all still unused) changes almost nothing.
- Leftovers of the meaning bug (8.13) are still visible, e.g. pace turns at −0.66 rad/s. Letting go retrains them.

**Growing the let-go brain** (`baby_dog_grow` v3; its "before" numbers match the let-go result, so it is the right
brain):
- v1 → grown, plain body: walk 0.32 → 0.29, pace 0.76 → 0.63, canter 0.54 → 0.54, run 0.99 → 0.89, turn-left
  1.84 → 1.62, turn-right −1.49 → −1.33, back −0.34 → −0.29, jump 0.21 → 0.25, sit +0.47 → +0.48, lie-down 0.70 →
  0.71;
- the same in the park with every sense on;
- the largest change is 0.22 (turn-left).

Kept as `results/baby_dog_stages/stage13_grown_letgo.npz` (md5 a239dd9a…), the start of the park run `baby_dog_park`
(60M steps; rehearses everything; with thrown objects, "ball", "come" and self-prediction).

**The park before any practice in it** (`baby_dog_park_smoke`, the grown real-dog brain; real tests from rest):
- told "ball": 11% reach it, and on average they drift away (2.46 → 3.34 m);
- told "come": 2% reach the owner;
- an object thrown while standing: hit 83%, against ~96% for a dog that doesn't move. 11% moved before it arrived
  (median 320 ms after the throw).

The run then crashed on its first update: the new-life reset assumed every brain state is 2-D, and the eye's last frame
is not. Fixed for any shape.

**The first park run** (`baby_dog_park`, 60M steps, 8 h, from the grown let-go brain, with throws, "ball" and "come";
real tests from rest):

| measure | before | after |
|---|---|---|
| told "ball": reached it | 3% | 2%: **not learned** |
| told "come": reached the owner | 6% | 0%: **not learned** |
| object thrown while standing: hit | 83% | 84%: **no dodging** |
| predicting its next body / hearing / smell / sight (÷ "no change") | 0.33 / 1.02 / 0.55 / 0.87 | **0.31 / 0.90 / 0.54 / 0.74**: every sense better than guessing |
| walk / pace / canter / run (m/s) | 0.30 / 0.63 / 0.54 / 0.82 | 0.21 / 0.51 / 0.38 / 0.64 |
| turn-left / turn-right (rad/s) | +1.60 / −1.36 | +1.62 / −1.07 |
| jump (forward), sit, lie-down | 0.27, +0.48, 0.72 | 0.02, +0.49, 0.74 |
| falls per 13 s | ~0 | ~0 |

- **Step 4 is done:** it predicts what every sense will do next better than guessing "no change".
- **Step 5's goals failed.** The likely reason: vision never reaches its decisions.
  - The optic lobe's output into the brain starts at zero.
  - Only the reward could open that path, and the reward is too weak and noisy to train vision.
  - The self-prediction trains the optic lobe's columns, but not the path from them into the central brain.
  - Hearing only hears the ball when it moves, and the two nostrils are 3 cm apart.
- **The fix being tried:** the voice. Learning to *say* "ball on the left" trains that path, as a parent pointing at
  a ball teaches a child to see it. If its ball-left / ball-right accuracy stays at chance, the information is not
  getting in.
- **Also found:** every practice run restarted the teacher's help at full, so this run briefly had a teacher again.
  Fixed: a brain that has been let go starts with the teacher at zero, permanently.
- Brain: `results/baby_dog_stages/stage14_park.npz` (md5 4f41909f…). Next run: `baby_dog_park2`, the same plus drives
  and the voice.

**The second park run** (`baby_dog_park2`, 60M steps, 8 h; with drives and the voice; the teacher stayed at zero):

| what it says about itself (balanced accuracy, 0.5 = chance) | start | end | where the information comes from |
|---|---|---|---|
| ball left / ahead / right | 0.50 / 0.50 / 0.53 | 0.53 / 0.50 / 0.51 | eye, LiDAR: new neurons |
| owner left / ahead / right | 0.50 / 0.50 / 0.52 | 0.55 / 0.50 / 0.50 | eye, LiDAR: new neurons |
| hungry | 0.50 | 0.51 (peak 0.60) | battery: new endocrine neurons |
| **moving** | 0.55 | **0.80** | joints and IMU: old neurons |

- **No goals or drives were learned:**
  - ball 2% → 0%, come 0% → 3%, hit 81% → 83%;
  - a hungry dog went to its food 0% of the time;
  - free-time exploring stayed at ~1–2 m² in 10 s.
- **Gaits recovered some speed:** walk 0.20 → 0.29, pace 0.48 → 0.59, canter 0.36 → 0.41, turn-right 0.99 → 1.08 rad/s.
  Their poses came back toward the real dog's (walk 0.067 → 0.026).
- **Lost:** back (0.22 → 0.08 m/s) and lie-down (trunk 0.74 → 0.93). Falls rose to 0.06–0.20 per 13 s.

**Why (bug 8.17):** the voice learned only what old neurons carry. The trained brain shows why:

| new sense | connections into the old brain | strength after training |
|---|---|---|
| vision | 7,753 | 1.2% |
| smell | 267 | 1.2% |
| inner state | 27 | 1.2% |
| body, hearing (old neurons) | | 92–96% |

My growth rule started every new connection at ~1% "to keep the old skills". At that strength, learning can barely
move them (gradient ~1%), so **the dog was blind, could not smell, and could not feel its hunger where it decides.**

**Fix:** new connections start at full strength, and new neurons are silent at rest (bias 0), so they speak only when
their sense drives them. `--open_senses` applies this to an already grown brain. Checked locally on the park2 brain,
told "ball", 4 dogs: how much its commands differ with eyes open versus shut went from 0.0045 (3% of the command size)
to **0.039 (22%)**, and the brain stays stable. Next run: `baby_dog_park3`, 35M steps (GPU budget), with the senses
open.

**The third park run** (`baby_dog_park3`, 35M steps, 5.3 h, from the park2 brain with the senses opened; no teacher):

| measure | start | end |
|---|---|---|
| voice: ball-left / ball-right | 0.50 / 0.56 | **0.58 / 0.58** |
| voice: owner-left / owner-right | 0.59 / 0.51 | 0.59 / 0.56 |
| voice: hungry | 0.50 | 0.65 at best |
| voice: ball-ahead, owner-ahead, hit | 0.50 | 0.50 (it always says "no") |
| told "ball" / "come": got there | 0% / 2% | 0% / 2% |
| object thrown: hit | 84% | 84% |
| hungry, nobody speaks: went to its food | 0% | 0% |
| walk / pace / canter / run (m/s) | 0.27 / 0.43 / 0.24 / 0.44 | 0.34 / 0.42 / 0.34 / 0.48 |
| turn-left / turn-right / back | 0.91 / 0.74 / 0.01 | 1.47 / 1.10 / 0.16 |
| falls per 13 s | 0.05–0.20 | **0.2–0.44** |

- **What the fix did:**
  - Opening the senses let what the eye and the battery carry into the central brain: the voice is above chance for
    the first time.
  - But only weakly (0.58–0.65), and not yet into what it does.
- **The cost:** opening the senses first knocked its skills back (park2's end against this run's start: pace
  0.59 → 0.43, canter 0.41 → 0.24, turn-left 1.42 → 0.91, back 0.08 → 0.01). Practice won most of it back, but falls
  rose.
- **"Ahead" and "hit" stay at exactly 0.50:** they are rare, and an unweighted naming loss lets "never" win.
- Brain: `results/baby_dog_stages/stage16_park3_open.npz`.

**What this says about the approach.** Everything is learned at once, by reward, through a 7,019-neuron recurrent brain:
- seeing;
- turning toward something;
- walking to it;
- dodging;
- eating;
- naming its state;
- keeping 13 words.

The reward for "ball" never gets a chance to train seeing, because nothing it does is ever caused by what it sees.

**Next, a puppy's order:**
1. **Seeing, through naming:** the naming loss weighted much more, with rare states weighted up, until "ball on the
   left" is said reliably.
2. **Looking:** "ball" pays for turning to face it before going there.
3. **Going to it,** then dodging, then eating when hungry.

Every lesson keeps rehearsing all the old words.

**Step 6, built and smoke-tested** (`baby_dog_drives_smoke`, 2M steps from the grown let-go brain; it ran end to end):
- **Free time:** nobody speaks, no word is heard.
- **Hunger:** some lives start with a low battery. Charging on the pad pays FOOD_PAY × hunger, and a flat battery
  costs.
- **Curiosity:** in free time, it is paid for learning progress, i.e. its own sight surprise falling (a fast average
  dropping below a slow one). Surprise itself is not paid, so unpredictable noise is not attractive. The bookkeeping
  lives in the brain's own state.
- **Two loopholes closed before running:**
  - a fall used to reset the battery, so falling could "feed" it;
  - switching to silence made the owner's voice sound.
- **Results:**
  - self-prediction against "no change": body 0.44 → 0.37, smell 0.62 → 0.60, sight 1.14 → 1.00, hearing 1.9 → 1.6
    (still worse than guessing);
  - before practice in free time, a hungry dog never went to its food (0%), and a dog nobody speaks to barely moves
    (0.9–2.2 m² in 10 s).
- **Hunger was too strong:** lives that started at 5% battery starved from the first second, the reward per second
  went to −0.35…−1.1, and the critic's error jumped from ~1 to 17–30. Now lives start at ≥10% battery, and starving
  costs 1/s.

**Step 4, built:** in the park, practice also trains the brain to predict its own senses.
- Each line's next change, as in babbling.
- Every eye point's next change, from the optic lobe and an efference copy of its motor neurons.
- Each is scored against guessing "no change".

### 6.4 Self-learning continuation (10 Oct): focused perception, then goals

**Owner's constraints:** entirely free compute; no checks or training on the laptop. The primary goal is
**self-learning**, not a controller whose movements or decisions are supplied by a teacher.

**Prepared, not yet verified on Kaggle:**
- Resume stage16_park3_open with teacher help fixed at zero (`--self_learning --let_go`). Never pass `--open_senses`
  again. Abort on mismatched checkpoint history, wiring, vocabulary or parameter shapes instead of resetting learning.
- Focus lessons in order: `perception`, `orient`, `approach`, then `all`. The environment, all senses, wobble, throws,
  battery and heat remain. Every lesson rehearses standing, all gaits and poses. Only the new objectives' training
  signals are staged; no motor demonstrations are added.
- Naming targets describe the state the brain sensed **before** its action, not the next state or a new scene after
  a fall reset. Predictions and naming gradients now respect the critic-only warm-up.
- Measure naming on scripted test scenes separate from the training rollouts. A class absent from those scenes is
  reported as not sampled, not as chance-level evidence. Tests reset recurrent state on the same new-life events as
  training. Task success excludes the period before the task was requested.
- Full evaluations are configurable; a short 2M-step pilot uses two to limit overhead, and saves its final checkpoint
  even when the step count is not divisible by the evaluation interval.

**Prediction before running:** stronger, balanced naming supervision may improve the weak visual representation.
This is believed, not established. A 2M-step pilot establishes whether this path runs without resetting skills;
it does not prove mastery. Do not advance to navigation until held-out ball bearings are reliable, and do not promote
a checkpoint that materially loses locomotion or raises falls. Every promotion needs the real tests from rest.

**Infrastructure:** the previous park3 job's status was COMPLETE and its output listing includes `baby_dog.npz`.
Automatic approval review rejected a new upload of the local brain as an unapproved sensitive-file export. That upload
did not happen. The continuation can attach the brain already on Kaggle through `kernel_sources`, without a local
checkpoint upload or replacing `bio-bot-brains`. Input identity will be checked and hashed on Kaggle.

**Self-learning scope:** this continuation learns from experience through PPO during Kaggle practice. Continuing to
learn after deployment requires a separate online learning mechanism; autonomous inference alone is not that claim.

**Submitted:** `karang1908/bio-bot-baby-dog-selflearn-perception-pilot`, version 1, 2M steps, 512 environments,
two full post-start evaluations, lr 5e-5, lesson `perception`, all park flags on, `--self_learning --let_go`.
Input is the prior park3 kernel output (no local model upload), with expected last checkpoint `after 35.0M steps`.
Kaggle accepted the job and reported QUEUED. No training result is available yet. Live logs and the status waiter
are remote API clients only; they do not run model checks or training on the laptop.

While queued, the live-log endpoint returned `500 Server Error: Internal Server Error` for
`/v1/kernels/logs/stream/karang1908/bio-bot-baby-dog-selflearn-perception-pilot`. This is a log-service failure,
not a training result. A free CPU preflight (`train/dog_preflight.py`) is prepared to check source syntax, remote
checkpoint identity, v2 parameter shapes, finite parameters and zero teacher help entirely on Kaggle.

**CPU preflight verified on Kaggle:** COMPLETE, all checks passed. It loaded the prior job's actual file at
`/kaggle/input/notebooks/karang1908/bio-bot-baby-dog-park3/baby_dog.npz`, 7,019 units and 502,438 connections;
SHA256 `4af773f477e7a173e74904d66d82943227ab2b96b5e2ee1454dc7921a7f0bf41`. Sources compile, parameter shapes
match the v2 core, all parameters are finite, and every recorded help value is zero. Report:
`results/baby_dog_selflearn_preflight/preflight.json`. These are source/checkpoint checks, not evidence of new skills.

**Free quota verified through the Kaggle API:** GPU used 0.00h, remaining 30.00h; TPU remaining 20.00h; next refresh
17 Oct 2026 00:00 UTC. The GPU pilot is still QUEUED while this is written.

After the T4 pilot remained queued for over 12 minutes, a second bounded trial was accepted in the free P100 queue:
`karang1908/bio-bot-baby-dog-selflearn-perception-p100`, 2M steps, 256 environments, seed 1. Other learning settings
and the already-hosted input are identical. Per-device environment count stays 256, as in the 2×T4 trial, to avoid
doubling memory on a single GPU. This is an independent repeat, not a checkpoint promotion. Both trials together
are a small initial expenditure of the verified free quota; no larger continuation is submitted before results.

Code review found a further monitoring issue: the installed CLI prints a failure message **after** the ERROR status,
but our waiter read only the last line. It could therefore wait forever on a failed job. `parse_status` now reads
the actual status line and preserves the reported failure. Prepared for a Kaggle CPU test with representative
QUEUED, COMPLETE, ERROR-with-message and CANCELLED responses; no tests run on the laptop.

**Updated CPU preflight verified:** version 2 completed on Kaggle. The source checks, checkpoint checks and all
four status-parser cases passed (`results/baby_dog_selflearn_preflight/preflight.json`). The original status clients
were stopped to reload this correction; only API monitoring stopped, not the Kaggle jobs. Replacement monitors will
fetch each trial's outputs after its terminal status, including failure logs. They run no model checks or training.

At the latest status read the T4 trial had remained QUEUED for 20 minutes and the P100 repeat for over 5 minutes.
No new learning or retention result can be claimed while Kaggle has not allocated either GPU worker.

Replacement API-only monitors are running with logs under
`results/kaggle/baby_dog_selflearn_perception_pilot/monitor.log` and
`results/kaggle/baby_dog_selflearn_perception_p100/monitor.log`. Each waits for the remote terminal status and then
fetches outputs, with overwrite enabled. There is no automatic promotion or larger training submission: the next
lesson depends on inspecting these results and checking retention. The dog curriculum remains incomplete.

Further static review caught a launch setting error before the P100 worker started: `--set seed=1` became
`seed: 1.0`, because the generic settings parser converted every number to float and `JOB_ARGS` bypasses argparse's
type conversion. JAX requires an integer random seed. Prepared `parse_value` to preserve integer literals and added
Kaggle CPU cases for seed, negative integers, learning rate, scientific step count and a string. The T4 job uses the
default integer seed and is unaffected.

**Verified:** CPU preflight version 3 passed on Kaggle, including settings types. The corrected P100 submission was
refused with `Kernel push error: Maximum batch GPU session count of 2 reached.` It was **not** accepted; the original
P100 version still occupies its slot and must finish or be cancelled before replacement. The T4 job changed to
RUNNING at about 17:48 UTC; a worker starting is not yet evidence of model updates or new learning. The attempted CLI
transfer interrupted the local API monitors; they were restarted after the owner said to continue. No model work ran
on the laptop. The launcher now treats a `Kernel push error:` as failure even when Kaggle exits zero.

The live-log endpoint also returned `500 Server Error: Internal Server Error` after the T4 worker was RUNNING.
The one-shot logs response was empty. Status monitoring still works; until outputs are persisted, neither model
update counts nor evaluation scores are visible. RUNNING must not be reported as verified skill improvement.

At 18:05 UTC, quota was GPU used 0.35h, remaining 29.65h; TPU remaining 20.00h. Both workers are now RUNNING.
The P100 source pulled back from Kaggle still contains `seed: 1.0`, confirming the refused push did not apply the
correction. JAX 0.7.2's official `threefry_seed` implementation rejects non-integer dtype. A temporary browser tab
could not access the private job in that browser session and was closed; no browser cancellation occurred. API
monitors remain responsible for fetching terminal logs. HANDOFF now records the stricter no-local-checks rule and
pending continuation so another session does not repeat the old workflow.

**Prepared: absolute drive levels, not yet verified.** Static review confirmed that `fresh_discovery` sets every
input's mean to its initial value, then `sense` reports only deviations from that adaptive mean. Different steady
initial charges are therefore identical on the battery nerve; steady motor temperatures have the same problem.
This removes information a learner needs and is not solved by more PPO steps.

The optional `--absolute_drives` appends 13 anonymous level nerves (charge plus 12 motor temperatures), preserving
all original 62 nerves, their random draws, adaptive features and model parameters. Temperature's instrument unit
is 100 °C; no hungry/hot label or threshold is an input. The additional levels use fixed smooth bounding rather
than adaptive centring. They still pass through the same learned brain and participate in its causal self-map.
The flag is saved in brain metadata and stays enabled on continuation once experienced. It is off in the running
pilot, which remains a controlled test of the earlier source. Prediction: hungry/full and cool/hot will become
distinguishable, but changes in attention may disturb existing skills. A new GPU trial and retention tests are
required before promotion. CPU preflight will first check steady levels, signed/scaled levels and unchanged
established adaptive features; no checks are run on the laptop.

**CPU verification passed on Kaggle:** preflight version 4 completed. The actual `level_features` helper
distinguishes steady charge and temperature, retains them over repeated samples, keeps established adaptive
values exactly, and preserves distinctions through anonymous signed/scaled inputs. Source compilation, checkpoint
identity, shapes, finite parameters, zero teacher help and launcher parser cases also passed. This verifies the
input feature mathematics and source syntax, not the new body wiring at runtime or learned behavior; a GPU trial
with full retention tests is still required. Current report: `results/baby_dog_selflearn_preflight/preflight.json`.

**Actual wiring verified on Kaggle CPU (preflight version 5):** using JAX 0.7.2 and the actual source's
`new_life`/`jnp_sign` functions, every established sensor's permutation, sign, scale, offset and modality matched
bit-for-bit before/after adding the 13 nerves; motor signs/tags also matched. Added nerve indices are disjoint and
retain different steady charges/temperatures through the actual anonymous wiring. The original seed problem was
reproduced exactly: `PRNG key seed must be an integer; got JitTracer<~float32[]>`. Integer seed 1 works. These tests
used only the Kaggle CPU, with no simulation or training. The full new body/brain behavior still needs a GPU trial.

A 2M-step level-sense trial is prepared in `results/kaggle/baby_dog_selflearn_levels_pilot/submit.sh` from the same
original stage16 and with the same seed/environment count as the legacy T4 pilot. It has **not been submitted**.
The launcher now supports Kaggle's official `--timeout` option; this trial will be capped at 10,800 seconds (3h)
so an unresponsive setup cannot use the whole weekly free quota. The existing sessions' limits were set at
submission. A short CPU submission will verify the timeout option before the next GPU trial uses it.

**Runtime-cap submission verified:** CPU preflight version 6 was accepted with `--timeout 600`, completed and passed
source/checkpoint/feature/parser checks. The actual JAX wiring and seed reproduction report from version 5 is
preserved separately as `results/baby_dog_selflearn_preflight/preflight_wiring_v5.json`; those unchanged wiring
checks were not repeated in version 6. At 18:53 UTC both GPU jobs still reported RUNNING. The complete raw status
responses contained no failure messages, and the live-log endpoint again returned 500 for both. No new learning
result is available or promoted. GPU worker status alone does not show whether setup, compilation or updates are
currently executing.

### 6.6 Self-learning pilot results (11 Oct, Dubai): retain stage16

**Verified, fetched with overwrite:** the legacy T4 perception pilot completed its 2M-step lesson. The input
SHA256 matches the original stage16 above; every baseline and post-training help value is zero. At 1M steps,
walking falls improved, but backward motion weakened. The final 2M result regressed substantially:

| measure | before | after 1M | after 2M |
|---|---:|---:|---:|
| walk falls / 13 s | 0.422 | 0.266 | 1.031 |
| run falls / 13 s | 0.375 | 0.297 | 0.828 |
| canter falls / 13 s | 0.219 | 0.250 | 0.688 |
| backward speed, m/s | -0.155 | -0.079 | -0.045 |
| hungry balanced accuracy | 0.606 | 0.618 | 0.580 |
| ball-ahead balanced accuracy | 0.500 | 0.597 | 0.548 |
| owner-ahead balanced accuracy | 0.500 | 0.525 | 0.583 |
| moving balanced accuracy | 0.679 | 0.712 | 0.745 |

Hit naming reached 0.931, but this is not learned dodging: the dodge test still had 70% hits. Hungry food-seeking
remained 0%; get-up remained 0% after 2–8 seconds; hot/fallen naming had no positive examples in these held-out
scenes and remains unmeasured. Full results are in `results/baby_dog_selflearn_perception_pilot/practice.json`.
**Rejected for promotion:** stronger naming did not preserve locomotion. The original stage16 remains best.
Only the final NPZ was kept by that job, so its intermediate 1M brain cannot be recovered from its JSON scores.

**P100-requested repeat failed before training**, as predicted from its saved source. Terminal log:
`TypeError: PRNG key seed must be an integer; got JitTracer<~float32[]>`.
There is no trained checkpoint in `results/baby_dog_selflearn_perception_p100`. The corrected parser is already
verified on Kaggle CPU; no repeat of the invalid version is needed. Both previous GPU slots have terminated.

**Next bounded trial:** restart the original hosted stage16, enable the previously CPU-verified absolute drive
senses, reduce lr from 5e-5 to 1e-5, and limit training to 1M steps with tests at 0.5M/1M. Keep each tested checkpoint
separately instead of overwriting the only earlier brain. This changes two factors, so it tests the combined path;
it cannot assign improvements to either factor alone. Prediction: smaller updates should reduce forgetting and
steady level inputs should make initial hunger distinguishable. Neither learned behavior is established yet.
No promotion or advance to orientation/navigation without retention and reliable perception. The trial retains all
old senses, wobble, teacher-zero rehearsal, park flags and the original input hash. Runtime cap remains 3h.
CPU preflight is being rerun for the small checkpoint-preservation edit before submission. No laptop checks or
training ran. Kaggle quota read at 22:12 UTC on 10 Oct: GPU 1.01h used / 28.99h remaining; TPU 20h remaining.

CPU preflight version 7 failed with `KeyError: 'expect_checkpoint'`: the submission omitted its required setting.
No GPU trial was started. Its failure log is saved as `results/baby_dog_selflearn_preflight/log_failed_v7.txt`.
The old version-6 report was still present locally because an output download does not delete old files absent
from a failed run. It was explicitly moved to `preflight_v6.json` so it cannot be mistaken for a current pass.
The corrected CPU submission restores `expect_checkpoint=after 35.0M steps`.

**Verified:** CPU preflight version 8 completed and passed. Current baby source SHA256:
`76ec3e75236a78da35f266e7bde3fd84370e97d40bc969e9ea860faa95d6eb26`; brain/world/client hashes remain unchanged
from version 6. Input identity, parameter preservation, zero help, feature mathematics and parser checks passed.
GPU job `karang1908/bio-bot-baby-dog-selflearn-levels-pilot`, version 1, was then accepted: 1M steps, lr 1e-5,
512 environments, integer seed 0, perception lesson, absolute drive senses, two post-start evaluations, timeout
10,800 seconds. This is a submitted experiment, not a verified improvement. An API-only waiter will fetch its
terminal outputs under `results/baby_dog_selflearn_levels_pilot`; monitor log is under `results/kaggle/`.

At 22:20 UTC the new job was RUNNING, and unlike the previous jobs its live-log stream worked. It reports JAX 0.7.2
on two CUDA devices; RESUME has the exact original input SHA256, integer seed 0, teacher-left stage, wobble 0.1,
`open_senses: false`, and `absolute_drives: true`. Parameter carry-over reports `new parts: none`. Setup reference
calibration has started. These are verified launch/preservation facts, not learned movement or perception results.

**Live 0.5M result (23:00 UTC):** the level-sense pilot completed its baseline and half-million-step tests with
every teacher-help value still zero. The new-sense baseline walk/back speeds were +0.333/-0.177 m/s; at 0.5M they
were +0.353/-0.175 m/s. Walking falls changed 0.42 -> 0.39, pace 0.45 -> 0.41, canter 0.23 -> 0.25, run 0.25 -> 0.33,
turn-left 0.23 -> 0.33, turn-right 0.31 -> 0.31, back 0.50 -> 0.50 per 13 s. This preserves backward motion much
better than the legacy pilot so far, but does not establish lower falls across all skills. Held-out hungry accuracy
was 0.57 -> 0.60; ball-ahead 0.50 -> 0.54; ball-right 0.64 -> 0.57; owner bearings remain weak. Hit naming improved
to 0.82, while actual dodge hits were 84%. Food-seeking remains 0% hungry, recovery remains 0%, hot/fallen naming
remains unsampled. The final 1M result and fetched JSON are pending; no checkpoint is promoted from live logs.

**Colab fallback proposed by owner:** official VS Code extension `google.colab` 0.9.7 is installed (package metadata
read only; no cells or laptop checks executed). No extra MCP/CLI is needed for ordinary remote notebook execution.
Use one Colab account alongside the existing Kaggle account, keeping verified checkpoints between sessions.
The proposal to rotate 4–5 Gmail accounts for uninterrupted quota is not adopted: the
[official Colab FAQ](https://research.google.com/colaboratory/faq.html) expressly prohibits multiple accounts to
work around access/resource limits, and free GPU availability/session lengths are variable (12h maximum, not a
guarantee). [Kaggle's terms](https://www.kaggle.com/terms) also prohibit multiple active Kaggle accounts. Neither
provider gives guaranteed nonstop free compute. A Colab notebook must use a remote GPU kernel and keep all
experienced senses/flags; checkpoint identity and retention must be checked on cloud compute. No Colab runtime,
model transfer or Colab training has been started. Quota at 23:00 UTC: GPU 1.77h used / 28.23h remaining; TPU 20h.

### 6.7 Level-sense pilot completed; checkpoint selected for continuation (11 Oct)

**Verified terminal result:** `baby_dog_selflearn_levels_pilot` completed; its API-only monitor fetched all outputs
with overwrite. No laptop model checks ran. It kept checkpoints at 491,520, 983,040 and 999,424 steps, in addition
to the final `baby_dog.npz`. Rounded test labels give both last checkpoints the text `after 1.0M steps`; use the
exact filename, hash and partial count to distinguish them.

| measure | new-sense baseline | 983,040 steps | final 999,424 steps |
|---|---:|---:|---:|
| walk falls / 13 s | 0.422 | 0.312 | 0.297 |
| pace falls / 13 s | 0.453 | 0.359 | 0.406 |
| canter falls / 13 s | 0.234 | 0.219 | 0.281 |
| run falls / 13 s | 0.250 | 0.328 | 0.297 |
| turn-left falls / 13 s | 0.234 | 0.266 | 0.359 |
| turn-right falls / 13 s | 0.312 | 0.266 | 0.266 |
| back falls / 13 s | 0.500 | 0.453 | 0.453 |
| backward speed, m/s | -0.177 | -0.162 | -0.155 |
| hungry balanced accuracy | 0.567 | 0.580 | 0.588 |
| ball-ahead balanced accuracy | 0.500 | 0.550 | 0.540 |

These are the same scripted validation scenes, not proof of generalization. Perception remains weak, food-seeking
and ball/come success are ~0%, recovery after 2–8 s is 0%, and lie-down/jump remain unlearned. No navigation lesson
or global-best promotion is justified. The smaller updates avoided the legacy pilot's broad collapse, but some
fall rates still worsened. The original stage16 remains preserved as the established best.

**Selected learning candidate:** `baby_dog_step_0000983040.npz`, the intermediate with the fewest aggregate falls
across the seven known movement/turn/back skills. It also retains more backward motion than the final checkpoint.
Free Kaggle CPU job `baby_dog_selflearn_levels_review` version 1 completed and passed: all learned parameters
finite, v2 shapes/words/voice/wiring preserved, teacher help zero, `absolute_drives: true`, `partial: 60/61`.
Its exact hosted path is
`/kaggle/input/notebooks/karang1908/bio-bot-baby-dog-selflearn-levels-pilot/baby_dog_step_0000983040.npz`;
SHA256 `18f6e62923d7b1c832fe6731784af830c37843d8fcd855b52d2944bd956b8dc1`.
Report: `results/baby_dog_selflearn_levels_review/preflight.json`.

**Prepared continuation, not yet submitted:** 2M more perception steps, lr 1e-5, 512 environments, integer seed 1,
four evaluations and separate snapshots, 3h runtime cap. Seed changes only fresh draws/experience and the critic;
all learned parameters are restored, and the body's wiring stays 20260927. It attaches the existing hosted pilot
output and names the selected file explicitly; no local checkpoint upload. Prediction: more varied experience
may improve naming without the previous broad forgetting. This remains an experiment; do not advance based on
elapsed steps alone, and do not promote a regressed checkpoint. All experienced senses/flags remain enabled.

The owner clarified the proposed resource pool as one Kaggle plus five Colab accounts. Colab has no fixed published
30h weekly allowance, so one account of each does not guarantee 60h; the account-rotation restriction in the linked
official FAQ still applies. No account rotation, Colab training or new MCP/CLI installation was performed.

**Submitted after verification:** `karang1908/bio-bot-baby-dog-selflearn-levels-continue` version 1 accepted with
the prepared settings above. Its API-only monitor waits and fetches under the matching `results/` job directory.
This continues the learning candidate; it does not promote it over stage16 or claim that perception is mastered.

**11 Oct 05:37 UTC status request:** the latest local monitor entry is RUNNING at 47.0 minutes (03:58:46 Dubai).
The direct current status request was denied permission to read the private kernel; a separate quota read reports
`Authentication required to call the Kaggle API.` No continuation output directory has been downloaded. Current
completion/failure is therefore unverified, not RUNNING by assumption. Refresh the Kaggle login with
`.venv/bin/kaggle auth login --force`, then check status and fetch terminal outputs; restart the API-only monitor
if the job is still running. Authentication loss does not cancel a remote job. No model checks or training ran
on the laptop during this status request.

### 6.8 Continuation completed but rejected; outcome-update guard proposed (11 Oct)

The owner renewed Kaggle authentication. `baby_dog_selflearn_levels_continue` is COMPLETE, and all outputs were
downloaded with overwrite. Its resume hash is exactly `18f6e62923d7b1c832fe6731784af830c37843d8fcd855b52d2944bd956b8dc1`,
seed 1 is an integer, absolute drive senses remain on, and teacher movement help is zero at every evaluation.

| measure | before continuation | final 2M |
|---|---:|---:|
| walk falls / 13 s | 0.328 | 0.500 |
| pace falls / 13 s | 0.406 | 0.484 |
| canter falls / 13 s | 0.219 | 0.297 |
| run falls / 13 s | 0.297 | 0.344 |
| turn-left falls / 13 s | 0.297 | 0.375 |
| turn-right falls / 13 s | 0.281 | 0.328 |
| back falls / 13 s | 0.391 | 0.578 |
| hungry balanced accuracy | 0.603 | 0.599 |
| ball-ahead balanced accuracy | 0.534 | 0.525 |
| owner-ahead balanced accuracy | 0.527 | 0.563 |

All seven movement fall rates are worse at the end; intermediate back-fall rates also regressed (0.64–0.81).
Ball success is 1/64, come success 0/64, hungry food-seeking 0/64, recovery 0%, and dodging still suffers 84% hits.
**Rejected for promotion and continuation.** Keep the original stage16 and the previously checked 983,040-step
level-sense candidate. Smaller learning rates slowed the broad forgetting but did not solve it. No new result
establishes self-learning task mastery. Full evidence: `results/baby_dog_selflearn_levels_continue/practice.json`.

**Prepared experiment: `--protect_skills`.** Hypothesis: the combined prediction/naming and outcome update can
push the shared brain away from the rehearsal objective. This mechanism is not yet measured or established as
the cause of the observed falls. A new optional rule obtains the outcome gradient and total gradient from the
same forward pass, applies the usual Adam update, then removes only the brain-step component whose dot product
with the outcome gradient is positive. Projection happens after Adam, so its rescaling/momentum cannot undo that
local constraint. Critic updates are unchanged; every learned brain parameter remains plastic. Existing wiring,
senses, wobble, reward meanings and zero teacher help remain. No movement demonstrations or runtime controller
are introduced. The flag and conflicting-step statistics are recorded in checkpoints/logs.

The constraint is only first-order on the current minibatch's estimated outcome loss. It does not guarantee
retention on later lives or any held-out task. Prediction before running: it may reduce interference while naming
learns in other directions; it may also stall naming, and it increases gradient computation. Test on the same
hosted 983,040-step input, seed 1, lr 1e-5, 2M-step budget and four evaluations as the rejected continuation for
a controlled comparison. Never promote from the mathematical constraint alone.

Free CPU job `baby_dog_selflearn_guard_preflight` is submitted first. It will run the actual helper under JAX jit,
check conflicting/tangent/descent/zero-gradient cases, exercise the two-pullback pattern and critic-only warm-up,
compile the edited sources, and verify the already-hosted candidate's identity/finite parameters/zero help.
All checks are on Kaggle CPU; no model checks or training run on the laptop. GPU experiment remains unsubmitted.

**Verified:** guard preflight version 1 completed on Kaggle CPU with JAX 0.7.2. The actual helper under jit removed
the conflicting component, kept tangent/descent/zero-gradient behavior, and the dual VJP equalled the direct total
gradient. Critic warm-up left the brain unchanged. Source compilation, checkpoint identity, finite parameters,
v2 shapes, words, voice and zero help also passed. Report:
`results/baby_dog_selflearn_guard_preflight/preflight.json`. New baby source SHA256:
`36cc62f4cc63e3e733c05b661f7737b9f7c4c46d4a9baba75941ce4ca22e44c9`;
brain/world/client hashes remain unchanged. These tests use small gradient examples, not a full dog training run
or proof of skill retention. Free quota read: GPU 2.99h used / 27.01h remaining; TPU 20h remaining.

**Submitted after the pass:** `karang1908/bio-bot-baby-dog-selflearn-guard-pilot` version 1 accepted. It matches
the rejected continuation's selected input, integer seed 1, lr 1e-5, 2M steps, 512 environments, four evaluations,
all park flags and 3h cap, with `protect_skills` enabled. Checkpoints and conflict statistics will be retained.
Its API-only monitor logs under `results/kaggle/baby_dog_selflearn_guard_pilot/monitor.log` and fetches terminal
outputs. Actual GPU execution, interference measurements and retention are still pending. No promotion.

**Live GPU observation, first 0.5M evaluation (11 Oct, about 11:05 Dubai):** the stream confirms the intended
input hash, integer seed 1, `absolute_drives: true`, `protect_skills: true`, two CUDA devices, and `new parts: none`.
All printed movement-help values remain zero. Rounded falls per 13 s are walk 0.23, pace 0.38, canter 0.25,
run 0.28, turn-left 0.25, turn-right 0.31, back 0.52. Walking improved versus the matched 0.33 baseline, while
backward motion worsened versus 0.39. Hungry naming is still 0.60; ball/food/recovery remain 0%, come is 1/64.
The two printed update summaries show 0% conflicting steps; they do not establish that the guard has helped.
Evidence is the live stream saved at `results/kaggle/baby_dog_selflearn_guard_pilot/live.log`. These are preliminary,
rounded log values; full checkpoint JSON and the remaining evaluations are pending. No checkpoint is promoted.

### 6.9 Guard trial completed; perception bottleneck and isolated voice lesson (11 Oct)

**Verified:** `baby_dog_selflearn_guard_pilot` version 1 is COMPLETE; the monitor downloaded every output with
overwrite. Input SHA256 is the intended `18f6e62923d7b1c832fe6731784af830c37843d8fcd855b52d2944bd956b8dc1`.
Teacher movement help remains zero throughout. Full results: `results/baby_dog_selflearn_guard_pilot/practice.json`.

| measure | this run's baseline | final 1,998,848 steps |
|---|---:|---:|
| walk falls / 13 s | 0.328 | 0.344 |
| pace falls / 13 s | 0.375 | 0.297 |
| canter falls / 13 s | 0.219 | 0.266 |
| run falls / 13 s | 0.313 | 0.297 |
| turn-left falls / 13 s | 0.266 | 0.297 |
| turn-right falls / 13 s | 0.313 | 0.266 |
| back falls / 13 s | 0.344 | 0.484 |
| mean of seven movement fall rates | 0.308 | 0.321 |
| hungry balanced accuracy | 0.591 | 0.601 |
| ball-ahead balanced accuracy | 0.563 | 0.559 |
| owner-ahead balanced accuracy | 0.517 | 0.525 |

Movement retention is much better than the previous continuation's 0.317 -> 0.415 mean falls, but backward
stability still worsens, and no tested checkpoint establishes a substantial perception improvement. The first
and 1.5M snapshots both average 0.317 falls; the 1M snapshot is worse at 0.382. All checkpoints are kept; none is
promoted. The logged 1.5M training update shows the guard active on 12% of its eight minibatch steps. This is not
a whole-run conflict rate or proof that projection caused the retention difference. Baselines differ slightly
between runs despite the same input/seed/settings; compare each result to its own baseline and do not treat
small differences in these 64-life tests as established gains.

**Correction to earlier interpretation of task scores:** `lesson=perception` deliberately disables approach,
food reward and curiosity. This pilot also does not enable `get_up`, so falls in practice reset automatically.
Final ball success 0/64, come 2/64, hungry food-seeking 0/64 and recovery 0% show unfinished goals, but are not
evidence that the dog failed a focused training attempt on those goals. The actual lesson's unmet objective is
improved perception. Hot/fallen labels are still absent from held-out positive states and remain unmeasured.

**Next bounded training experiment, prepared before running:** a temporary voice-readout lesson from the
previously verified 983,040-step level-sense candidate. The last pilots used only lr 1e-5 for both the large
shared brain and its voice readout. Give only `say_w` and `say_b` lr 3e-4, with shared/motor lr 0.0 for this short
lesson; the critic still learns at 3e-4. All senses, wobble, words, wiring, hazards and skill rehearsal remain.
This is supervised naming from the brain's own activity, not proof of self-taught navigation or deployed online
learning. No sensor-to-label bypass or external movement controller is added. The shared brain can resume
plasticity in later action lessons; freezing it temporarily is a curriculum decision, not a runtime restriction.

Prediction: if useful charge/location information already reaches central/descending activity, the faster voice
readout may expose it without changing motor weights. If naming stays near chance, increasing readout learning
alone is insufficient; investigate the representation/observation curriculum rather than repeating it blindly.
Plan: integer seed 1, 1M steps, two evaluation intervals, 512 environments, T4 requested, 3h cap. The guard is off
because shared updates are zero. Default optimizer behavior is preserved when `voice_lr` is omitted. Learning
rates are stored in resume/checkpoint metadata. The remote run must write `voice_audit.json` proving every
non-voice parameter array stayed exactly unchanged and all brain parameters remain finite.

Free Kaggle CPU preflight `baby_dog_selflearn_voice_preflight` version 1 is accepted. It tests the actual optimizer
under jit with nested motor parameters: zero shared updates, learning voice/critic, exact default-optimizer
parity and rejection of a missing voice head. Source and input-identity checks also run there. GPU submission
waits for that result. No checks or training are run on the laptop.

**Verified CPU pass:** version 1 completed with JAX 0.7.2 and Optax 0.2.8. The actual optimizer under jit preserved
nested non-voice weights exactly across four updates while both voice arrays and the critic changed. Default
optimizer updates were bit-for-bit identical to the previous construction. Missing-voice rejection, source
compilation, launcher types/statuses, exact input identity, finite parameters, wiring/words/voice/zero help passed.
Report: `results/baby_dog_selflearn_voice_preflight/preflight.json`. Baby source SHA256:
`63d575bad8d18004305e4f764c0fc0c3f2a5ce26bc01f323084c21e3b6cc9156`; brain/world/client sources unchanged.
These are optimizer/source/input checks, not measured perception gains or full dog-training integration.

**GPU submitted after the pass:** `karang1908/bio-bot-baby-dog-selflearn-voice-pilot` version 1 accepted with the
prepared 1M-step configuration. Its API-only monitor fetches terminal outputs automatically; monitor and live
stream logs are under `results/kaggle/baby_dog_selflearn_voice_pilot/`. Actual training, exact-weight audit and
perception results are pending. No automatic promotion, no local training/checks, no Colab job or account rotation.

### 6.10 Voice lesson complete; evaluation correction and active orientation (11 Oct)

The user asked for timed monitoring. The voice job had already completed at 12:34 Dubai, and its API monitor
downloaded every output. Reading the results at 14:23 showed:
- the Kaggle `voice_audit.json` confirms all 55 non-voice parameter arrays stayed exactly unchanged;
- both voice parameter arrays changed and all parameters are finite;
- hungry balanced accuracy 0.596 -> 0.589; owner-ahead 0.522 -> 0.556; ball-ahead 0.549 -> 0.629 at the final
  999,424-step snapshot, but intermediate ball-ahead scores are about 0.50. Faster readout training alone did
  not establish consistent perception improvement. No promotion and no blind continuation.

An important new observation: movement fall rates vary between evaluations despite unchanged motor/shared
weights (e.g. pace 0.344 -> 0.469). Earlier single-group evaluations cannot justify strong claims from small
differences. The cause of run-to-run numerical/simulation variability is not established. Do not infer that
the unchanged brain forgot or improved movement from this voice lesson's fall fluctuations.

**Confirmed source mismatch, corrected:** `make_scripted.life` reset the body at a new life but retained the
old recurrent brain state. Actual practice and the park tests clear recurrent state on `new_life`. Scripted
movement/pose/get-up tests now apply the same shared `reset_brain_state` before the brain tick. No senses,
wobble, wiring or learned parameters change. This correction does not establish the cause of all numerical
variability. Previous movement numbers used the older test protocol; compare future checkpoints to their
own newly measured baseline rather than treating old numbers as directly interchangeable.

The next run uses three independent 64-life cohorts per evaluation (seed offsets 0,100,200), averages test
metrics, sums sampled class counts, and keeps absent-class accuracy null. The voice accuracy is a mean of
cohort accuracies with both classes sampled, not a claim that adjacent frames are independent samples.
Park task reports also record start/end facing cosine and the fraction aligned within 0.35 rad in the last
second, so an orientation lesson is judged by the outcome it actually trains. Falls and distance remain.

**Next bounded outcome-learning experiment:** `lesson=orient` from the established 983,040-step learning
candidate (hash `18f6e62923d7b1c832fe6731784af830c37843d8fcd855b52d2944bd956b8dc1`), not the voice-only output.
1M steps, seed 1, shared lr 5e-6 / voice lr 3e-4, outcome guard enabled, two evaluation intervals, three cohorts,
512 environments, T4 requested, 3h cap. Shared brain plasticity resumes. Teacher movement help stays zero;
every earlier skill is rehearsed. The existing orientation reward pays for turning to face the named object,
without a demonstrated action. Approach, food reward and curiosity remain disabled in this focused lesson;
do not claim it trains navigation/food/recovery. The voice labels are still taught. This also provides active
sensory experience instead of holding all new task rewards off until naming passes an unvalidated threshold.

Prediction before running: task-facing may improve through outcome learning even while verbal reports are
weak; retention and new perception are uncertain. Judge facing alongside falls and paired baselines. Do not
promote automatically. All checks and training remain on free Kaggle compute.

Evaluation CPU preflight v1 had an incorrect input path; v2 corrected it before GPU training. The timer caught
v2 COMPLETE. Its actual jit reset, scripted-call placement, aggregation, compilation and input checks must be
read before GPU submission. The first source patch also failed to match the get-up line and made no edits;
the subsequent patch used the correct context. These failures consumed no training GPU time.

**Verified:** evaluation preflight v2 passed on Kaggle CPU with JAX 0.7.2. Actual reset behavior under jit,
scripted reset-before-tick placement, fresh-cohort aggregation/class-count/null handling, source compilation,
launcher checks and exact input identity/finite parameters/zero help passed. Report:
`results/baby_dog_selflearn_eval_preflight/preflight.json`. Baby source SHA256:
`a45385fe785db0ef0acee0c7a1659ca4aa4bab64514930a4f837754197d4bbcf`; brain/world/client unchanged.
GPU integration and the new facing measurements are still pending. This pass is not proof of perception or
orientation learning, and does not establish that GPU simulation is deterministic.

**Submitted after the pass:** `karang1908/bio-bot-baby-dog-selflearn-orient-pilot` version 1 accepted with the
prepared 1M-step orientation configuration. Status and live streams are under
`results/kaggle/baby_dog_selflearn_orient_pilot/`; terminal results will be fetched automatically. The agent is
staying active with clock-based checks about once a minute, as requested. Execution and learning results pending.

**Timed monitoring outcome:** orientation job v1 completed at 15:52 Dubai after 77.3 minutes of monitor time
(including queue/setup), and every checkpoint/report was downloaded automatically. Initial log streaming
returned HTTP 500 while queued, then worked after RUNNING. A later live connection stopped advancing after
warm-up while status polling continued; reconnecting recovered the 0.5M and 1M evaluations. Fresh connections
were used for remaining checks. One-shot persisted logs were empty during RUNNING. Treat a quiet live stream
as potentially stale; compare against fresh status and reconnect rather than declaring the training stalled.

| averaged metric, three 64-life cohorts | fresh baseline | final 999,424 steps |
|---|---:|---:|
| mean seven movement falls / 13 s | 0.298 | 0.303 |
| walk falls / 13 s | 0.297 | 0.307 |
| pace falls / 13 s | 0.359 | 0.328 |
| back falls / 13 s | 0.391 | 0.391 |
| ball aligned in final second | 8.93% | 10.79% |
| owner aligned in final second | 9.96% | 10.18% |
| ball final facing cosine | -0.075 | +0.003 |
| owner final facing cosine | -0.078 | -0.067 |
| hungry balanced accuracy | 0.572 | 0.572 |
| ball-ahead balanced accuracy | 0.545 | 0.541 |
| owner-ahead balanced accuracy | 0.508 | 0.537 |

All movement help is zero at all evaluations. Retention is largely stable, but target-facing remains weak;
small alignment changes do not establish a useful learned orientation skill. No checkpoint is promoted or
selected for restart. The established stage16 and the verified pre-voice 983,040-step learning candidate remain.
The two final curve labels both round to 1M; snapshots are 983,040 and 999,424 steps. Full evidence:
`results/baby_dog_selflearn_orient_pilot/practice.json`. The final generic help-deadline log line says “since update
42”, but the loaded let-go branch had help zero from the start, confirmed by every help dictionary and summary.

**Next measured question, before more GPU training:** do anonymous steady charge levels alter the sensory and
central/descending activity used by the voice? Free CPU job `baby_dog_selflearn_sensor_path` is prepared with a
20-minute cap. It runs the actual frozen recurrent brain for 200 ticks, with its real wiring and learned weights,
four charge levels (0.1,0.2,0.4,1.0), and stand/quiet word contexts. Other body features are synthetic adapted
zeros; camera is black and LiDAR has no returns. The added heat levels stay at ambient. It measures paired
internal-sense/central/voice/motor responses. No body, actions, optimizer or weight updates are run.

Hypotheses, not conclusions: charge may fail to activate usable pathways, or usable information may be drowned
out by other activity/learning noise. A weak response in this controlled static probe does not prove absent
information in moving scenes or establish a causal learning defect. Do not alter activation functions, reset
learned senses or repeat long GPU runs based on an unmeasured explanation. Probe results are pending.

**Verified CPU response:** the probe completed and its report is in
`results/baby_dog_selflearn_sensor_path/preflight.json`. All 30 internal-sense neurons were active at 4 s in both
contexts. Charge input contrast is 0.357; inner-neuron rate contrast is 0.128 (stand) / 0.148 (quiet). Central
rate contrast grows with time: at 0.2 s max 0.00063 / 0.00039; at 4 s max 0.0101 / 0.2097. Hungry voice-logit
differences at 4 s are only ~0.0002 / 0.0072, and all four charge levels are reported as not hungry then.
Thus charge reaches the actual neural circuit in this static probe. A completely disconnected or wholly silent
internal-sense path is not supported by these observations. Do not “fix” it by resetting gains/biases or changing
activation functions on that assumption. Probe inputs are synthetic; these are not real-scene classification
accuracy or proof of a causal training bottleneck. No weight changes or physics were run.

**Next controlled training hypothesis:** the default recurrent backpropagation window is only 32 ticks = 0.64 s.
Test 128 ticks = 2.56 s, reducing environments from 512 to 128 so each update still has 16,384 total samples.
Same verified starting brain, integer seed, orientation lesson, 1M budget, lr 5e-6 / voice 3e-4, guard, zero help,
three evaluation cohorts and 3h cap. This changes sequence length and the number of concurrent lives, not brain
wiring, activation, senses, wobble or training-source code. Longer response development suggests investigating
credit assignment, but does not prove that a short window caused the stall. Longer history may help, may leave
results unchanged, may cost more wall time, and may still exceed memory despite constant samples/update.
Plan is in `results/kaggle/baby_dog_selflearn_long_history_pilot/submit.sh`; no automatic promotion.

**Submitted:** `karang1908/bio-bot-baby-dog-selflearn-long-history-pilot` version 1 accepted. Its API status
monitor fetches terminal outputs; the agent continues active clock checks. Source/input/seed/lesson/rates/evaluation
protocol are unchanged from the verified orientation run; only unroll/envs differ. Execution/results pending.

**Owner's version-control instruction, 11 Oct:** “also commit and push file by file after every chnage”.
From now on, every changed source/document file is committed individually and pushed to `main` immediately.
The existing pending files are being published the same way. No co-author trailer; generated results, learned
checkpoints and credentials remain ignored. Reading diffs and Git operations do not run source/model checks;
all model validation and training remain on free Kaggle compute. Notebook monitoring continues during Git work.

**Longer-history startup:** status changed from QUEUED to RUNNING at 16:40 Dubai, after about 31 minutes of
queue/monitor time. The live log confirms JAX 0.7.2 on two CUDA devices, the exact `18f6e629…` input hash,
integer seed 1, orientation/self-learning/level-sense/guard flags, three evaluation cohorts and the prepared
learning rates. Carry-over reports `new parts: none`. Package resolver and optional Warp-import warnings appeared;
execution continued through weight carry-over. Fresh baselines, training integration and outcomes remain pending.

## 7. Open problems and next steps

The full plan is in `DogMind.md` §7: let go, the full robot's senses, a larger fly brain, sensory babbling, a world
with a ball and thrown objects, drives, thinking, and the never-taught tests.

1. **Getting up after a fall: not learned.** The teacher had no demonstration, and the ±0.1 wobble is far too small to
   discover a roll-over by chance.
   - Now one of the never-taught tests (`DogMind.md` §2.2): only being back on its feet is rewarded, never how.
   - With curiosity and the energy cost, it must find a way itself.
2. **Free time** (nobody speaks): wander, rest, stay up. Planned as curiosity and drives (`DogMind.md` §2.2, §7 step 6).
3. **Words are only partly understood on their own.** Untaught word pairs work only partly (untaught lift: 0.57×
   pose error).
4. **The human body**, then both bodies in one brain: the research question itself. Not started.
5. **Studio integration:** a mode that runs this brain on the dog with voice or text commands. Not started.
6. **GPU budget:** about 29 of the 30 weekly Kaggle hours were used this week (reset around 3 Oct).

---

## 8. Mistakes and bugs (so they are not repeated)

1. **Stale downloads.** `kaggle kernels output` skips any file that already exists locally with the same size. Twice, a
   new brain of the same shape silently kept the previous run's file:
   - the stage-4b "walking" brain was really the failed first try at 4b;
   - "practice round 3" was really round 2.

   As a result, one film and the first get-up run used the wrong brain. **Fixed:** `fetch` always passes `-o`. Every
   saved brain was checked by content hash and by its own stored stage name.
2. **Removing the wobble at test time.** It stood still. Rule: anything it learned with stays.
3. **A misleading check.** "Walk" was measured mid-activity, not from rest, and gave +0.205 m/s where the real test gave
   ~0. Rule: every checkpoint runs the real scripted test from a fresh start.
4. **Forgetting.** The poses lesson left the gaits out, and walking fell from 0.39 to 0.03 m/s. Rule: every lesson
   rehearses everything it already knows.
5. **Reward loopholes.**
   - The stepping reward averaged over feet, so holding one foot up paid. It now uses the laziest foot.
   - The early "track" signal scored standing still above walking.
6. **Falls.** Lying down was counted as a fall and reset the dog mid-pose. A fall now means tipped over only (on its
   side or back). Separately, the body code's comment claimed a fall "costs ~10 s of standing"; it cost 0.2 s. Practice
   adds its own 2 s fall cost, and the comment is corrected.
7. **Two small crashes:**
   - a typo (`vv` used outside its loop);
   - numpy float32 values not accepted by the JSON writer.

   Both were caught on Kaggle, because nothing is run on the Mac.
8. **Kaggle limits.**
   - At most 2 batch GPU jobs at once; a replaced job can hold a slot.
   - The OAuth login expires after about 3 h.
   - Logs can only be read once a job ends.
9. **Kaggle changed its machine image (Oct 2026)** to jax 0.11.1 on Python 3.13. Installing jax 0.7.2 on top left
   jaxlib 0.11.1 in place, and the mix failed to import (`ImportError: xla_pmap_p`). **Fixed:** the whole set (jax[cuda12],
   jaxlib, mujoco, mujoco-mjx 0.7.2 / 3.14.0) is pinned and checked by version before anything is imported.
10. **A ready-made retarget taken on trust.** Kine2Go (an existing Go2 retarget of dog motion) had duplicate and
    mislabelled clips. The raw MANN recordings were retargeted here instead, and labelled from the motion itself.
11. **Foot contacts too strict.** One ground height for all paws marked most real-dog steps as "in the air". Each paw now
    gets its own ground level (its 5th-percentile height) plus a speed check.
12. **The teacher never really let go.** The harness left in stage 5, but the teacher's correction stayed in every
    practice's update. The "self-taught" walk was mostly the teacher's. Found on 3 Oct; this is what `--let_go` fixes.
13. **Real-dog meanings carried the recordings' drift.** Each word's meaning was the recordings' mean motion. So
    "pace" asked for a −0.13 rad/s turn as well (the dog circled the capture room), and the tests scored every moving
    word as a turn, because its turning part was not zero. **Fixed:** a moving word means forward at the dog's speed,
    and a turning word means turning on the spot.
14. **A tipped dog sank through the floor.** The Menagerie MJX Go2's only trunk collision shape is a 5.7 cm sphere,
    and only the paws touched the ground. Tipped on its side and holding its pose, the trunk ended up 27 cm *under*
    the floor. Getting up was physically impossible, which explains the 0% in stage 9 and in every get-up test since.
    **Fixed:** a box the size of the real trunk, plus the head, now rest on the ground.
    - Checked: standing height is unchanged (0.247 m), and a tipped dog lies at +0.06 m.
    - The teacher's sit is unchanged; the lie-down settles at 0.39 instead of 0.37, with the belly on the ground.
15. **The park's fence was an endless plane.** A collision plane is a half-space. The thrown object, "parked" at
    (50, 50) between throws, sat 44 m deep inside it, and the physics exploded 3 s after every throw.
    - My first local tests ran for exactly 3.0 s and stopped one step before it.
    - **Fixed:** the fence's collision shape is now finite rails behind the visible boards.
    - Lesson: run a check past every timer in the code under test.
16. **A job too big for Kaggle.** Embedding the 1.8 MB `core_v2.npz` made the job script 3.25 MB, and Kaggle refused
    it with a bare `400 Bad Request`. The cores now travel in the private dataset `bio-bot-cores`.
17. **New senses wired in at 1% strength.** To keep the grown brain's skills, every connection v1 never had started
    at ~1% gain. The eye, the nose and the battery then never reached the old brain: after 120M steps of practice
    their gains were still 1.2%. The voice stayed at chance for everything they carry, and was 0.80 for what the old
    body neurons carry. Lesson: a weak start is a closed door, not a gentle one, when the gradient through it is just
    as weak.


---

## 9. How to reproduce

```bash
# stage by stage (each loads the previous brain from the private dataset bio-bot-brains)
.venv/bin/python scripts/kaggle_job.py push baby_dog --embed train/brain_dog.py --embed data/malecns/core_v1.npz \
    --set stage=babble                                  # then: fetch, upload, next stage with --dataset bio-bot-brains
.venv/bin/python scripts/kaggle_job.py push baby_dog --job baby_dog_practice --embed train/brain_dog.py \
    --embed data/malecns/core_v1.npz --dataset bio-bot-brains --set stage=practice --set lr=1e-4 --steps 50e6
.venv/bin/python scripts/kaggle_job.py wait  baby_dog --job baby_dog_practice
.venv/bin/python scripts/kaggle_job.py fetch baby_dog --job baby_dog_practice     # always a fresh download (-o)
.venv/bin/python scripts/kaggle_job.py upload baby_dog                            # results/baby_dog/baby_dog.npz -> dataset

# the real dog: retarget the mocap (light, runs locally), then practise with it (needs the dataset bio-bot-dog-mocap)
.venv/bin/python train/dog_mocap.py                  # data/dog_mocap/*.bvh -> data/dog_mocap/go2_dog_motions.npz
.venv/bin/python scripts/kaggle_job.py push baby_dog --job baby_dog_realdog --embed train/brain_dog.py \
    --embed data/malecns/core_v1.npz --dataset bio-bot-brains --dataset bio-bot-dog-mocap --set stage=practice \
    --flag dog --set lr=1e-4 --steps 60e6
# letting go (DogMind.md step 1): the same, plus --flag let_go
# the park and every sense, checked (step 2); the grown brain (step 3); practice in the park (steps 4-5)
.venv/bin/python train/dog_world.py                   # pictures of what the dog senses -> results/dog_world/
.venv/bin/python -c "from brain.core import build_v2; build_v2()"      # data/malecns/core_v2.npz (dataset bio-bot-cores)
.venv/bin/python scripts/kaggle_job.py push baby_dog --job baby_dog_world --embed train/brain_dog.py \
    --embed train/dog_world.py --embed data/malecns/core_v1.npz --set stage=world_check
.venv/bin/python scripts/kaggle_job.py push baby_dog --job baby_dog_grow --embed train/brain_dog.py \
    --embed train/dog_world.py --dataset bio-bot-brains --dataset bio-bot-dog-mocap --dataset bio-bot-cores \
    --set stage=grow --flag dog
.venv/bin/python scripts/kaggle_job.py push baby_dog --job baby_dog_park --embed train/brain_dog.py \
    --embed train/dog_world.py --dataset bio-bot-brains --dataset bio-bot-dog-mocap --dataset bio-bot-cores \
    --set stage=practice --set core=v2 --flag dog --flag let_go --flag park --flag throws --flag tasks --set lr=1e-4
```

Stage order: babble → name → act (×2) → walk_words → walk → start_stop → gaits → no_parent → practice → poses →
practice (all skills together) → practice --dog (the real dog) → practice --dog --let_go. Films come from `--set stage=film_walk`.
