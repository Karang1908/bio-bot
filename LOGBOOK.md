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

**The full run** (`baby_dog_letgo`, 60M steps, from the real-dog brain) runs on the corrected body (bug 8.14) and with
corrected meanings (bug 8.13).

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
left/right asymmetries, the new neurons, and hearing neurons no longer receiving body lines. The real measure, the
closed-loop skill test before and after, runs on Kaggle.

**Step 4, built:** in the park, practice also trains the brain to predict its own senses.
- Each line's next change, as in babbling.
- Every eye point's next change, from the optic lobe and an efference copy of its motor neurons.
- Each is scored against guessing "no change".

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
```

Stage order: babble → name → act (×2) → walk_words → walk → start_stop → gaits → no_parent → practice → poses →
practice (all skills together) → practice --dog (the real dog) → practice --dog --let_go. Films come from `--set stage=film_walk`.
