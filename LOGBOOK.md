# Logbook: teaching the fly-connectome brain to control the Go2 dog

This is the running record of every experiment in the "one brain, many bodies" work: what was tried, what came out,
what it meant, and what went wrong. The newest results are in §4, and the plan in §6.

- **Code:** `train/brain_dog.py` (body, brain, teacher, early runs); `train/baby_dog.py` (the baby curriculum);
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
| 4b "walk" | One word, with the teacher counting the steps aloud and then fading out | lesson measure (mid-activity) | −0.02 → +0.205 m/s. **Caveat:** the brain filmed afterwards was the wrong file (bug 7.1), so how this brain behaved from rest was never actually tested. |
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

## 6. Open problems and next steps

1. **Getting up after a fall: not learned.** The teacher has no demonstration, and the ±0.1 wobble is far too small to
   discover a roll-over by chance. Next: the teacher first finds a roll-over on the body by practice (tuck the legs on
   one side, push with the others), then shows and guides it, as it did for the trot and the poses.
2. **Free time** (nobody speaks): wander, rest, stay up. Not started.
3. **Words are only partly understood on their own.** Untaught word pairs work only partly (untaught lift: 0.57×
   pose error).
4. **The human body**, then both bodies in one brain: the research question itself. Not started.
5. **Studio integration:** a mode that runs this brain on the dog with voice or text commands. Not started.
6. **GPU budget:** about 29 of the 30 weekly Kaggle hours were used this week (reset around 3 Oct).

---

## 7. Mistakes and bugs (so they are not repeated)

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

---

## 8. How to reproduce

```bash
# stage by stage (each loads the previous brain from the private dataset bio-bot-brains)
.venv/bin/python scripts/kaggle_job.py push baby_dog --embed train/brain_dog.py --embed data/malecns/core_v1.npz \
    --set stage=babble                                  # then: fetch, upload, next stage with --dataset bio-bot-brains
.venv/bin/python scripts/kaggle_job.py push baby_dog --job baby_dog_practice --embed train/brain_dog.py \
    --embed data/malecns/core_v1.npz --dataset bio-bot-brains --set stage=practice --set lr=1e-4 --steps 50e6
.venv/bin/python scripts/kaggle_job.py wait  baby_dog --job baby_dog_practice
.venv/bin/python scripts/kaggle_job.py fetch baby_dog --job baby_dog_practice     # always a fresh download (-o)
.venv/bin/python scripts/kaggle_job.py upload baby_dog                            # results/baby_dog/baby_dog.npz -> dataset
```

Stage order: babble → name → act (×2) → walk_words → walk → start_stop → gaits → no_parent → practice → poses →
practice (all skills together). Films come from `--set stage=film_walk`.
