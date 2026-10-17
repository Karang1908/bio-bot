# Agent instructions for bio-bot

These standing project rules come from the owner's instructions and the recorded experiments.
Within project guidance, the latest explicit owner instruction takes precedence over older documentation.
Do not treat historical plans or pending-status notes as current authorization or measured success.

## Read before working

- Read `HANDOFF.md` for the current state, exact checkpoints, running jobs and takeover instructions.
- Read the relevant sections of `LOGBOOK.md`, including mistakes and failed trials, before changing or running
  training. Read `DogMind.md` for the agreed learning plan and `TechnicalDesign.md` / `Requirements.md` for context.
- Keep standing rules here, current state in `HANDOFF.md`, and experiment evidence in `LOGBOOK.md`. Follow a
  narrower documentation request when the owner specifies one, such as updating only the handoff.
- Older instructions allowing Mac tests or training, reshuffling wiring every life, or waiting for permission to
  commit are superseded by the rules below. Older quota numbers and notebook statuses must be verified afresh.

## Objective and learning design

- **Self-learning is the main goal.** The fly brain must discover its body and world and learn useful behavior.
  Complete the robot dog first; then the same brain learns the humanoid and ultimately distinguishes and controls
  both bodies. Do not replace this with separate learned controllers for each body.
- The fly brain is the decision-making layer. Actions, goals, plans and self-reports must come from it. Do not add
  an external LLM or a hand-written runtime controller that thinks or chooses movements for the dog.
- Teach like a baby: body discovery first, then names, actions and skills. The brain is not told which anonymous
  input/output line corresponds to a body part or physical measurement. Anatomical modality routing and ordered
  eye/LiDAR inputs are intentional; they are not permission to expose privileged labels to the runtime policy.
- A teacher may show, assist and name things during initial learning, then its movement help, guide and style
  pull must fade completely. When continuing an existing let-go/self-learning brain, **movement help stays zero
  from the start**. Never restart the teacher at full help.
- After letting go, reward what the dog achieves, rather than prescribing joint motions or gait form. Preserve
  a useful self-discovered movement even when it differs from the real dog's appearance.
- Keep the real-dog goal: balance, gaits, turns, backing, poses, recovery, free time, perception, navigation,
  dodging, drives, memory and decision-making. Do not quietly drop difficult skills or any agreed robot senses.
- Distinguish offline practice, supervised naming, runtime sensory adaptation and deployed online weight learning.
  A completed notebook or a trained voice head does not establish continuous self-learning after deployment.
- Discuss consciousness honestly as measurable functional ingredients. Never claim consciousness or task mastery
  without evidence, or describe a built but unlearned capability as complete.

## Preserve the brain and its experience

- **One brain, one fixed body wiring.** Preserve established input/motor mappings across lives and continuations.
  New nerves must not change the existing random draws, signs, scales, ordering or motor mappings.
- Keep everything the brain experienced while learning at evaluation and runtime: all senses, body interfaces,
  words, learned voice, motor wobble (±0.1), and the relevant environment and hazards. Do not simplify the body or
  remove exploration noise to obtain a better-looking result. The teacher is a training signal and may leave.
- **Every lesson rehearses all earlier skills.** Keep gaits, turns, backing and poses while teaching new goals;
  a previous lesson without gait rehearsal erased walking. Measure retention as well as the new skill.
- Resume the exact intended checkpoint. Preserve learned weights, core, vocabulary, readout, wiring and sensory
  metadata; fail explicitly if a continuation would silently initialize missing or mismatched parameters.
- Do not run `--open_senses` again on an already-open stage16 or later brain: it resets learned gains and biases.
  Do not reset activations or senses based on an unmeasured explanation of a learning failure.
- Keep the established best and learning restart distinct. Preserve intermediate tested checkpoints and rejected
  outputs for comparison. Promote or select a new restart only from measured learning and retention evidence.

## Compute and local-machine limits

- **Fully free compute only: budget $0.** Use free Kaggle CPU/GPU resources; Colab is a free fallback. Do not enable
  paid services, upgrades, subscriptions, credits or billing to continue training.
- **No checks or training on the owner's laptop, including light checks.** Do not run local tests, syntax checks,
  lint/type checks, neural forward passes, simulations, benchmarks or training. Do not create local JAX/MJX
  environments to work around this restriction. All validation and training belong on free remote compute.
- Local file reading/editing, result-JSON analysis, Git operations and remote API/CLI monitoring are permitted.
  Use the existing `.venv` as an API client, not a local model runtime. Inspect active Git hooks before committing
  so a Git operation does not accidentally run prohibited local checks.
- Use one account per provider. Do not rotate Gmail/Colab/Kaggle accounts to evade quotas or restrictions. Do not
  promise fixed Colab GPU hours, a guaranteed combined allowance or nonstop free availability.
- Check current quota (via `.venv/bin/kaggle quota`), authentication and notebook status before planning or submitting.
  Use bounded trials, explicit runtime caps, intermediate checkpoints and remote CPU preflight where needed before spending GPU time.
  Prefer short experiments that can demonstrate learning over long blind continuations.
- Prefer existing hosted checkpoint outputs through `--kernel-source`. Do not routinely upload local checkpoints
  or overwrite the shared brain dataset. An earlier local checkpoint export was rejected by approval review;
  reuse of already-hosted outputs was the accepted alternative. Do not bypass an approval rejection.
- Never commit or expose credentials, OAuth codes or tokens. Keep generated results, datasets, logs and learned
  checkpoints under the existing ignore rules; do not force-add them as part of ordinary source commits.
- Ordinary remote notebook execution already has a Kaggle CLI and the official VS Code Colab extension available.
  Add another MCP/CLI only when a concrete capability is missing; do not install tools speculatively.

## Experiments and evidence

- Predict the expected result before running, separate verified facts from hypotheses/assumptions, and reproduce
  a suspected defect before claiming a fix. Report actual failures accurately and preserve their logs.
- Record the source/configuration, seed, exact input path and hash, flags, learning rates, runtime cap, actual step
  counts, output paths and conclusions. Preserve integer seeds when passing job arguments outside argparse.
- Use fresh scripted tests from rest, without teacher help, with the learned senses and wobble intact. Reset
  recurrent state when the body starts a new life, consistently in practice and evaluation.
- Compare each run to its own fresh baseline. Use independent evaluation cohorts when variation matters. Do not
  confuse adjacent frames with independent lives, missing-class accuracy with zero, or different test protocols
  with interchangeable measurements.
- Judge a lesson by what it actually trains. Disabled approach/food/recovery objectives remain unfinished; their
  low scores do not prove that a focused perception/orientation lesson failed to learn those disabled objectives.
- Distinguish checkpoint filename, hash and actual steps: multiple snapshots can have the same rounded label.
  Fetch outputs with overwrite to avoid the known same-size stale-download problem.
- Do not infer a causal explanation or mastery from small metric changes, training loss alone, one cohort or a
  static synthetic probe. Verify actual checkpoint behavior and retention before claiming success.

## Notebook monitoring and communication

- Keep the owner updated in short, plain language: what changed, what was measured, what failed and what remains.
  Be careful and candid; do not label a whole training goal complete because one job finished.
- When asked to track a notebook, use an active clock timer with checks roughly once a minute, alongside API
  status monitoring and terminal-output fetching. Keep individual blocking waits below 60 seconds.
- Parse actual terminal states, including ERROR with a trailing failure message and cancellation. A CLI exit code
  of zero can still contain a push failure. Do not relaunch blindly, start duplicate GPU jobs, or equate queue and
  setup time with time spent learning.
- Live log streams can become stale while training continues. Compare fresh status and reconnect the local
  reader when output stops advancing; reconnecting the reader must not cancel the remote notebook. One-shot
  persisted logs may be empty while a notebook is running.
- Renew authentication only after an observed authentication failure; do not assume a fixed three-hour expiry.
  If user login is needed, explain the specific failure and ask for the needed login step without exposing tokens.
- Fetch and read terminal reports before declaring results. Record whether monitoring ended, whether any job is
  still active, and whether any checkpoint was promoted. Do not promise agent notifications after ending a turn.
- Stay within the current requested task. Routine engineering decisions are delegated; ask only for genuine
  missing information or a decision that would conflict with a standing constraint. Avoid unrelated diagnostics
  or repeated permission requests for already-authorized work.

## Git and handoff

- **Work on `main` only.** Do not create feature branches or worktrees on another branch.
- **After every file change, commit that file individually and push it immediately to `main`.** The owner has
  already authorized this workflow; do not ask again. Finish that file's commit/push before changing another file.
  Use the repository's style, such as `feat(train): ...`, `docs(logbook): ...` and `docs(handoff): ...`.
- Never add a `Co-Authored-By` trailer or an AI co-author attribution.
- Keep unrelated user changes intact. Do not reset history, force-push or stage other files to simplify a commit.
  Report a failed commit/push honestly rather than claiming publication succeeded.
- At takeover, update `HANDOFF.md` with completed work, exact best/restart checkpoints, verified results, rejected
  trials, known mistakes, active jobs/monitors, remaining work and the next measured question. Make the latest state
  clearly override historical pending notes. Commit and push the handoff separately like any other changed file.
