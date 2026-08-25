# Four-GPU T0b-Latest Experiment Contract

## Goal

Run and supervise the matched EnvScaler comparison on GPUs 4-7 until all
four shards complete, while keeping the agreed GPUs occupied by useful work
and preserving every existing result.

## Frozen Experiment Definition

The new arm is `t0b_latest_k5_paper`.

- Task pool: the same 200-task tail pool used by `tier0`, `tier0c_paper`, and
  `attt_paper`.
- Base model, decoding settings, seed behavior, and harness: unchanged.
- Candidate: the most recent environment observation only.
- Schedule: select one candidate every 5 settled, non-terminal agent steps;
  at most 5 candidate selections per episode.
- Extra evidence: none. Do not build or inject an `action -> ok/ERROR` block.
- Context arm: score and train the latest observation after the complete
  naturally occurring trajectory prefix through the action that produced it.
- Control score: score the exact same observation as a standalone user
  message, matching the local aTTT reproduction's serialization.
- Token price:
  `relu(logp(observation token | trajectory prefix) -
  logp(observation token | observation prefix alone))`.
- Loss support: only content tokens of the latest observation. Earlier
  messages are context with zero loss.
- Dose: LoRA rank 8, alpha 16, learning rate 5e-4, two gradient steps per
  successful update.
- Adapter lifetime: accumulate within one episode; reset between episodes.
- Observation provenance, not chat role, defines the candidate. EnvScaler
  parse and invalid-action observations returned as `user` are eligible.

The existing `tier0` arm remains the historical ERROR-triggered prefix arm.
Do not relabel or overwrite its artifacts.

## Frozen Matched-Action Signed Follow-up

The user-approved follow-up arm is `signed_action_ul002_paper`.

- Candidate schedule: every 5 settled non-terminal steps, at most 5 windows
  per episode; adapter state accumulates only within the episode.
- Dose: LoRA rank 8, alpha 16, nominal learning rate 5e-4, and two optimizer
  steps per accepted update.
- For each executed action `A` and returned observation `O`, both scoring
  branches contain the exact same action. The only contrast is actual result
  `O` versus `[RESULT MASKED]`.
- Controlled token residual:
  `delta = logp(A | P, A, O) - logp(A | P, A, [RESULT MASKED])`.
- Positive branch: only status-ok actions, with
  `w_pos = clip(relu(delta), 0, 1.55)`.
- Negative branch: only explicit ERROR actions, with
  `w_neg = clip(relu(-delta), 0, 4.51)` and unlikelihood coefficient 0.02.
- Tool-call framing, JSON keys, and tool-name tokens are excluded from both
  branches. Argument values and other semantic action content remain.
- Each branch divides by the fixed number of action content tokens in its
  status group, not by the sum of residual weights. Residual magnitude is
  therefore retained in the scalar objective.
- Objective:
  `L = L_pos + 0.02 * L_unlikelihood + 0.01 * KL(pre_update || current)`.
- Global trainable-gradient clipping is 0.5. Any nonfinite loss, gradient, or
  parameter rejects the entire candidate update.
- Exact active-position post-update KL must not exceed 0.02. A rejected
  attempt restores the pre-update LoRA and retries at one quarter of the
  previous learning rate, for at most three backtracks.
- Direction gate: above effective branch dose 1e-5, status-ok weighted action
  log-prob may not decrease by more than 1e-5 and ERROR weighted action
  log-prob may not increase by more than 1e-5. Violations backtrack exactly as
  KL violations do.
- A candidate adapter is saved and hot-loaded only after every safety gate
  passes. Adapter-load failure restores the pre-update LoRA.

## Frozen Residual + 3-Gram Follow-up

The follow-up arm is `latest_resid_ngram_standalone_paper`.

- Candidate, cadence, maximum selections, dose, optimizer steps, task pool,
  and adapter lifetime are identical to `attt_paper`.
- Train the latest observation as a standalone user message.
- Compute residual weights from the full natural prefix versus that standalone
  message, then multiply tokenwise by aTTT's prior-update 3-gram repetition
  weights: `relu(logp_full - logp_standalone) * max(0.05, 1 / (1 + f_j))`.
- Update the per-episode 3-gram history only after a successful optimizer
  update. Reset both the adapter and history between episodes.
- Do not change the residual sign rule, add evidence, or train trajectory
  prefix tokens.

## Required Diagnostics

Each task record must expose:

- binary success and graded reward;
- steps, candidate selections, optimizer-bearing updates, and update errors;
- for each selected observation: token count, positive-residual fraction,
  mean residual, mean positive weight, and whether an optimizer update ran.

## Run Loop

1. Read this contract, `progress.md`, and the latest `log.md` entry.
2. Inspect the remote driver, worker PIDs, shard progress, worker errors,
   vLLM health, and `nvidia-smi` for GPUs 4-7.
3. Never disturb a live valid worker. Queue the new arm after any valid active
   arm unless the active arm is irrecoverably failed.
4. If an in-scope process is dead, verify its PID file, command, run directory,
   completed task IDs, and logs before a resume-safe restart.
5. Confirm all four shards make progress; GPU utilization may legitimately
   dip during environment execution, scoring, adapter save, or hot-swap.
6. After completion, read every shard, reject error/incomplete rows, and
   compute paired comparisons on identical task IDs.
7. Update `progress.md` and append evidence to `log.md` on every monitoring
   pass.

## Allowed Actions

- Read local and remote repository, process, GPU, log, and result state.
- Add narrowly scoped worker, tests, driver, and monitoring files for this
  experiment.
- Run local static checks and tests.
- Copy only the reviewed experiment files to the user's established GPU
  checkout.
- Launch the agreed arm on GPUs 4-7 after confirming they are free or queue it
  behind the valid current arm.
- Resume a dead in-scope worker using its existing run directory and
  task-ID-based skip logic.
- Restart a dead in-scope vLLM server only after verifying no live worker is
  using it and the model/template fingerprint matches this contract.

## Forbidden Actions

- Do not delete, truncate, rename, or overwrite existing result directories.
- Do not stop a live valid experiment merely to start this arm sooner.
- Do not use GPUs outside 4-7 or interfere with other users' processes.
- Do not stop, reset, delete, resize, or recycle the GPU instance.
- Do not modify credentials, billing, permissions, drivers, CUDA, or system
  packages.
- Do not push, publish, merge, or commit unless the user separately asks.
- Do not report a result before all four shards and the paired summaries have
  been read.

## Done Criteria

- All four GPUs have contributed a completed shard for the same 200-task pool.
- There are 200 unique successful records and zero unresolved worker errors.
- Configuration and runtime diagnostics match the frozen definition.
- Paired success, reward, rescue/harm counts, and exact McNemar comparisons
  against `std`, `retry`, `attt_paper`, and relevant T0 arms are computed.
- `progress.md` and `log.md` contain the completion evidence.
