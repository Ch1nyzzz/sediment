# Four-GPU T0b-Latest Log

## 2026-08-24 02:45 EDT | contract and discovery

- Inputs checked: current discussion, aTTT paper definition, local worker and
  driver code, launcher GPU mapping, git status, SSH config, active processes,
  and Brev CLI.
- Actions taken: froze the matched experiment and safe monitoring boundaries.
- Verification: GPUs 4-7 and ports 8104-8107 are encoded in `launch_p1.sh`;
  no active local SSH session identifies the remote; Brev is logged out.
- Files changed: loop contract, progress, and log only.
- Decision needed: none yet.
- Next: implement and test `t0b_latest_k5_paper`, then continue endpoint
  discovery and remote audit.

## 2026-08-24 02:50 EDT | implementation verification

- Inputs checked: aTTT Env worker, resident trainer, vLLM scoring path,
  chat-template prefix contract, and EnvScaler observation-role behavior.
- Actions taken: added `tier0b_latest_worker.py`, focused unit tests, and the
  single-arm `p3_driver12.sh` launcher.
- Verification: 82 tests passed; Ruff, Python compilation, `git diff --check`,
  and `bash -n` all passed.
- Files changed: new worker, test, driver, and loop state only.
- Decision needed: remote endpoint remains unresolved.
- Next: identify the established machine, inspect driver11 and GPUs 4-7,
  perform a bounded GPU smoke, then queue the full matched arm.

## 2026-08-24 02:51 EDT | remote smoke and full launch

- Inputs checked: exact remote dependency hashes, previous driver completion,
  live worker list, vLLM health, and GPU 4-7 state. GPU 0-3 were not touched.
- Actions taken: synced only the new worker, test, and driver; ran remote focused
  tests; executed a one-GPU bounded smoke; unloaded its two temporary adapters;
  launched the full four-shard run.
- Verification: remote tests 3/3 passed. Smoke selected observations at steps 5
  and 10, produced positive residual weights for 90.0% and 95.5% of target
  tokens, completed two optimizer losses per selection, had zero update errors,
  and successfully hot-loaded adapters into vLLM.
- Active run: `results/t0b_latest_k5_paper`, driver PID 1215005, worker PIDs
  1216007/1216014/1216016/1216022 on GPUs 4/5/6/7.
- Decision needed: none.
- Next: monitor progress and errors until 200 unique rows, then compute paired
  results from completed artifacts.

## 2026-08-24 05:27 EDT | completed factorial audit

- Actions taken: monitored `t0b_latest_k5_paper` to 200/200, audited its
  surprising 34/200 result, implemented and tested the two missing factorial
  cells, ran real-GPU smokes, then supervised two additional four-GPU 200-task
  runs to completion.
- Verification: all three new arms have 200 unique rows, zero top-level errors,
  and zero update errors. Both sequential driver completion markers are present.
- 2x2 successes: ngram/standalone 45, residual/standalone 40, residual/full 34,
  ngram/full 8. Full versus standalone under ngram is +2/-39 (p=7.84e-10).
  Residual versus ngram under full context is +29/-3 (p=2.56e-6).
- Interpretation: full-trajectory observation training corrupts the action
  policy in EnvScaler. Residual weighting mitigates that damage but does not
  beat aTTT in standalone context.
- Files changed: factorial worker/test/driver, analysis and monitor scripts,
  durable report/progress/log, and local copies of raw JSONL artifacts.
- Decision needed: whether to take aTTT to 8B or redesign the gradient target.

## 2026-08-24 11:54 EDT | residual plus 3-gram follow-up smoke

- Inputs checked: completed factorial artifacts, existing aTTT 3-gram helper,
  old worker/driver process state, vLLM health, and GPU 4-7 ownership.
- Actions taken: added the `residual_ngram` weighting mode, a focused unit
  test, and `p3_driver14.sh`; synced only those reviewed files after hash
  verification; ran a three-task smoke on GPU 4.
- Verification: 86 local tests and 4 remote focused tests passed. The smoke
  completed 3/3 rows with 10 successful updates, zero update errors, and two
  optimizer losses per update. On the first task, later updates had mean
  3-gram weights 0.642 and 0.789 versus 1.0 with empty history.
- Decision needed: none.
- Next: launch the 200-task four-shard arm and monitor to completion.

## 2026-08-24 11:55 EDT | residual plus 3-gram full launch

- Inputs checked: smoke completion, no remaining factorial worker, absent full
  result directory, and healthy vLLM endpoints on ports 8104-8107.
- Actions taken: launched `p3_driver14.sh` detached on GPUs 4-7.
- Verification: driver PID 3854318; worker PIDs
  3855385/3855389/3855394/3855399 correspond to shards 0/1/2/3 and explicitly
  carry `--weighting residual_ngram --train-context standalone`.
- Decision needed: none.
- Next: monitor row growth, update errors, processes, logs, and GPU state until
  all 200 tasks complete.

## 2026-08-24 12:01 EDT | residual-density audit and signed follow-up design

- Inputs checked: all 401 diagnostics from the completed
  `latest_resid_standalone_paper` arm and the exact weighted-CE normalization.
- Verification: candidate-mean positive residual coverage is 87.8%; token-
  weighted coverage is 84.4%; zero candidates have no positive token and only
  two candidates are below 50% coverage. Weighted CE divides by total weight,
  so small absolute residual magnitude does not reduce the overall loss scale.
- Finding: signal sparsity is not a supported explanation. Weight concentration
  is more plausible: candidate max/mean has median 5.5, p90 9.7, and maximum
  42.7.
- Follow-up boundary: do not use negative-weight CE. After the 3-gram arm,
  inspect negative-residual token semantics before a bounded unlikelihood
  smoke with a small separately normalized coefficient, gradient clipping,
  KL-to-pre-update anchoring, nonfinite checks, and rollback.

## 2026-08-24 12:04 EDT | full-run checkpoint

- Progress: 26/200 rows across shards 7/5/7/7.
- Health: all four workers alive, error hits 0, and every GPU has shown active
  training utilization since launch.
- Decision needed: none.
- Next: continue monitoring without modifying the live workers.

## 2026-08-24 12:15 EDT | near-halfway checkpoint

- Progress: 95/200 rows across shards 24/16/29/26.
- Health: all four workers alive, error hits 0; every shard has continued to
  grow and all four GPUs have shown optimizer activity.
- Mechanism check at 35 rows: 77/77 diagnostics trained successfully; 47 had
  mean 3-gram weight below 1 and the aggregate mean was 0.820.
- Decision needed: none.
- Next: continue to 200 rows and defer outcome analysis until completion.

## 2026-08-24 12:44 EDT | residual plus 3-gram completion

- Verification: 200 unique rows, 50 per shard, zero top-level and update
  errors, 387/387 successful updates, two losses per update, and a normal
  driver completion marker.
- Outcome: 40 successes, env_191 26/50, mean reward 0.628674. Versus residual
  standalone the paired result is +6/-6 (p=1.000); versus aTTT paper +4/-9
  (p=0.267).
- Decision: 3-gram repetition discount is a null addition to residual pricing.

## 2026-08-24 12:52 EDT | zero-training residual root-cause audit

- Actions taken: audited task-level residual/outcome associations; added and
  ran a scoring-only 20-task worker on GPUs 4-7; verified prompt-logprob parser
  behavior; copied raw token traces locally. No adapter was trained or loaded.
- Verification: 44 observations, 1,197 tokens, 91.2% positive deltas. Full
  context assigns >99% probability to 87.5% of tokens, which contribute 93.5%
  of positive weight mass. Bare standalone assigns <1% probability to tokens
  contributing 74.0% of positive mass.
- Finding: the proxy detects deterministic action-conditioned tool-return
  fields and bare-prompt mismatch, not evidence changing belief. Weighted-CE
  normalization also removes candidate-level residual magnitude.
- Decision: do not train unlikelihood. Redefine the evidence contrast and
  counterfactual before another training arm.

## 2026-08-24 13:23 EDT | action-hint and matched-action control audit

- Actions taken: implemented a scoring-only worker that evaluates identical
  action tokens under base, exact-action-plus-masked-result, and
  exact-action-plus-actual-result prompts; ran 40 tasks across GPUs 4-7 and
  copied all four raw shards locally. No training or adapter load occurred.
- Verification: 40 unique tasks, 581 actions (494 ok, 87 ERROR), 36,579 tokens,
  zero top-level errors; 88 local tests and the remote focused test pass.
- Copy finding: the pure action hint gives at least one positive token to
  100% of ERROR actions; 13.82% of their tokens and 25.29% of their action
  means are positive. Whole-action mean delta is nevertheless negative.
- Control finding: matched-action positive-token coverage is 5.43% overall,
  5.32% for ok, and 6.04% for ERROR. ERROR conditional positive magnitude is
  1.467 versus 0.255 for ok; ok-above-ERROR AUC is 0.420.
- Saturation: 97.63% of base action tokens already have probability above 99%,
  and 93.67% of controlled deltas have absolute value below 1e-4. The residual
  is dominated by sparse outliers, with the largest on ERROR actions.
- Decision: the same-action control is a useful diagnostic but not a valid
  training target. Do not launch it with ReLU-weighted CE.

## 2026-08-24 14:04 EDT | direction-gated signed-action launch

- Design: status-ok actions receive capped positive controlled-residual CE;
  explicit ERROR actions receive capped negative-residual unlikelihood at
  lambda 0.02. Both branches use fixed token-count denominators, exclude
  tool/JSON framing, and share the same-action masked-result control.
- Token audit: after the real Qwen tokenizer mask, 1.10% of ok tokens carry
  positive weight and 6.51% of ERROR tokens carry negative weight; 87.36% of
  ERROR actions have negative mass. Highest ERROR weights are failed entity
  values rather than tool-call framing or function names.
- Safety calibration: fresh AdamW can make a large step from a tiny loss. One
  positive-only update produced KL 0.452 at 5e-4 and 0.122 at 1.25e-4; both
  were restored and rejected. It passed at 3.125e-5 with KL 4.68e-6.
- ERROR smoke: an accepted window with three ERROR actions raised ok weighted
  log-prob by 0.02760 and lowered ERROR weighted log-prob by 0.04495 at KL
  0.00180. A later direction-gated smoke with two ERROR actions passed at
  3.125e-5, KL 0.00101, ok change +0.000012, ERROR change -0.02699.
- Verification: 94 local tests, Ruff, compilation, shell syntax, and diff
  checks passed; 6 remote focused tests passed. No smoke had a top-level or
  update error, nonfinite value, OOM, or adapter-load failure.
- Launch: four 50-task shards started on GPUs 4-7 with PIDs
  2309623/2309631/2309637/2309643. The run directory was absent beforehand;
  all four existing vLLM servers were reused without restart.

## 2026-08-24 15:18 EDT | signed-action completion

- Completion: four shards reached 50/50 and all workers exited normally. The
  run has 200 unique rows, zero top-level errors, 417 selected windows, 399
  accepted updates, 13 inactive windows, five direction-gated safe skips, and
  zero unexpected update errors or adapter-load rollbacks.
- Safety: 191 attempts were rejected for KL, 10 for positive-direction
  violation, and 11 for negative-direction violation. Maximum accepted KL was
  0.019671; 7/798 accepted optimizer steps hit the 0.5 gradient clip; no
  nonfinite or OOM occurred.
- Mechanism: 95 material negative updates changed ERROR weighted log-prob by
  -0.04925 on average; 167 material positive updates changed ok weighted
  log-prob by +0.01282. No accepted material update violated its sign gate.
- Outcome: 41/200 successes, env_191 26/50, mean reward 0.608392. Versus std
  +6/-3 (p=0.508); retry +7/-3 (p=0.344); T0c paper +6/-6 (p=1.000); aTTT
  paper +2/-6 (p=0.289). Mean reward is 0.026710 below aTTT paper.
- Decision: stable negative suppression is demonstrated, behavioral benefit is
  not. Do not attribute a task-success gain to unlikelihood without an exact
  lambda=0 matched control; prefer a repaired-action target for directed mass.

## 2026-08-24 15:21 EDT | final verification and service handoff

- Local verification: 94/94 tests passed; Ruff, Python compilation, driver
  shell syntax, `git diff --check`, and independent result re-analysis passed.
- Remote health: the existing Qwen vLLM services remain resident on GPUs 4-7
  at ports 8104-8107, each using about 34.9 GB; utilization was idle after the
  four training workers exited normally. GPUs 0-3 were not modified.
- Handoff: raw JSONL, worker logs, config, analysis scripts, and the written
  report are preserved locally. No cloud instance was created, stopped, or
  destroyed.
