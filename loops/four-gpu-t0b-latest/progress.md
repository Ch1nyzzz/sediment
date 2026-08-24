# Progress: Four-GPU T0b-Latest

## Current Goal

Diagnose why the latest-observation residual arm does not outperform aTTT,
without launching an unlikelihood training arm.

## Current State

- Experiment contract frozen on 2026-08-24.
- `scripts/tier0b_latest_worker.py` implements the no-extra-evidence,
  latest-observation, K=5 matched arm.
- `scripts/p3_driver12.sh` is a single-arm, resume-safe four-GPU launcher.
- Local verification: 82 tests passed; Ruff, `git diff --check`, Python
  compilation, and shell syntax checks passed.
- Local checkout contains pre-existing uncommitted work that must be preserved.
- Existing launch convention uses GPUs 4-7 and ports 8104-8107.
- Remote alias `reds-lab` resolves to the established eight-A100 machine;
  GPUs 0-3 remain outside this experiment's scope.
- `p3_driver11.sh` completed before this launch; no previous worker remained on
  GPUs 4-7.
- Remote dependency hashes matched the locally tested checkout, and the three
  new files matched after targeted sync.
- Real-GPU smoke passed: two selected candidates, two successful updates, two
  optimizer losses per update, zero update errors, and successful vLLM adapter
  loads.
- `t0b_latest_k5_paper` and both missing 2x2 cells completed on GPUs 4-7.
- The sequential factorial driver exited normally after two 200-task runs.
- `latest_factorial_worker.py` now supports tokenwise residual times aTTT
  3-gram weights in standalone context; 86 local tests and 4 remote focused
  tests pass, with matching local/remote hashes.
- A three-task real-GPU smoke finished normally: 10/10 selected candidates
  produced updates, every update had two optimizer losses, and there were zero
  update errors. Repetition weights dropped below 1 after history accumulated.
- The full four-shard run launched at 2026-08-24 11:55 EDT (15:55 UTC) with worker PIDs
  3855385/3855389/3855394/3855399 on GPUs 4/5/6/7. All four shared vLLM
  endpoints were healthy at launch.
- The full residual-plus-3-gram arm completed 200/200 with zero errors and a
  normal driver completion marker. It scored 40 successes, tying residual-only.
- A 20-task, zero-training token audit completed across GPUs 4-7 with zero
  errors. It identified deterministic action-to-tool-response predictability
  and the bare-standalone counterfactual as the dominant residual source.
- A 40-task, zero-training matched-action audit completed across GPUs 4-7:
  581 actions, 36,579 action tokens, and zero errors. It confirms an exact
  action-hint shortcut but shows that sharing the action on both sides still
  assigns stronger positive residual outliers to ERROR than to ok actions.
- The ERROR-gated signed follow-up passed 94 local tests, 6 remote focused
  tests, real-token weight audit, positive-only KL calibration, and a real
  ERROR update smoke. The final direction-gated smoke lowered ERROR weighted
  log-prob by 0.02699 while raising ok weighted log-prob by 0.000012 at KL
  0.00101.
- `signed_action_ul002_paper` launched on GPUs 4-7 at 2026-08-24 14:04 EDT
  with worker PIDs 2309623/2309631/2309637/2309643. All four shared vLLM
  endpoints were healthy and the result directory was new.

## Last Result

- Date: 2026-08-24
- Summary: The 200-task ERROR-gated signed-action arm completed safely but did
  not beat aTTT paper: 41 versus 45 successes and mean reward 0.608392 versus
  0.635103.
- Evidence: 200 unique rows, zero top-level/unexpected update errors, 399
  accepted updates, five direction-gated safe skips, and all raw JSONL/config
  and worker logs copied to local artifacts.

## Next Steps

1. Do not claim that negative residual unlikelihood improves task success from
   this arm; it passed its mechanical direction tests but lost to aTTT paper.
2. If causal isolation is worth another 200-task run, use an exact lambda=0
   control with every other matched-action, mask, KL, and backtracking setting
   frozen.
3. Prefer a paired failed-action versus hindsight-repaired-action objective if
   the goal is to direct released probability mass toward a useful behavior.

## Blockers

- None.

## Human Decisions Needed

- Whether to spend another run on the exact lambda=0 causal control or move to
  a failed-versus-repaired pairwise target.
