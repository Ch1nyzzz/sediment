# Stable memory signal family-held-out pilot report

- Completed: 2026-08-28 01:50:48 UTC
- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Verdict: **the preregistered efficacy claim gate failed; the current method
  is not supported and should not be scaled.**

## What was tested

The pilot asked whether a permutation-invariant learner over a complete
16-memory window could denoise single-donor KL proposals into a stable
low-dimensional LoRA update. Future family-held-out reward supplied the
train-only target. The frozen outer comparison had six arms: `noop`,
discovery-selected `single`, `heuristic_joint`, parameter `mean`, `stable`, and
cluster-deranged `shuffled` memory.

The completed design contained 10 meta-train and 12 outer-test windows. Each
side reduced to four independent source-memory task clusters; request-seed
variants were averaged within cluster before the primary comparison. The update
basis used 25 single proposals from three train fitting clusters, with the
fourth train cluster excluded for validation. The effective basis rank was 16
and captured 0.8196 of fitting-proposal energy.

All 22 proposal windows passed the completed-artifact audit: 10 train, 12 test,
zero missing, zero failed. The checkpoint was frozen with
`test_windows_unread=12`, `outer_test_labels_loaded=false`, and a deployment
gate calibrated only on the train-validation cluster. Test projection happened
after that checkpoint existed.

## Primary result

The table reports the primary equal-source-cluster mean confirmation gain.

| Arm | Mean gain |
| --- | ---: |
| noop | 0.000000 |
| single | 0.024126 |
| heuristic joint | 0.000557 |
| mean | **0.029614** |
| stable | 0.010156 |
| shuffled | 0.014028 |

Stable-relative comparisons were:

| Comparison | Mean | Source-cluster bootstrap 95% | Exact one-sided p |
| --- | ---: | ---: | ---: |
| stable - noop | +0.010156 | [-0.003229, 0.034235] | 0.5000 |
| stable - single | -0.013969 | [-0.089425, 0.037876] | 1.0000 |
| stable - heuristic joint | +0.009600 | [-0.063722, 0.082921] | 0.4375 |
| stable - mean | -0.019458 | [-0.077728, 0.052080] | 1.0000 |
| stable - shuffled | -0.003872 | [-0.010391, 0.002397] | 1.0000 |

The shuffled control itself had `shuffled - noop = +0.014028`, bootstrap
`[-0.004550, 0.039323]`, one-sided `p=0.25`. Therefore all three central claim
requirements failed:

1. stable did not beat the single and mean donor baselines;
2. stable did not beat shuffled memory;
3. the shuffled advantage did not disappear.

The catastrophe and leave-one-confirmation-family checks passed, but they do
not rescue the claim. All primary bootstrap lower bounds were not positive and
the exact-p precision diagnostic failed. With only four outer clusters the
pilot could not attain conventional one-sided `p<0.05` even in the best possible
sign pattern; the observed comparisons were weaker than that design ceiling.

## Why the stable and shuffled gains are not update effects

Only one of the nine fitting windows had a positive stable target, and the
single held-out validation window had none. The frozen uncertainty threshold
therefore became zero. The deployment gate rejected every test prediction:
`stable_write_rate=0`.

This was verified at the parameter boundary, not inferred from a metric:

- all 24 stable/shuffled predictions had `gate_passed=false`;
- every emitted coefficient was exactly zero (`max_abs=0`);
- all 24 synthesized adapter files represented one identical tensor state;
- their tensor keys, dtypes, and values were exactly equal to the reference
  adapter (`max_abs_tensor_delta=0`).

Consequently the observed nonzero stable and shuffled paired gains cannot be
caused by a learned update. They measure residual replay/order/engine
nondeterminism between two evaluations of parameter-identical adapters. This
also explains why shuffled appeared positive. The current analytic `noop=0`
arm did not empirically estimate this identity-replay noise; a future protocol
should include a separately named reference-vs-reference replay arm.

The safe no-op behavior worked as designed, but this pilot contains no evidence
that the learner extracted a useful memory-dependent update.

## Mechanical amendment and integrity limits

After all 22 proposal windows passed audit, the first train-only basis fit
stopped before checkpoint creation or test projection with
`ValueError: all oracle updates are numerically zero`. The updates were not
zero: every inspected single proposal changed 360 tensors, with an example
delta norm of 0.78446. The LoRA vector width was 13,565,952, so the old
float32 tolerance multiplier `eps * max(matrix.shape)` equaled 1.6171875. It
therefore exceeded the largest singular value by construction.

Before any outer-test label was loaded, the tolerance was corrected to scale by
the observable singular-spectrum size. A regression test reproducing a width
greater than `1/eps` passed locally and remotely. The exact old/new hashes and
failure evidence are in
[`numerical_rank_amendment.json`](evidence/numerical_rank_amendment.json).

This is a real preregistration limitation: `sediment/compiler/basis.py` was
omitted from the original frozen `code_sha256` dependency list. The frozen
protocol hash therefore remained unchanged even though an imported dependency
changed. The amendment records the omission and new hash; future freezes must
hash the complete transitive runtime closure.

There is also a frozen prose erratum: the preregistration says `fit-train`
accepts “12 train manifests.” The frozen plan, code gates, command, audit, and
actual run consistently used **10 train windows and 12 test windows**. The
hashed preregistration file was not edited post hoc.

The final local suite passed 209 tests with three warnings; the post-amendment
minimal basis/stable suite passed 14 tests both locally and on the remote host.
No proposal collector, six-arm evaluator, OOM, traceback, or connection-refused
error remained in the completed run.

## Resource and artifact boundary

The run reused the existing `reds-lab` 8xA100 grant. GPUs 0-3 and their unrelated
TP4 workload were untouched. GPUs 4-7 hosted the four resident 4B endpoints;
after completion ports 8104-8107 were healthy and each GPU returned to 34,939
MiB resident usage. Temporary 8B services were not left running.

Repository-facing evidence is aggregate-only. It contains the frozen protocol
and family partition, structural proposal audit, train/checkpoint/test aggregate
reports, amendment, checksums, and public six-arm aggregate. It excludes raw
trajectories, JSONL manifests, task-level paired rows, worker logs, synthesized
adapters, and checkpoints. Those working artifacts remain preserved under
`/data/erv1n/sediment/results/stable_signal_v2_20260827`.

## Decision

Do not enlarge this test set or weaken the write/uncertainty gate based on the
observed outer test. The next admissible experiment should be a newly frozen
pilot that:

1. expands train-only source-memory clusters until positive confirmation targets
   occur in more than one fitting cluster and at least one validation cluster;
2. preregisters an empirical reference-vs-reference replay control so no-op
   noise is measured rather than fixed analytically at zero;
3. freezes a fresh, untouched family-held-out test after the train-side
   feasibility criterion is met; and
4. hashes the full imported runtime closure, including basis/state utilities.

Until those conditions are met, the correct output of the current system is the
no-op that it actually deployed.
