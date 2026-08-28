# Stable memory signal meta-learning: preregistered experiment

Date: 2026-08-27

## 1. Scientific target

The target is not to reproduce one donor-KL update. A donor intervention is a
noisy sensor reading: it says how one complete memory changed the frozen policy
on a small set of current-window states. The meta-learner must infer a lower
variance update that improves genuinely future, family-held-out reward.

For a completed window of 16 trajectories,

\[
M_w=\{\tau_1,\ldots,\tau_{16}\},\qquad
q_\phi(z_w\mid M_w)=(\mu_w,\sigma_w,p_{\rm write}),\qquad
\Delta\theta_w=U\mu_w.
\]

`U` is a frozen low-dimensional update basis fitted only on donor proposals from
the meta-train fitting clusters; the train-side validation cluster is excluded
from basis fitting as well as supervised optimization. Future tasks label the
training window but are never compiler input.
The deployment decision is a no-op when the write probability is low or the
multi-view uncertainty is high.

The falsifiable hypothesis is:

> On source families and future-probe families excluded from meta-training, the
> stable-signal arm has higher paired future reward than noop, discovery-selected
> single donor-KL, heuristic joint donor-KL, and the unweighted donor mean; the
> corresponding shuffled-memory advantage disappears.

The old single-discovery-winner label is retired. The three-window pilot showed
why: all three initial winners were positive, but repeat-confirmed cross-family
gains became -0.094967, +0.056517, and 0. The pooled primary gain was -0.012817.

## 2. Leakage-free partitions

The four frozen streams contain 1,700 completed trajectories each and 34
families. Family allocation is frozen before any proposal reward is inspected.
A deterministic seed partitions the 34 families into four disjoint roles:

- 11 meta-train member families;
- 8 meta-train probe families, split 4 discovery / 4 confirmation;
- 7 meta-test member families;
- 8 meta-test probe families, split 4 discovery / 4 confirmation.

The partitioner deterministically balances completed-trajectory success counts,
because those are deployment-time memory facts, but does not inspect candidate gains.
It emits the exact family lists and a SHA-256 digest before GPU collection.

The corrected feasibility prepass enumerated 8--14 train-member families while
holding all four probe roles at four families. The first plan required only two
successful members. Its real train smoke completed, but its first test window
had four successful donors from one family and only three cross-family target
states, so the collector correctly rejected it before training a test proposal
or generating a test reward. No completed train gain was inspected. That v3
root is retained as mechanical-failure evidence and is excluded from fitting.

The planner now mirrors the collector's reward-free structural rule: among the
first four successful members, at least two donors must each have at least four
other-family assistant target states. Under that rule a uniform 3-train / 3-test
window count in every request-seed stream is impossible for every family
capacity. The frozen pilot therefore uses all structurally valid windows from
the only allocation with at least ten train windows, twelve test windows, and
four independent source-memory task clusters on both sides: 12 train-member / 6
test-member families. Train request-seed replication is explicitly ragged
(3,2,2,3 windows across streams); test remains (3,3,3,3). Cluster-balanced
training removes replication-count weighting. The corrected partition digest is
`4d392eea09a3c140cfb2ca11356b62f6acc98c0588e8b73aa445ec5911fe19ce`; the
22-window plan digest is
`7b216a5a90b9d3518449ca29af12a79a2d2ce09f445e40c0b0a5aba84b27d548`.
Both manifests state `candidate_rewards_inspected=false`.

Windows are constructed after filtering to their member-family partition while
preserving the original within-stream order. Probe trajectories retain their
original positions and must occur strictly after the last member in the source
stream. A window is eligible only when it has exactly 16 members, at least two
successful complete donors, and all required discovery/confirmation families
have later tasks. Member, discovery, and confirmation task IDs and families are
pairwise disjoint.

The held-out test is strict in two ways: neither its member families nor its
probe families may occur in meta-training. The update basis, feature
normalization, write threshold, and uncertainty threshold are frozen before the
first meta-test candidate is evaluated.

The four frozen streams are request-seed variants over a shared task pool. In
the corrected 22-window pilot, the 12 outer-test windows reduce to four unique
source-memory task clusters with replication sizes 4, 4, 3, and 1; the 10 train
windows reduce to four source-memory task clusters with sizes 4, 4, 1, and 1.
Request-seed variants are augmentation/replication, not independent experimental
units. Exact source task-set hashes are saved, train-side validation holds out a
whole hash cluster, the update basis excludes that cluster, and all primary test
statistics first average within these clusters.

The data compiler has two irreversible-order modes. `fit-train` accepts only the
12 train manifests, fits the basis, and writes a train-only file. The training
process refuses any `partition=test` record; it learns the expected outer count
only from the frozen plan plus the reward-free completed-artifact audit.
`project-test` refuses to parse the 12 test manifests until a checkpoint with
`outer_test_labels_loaded=false` and the matching basis hash exists. Thus “test
labels unread” means they never entered the fitting process, not merely that a
loss function happened to ignore already-parsed fields.

## 3. Proposal and label construction

Each eligible window produces at most four single donor-KL proposals. Every
proposal uses one complete successful donor and exactly four family-diverse
target states from the same completed window. It starts from the same reference
LoRA and receives the same two optimizer steps. Donor text is used only to make
the offline intervention teacher; it is absent from the student prompt.

The existing heuristic joint remains a baseline: for each target state it keeps
the donor causing the largest action-distribution shift, without concatenating
multiple memories. The simple-mean baseline is the equal parameter-coordinate
mean of valid single proposals relative to the common reference.

Every non-noop proposal is scored on:

- one paired, greedy pass over the four discovery families;
- three interleaved paired A/B repeats over the four disjoint confirmation
  families, reversing base/candidate order on odd repeats;
- per-family deltas, not individual decoding calls, as the uncertainty units.

For proposal `a`, let `g_wa` be the mean confirmation family gain and `se_wa`
its standard error over family means. The conservative score is

\[
\ell_{wa}=g_{wa}-\kappa\,se_{wa},\qquad \kappa=1.
\]

A proposal receives positive target weight only when `ell_wa > 0` and its worst
confirmation-family mean is at least -0.20. The stable coefficient target is a
reward-weighted barycenter of the surviving single-donor proposal coordinates:

\[
z_w^*=\frac{\sum_a [\ell_{wa}]_+ z_{wa}}
                 {\sum_a [\ell_{wa}]_+}.
\]

If no proposal survives, `z_w* = 0` and the write label is false. Thus future
reward supplies direction and the donor updates supply candidate coordinates;
no donor is treated as ground truth. Discovery reward is never used to form the
stable target. It is reserved for the single-donor baseline and diagnostics.

## 4. Multi-view denoising model

The compiler is a permutation-invariant set encoder over all 16 completed
memories. Each member includes deployment features. Successful donors also carry
their projected donor-KL proposal coordinates and intervention summaries such as
action shift, teacher support, and cross-family target coverage. Failures remain
in the set and therefore can affect the write/no-write decision, but have a zero
proposal mask.

For every training window the deterministic view generator creates:

1. the full 16/16 set;
2. one leave-one-source-family-out view per represented family when at least two
   members and one donor remain;
3. four seeded 8/16 subsamples, retaining chronological identity but presenting
   the selected members as a set.

The model returns a coefficient mean, diagonal log variance, and write
probability. To keep capacity defensible with only three fitting clusters, the
mean is restricted to a learned positive 0--2x scaling of an attention-weighted
convex combination of the current window's donor-proposal coordinates. It cannot
emit an arbitrary direction outside those proposals. Failures and other
non-proposal members still enter the full-window encoder and can suppress the
write. Training combines:

- heteroscedastic coefficient loss against the conservative reward target;
- write/no-write binary loss;
- agreement between full-view and resampled-view coefficient means;
- a no-op norm penalty for confirmation-negative windows;
- uncertainty calibration from target error plus across-view dispersion.

The pilot freezes a requested rank-16 basis (capped by numerical rank),
32-dimensional member/window hidden layers,
attention temperature 1, deployment top-8 member selection, and a maximum
coefficient-vector norm of 12. Optimization is 100 deterministic AdamW epochs
with batch size 16, learning rate 1e-3, weight decay 1e-4, and seed 20260827.
The write, view-consistency, and negative-window no-op loss weights are 0.5,
0.25, and 0.25. One of the four train source-memory task clusters (selected only
by the seed) is held out in full for train-side validation and gate calibration.
No request-seed variant of that task set remains in the fitting split.
Within the fitting split, each epoch selects one seeded variant from every
remaining task cluster and cycles variants across epochs, so a four-seed cluster
does not receive four times the optimization weight of a singleton cluster.

At deployment, total uncertainty is predicted variance plus the variance of the
view means. The write-probability threshold is fixed at 0.5. The uncertainty
threshold is the 75th percentile among positive meta-train validation-cluster
windows (and becomes zero if none are positive). Both are saved in the checkpoint
before test rewards are unblinded. The stable arm writes only when both thresholds
pass. An untrained checkpoint is an exact no-op.

Every raw donor proposal is a rank-8 LoRA update with alpha 8, learning rate
1.5e-4, reverse top-20 KL distillation, base-anchor coefficient 0.5, maximum
sequence length 12,288, and exactly two optimizer steps. These proposal settings
and all compiler settings above are fixed before the first proposal collection.

## 5. Six matched arms

All arms share the same reference adapter, task order, greedy decoding, probe
IDs, A/B order schedule, and update dose where training is involved.

| arm | update |
| --- | --- |
| `noop` | common reference, zero update |
| `single` | single donor with best discovery gain; confirmation untouched |
| `heuristic_joint` | existing action-shift joint donor-KL update |
| `mean` | equal mean of valid single donor parameter deltas |
| `stable` | full-window meta-learner mean after write/uncertainty gates |
| `shuffled` | same frozen meta-learner on a deterministic different test window |

The shuffled arm is evaluated on the original window's probes. It therefore
controls whether any generic update helps, while breaking the relationship
between the memory set and its future. No arm may select using confirmation or
test reward.

## 6. Staging and claim gate

1. **Local mechanical tests:** schema, partition digest, view invariance,
   conservative no-op labels, zero-initialized model, and six-arm accounting.
2. **Two-window GPU smoke:** one train and one test window; prove completed
   optimizer steps, finite/nonzero proposal deltas, probe isolation, synthesized
   adapter loading, and endpoint cleanup. It makes no efficacy claim.
3. **Pilot:** collect 10 meta-train and 12 meta-test eligible windows across the
   four streams, representing four train and four test source-memory task
   clusters. Run train-only basis fitting and freeze the model and all gates
   before test projection or evaluation.
4. **Scale only if justified:** expand the preregistered test set if the stable
   arm is mechanically sound and its point estimates are not dominated by a
   single family.

The primary statistic is the source-memory-task-cluster paired mean confirmation
gain of `stable` relative to each of `noop`, `single`, `heuristic_joint`, and
`mean`: first average the four request-seed variants of each source task set,
then average the four independent outer clusters. Report source-cluster
bootstrap intervals and an exact paired sign/permutation test. With only four
outer clusters, the smallest attainable one-sided exact p-value is 0.0625, so
this pilot still cannot establish conventional 0.05 statistical significance.
Also report
window-level descriptives, per-family gains, worst-family gain, write rate,
uncertainty, and `stable - shuffled`.

The method is not supported unless all of the following hold on frozen
meta-test windows:

- each point estimate `stable - {noop, single, heuristic_joint, mean}` is
  strictly positive;
- `stable - shuffled` is strictly positive, while the shuffled-memory point
  estimate has no positive aggregate advantage over noop (`shuffled - noop <= 0`);
- improvement remains positive after omitting each confirmation family in turn,
  and the stable arm's worst-family mean stays at or above the preregistered
  -0.20 catastrophe floor;
- all scope, task/family isolation, adapter-delta, and completed-update audits
  pass.

These are frozen point-estimate gates for the small pilot. Separately report
whether every primary source-cluster-bootstrap lower bound is positive, every exact
one-sided sign-flip p-value is below 0.05, and the shuffled-minus-noop interval
contains zero. Those sampling-precision diagnostics do not turn failure to
reject the shuffled null into evidence that the null is true.

Passing the two-window smoke only proves the experimental machinery. Passing the
pilot with wide intervals is directional evidence, not statistical confirmation.

## 7. Artifact boundary

Local/remote working storage may contain raw buffers, task-level paired results,
adapters, and worker logs. The repository-facing result package contains only
the frozen partition manifest, configuration/digests, aggregate arm table,
clustered statistics, audits, and the written report. Raw trajectories and
worker logs are not published.
