# T0b-Latest Paper-Dose Result

## Completed arm

- Run: `t0b_latest_k5_paper`
- Target: latest nonterminal environment observation, serialized as a user
  message.
- Schedule: every 5 settled steps, at most 5 selections per episode.
- Dose: LoRA r=8, alpha=16, lr=5e-4, 2 optimizer steps per update.
- Training context: full natural trajectory through the producing action.
- Token weights: `relu(logp_full_context - logp_standalone)` on observation
  content tokens only.

## Integrity

- 200 rows and 200 unique task IDs.
- 387 selected candidates and 387 successful adapter updates.
- Every trained diagnostic contains exactly two optimizer losses.
- Zero top-level errors and zero update errors.
- Mean positive-token fraction: 0.895; no candidate had zero positive tokens.
- Source observation roles: 368 tool, 19 user.

## Outcome

| Arm | Success | env_191 | Mean reward |
| --- | ---: | ---: | ---: |
| std | 38/200 | 25/50 | 0.608 |
| retry | 37/200 | 24/50 | 0.620 |
| T0b | 41/200 | 26/50 | 0.624 |
| aTTT paper | 45/200 | 29/50 | 0.635 |
| T0b-latest paper | 34/200 | 22/50 | 0.562 |

Paired comparisons for T0b-latest paper:

| Baseline | Rescues / harms | Net | Exact McNemar p | Mean reward delta |
| --- | ---: | ---: | ---: | ---: |
| std | +8 / -12 | -4 | 0.503 | -0.0459 |
| retry | +7 / -10 | -3 | 0.629 | -0.0583 |
| original T0b | +4 / -11 | -7 | 0.118 | -0.0617 |
| aTTT paper | +3 / -14 | -11 | 0.0127 | -0.0732 |

## Current interpretation

This matched bundle does not rescue the residual-pricing claim.  Against aTTT
paper it is significantly worse on binary success and also worse on graded
reward.  The run's complete update evidence rules out a launch failure or an
undertrained adapter as the explanation.

The arm changes two things relative to aTTT paper: token weights and the
training context.  Those must be separated before assigning the failure to
residual pricing alone.  A strong clue is that full-context training often
reported nearly-zero loss: the selected tokens are precisely those made easy
by the producing action and trajectory.  In a real-GPU smoke, keeping residual
weights but training the target standalone restored substantial losses (for
example 7.75 to 5.47 over two steps).

## Completed 2x2

| Token weights | Standalone context | Full trajectory context |
| --- | ---: | ---: |
| aTTT 3-gram | 45/200, r=0.635 | 8/200, r=0.426 |
| residual | 40/200, r=0.631 | 34/200, r=0.562 |

All four cells use the same latest observation, K=5/max-5 schedule, paper
dose, two optimizer steps, and within-episode adapter lifetime.

Paired effects:

- Context effect with 3-gram weights, full versus standalone: +2/-39, net -37,
  exact McNemar p=7.84e-10, mean reward delta -0.2089.
- Context effect with residual weights, full versus standalone: +6/-12,
  net -6, p=0.238, mean reward delta -0.0692.
- Weight effect in standalone context, residual versus 3-gram: +2/-7,
  net -5, p=0.180, mean reward delta -0.0040.
- Weight effect in full context, residual versus 3-gram: +29/-3,
  net +26, p=2.56e-6, mean reward delta +0.1356.

The descriptive success-count interaction is +31: full-context training costs
37 successes under 3-gram weighting but 6 under residual weighting.  This is a
large interaction, not a residual main-effect win.

## Final interpretation

Full-trajectory observation training is the damaging component in EnvScaler.
It trains an action-conditioned environment predictor and then applies the same
adapter to future agent-action generation.  The 3-gram/full cell shows the
strongest interference: 8/200, only 4/50 on env_191, and 309 updates before
trajectories shortened through early failure.

Residual weighting is protective inside the bad full-context regime because it
concentrates loss on a subset of tokens.  It does not beat aTTT in the healthy
standalone regime: 40 versus 45, with near-identical mean reward (0.631 versus
0.635) and a nonsignificant paired binary difference.

Therefore the evidence does not support using the complete trajectory as the
training context for EnvScaler, nor a positive claim that residual pricing
beats unconditional aTTT NTP.  The defensible result is narrower: residual
weighting mitigates contextual observation-training damage, while aTTT's
standalone latest-observation formulation remains the best tested arm.

## Residual plus 3-gram follow-up

`latest_resid_ngram_standalone_paper` completed 200 unique rows with 387/387
successful updates, exactly two optimizer losses per update, and zero top-level
or update errors. It scored 40/200, 26/50 on env_191, and mean reward 0.628674.

- Versus residual/standalone: +6/-6, net 0, exact McNemar p=1.000, mean reward
  delta -0.002424.
- Versus aTTT paper: +4/-9, net -5, p=0.267, mean reward delta -0.0064285.
- Versus std: +7/-5, net +2, p=0.774, mean reward delta +0.020899.

The 3-gram factor was active: 213/387 candidates had mean repetition weight
below 1 and the overall mean was 0.851. Its lack of effect is therefore an
algorithmic null result, not an inactive-code failure.

## Root-cause audit

The positive residual is not sparse. Across the completed residual/standalone
arm, candidate-mean positive coverage is 87.8%, token-weighted coverage is
84.4%, no candidate has zero positive support, and only two candidates are
below 50%.

A separate zero-training audit ran 20 tasks and recorded 44 observations with
1,197 content tokens. It found:

- 91.2% of token deltas are positive;
- 87.5% of tokens have full-context probability above 99%;
- 93.5% of all positive residual mass comes from those near-deterministic
  full-context tokens;
- 74.0% of positive mass comes from tokens whose bare-standalone probability
  is below 1%.

The largest weights fall on deterministic tool-response fields and echoes such
as `success`, `Maintenance record`, `for machine`, machine identifiers, and
`LOC`. The current contrast therefore measures how well the action/history
predicts the structured tool return relative to an unnatural bare user
message. It is action-conditioned observation predictability, not hindsight
evidence changing a belief.

Post-hoc task analysis agrees. Residual magnitude tracks baseline familiarity
more than update value: mean residual weight correlates +0.252 with std reward
but -0.155 with reward lift over std; first-update delta correlates +0.241 with
std reward but -0.214 with lift. Residual features have approximately zero
association with beating aTTT.

Finally, weighted CE divides by total token weight. Multiplying every token in
a candidate by a common residual strength leaves the gradient unchanged. All
401 scheduled candidates in residual/standalone trained successfully, so the
implementation has no candidate-level residual gate; it only redistributes
mass within an almost-dense NTP target.

The matched latest-observation arm should therefore be described as a
conditional-predictability weighting ablation, not as the original T0b method.
Original T0b used a settled `action -> ok/ERROR` evidence block and fired on
100/200 hard tasks; removing that evidence to align with aTTT removed the
mechanism whose value the residual claim is about.

## Matched-action control audit

We tested whether placing the exact previous action in the hindsight prompt
creates a copy shortcut, and then tested the smallest proposed correction:
give both sides the identical action and vary only the result field.

For every executed action `A` with returned observation `O`, the scoring-only
audit evaluated the same assistant action tokens under three prompts:

- `base = log p(A | P)`;
- `masked = log p(A | P + action=A, result=[MASKED])`;
- `actual = log p(A | P + action=A, result=O)`.

This gives a pure hint contrast `masked - base`, the previous polluted
contrast `actual - base`, and the matched-action contrast
`actual - masked`. The audit used 40 tasks, 581 actions (494 ok, 87 ERROR),
and 36,579 action tokens. It performed no optimizer update and loaded no
adapter.

The copy-shortcut hypothesis is supported at the token-subset level. Under the
pure action hint, 98.97% of all actions and 100% of ERROR actions contain at
least one positive token. Among ERROR actions, 13.82% of tokens are positive
and 25.29% of actions have positive mean delta. It is not true that the whole
wrong action always becomes more probable: the ERROR mean delta remains
negative overall (-0.144).

The matched-action control removes some direct copying but does not recover a
valid credit signal:

| subset | positive tokens | actions with any positive token | positive-mean actions | mean positive weight conditional on positive |
| --- | ---: | ---: | ---: | ---: |
| all | 5.43% | 92.94% | 16.35% | 0.466 |
| ok | 5.32% | 92.31% | 14.57% | 0.255 |
| ERROR | 6.04% | 96.55% | 26.44% | 1.467 |

Thus positive residual coverage is sparse but not absent: the median action
has about 5% positive tokens, and only 41/581 actions have none. The important
failure is directionality. ERROR actions have slightly more positive tokens,
almost twice the positive-mean action rate, and 5.8 times the conditional
positive magnitude of ok actions. The matched signal's AUC for ranking ok
above ERROR is 0.420, below random. Its largest positive outliers are ERROR
examples on tool-call framing, function-name, identifier, and number tokens.

All three prompts are also saturated: 97.63% of base, 97.14% of masked, and
96.19% of actual action tokens already have probability above 99%. In the
matched contrast, 93.67% of deltas have absolute value below 1e-4. The control
therefore subtracts two nearly deterministic teacher-forced predictions and
leaves tiny prompt differences plus a small number of very large outliers.

Decision: do not train this matched-action/ReLU arm. Because weighted CE is
normalized by the sum of positive weights, 5.43% coverage would not merely
make the update weak; it would concentrate a full-sized update onto a few
high-variance tokens, disproportionately from ERROR actions. A further design
must first pass a scoring-only directionality gate: ok evidence should receive
more support than ERROR, and the result must not be dominated by echoed raw
action tokens.

## ERROR-gated signed-action result

The user-approved signed follow-up retained the matched-action control but
used outcome status for direction. Status-ok action tokens received capped
positive-residual CE; explicit ERROR action tokens received capped
negative-residual unlikelihood at lambda 0.02. Both branches used fixed
action-token denominators, masked chat/JSON/tool-name tokens, and were guarded
by exact active-position KL, gradient clipping, transactional LoRA rollback,
learning-rate backtracking, and signed post-update direction checks.

The complete four-shard run produced 200 unique rows with zero top-level
errors. Outcome:

| arm | success | env_191 | mean reward |
| --- | ---: | ---: | ---: |
| signed action UL 0.02 | 41/200 | 26/50 | 0.608392 |
| std | 38/200 | 25/50 | 0.607775 |
| retry | 37/200 | 24/50 | 0.620160 |
| T0b historical | 41/200 | 26/50 | 0.623577 |
| T0c | 40/200 | 26/50 | 0.614790 |
| T0c paper dose | 41/200 | 27/50 | 0.597358 |
| aTTT paper | 45/200 | 29/50 | 0.635103 |
| residual standalone | 40/200 | 26/50 | 0.631098 |

Paired results:

- versus std: +6/-3, net +3, exact McNemar p=0.508, reward +0.000617;
- versus retry: +7/-3, net +4, p=0.344, reward -0.011768;
- versus T0c paper: +6/-6, net 0, p=1.000, reward +0.011035;
- versus T0b historical: +4/-4, net 0, p=1.000, reward -0.015185;
- versus aTTT paper: +2/-6, net -4, p=0.289, reward -0.026710;
- versus residual standalone: +5/-4, net +1, p=1.000, reward -0.022706.

The safety mechanism did what it was designed to do. Of 417 selected windows,
399 produced accepted updates, 13 had no active signed weight, and 5 were
safely skipped after all four learning rates failed a direction gate. There
were no unexpected update errors or adapter-load rollbacks. Across accepted
updates, 191 attempts were rejected for KL, 10 for decreasing a material ok
branch, and 11 for increasing a material ERROR branch. The maximum accepted
KL was 0.019671. Seven of 798 accepted optimizer steps hit the 0.5 gradient
clip; no loss, gradient, or parameter became nonfinite.

Mechanically, the signed objective also moved in the requested direction:
95 material negative-branch updates lowered ERROR weighted log-prob by 0.04925
on average, and 167 material positive-branch updates raised ok weighted
log-prob by 0.01282. There were zero accepted material direction violations.

This mechanical success did not translate into a better policy. The arm tied
T0b and T0c-paper in binary success, remained four net tasks behind aTTT paper,
and had substantially lower graded reward than aTTT and the standalone
residual controls. Post hoc, 54 tasks had a material negative-branch update;
the signed arm solved 5/54, exactly the same as std, versus 7/54 for T0c paper
and 6/54 for aTTT paper. This subset is endogenous and harder, so it is not a
causal comparison, but it provides no evidence that suppressing failed tokens
rescued the tasks that positive-only learning could not solve.

The bounded conclusion is therefore: ERROR-gated residual unlikelihood is
stable and can reliably lower the selected failed-action tokens, but lowering
a bad action does not specify where its probability mass should go. This run
does not support the claim that the negative branch improves EnvScaler task
success. An exact lambda=0 arm under the same matched control, masks, KL, and
backtracking would be required to isolate the causal contribution of
unlikelihood; existing T0c controls differ in more than lambda.
