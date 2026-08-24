# Signed Residual Follow-up: Safety-Gated Design

## Evidence boundary

The completed residual/standalone arm does not have sparse positive support:
84.4% of observation content tokens are positive after weighting by token
count. The follow-up therefore tests whether signed negative information adds
value; it is not justified as a fix for missing positive gradients.

## Do not use signed CE

Never implement the negative branch as a negative coefficient on token NLL.
That objective is unbounded and can drive the target logit toward negative
infinity. Positive and negative branches must be separately normalized so the
number or magnitude of negative tokens cannot silently change the dose.

## Preferred negative target

Do not initially apply unlikelihood to the observed token itself. A negative
residual means the actual observed token was less likely after conditioning on
the trajectory, but the token still occurred. Suppressing it may teach the
model to deny surprising real observations.

Instead, inspect the top-k alternatives at each observation position under:

- the standalone observation prefix, `p_s(v)`;
- the full natural trajectory prefix, `p_f(v)`.

Use as unlikelihood targets only counterfactual tokens for which the evidence
suppresses the prior prediction:

`u(j,v) = g_j * clip(relu(log p_s(v) - log p_f(v)), 0, c_neg)`.

Here `g_j` is the same aTTT 3-gram repetition discount. This negative branch
encodes what the evidence ruled out without penalizing the ground-truth
observation token merely because it was surprising.

## Objective

On the standalone latest-observation context:

`L = L_pos + lambda_ul * L_unlikely + beta_kl * KL(p_pre || p_theta)`.

- `L_pos`: current positive residual times 3-gram weighted CE.
- `L_unlikely`: `-log(1 - p_theta(v))` on evidence-suppressed top-k
  counterfactual tokens, normalized independently from `L_pos`.
- `p_pre`: the episode adapter immediately before this update.
- KL support: observation positions used by either branch; exact vocabulary KL
  if memory permits, otherwise a documented top-k-plus-other bucket KL.

## Staged gates

1. **Token audit, no training.** Log decoded top negative targets, their two
   probabilities, residual, position, and observation snippet. Categorize at
   least 200 targets as grounded entities/values, syntax, explicit error text,
   or contradicted alternatives. Do not proceed if target identity or alignment
   is ambiguous.
2. **Four-dose safety sweep.** Run the same small task set with
   `lambda_ul = 0, 0.02, 0.05, 0.10`; retain paper dose for the positive branch.
   This stage establishes stability, not efficacy.
3. **Matched efficacy run.** Only a dose that passes every guard may receive a
   200-task run paired against residual+3-gram, residual-only, aTTT paper, and
   std.

## Mandatory guards

- Cap positive and negative residual weights at audit-derived percentiles.
- Global trainable-parameter gradient norm clip at 0.5; log pre-clip norm.
- Cache the pre-update adapter state and roll back the entire candidate on any
  nonfinite loss/gradient/parameter, adapter-save error, or KL trust-region
  violation.
- Calibrate the KL threshold from `lambda_ul=0` control updates before choosing
  a bound; do not invent a threshold that rejects the existing positive arm.
- Record positive CE, unlikelihood loss, KL, total loss, grad norm, post-step
  KL, rollback reason, and adapter load success for every substep.
- Never load a rejected adapter into vLLM. Reset adapter and history between
  episodes exactly as in the current paper-dose arms.
