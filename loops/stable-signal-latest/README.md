# Stable memory signal pilot (2026-08-28)

Verdict: **not supported; do not scale the current stable-signal model**.

The family-held-out 10-train / 12-test pilot completed all 22 proposal windows
and the preregistered six-arm evaluation. The stable arm failed the efficacy
gate: it did not beat the discovery-selected single, donor mean, or
shuffled-memory control. Its deployment write rate was zero.

- [`report.md`](report.md): scientific result, integrity limits, and next-step
  decision.
- [`evidence/aggregate_public.json`](evidence/aggregate_public.json): public
  six-arm aggregate without per-window pairs.
- [`evidence/integrity.json`](evidence/integrity.json): checksums, blinding,
  no-op identity, test, service, and artifact-boundary evidence.
- [`evidence/`](evidence/): frozen protocol/partition, structural audit,
  train/checkpoint/test aggregate reports, and the pre-unblinding numerical-rank
  amendment.

No raw trajectories, JSONL manifests, task-level A/B pairs, synthesized
adapters, checkpoints, or worker logs are included. They remain in the remote
working root recorded in the report.
