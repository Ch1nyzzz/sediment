# sediment — residual-gated streaming test-time training for agents

Experiences flow through; only what passes the gate settles into the weights.

An agent serves a task stream (predict-then-update, G=1 per task). After each
task, a hindsight pass prices the experience via belief residuals (per-token
deltas from scoring the trajectory with vs. without an evidence block built
from retrieved past experiences + the task's own outcome). A three-stage gate
(magnitude + recurrence ledger, behavioral replay, transfer probe) decides
what gets distilled into a persistent session LoRA. Goal: the agent gets
stronger the more it is used — without rewards, without offline RL.

Docs (Chinese): `docs/BACKGROUND.md` (positioning & claims),
`docs/EXPERIMENTS.md` (P1–P4), `docs/SYSTEM.md` (4-GPU design),
`docs/LITERATURE.md` (survey notes). Module APIs: `CONTRACTS.md`.

## Quickstart (mock, no GPU)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/run_stream.py --engine mock --tasks 8 --window 4
```

GPU path (vLLM serving + peft trainer) is lazy-imported; see `docs/SYSTEM.md`.

## Layout

```
sediment/types.py       shared dataclasses (contract; do not fork per-module)
sediment/config.py      StreamConfig
sediment/engine/        Engine protocol, MockEngine, vLLM client (multi-LoRA, scoring)
sediment/envs/          Env protocol, ToyOrderEnv, EnvScaler adapter
sediment/rollout/       single-rollout agent loop (G=1)
sediment/buffer.py      experience buffer + retrieval (the "group" is the stream's history)
sediment/experience.py  evidence block builder (retrieved + own outcome)
sediment/hindsight.py   2-prefill scoring -> per-token deltas -> w_t
sediment/spans.py       chat-template role/token span alignment
sediment/gate.py        magnitude/recurrence + behavioral replay + transfer probe
sediment/trainer.py     w_t-weighted distillation into LoRA (torch lazy; stub for tests)
sediment/registry.py    versioned adapter registry (runtime hot-load friendly)
sediment/merge.py       candidate -> session merge (EMA)
sediment/scheduler.py   windowed predict-then-update pipeline (staleness = 1 window)
sediment/router.py      sticky episode routing over engine pool
sediment/eval.py        W-AUC, gain, snapshot runner, report writers
scripts/run_stream.py   streaming experiment CLI
scripts/run_p1_probe.py per-task internalization probe (P1)
```

Ported/adapted from `../resid` (P0 probe) and `../resid/resid_verl`
(hindsight + credit_belief loss, de-verl'd).
