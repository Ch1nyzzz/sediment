# Module contracts

Law for all writers. `sediment/types.py` and `sediment/config.py` are frozen —
implement against them, never edit them. All heavy deps (torch, peft,
transformers, vllm) are lazy-imported inside functions; importing any
sediment module must succeed with only numpy + stdlib. Tests use MockEngine
and the stub trainer only. Style: match ../resid — English docstrings, type
hints, small modules, no dead code.

## engine (sediment/engine/)

```python
class Engine(Protocol):                      # base.py
    def generate(self, messages: list[Message], *, adapter: str = "base",
                 temperature: float, max_tokens: int) -> str: ...
    def score(self, messages: list[Message], *, adapter: str = "base"
              ) -> list[list[float]]:
        """Teacher-forced per-token logprobs, one list per message,
        aligned with the tokenization used by sediment.spans."""
    def load_adapter(self, version: AdapterVersion) -> None: ...
```
`mock.py`: MockEngine(policy=None, scorer=None) — deterministic, injectable
scripted policy/scoring like ../resid/resid/engine/mock.py (port + extend
with the adapter kwarg; different adapter names may route to different
injected policies so gate A/B is testable). Whitespace tokenizer, exposed as
`mock_tokenize(text) -> list[str]` for spans tests.
`vllm_client.py`: OpenAI-compatible client against a vLLM server: chat
completions with `model=<adapter name or base>`, prompt_logprobs-style
scoring (port the approach from ../resid/resid/engine/vllm_engine.py — read
it), `load_adapter` via POST /v1/load_lora_adapter. Constructor takes
base_url; no vllm import needed (pure HTTP via urllib). Server launch is out
of scope (documented command string constant is enough).

## envs + rollout

Port from ../resid/resid/envs/ and ../resid/resid/rollout/agent_loop.py:
`Env` protocol (reset(task) -> list[Message]; step(action_text) ->
(messages, done, reward)), `ToyOrderEnv` (hidden cancel rule),
`EnvScalerAdapter` (guarded: skip if third_party absent; path via cfg.data_dir).
`run_episode(engine, env, task, cfg, *, adapter: str, experience:
ExperienceBlock | None) -> Trajectory` — single rollout (G=1); when
`experience` is given its text is injected into the first user message
(reuse ../resid/resid/experience/context.py injection format). Tasks are
dicts: {"task_id", "env_family", "payload"...}; `make_toy_tasks(n, seed)
-> list[dict]` lives in envs.

## experience + hindsight + spans

`experience.build_block(retrieved: list[Trajectory], own: Trajectory | None,
cfg) -> ExperienceBlock` — numbered, outcome-tagged action->result lines,
results truncated to cfg.max_result_chars; `own` contributes ONLY an outcome
summary (final reward + tail of final feedback, no trajectory body) — port
the copy-safe design from ../resid/resid_verl/hindsight.py
build_group_experience and ../resid/resid/experience/context.py.
`spans.py`: port ../resid/resid/measure/spans.py — incremental chat-template
role mapping + suffix alignment; must work with mock_tokenize for tests and
accept a tokenizer callable.
`hindsight.score(engine, traj, block, cfg) -> HindsightResult` — two
engine.score calls (with/without block injected), align suffix spans, deltas
= with - without; obs_surprise = mean(max(d,0)) over tool tokens; act_gain =
mean(d) over assistant tokens.
`hindsight.to_train_sample(traj, hr, cfg) -> TrainSample` — weights: tool
tokens relu(delta), assistant tokens relu(delta), floor cfg.w_floor,
prompt/system 0.0.

## buffer

`class Buffer(path)`: `.add(traj)` (append jsonl + in-memory),
`.retrieve(task: dict, k) -> list[Trajectory]` — score = same env_family
(strong) + token-overlap of task payload text (weak), excludes same task_id;
`.load()` classmethod; `.replay_states(n, seed) -> list[list[Message]]` —
sample stored decision-point prefixes (messages up to a random assistant
turn) for gate G2.

## gate

`class Gate(cfg)`:
`.propose(hr: HindsightResult, traj: Trajectory) -> bool` — G1: obs_surprise
>= gate_min_surprise AND recurrence ledger (per env_family count of
above-threshold surprises this stream) >= gate_recurrence.
`.validate(candidate: UpdateCandidate, parent: AdapterVersion,
engine, replay_states, probe_tasks, run_probe) -> GateDecision` — G2:
generate on replay states with parent vs candidate adapter, fraction of
changed actions >= gate_min_behavior_change; G3: `run_probe(adapter_name,
tasks) -> float` success rate, delta vs parent >= gate_min_probe_delta.
Pure logic; all engine/probing passed in as callables so it is mock-testable.

## trainer + registry + merge

`trainer.train_candidate(samples: list[TrainSample], parent: AdapterVersion,
cfg, workdir) -> UpdateCandidate` — cfg.trainer=="stub": write
`{workdir}/{candidate_id}/adapter_meta.json` (task ids, weight stats), no
torch; "torch": lazy-import peft/transformers, LoRA(cfg.lora_r, alpha,
target q/v/gate/up/down), weighted CE with per-token weights from
TrainSample, cfg.epochs over samples, save_pretrained.
`class Registry(dir)`: `.base() -> AdapterVersion("v0000", None, None)`,
`.publish(candidate | path, parent, provenance) -> AdapterVersion` (next
vNNNN, copy/point to dir, write meta.json, update `current` symlink),
`.current() -> AdapterVersion`, `.get(name)`, `.history() -> list`.
`merge.merge(session: AdapterVersion, candidate: UpdateCandidate, cfg,
registry) -> AdapterVersion` — stub mode: publish candidate as new version
with provenance union; torch mode: EMA over LoRA tensors (alpha =
cfg.merge_alpha) then publish.

## scheduler + router

`class Router(engines: list[Engine])`: `.for_task(task_id) -> Engine`
(stable hash); `.all()`.
`scheduler.run_stream(tasks, engines, buffer, gate, trainer_fn, registry,
cfg, *, run_probe=None, on_window=None) -> list[StreamRecord]` — asyncio;
windows of cfg.window_size; each window: (1) first attempts concurrently
with `registry.current()` (predict-then-update: record success BEFORE any
update from this window), (2) per task: retrieve -> build_block -> hindsight
-> gate.propose; failed+proposed tasks may retry once with block in context
(record retried, not in curve), (3) accepted tasks' TrainSamples ->
trainer_fn -> candidate -> gate.validate -> merge/publish. Async wrapping of
sync engine calls via asyncio.to_thread. on_window(window_idx, records)
callback for snapshots. Deterministic under seed with MockEngine.

## eval + scripts

`eval.wauc(records) -> float` (mean success over stream, trapezoid over
window means); `eval.gain(records, baseline_records) -> float`;
`eval.write_report(out_dir, cfg, records, registry)` -> stream.jsonl +
summary.json (+ printed table).
`scripts/run_stream.py`: argparse over StreamConfig fields (at least
--engine --tasks --window --out --seed --trainer); mock path: ToyOrderEnv
tasks, MockEngine, stub trainer; prints summary.
`scripts/run_p1_probe.py`: per-task loop (attempt -> block -> hindsight ->
train stub/torch -> re-attempt), reports per-task gain jsonl. Keep thin.

## tests

Each writer ships `tests/test_<module>.py` (pytest, mock/stub only, no
network). Scheduler test: 8 toy tasks, window 4, scripted MockEngine where
adapter "v0001" answers a task family correctly that "base" fails ->
asserts predict-then-update ordering (window k successes computed with
pre-window adapter), a publish happens, records/jsonl written.
