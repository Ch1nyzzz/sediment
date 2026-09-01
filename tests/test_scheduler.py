"""Scheduler + Router tests: predict-then-update ordering, publish, jsonl records.

Collaborator modules (envs, rollout, experience, hindsight, merge) are being
written concurrently, so contract-shaped stubs are injected via sys.modules;
buffer/gate/registry/trainer are stubbed through run_stream's own parameters.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import types
from typing import Any, Optional

import pytest

import sediment
import sediment.harness  # noqa: F401  (binds sediment.envs.base before the stub replaces sediment.envs)
from sediment.config import StreamConfig
from sediment.router import Router
from sediment.scheduler import run_stream
from sediment.types import (
    AdapterVersion,
    ExperienceBlock,
    GateDecision,
    HindsightResult,
    Message,
    StreamRecord,
    TrainSample,
    Trajectory,
    UpdateCandidate,
)

HARD = "toy-hard"  # solved only by adapter v0001+
EASY = "toy-easy"


# ---------------------------------------------------------------- stubs

class ScriptedMockEngine:
    """Deterministic Engine: HARD family answered correctly only by >= v0001."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []  # (task_id, adapter) per generate

    def generate(self, messages: list[Message], *, adapter: str = "base",
                 temperature: float = 0.0, max_tokens: int = 64) -> str:
        user = next(m for m in reversed(messages) if m.role == "user")
        # Retry attempts carry "block...\n\n" before the task line: parse last line.
        task_id, family = user.content.splitlines()[-1].split("|")[:2]
        self.calls.append((task_id, adapter))
        if family == HARD and adapter in ("base", "v0000"):
            return "WRONG"
        return "CORRECT"

    def score(self, messages: list[Message], *, adapter: str = "base") -> list[list[float]]:
        return [[0.0] for _ in messages]

    def load_adapter(self, version: AdapterVersion) -> None:
        pass


class StubToyOrderEnv:
    """Contract Env: reset(task) -> messages; step(text) -> (msgs, done, reward)."""

    def reset(self, task: dict[str, Any]) -> list[Message]:
        return [Message("system", "sys"),
                Message("user", f"{task['task_id']}|{task['env_family']}|{task.get('payload', '')}")]

    def step(self, action_text: str) -> tuple[list[Message], bool, float]:
        ok = action_text == "CORRECT"
        return [Message("tool", "ok" if ok else "bad")], True, 1.0 if ok else 0.0


def stub_run_episode(engine, env, task, cfg, *, adapter: str,
                     experience: Optional[ExperienceBlock] = None,
                     stepwise=None, **_kwargs) -> Trajectory:
    messages = env.reset(task)
    if experience is not None:  # block injected into first user message (real tag format)
        messages[-1] = Message("user", f"<previous_attempts>\n{experience.text}\n</previous_attempts>"
                                       f"\n\n{messages[-1].content}")
    if stepwise is not None:
        stepwise.pre_generate(messages)
    out = engine.generate(messages, adapter=adapter,
                          temperature=cfg.temperature, max_tokens=cfg.max_tokens)
    messages = messages + [Message("assistant", out)]
    obs, done, reward = env.step(out)
    messages = messages + obs
    return Trajectory(task_id=task["task_id"], env_family=task["env_family"],
                      messages=messages, reward=reward, success=reward > 0,
                      adapter=adapter, steps=1,
                      meta=({"stepwise_events": list(stepwise.events)}
                            if stepwise is not None else {}))


def stub_build_block(retrieved: list[Trajectory], own: Optional[Trajectory],
                     cfg: StreamConfig, **kwargs) -> ExperienceBlock:
    del kwargs
    return ExperienceBlock(text=f"block:{own.task_id if own else 'none'}",
                           source_task_ids=[t.task_id for t in retrieved],
                           includes_own_outcome=own is not None)


def stub_score(engine, traj: Trajectory, block: ExperienceBlock,
               cfg: StreamConfig) -> HindsightResult:
    s = 0.0 if traj.success else 1.0  # failures are maximally surprising
    return HindsightResult(task_id=traj.task_id, spans=[], obs_surprise=s, act_gain=s)


def stub_to_train_sample(traj: Trajectory, hr: HindsightResult,
                         cfg: StreamConfig) -> TrainSample:
    return TrainSample(task_id=traj.task_id, messages=traj.messages,
                       token_weights_by_msg=[[0.0] for _ in traj.messages])


def stub_merge(session: AdapterVersion, candidate: UpdateCandidate,
               cfg: StreamConfig, registry) -> AdapterVersion:
    return registry.publish(candidate, session, list(candidate.task_ids))


class StubBuffer:
    def __init__(self) -> None:
        self.items: list[Trajectory] = []

    def add(self, traj: Trajectory) -> None:
        self.items.append(traj)

    def retrieve(self, task: dict[str, Any], k: int, *, scope: str = "all",
                 score_mode: str = "task", offset: int = 0,
                 diversity: str = "task") -> list[Trajectory]:
        del score_mode, offset, diversity
        return []

    def replay_states(self, n: int, seed: int) -> list[list[Message]]:
        return []


class StubGate:
    def __init__(self, cfg: StreamConfig) -> None:
        self.cfg = cfg
        self.validated = 0

    def propose(self, hr: HindsightResult, traj: Trajectory) -> bool:
        return hr.obs_surprise >= self.cfg.gate_min_surprise

    def validate(self, candidate, parent, engine, replay_states,
                 probe_tasks, run_probe=None) -> GateDecision:
        self.validated += 1
        return GateDecision(candidate_id=candidate.candidate_id, passed=True,
                            magnitude=1.0, reason="stub-pass")


class StubRegistry:
    def __init__(self) -> None:
        self._versions = [AdapterVersion("v0000", None, None)]
        self.publishes = 0

    def base(self) -> AdapterVersion:
        return self._versions[0]

    def current(self) -> AdapterVersion:
        cur = getattr(self, "_cur", None)
        return next(v for v in self._versions if v.name == cur) if cur else self._versions[-1]

    def publish(self, candidate: UpdateCandidate, parent: AdapterVersion,
                provenance: list[str]) -> AdapterVersion:
        v = AdapterVersion(f"v{len(self._versions):04d}", candidate.adapter_path,
                           parent.name, list(provenance))
        self._versions.append(v)
        self._cur = None
        self.publishes += 1
        return v

    def history(self) -> list[AdapterVersion]:
        return list(self._versions)

    def _set_current(self, name: str) -> None:
        self._versions.append(next(v for v in self._versions if v.name == name))
        self._versions[-1:] = []  # history unchanged; current = named version
        self._cur = name


def stub_trainer(samples: list[TrainSample], parent: AdapterVersion,
                 cfg: StreamConfig, workdir: str) -> UpdateCandidate:
    return UpdateCandidate(candidate_id=f"cand-{parent.name}",
                           task_ids=[s.task_id for s in samples],
                           adapter_path=f"{workdir}/cand-{parent.name}",
                           parent=parent.name)


# ---------------------------------------------------------------- fixtures

@pytest.fixture()
def stub_modules(monkeypatch):
    """Install contract-shaped collaborator modules for scheduler's lazy imports."""
    def install(name: str, **attrs) -> types.ModuleType:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        monkeypatch.setitem(sys.modules, name, mod)
        short = name.split(".")[1]
        if name.count(".") == 1:  # direct child: also patch package attribute
            monkeypatch.setattr(sediment, short, mod, raising=False)
        return mod

    install("sediment.envs", ToyOrderEnv=StubToyOrderEnv)
    install("sediment.experience", build_block=stub_build_block)
    install("sediment.hindsight", score=stub_score, to_train_sample=stub_to_train_sample)
    install("sediment.merge", merge=stub_merge)
    agent_loop = install("sediment.rollout.agent_loop", run_episode=stub_run_episode)
    install("sediment.rollout", agent_loop=agent_loop)


def make_tasks(n: int) -> list[dict[str, Any]]:
    """Alternating easy/hard toy tasks: odd indices are the HARD family."""
    return [{"task_id": f"t{i}", "env_family": HARD if i % 2 else EASY,
             "payload": f"order-{i}"} for i in range(n)]


def make_cfg(tmp_path) -> StreamConfig:
    return StreamConfig(window_size=4, retry_on_fail=True, gate_min_surprise=0.05,
                        retrieval_k=2, gate_replay_states=4, seed=0,
                        temperature=0.0, out_dir=str(tmp_path))


def run_once(tmp_path, n_tasks: int = 8, n_engines: int = 2):
    cfg = make_cfg(tmp_path)
    engines = [ScriptedMockEngine() for _ in range(n_engines)]
    buffer, gate, registry = StubBuffer(), StubGate(cfg), StubRegistry()
    seen: list[tuple[int, int]] = []
    records = run_stream(make_tasks(n_tasks), engines, buffer, gate, stub_trainer,
                         registry, cfg,
                         on_window=lambda i, recs: seen.append((i, len(recs))))
    return records, engines, buffer, gate, registry, seen


# ---------------------------------------------------------------- router tests

def test_router_stable_hash_and_all():
    engines = [object(), object(), object()]
    router = Router(engines)
    import hashlib
    for tid in ("t0", "t1", "abc", "task-xyz"):
        expect = engines[int(hashlib.sha1(tid.encode()).hexdigest(), 16) % 3]
        assert router.for_task(tid) is expect
        assert router.for_task(tid) is router.for_task(tid)
    assert router.all() == engines
    assert router.all() is not engines  # copy
    with pytest.raises(ValueError):
        Router([])


# ---------------------------------------------------------------- scheduler tests

def test_rollout_workers_overrides_asyncio_default_pool(monkeypatch):
    import asyncio
    import sediment.scheduler as scheduler

    lock = threading.Lock()
    active = peak = 0

    async def fake_stream(*args, **kwargs):
        async def one():
            def work():
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.05)
                with lock:
                    active -= 1

            await asyncio.to_thread(work)

        await asyncio.gather(*(one() for _ in range(40)))
        return []

    monkeypatch.setattr(scheduler, "_stream", fake_stream)
    cfg = StreamConfig(extra={"rollout_workers": 40})
    assert scheduler.run_stream([], [], None, None, None, None, cfg) == []
    assert peak == 40


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "64"])
def test_rollout_workers_must_be_positive_integer(value):
    cfg = StreamConfig(extra={"rollout_workers": value})
    with pytest.raises(ValueError, match="positive integer"):
        run_stream([], [], None, None, None, None, cfg)


def test_stream_predict_then_update(stub_modules, tmp_path):
    records, engines, buffer, gate, registry, seen = run_once(tmp_path)

    assert len(records) == 8
    assert [r.window for r in records] == [0] * 4 + [1] * 4

    # Window 0 served entirely by the pre-update adapter: HARD tasks fail.
    w0, w1 = records[:4], records[4:]
    assert [r.adapter for r in w0] == ["v0000"] * 4
    assert [r.success for r in w0] == [True, False, True, False]

    # The publish happened inside window 0 ...
    assert registry.publishes == 1
    assert registry.current().name == "v0001"
    assert [v.name for v in registry.history()] == ["v0000", "v0001"]

    # ... yet every generate for window-0 tasks (attempts + retries) used v0000.
    calls = [c for e in engines for c in e.calls]
    w0_ids = {f"t{i}" for i in range(4)}
    assert {a for tid, a in calls if tid in w0_ids} == {"v0000"}
    assert {a for tid, a in calls if tid not in w0_ids} == {"v0001"}

    # Window 1 served by the published adapter: HARD family now solved.
    assert [r.adapter for r in w1] == ["v0001"] * 4
    assert [r.success for r in w1] == [True, True, True, True]

    # Retry marked but success/reward of the record untouched.
    for r in w0:
        hard = r.meta["env_family"] == HARD
        assert r.retried is hard
        assert r.gated_in is hard
        if hard:
            assert r.success is False and r.reward == 0.0
            assert "retry" in r.timings and "train" in r.timings
    for r in w1:  # nothing failed, nothing proposed or gated
        assert not r.retried and not r.gated_in and r.meta["proposed"] is False

    # Buffer holds every trajectory: 8 first attempts + 2 retries.
    assert len(buffer.items) == 10
    retries = [t for t in buffer.items if t.is_retry]
    assert sorted(t.task_id for t in retries) == ["t1", "t3"]
    assert gate.validated == 1
    assert seen == [(0, 4), (1, 4)]


def test_records_jsonl_roundtrip(stub_modules, tmp_path):
    records, *_ = run_once(tmp_path)
    path = tmp_path / "stream.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")
    loaded = [json.loads(line) for line in open(path)]
    assert [d["task_id"] for d in loaded] == [f"t{i}" for i in range(8)]
    assert all("attempt" in d["timings"] and "hindsight" in d["timings"] for d in loaded)
    assert [d["adapter"] for d in loaded] == ["v0000"] * 4 + ["v0001"] * 4


def test_stream_deterministic(stub_modules, tmp_path):
    def strip(recs: list[StreamRecord]) -> list[dict]:
        out = []
        for r in recs:
            d = r.to_dict()
            d.pop("timings")  # wall-clock only nondeterminism
            out.append(d)
        return out

    a, *_ = run_once(tmp_path, n_engines=3)
    b, *_ = run_once(tmp_path, n_engines=3)
    assert strip(a) == strip(b)


def test_partial_window_and_no_publish(stub_modules, tmp_path):
    cfg = make_cfg(tmp_path)
    tasks = [{"task_id": f"e{i}", "env_family": EASY, "payload": ""} for i in range(6)]
    engines = [ScriptedMockEngine()]
    registry = StubRegistry()
    records = run_stream(tasks, engines, StubBuffer(), StubGate(cfg), stub_trainer,
                         registry, cfg)
    assert [r.window for r in records] == [0, 0, 0, 0, 1, 1]
    assert all(r.success for r in records)
    assert registry.publishes == 0  # no failures -> no proposals -> no train
    assert all(r.adapter == "v0000" for r in records)


def test_unknown_env_family_raises(stub_modules, tmp_path):
    cfg = make_cfg(tmp_path)
    tasks = [{"task_id": "x0", "env_family": "warp", "payload": ""}]
    with pytest.raises(ValueError, match="env_family"):
        run_stream(tasks, [ScriptedMockEngine()], StubBuffer(), StubGate(cfg),
                   stub_trainer, StubRegistry(), cfg)


def test_serve_experience_injects_block_into_first_attempt(stub_modules, tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.serve_experience = True
    cfg.gate_min_surprise = 999  # ICL arm: no proposal, no training
    buffer, gate, registry = StubBuffer(), StubGate(cfg), StubRegistry()
    records = run_stream(make_tasks(8), [ScriptedMockEngine()], buffer, gate,
                         stub_trainer, registry, cfg)
    assert registry.publishes == 0
    firsts = [t for t in buffer.items if not t.is_retry]
    assert len(firsts) == 8
    for t in firsts:  # served block is stripped again before buffer/hindsight (student view)
        user = next(m for m in t.messages if m.role == "user")
        assert not user.content.startswith("block:none") and user.content.startswith("t")
    assert all(r.meta["served_experience"] == [] for r in records)


def test_pure_memory_hindsight_excludes_current_outcome(stub_modules, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    cfg.hindsight_include_own_outcome = False
    seen_own = []

    def capture_block(retrieved, own, cfg, **kwargs):
        seen_own.append(own)
        return stub_build_block(retrieved, own, cfg, **kwargs)

    monkeypatch.setattr(sys.modules["sediment.experience"], "build_block", capture_block)
    run_stream(make_tasks(4), [ScriptedMockEngine()], StubBuffer(), StubGate(cfg),
               stub_trainer, StubRegistry(), cfg)
    assert len(seen_own) == 4
    assert all(own is None for own in seen_own)


def test_binary_mode_proposes_by_gate_count_not_surprise(stub_modules, tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.weight_mode = "binary"
    cfg.min_gate_tokens = 0     # stub samples carry zero gated tokens -> only 0 passes
    cfg.gate_min_surprise = 999  # G1 would reject everything; binary path must ignore it
    cfg.retry_on_fail = False
    buffer, gate, registry = StubBuffer(), StubGate(cfg), StubRegistry()
    records = run_stream(make_tasks(4), [ScriptedMockEngine()], buffer, gate,
                         stub_trainer, registry, cfg)
    assert all(r.meta["proposed"] for r in records) and all(r.meta["n_gate"] == 0 for r in records)
    assert registry.publishes == 1


def test_stepwise_feedback_stream_trains_successes_and_failures(stub_modules, tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.stepwise_feedback_distill = True
    cfg.kl_states = "first"
    cfg.kl_teacher = "local"
    cfg.kl_target = True
    cfg.weight_mode = "sft"
    cfg.propose_rule = "all"
    cfg.retry_on_fail = False
    cfg.turn_decay = 0.6
    trained: list[list[TrainSample]] = []

    def capture_trainer(samples, parent, cfg, workdir):
        trained.append(list(samples))
        return stub_trainer(samples, parent, cfg, workdir)

    buffer, gate, registry = StubBuffer(), StubGate(cfg), StubRegistry()
    records = run_stream(make_tasks(4), [ScriptedMockEngine()], buffer, gate,
                         capture_trainer, registry, cfg)

    # All four first attempts contribute exactly one executed assistant turn,
    # including the two naturally correct and two naturally failed episodes.
    assert len(trained) == 1
    samples = trained[0]
    assert sorted(s.task_id for s in samples) == [f"t{i}@turn00" for i in range(4)]
    assert [r.success for r in records] == [True, False, True, False]
    assert all(r.meta["proposed"] for r in records)
    assert all(r.meta["stepwise"]["supervised_turns"] == 1 for r in records)
    assert all("behavior_score" in r.timings for r in records)
    assert all(s.loss_scale == 1.0 for s in samples)
    assert all(s.behavior_logprobs_by_msg[-1] == [0.0] for s in samples)
    for sample in samples:
        succeeded = int(sample.task_id.removeprefix("t").split("@")[0]) % 2 == 0
        feedback = sample.teacher_contexts["feedback"]
        assert ("ok" if succeeded else "bad") in feedback
        assert sample.messages[-1].role == "assistant"
        assert sample.token_weights_by_msg[-1] == [1.0]
    assert registry.publishes == 1


def test_stepwise_feedback_failure_only_excludes_successes(stub_modules, tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.stepwise_feedback_distill = True
    cfg.stepwise_feedback_failures_only = True
    cfg.kl_states = "first"
    cfg.kl_teacher = "local"
    cfg.kl_target = True
    cfg.weight_mode = "sft"
    cfg.propose_rule = "all"
    cfg.retry_on_fail = False
    trained: list[list[TrainSample]] = []

    def capture_trainer(samples, parent, cfg, workdir):
        trained.append(list(samples))
        return stub_trainer(samples, parent, cfg, workdir)

    records = run_stream(make_tasks(4), [ScriptedMockEngine()], StubBuffer(),
                         StubGate(cfg), capture_trainer, StubRegistry(), cfg)

    assert len(trained) == 1
    assert sorted(s.task_id for s in trained[0]) == ["t1@turn00", "t3@turn00"]
    assert [r.meta["proposed"] for r in records] == [False, True, False, True]
    assert records[0].meta["stepwise"]["skipped"] == "successful_first_attempt"
    assert records[2].meta["stepwise"]["skipped"] == "successful_first_attempt"


def test_rollback_republishes_previous_version_when_probe_regressed(stub_modules, tmp_path, monkeypatch):
    """Window 0 merge regresses the probe -> window 1 re-probes v0000; it wins ->
    v0000's weights are republished and window 1's candidate is discarded."""
    from sediment.types import GateDecision
    cfg = make_cfg(tmp_path)
    cfg.retry_on_fail = False
    cfg.rollback_thr = 0.05
    cfg.gate_min_probe_delta = -999
    cfg.weight_mode, cfg.min_gate_tokens = "binary", 0  # every task proposes
    rates = {"v0000": 0.9, "cand-v0000": 0.5, "v0001": 0.5, "cand-v0001": 0.4}
    probed = []

    def run_probe(name, ptasks):
        probed.append(name)
        return rates.get(name, 0.0)

    def validate(self, candidate, parent, engine, replay_states, probe_tasks, run_probe=None):
        c, p = run_probe(candidate.candidate_id, probe_tasks), run_probe(parent.name, probe_tasks)
        return GateDecision(candidate_id=candidate.candidate_id, passed=True, magnitude=1.0,
                            probe_delta=c - p, probe_rate=c, parent_rate=p, reason="stub")
    monkeypatch.setattr(StubGate, "validate", validate)
    buffer, gate, registry = StubBuffer(), StubGate(cfg), StubRegistry()
    run_stream(make_tasks(8), [ScriptedMockEngine()], buffer, gate, stub_trainer, registry, cfg,
               run_probe=run_probe, probe_tasks=[{"task_id": "p0"}])
    # window 0: v0001 published (probe 0.9 -> 0.5 flags a re-check); window 1:
    # v0000 re-probed, wins -> current rolled back to base, candidate discarded
    assert [v.name for v in registry.history()] == ["v0000", "v0001"]
    assert registry.current().name == "v0000"
    assert "v0000" in probed[2:]  # re-probed at window 1


# ---------------------------------------------------------------- kl_states=retry

class BlockAwareMockEngine(ScriptedMockEngine):
    """HARD is solved by v0000 only when the own-failure block is in context."""

    def generate(self, messages, *, adapter="base", temperature=0.0, max_tokens=64):
        user = next(m for m in reversed(messages) if m.role == "user")
        if "<previous_attempts>" in user.content and "block:" in user.content:
            self.calls.append((user.content.splitlines()[-1].split("|")[0], adapter))
            return "CORRECT"
        return super().generate(messages, adapter=adapter, temperature=temperature,
                                max_tokens=max_tokens)


class FeedbackAwareMockEngine(ScriptedMockEngine):
    """HARD is solved only when turn-1 previous-attempt feedback is visible."""

    def generate(self, messages, *, adapter="base", temperature=0.0, max_tokens=64):
        if any("<experience_hint previous_attempt>" in m.content for m in messages):
            user = next(m for m in reversed(messages) if m.role == "user")
            self.calls.append((user.content.splitlines()[-1].split("|")[0], adapter))
            return "CORRECT"
        return super().generate(messages, adapter=adapter, temperature=temperature,
                                max_tokens=max_tokens)


def _retry_cfg(tmp_path, **kw) -> StreamConfig:
    cfg = make_cfg(tmp_path)
    cfg.weight_mode, cfg.kl_states, cfg.own_view = "sft", "retry", "full"
    cfg.propose_rule = "advantage"
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _run_retry_mode(tmp_path, engine, **kw):
    cfg = _retry_cfg(tmp_path, **kw)
    trained: list[list[TrainSample]] = []

    def trainer(samples, parent, cfg, workdir):
        trained.append(list(samples))
        return stub_trainer(samples, parent, cfg, workdir)

    buffer, gate, registry = StubBuffer(), StubGate(cfg), StubRegistry()
    records = run_stream(make_tasks(4), [engine], buffer, gate, trainer, registry, cfg)
    return records, buffer, registry, trained


def test_kl_states_retry_trains_on_the_stripped_redo(stub_modules, tmp_path):
    records, buffer, registry, trained = _run_retry_mode(tmp_path, BlockAwareMockEngine())
    # metric = first attempt (HARD fails under v0000); the redo succeeded
    assert [r.success for r in records] == [True, False, True, False]
    for r in records:
        hard = r.meta["env_family"] == HARD
        assert r.retried is hard and r.meta["proposed"] is hard and r.gated_in is hard
        if hard:
            assert r.meta["redo_success"] is True and r.meta["redo_reward"] == 1.0
            assert "hindsight" in r.timings  # the REDO was scored, not the first attempt
        else:
            assert "hindsight" not in r.timings
    # the train samples are the redos in the student view: successful (tool "ok"),
    # with the own-failure block stripped from the first user message
    assert len(trained) == 1 and [s.task_id for s in trained[0]] == ["t1", "t3"]
    for s in trained[0]:
        assert s.messages[-1].content == "ok"
        user = next(m for m in s.messages if m.role == "user")
        assert "<previous_attempts>" not in user.content and user.content.startswith(s.task_id)
    # buffer stores the redo stripped as well, marked as a retry
    retries = [t for t in buffer.items if t.is_retry]
    assert sorted(t.task_id for t in retries) == ["t1", "t3"]
    assert all("<previous_attempts>" not in t.messages[1].content for t in retries)
    assert all(t.meta["served_block_chars"] > 0 for t in retries)
    assert registry.publishes == 1


def test_kl_states_retry_advantage_drops_failed_redos_all_keeps_them(stub_modules, tmp_path):
    # plain engine: the redo fails too -> advantage proposes nothing
    records, _, registry, trained = _run_retry_mode(tmp_path, ScriptedMockEngine())
    assert all(r.meta["proposed"] is False for r in records)
    assert [r.retried for r in records] == [False, True, False, True]
    assert registry.publishes == 0 and trained == []
    # propose_rule=all: the failed redo is still a KL sample
    records, _, registry, trained = _run_retry_mode(tmp_path, ScriptedMockEngine(),
                                                     propose_rule="all")
    assert [r.meta["proposed"] for r in records] == [False, True, False, True]
    assert all(r.meta["redo_success"] is False for r in records if r.retried)
    assert len(trained) == 1 and [s.task_id for s in trained[0]] == ["t1", "t3"]
    assert registry.publishes == 1


def test_trajectory_dpo_pairs_complete_redo_with_complete_first_attempt(
        stub_modules, tmp_path):
    records, _, _, trained = _run_retry_mode(
        tmp_path, BlockAwareMockEngine(), fork_objective="trajectory_dpo")
    assert len(trained) == 1 and len(trained[0]) == 2
    for sample in trained[0]:
        assert sample.rejected is None
        assert sample.rejected_messages is not None
        assert sample.messages[-1].content == "ok"  # complete successful redo
        assert sample.rejected_messages[-1].content == "bad"  # complete failed first
        assert sum(m.role == "assistant" for m in sample.messages) >= 1
        rec = next(r for r in records if r.task_id == sample.task_id)
        assert rec.meta["preference_scope"] == "trajectory"


def test_feedback_redo_uses_four_state_aligned_attempts_without_principles(
        stub_modules, tmp_path):
    records, _, _, trained = _run_retry_mode(
        tmp_path,
        FeedbackAwareMockEngine(),
        fork_objective="trajectory_dpo",
        redo_block_samples=0,
        redo_stepwise_samples=0,
        redo_feedback_samples=4,
        stepwise_experience=False,
        stepwise_extract=False,
    )
    assert len(trained) == 1 and len(trained[0]) == 2
    for sample in trained[0]:
        rec = next(r for r in records if r.task_id == sample.task_id)
        assert rec.meta["redo_group"] == {
            "n": 4,
            "n_success": 4,
            "chosen": 0,
            "chosen_context": "previous_attempt_feedback",
            "success_idx": [0, 1, 2, 3],
        }
        assert rec.meta["attempt_feedback_hints"] == 1
        assert all("previous_attempt" not in m.content for m in sample.messages)


def test_own_view_full_requires_retry_states(stub_modules, tmp_path, capsys):
    cfg = make_cfg(tmp_path)
    cfg.own_view = "full"  # kl_states stays "first": scored tokens sit in the block
    # 08-31: demoted from ValueError to a printed warning (multi-turn agentic
    # failures need the full trajectory as feedback; copy risk is monitored)
    run_stream(make_tasks(2), [ScriptedMockEngine()], StubBuffer(), StubGate(cfg),
               stub_trainer, StubRegistry(), cfg)
    assert "copy risk" in capsys.readouterr().out
    cfg.own_view, cfg.kl_states = "outcome", "sometimes"
    with pytest.raises(ValueError, match="kl_states"):
        run_stream(make_tasks(2), [ScriptedMockEngine()], StubBuffer(), StubGate(cfg),
                   stub_trainer, StubRegistry(), cfg)
