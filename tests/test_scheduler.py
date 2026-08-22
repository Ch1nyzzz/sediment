"""Scheduler + Router tests: predict-then-update ordering, publish, jsonl records.

Collaborator modules (envs, rollout, experience, hindsight, merge) are being
written concurrently, so contract-shaped stubs are injected via sys.modules;
buffer/gate/registry/trainer are stubbed through run_stream's own parameters.
"""
from __future__ import annotations

import json
import sys
import types
from typing import Any, Optional

import pytest

import sediment
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
                     experience: Optional[ExperienceBlock] = None) -> Trajectory:
    messages = env.reset(task)
    if experience is not None:  # block injected into first user message
        messages[-1] = Message("user", f"{experience.text}\n\n{messages[-1].content}")
    out = engine.generate(messages, adapter=adapter,
                          temperature=cfg.temperature, max_tokens=cfg.max_tokens)
    messages = messages + [Message("assistant", out)]
    obs, done, reward = env.step(out)
    messages = messages + obs
    return Trajectory(task_id=task["task_id"], env_family=task["env_family"],
                      messages=messages, reward=reward, success=reward > 0,
                      adapter=adapter, steps=1)


def stub_build_block(retrieved: list[Trajectory], own: Optional[Trajectory],
                     cfg: StreamConfig) -> ExperienceBlock:
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

    def retrieve(self, task: dict[str, Any], k: int) -> list[Trajectory]:
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
        return self._versions[-1]

    def publish(self, candidate: UpdateCandidate, parent: AdapterVersion,
                provenance: list[str]) -> AdapterVersion:
        v = AdapterVersion(f"v{len(self._versions):04d}", candidate.adapter_path,
                           parent.name, list(provenance))
        self._versions.append(v)
        self.publishes += 1
        return v

    def history(self) -> list[AdapterVersion]:
        return list(self._versions)


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
