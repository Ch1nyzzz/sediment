"""Tests for sediment.eval and the run_stream / run_p1_probe CLIs.

Script smoke tests inject stub sibling modules into sys.modules so this file
only exercises the eval+scripts module (siblings are written concurrently).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sediment import eval as ev
from sediment.config import StreamConfig
from sediment.types import (AdapterVersion, ExperienceBlock, GateDecision,
                            HindsightResult, Message, StreamRecord, Trajectory,
                            TrainSample, UpdateCandidate)


def rec(task_id: str, window: int, success, adapter: str = "v0000") -> StreamRecord:
    return StreamRecord(task_id=task_id, window=window, adapter=adapter,
                        success=success, reward=1.0 if success else 0.0)


# ---------------------------------------------------------------- eval metrics

def test_window_means_and_none_success():
    records = [rec("a", 0, True), rec("b", 0, None), rec("c", 1, True)]
    assert ev.window_means(records) == {0: 0.5, 1: 1.0}


def test_wauc_trapezoid():
    records = (
        [rec(f"a{i}", 0, False) for i in range(2)]
        + [rec("b0", 1, True), rec("b1", 1, False)]
        + [rec(f"c{i}", 2, True) for i in range(2)]
    )
    # window means 0.0, 0.5, 1.0 over x=[0,1,2] -> normalized trapezoid 0.5
    assert ev.wauc(records) == pytest.approx(0.5)


def test_wauc_single_window_and_empty():
    assert ev.wauc([rec("a", 0, True), rec("b", 0, False)]) == pytest.approx(0.5)
    assert ev.wauc([]) == 0.0


def test_wauc_accepts_jsonl_dicts():
    rows = [{"task_id": "a", "window": 0, "success": True},
            {"task_id": "b", "window": 1, "success": False}]
    assert ev.wauc(rows) == pytest.approx(0.5)


def test_gain():
    ours = [rec("a", 0, True), rec("b", 0, True), rec("c", 1, True), rec("d", 1, False)]
    base = [rec("a", 0, True), rec("b", 0, False), rec("c", 1, False), rec("d", 1, False)]
    assert ev.gain(ours, base) == pytest.approx(0.5)
    assert ev.gain([], base) == 0.0
    assert ev.gain(ours, []) == 0.0


class FakeRegistry:
    def __init__(self, versions):
        self._versions = versions

    def history(self):
        return list(self._versions)


def test_write_report(tmp_path, capsys):
    cfg = StreamConfig(num_tasks=4, window_size=2, run_id="t")
    records = [rec("a", 0, True), rec("b", 0, False),
               rec("c", 1, True, adapter="v0001"), rec("d", 1, True, adapter="v0001")]
    registry = FakeRegistry([AdapterVersion("v0000", None, None),
                             AdapterVersion("v0001", "p", "v0000")])
    summary = ev.write_report(tmp_path, cfg, records, registry)

    lines = (tmp_path / "stream.jsonl").read_text().splitlines()
    assert len(lines) == 4
    assert json.loads(lines[0])["task_id"] == "a"
    on_disk = json.loads((tmp_path / "summary.json").read_text())
    assert on_disk == summary
    assert summary["config"]["num_tasks"] == 4
    assert summary["wauc"] == pytest.approx(0.75)
    assert summary["success_rate"] == pytest.approx(0.75)
    assert summary["window_means"] == {"0": 0.5, "1": 1.0}
    assert summary["adapters"] == ["v0000", "v0001"]
    out = capsys.readouterr().out
    assert "window" in out and "v0001" in out


# ------------------------------------------------------------- script smokes

def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _install_stubs(monkeypatch) -> None:
    """Register contract-shaped stubs for every sibling module the scripts use."""
    import sediment

    def make_toy_tasks(n, seed=0):
        return [{"task_id": f"toy-{i:03d}", "env_family": "toy_order",
                 "payload": {"i": i}} for i in range(n)]

    class ToyOrderEnv:
        def reset(self, task):
            return [Message("user", "task")]

        def step(self, action_text):
            return [], True, 1.0

    class MockEngine:
        def __init__(self, policy=None, scorer=None):
            self.loaded = {}

        def generate(self, messages, *, adapter="base", temperature=0.7,
                     max_tokens=2048):
            return "ok"

        def score(self, messages, *, adapter="base"):
            return [[-1.0] * 3 for _ in messages]

        def load_adapter(self, version):
            self.loaded[version.name] = version

    class Buffer:
        def __init__(self, path):
            self.path, self.items = path, []

        def add(self, traj):
            self.items.append(traj)

        def retrieve(self, task, k):
            return []

        def replay_states(self, n, seed=0):
            return []

    class Gate:
        def __init__(self, cfg):
            self.cfg = cfg

        def propose(self, hr, traj):
            return False

        def validate(self, candidate, parent, engine, replay_states,
                     probe_tasks, run_probe):
            return GateDecision(candidate_id=candidate.candidate_id,
                                passed=False, magnitude=0.0)

    class Registry:
        def __init__(self, dir):
            self.dir = dir
            self._versions = [AdapterVersion("v0000", None, None)]

        def base(self):
            return self._versions[0]

        def current(self):
            return self._versions[-1]

        def history(self):
            return list(self._versions)

        def get(self, name):
            return next(v for v in self._versions if v.name == name)

        def publish(self, candidate, parent, provenance):
            parent_name = parent.name if hasattr(parent, "name") else str(parent)
            v = AdapterVersion(f"v{len(self._versions):04d}", "path",
                               parent_name, list(provenance))
            self._versions.append(v)
            return v

    def train_candidate(samples, parent, cfg, workdir):
        return UpdateCandidate(candidate_id="cand-0001",
                               task_ids=[s.task_id for s in samples],
                               adapter_path=str(workdir), parent=parent.name)

    def run_stream(tasks, engines, buffer, gate, trainer_fn, registry, cfg, *,
                   run_probe=None, probe_tasks=None, on_window=None, **_kw):
        w = max(1, cfg.window_size)
        records = [StreamRecord(task_id=t["task_id"], window=i // w,
                                adapter=registry.current().name,
                                success=i % 2 == 0, reward=float(i % 2 == 0))
                   for i, t in enumerate(tasks)]
        if on_window is not None:
            for widx in sorted({r.window for r in records}):
                on_window(widx, [r for r in records if r.window == widx])
        return records

    def run_episode(engine, env, task, cfg, *, adapter="base", experience=None):
        ok = adapter != "base"  # the trained candidate flips the outcome
        return Trajectory(task_id=task["task_id"], env_family=task["env_family"],
                          messages=[Message("user", "t"), Message("assistant", "a")],
                          reward=1.0 if ok else 0.0, success=ok,
                          adapter=adapter, steps=1)

    def build_block(retrieved, own, cfg):
        return ExperienceBlock(text="<previous_attempts>x</previous_attempts>",
                               source_task_ids=[own.task_id] if own else [],
                               includes_own_outcome=own is not None)

    def hindsight_score(engine, traj, block, cfg):
        return HindsightResult(task_id=traj.task_id, spans=[],
                               obs_surprise=0.25, act_gain=0.1)

    def to_train_sample(traj, hr, cfg):
        return TrainSample(task_id=traj.task_id, messages=traj.messages,
                           token_weights_by_msg=[[0.0], [1.0]])

    mods = {
        "sediment.envs": _module("sediment.envs", make_toy_tasks=make_toy_tasks,
                                 ToyOrderEnv=ToyOrderEnv),
        "sediment.engine": _module("sediment.engine"),
        "sediment.engine.mock": _module("sediment.engine.mock", MockEngine=MockEngine),
        "sediment.buffer": _module("sediment.buffer", Buffer=Buffer),
        "sediment.gate": _module("sediment.gate", Gate=Gate),
        "sediment.registry": _module("sediment.registry", Registry=Registry),
        "sediment.trainer": _module("sediment.trainer", train_candidate=train_candidate),
        "sediment.scheduler": _module("sediment.scheduler", run_stream=run_stream),
        "sediment.rollout": _module("sediment.rollout", run_episode=run_episode),
        "sediment.experience": _module("sediment.experience", build_block=build_block),
        "sediment.hindsight": _module("sediment.hindsight", score=hindsight_score,
                                      to_train_sample=to_train_sample),
    }
    mods["sediment.engine"].mock = mods["sediment.engine.mock"]
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
        parent, _, child = name.rpartition(".")
        if parent == "sediment":
            monkeypatch.setattr(sediment, child, mod, raising=False)


def _load_script(name: str) -> types.ModuleType:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_stream_main_smoke(tmp_path, monkeypatch):
    _install_stubs(monkeypatch)
    mod = _load_script("run_stream")
    summary = mod.main(["--engine", "mock", "--tasks", "6", "--window", "3",
                        "--out", str(tmp_path), "--run-id", "t1", "--seed", "7",
                        "--set", "temperature=0.11"])

    run_dir = tmp_path / "t1"
    assert (run_dir / "stream.jsonl").exists()
    assert (run_dir / "summary.json").exists()
    rows = [json.loads(l) for l in (run_dir / "stream.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    assert {r["window"] for r in rows} == {0, 1}
    assert rows[0]["adapter"] == "v0000"

    cfg = summary["config"]
    assert (cfg["engine"], cfg["num_tasks"], cfg["window_size"]) == ("mock", 6, 3)
    assert cfg["seed"] == 7 and cfg["temperature"] == 0.11
    # state paths were isolated under the run dir
    assert cfg["buffer_path"] == str(run_dir / "buffer.jsonl")
    assert cfg["registry_dir"] == str(run_dir / "registry")
    # metrics recomputable from the jsonl rows
    assert summary["wauc"] == pytest.approx(ev.wauc(rows)) == pytest.approx(0.5)
    assert summary["adapters"] == ["v0000"]
    assert json.loads((run_dir / "summary.json").read_text()) == summary


def test_run_stream_set_rejects_unknown_field(tmp_path, monkeypatch):
    _install_stubs(monkeypatch)
    mod = _load_script("run_stream")
    with pytest.raises(SystemExit):
        mod.main(["--engine", "mock", "--set", "not_a_field=1"])


def test_run_stream_initial_adapter_for_heldout_eval(tmp_path, monkeypatch):
    _install_stubs(monkeypatch)
    mod = _load_script("run_stream")
    adapter = tmp_path / "final-adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    summary = mod.main([
        "--engine", "mock", "--tasks", "4", "--window", "2",
        "--out", str(tmp_path), "--run-id", "heldout-final",
        "--set", f"initial_adapter_path={adapter}",
    ])
    rows = [json.loads(line) for line in
            (tmp_path / "heldout-final" / "stream.jsonl").read_text().splitlines()]
    assert {row["adapter"] for row in rows} == {"v0001"}
    assert summary["adapters"] == ["v0000", "v0001"]


def test_heldout_pair_analysis_is_paired_and_family_aware():
    mod = _load_script("analyze_heldout_pair")
    frozen = [
        {"task_id": "env_183_rl-task_1", "success": True},
        {"task_id": "env_183_rl-task_2", "success": False},
        {"task_id": "env_184_rl-task_1", "success": True},
    ]
    final = [
        {"task_id": "env_183_rl-task_1", "success": True},
        {"task_id": "env_183_rl-task_2", "success": True},
        {"task_id": "env_184_rl-task_1", "success": False},
    ]
    result = mod.compare(frozen, final)
    assert result["delta_successes"] == 0
    assert result["paired"] == {
        "final_only": 1, "frozen_only": 1, "ties": 1, "two_sided_exact_p": 1.0,
    }
    assert result["by_family"]["env_183"]["delta_successes"] == 1
    assert result["by_family"]["env_184"]["delta_successes"] == -1


def test_run_p1_probe_main_smoke(tmp_path, monkeypatch):
    _install_stubs(monkeypatch)
    mod = _load_script("run_p1_probe")
    records = mod.main(["--engine", "mock", "--tasks", "3",
                        "--out", str(tmp_path), "--run-id", "p1"])

    out_path = tmp_path / "p1" / "p1.jsonl"
    assert out_path.exists()
    rows = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert rows == records and len(rows) == 3
    for row in rows:
        assert set(row) == {"task_id", "first_success", "retry_success",
                            "obs_surprise"}
        assert row["first_success"] is False  # base fails
        assert row["retry_success"] is True  # trained candidate succeeds
        assert row["obs_surprise"] == pytest.approx(0.25)
