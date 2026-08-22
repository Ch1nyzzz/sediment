"""Gate tests: G1 ledger recurrence, G2 behavioral A/B, G3 probe, skips."""
from __future__ import annotations

from typing import Callable, Optional

from sediment.config import StreamConfig
from sediment.gate import Gate
from sediment.types import (
    AdapterVersion,
    HindsightResult,
    Message,
    Trajectory,
    UpdateCandidate,
)


class MockEngine:
    """Engine stub: routes each adapter name to an injected policy."""

    def __init__(self, policies: dict[str, Callable[[list[Message]], str]]):
        self.policies = policies
        self.calls: list[str] = []  # adapter names, in call order

    def generate(self, messages: list[Message], *, adapter: str = "base",
                 temperature: float, max_tokens: int) -> str:
        self.calls.append(adapter)
        return self.policies[adapter](messages)


def _cfg(**kw) -> StreamConfig:
    base = dict(gate_min_surprise=0.05, gate_recurrence=2,
                gate_min_behavior_change=0.1, gate_min_probe_delta=0.0)
    base.update(kw)
    return StreamConfig(**base)


def _hr(surprise: float, task_id: str = "t") -> HindsightResult:
    return HindsightResult(task_id=task_id, spans=[], obs_surprise=surprise,
                           act_gain=0.0)


def _traj(env_family: str = "toy", task_id: str = "t") -> Trajectory:
    return Trajectory(task_id=task_id, env_family=env_family, messages=[])


def _candidate(cid: str = "cand-1") -> UpdateCandidate:
    return UpdateCandidate(candidate_id=cid, task_ids=["t1"],
                           adapter_path="/tmp/x", parent="v0000")


PARENT = AdapterVersion(name="v0000", path=None, parent=None)


def _states(contents: list[str]) -> list[list[Message]]:
    return [[Message("user", c)] for c in contents]


# ---------------------------------------------------------------- G1 ledger

def test_propose_requires_recurrence():
    gate = Gate(_cfg(gate_recurrence=2))
    assert gate.propose(_hr(0.1), _traj("toy")) is False  # count 1 < 2
    assert gate.propose(_hr(0.3), _traj("toy")) is True   # count 2
    assert gate.propose(_hr(0.2), _traj("toy")) is True   # stays fired
    assert gate.ledger == {"toy": [0.1, 0.3, 0.2]}


def test_propose_below_threshold_never_counts():
    gate = Gate(_cfg(gate_min_surprise=0.05, gate_recurrence=1))
    assert gate.propose(_hr(0.04), _traj("toy")) is False
    assert gate.ledger == {}  # below-threshold event not recorded
    assert gate.propose(_hr(0.05), _traj("toy")) is True  # >= is inclusive


def test_propose_ledger_is_per_env_family():
    gate = Gate(_cfg(gate_recurrence=2))
    assert gate.propose(_hr(0.1), _traj("a")) is False
    assert gate.propose(_hr(0.1), _traj("b")) is False  # families independent
    assert gate.propose(_hr(0.1), _traj("a")) is True
    assert set(gate.ledger) == {"a", "b"}
    assert len(gate.ledger["a"]) == 2 and len(gate.ledger["b"]) == 1


# ---------------------------------------------------------- G2 behavioral

def test_validate_behavioral_change_passes():
    # Candidate answers differently on the two "hard" states -> 2/4 = 0.5.
    engine = MockEngine({
        "v0000": lambda ms: "act-A",
        "cand-1": lambda ms: "act-B" if "hard" in ms[-1].content else "act-A",
    })
    gate = Gate(_cfg())
    dec = gate.validate(_candidate(), PARENT, engine,
                        _states(["easy1", "hard1", "easy2", "hard2"]),
                        [], None)
    assert dec.passed is True
    assert dec.behavioral_changed == 0.5
    assert dec.candidate_id == "cand-1"
    assert set(engine.calls) == {"v0000", "cand-1"}  # A/B used both adapters
    assert "G2 pass" in dec.reason


def test_validate_behavioral_no_change_fails():
    engine = MockEngine({"v0000": lambda ms: "same",
                         "cand-1": lambda ms: "same"})
    gate = Gate(_cfg())
    dec = gate.validate(_candidate(), PARENT, engine,
                        _states(["s1", "s2"]), [], None)
    assert dec.passed is False
    assert dec.behavioral_changed == 0.0
    assert "G2 fail" in dec.reason


# --------------------------------------------------------------- G3 probe

def test_validate_probe_delta_pass_and_fail():
    engine = MockEngine({"v0000": lambda ms: "a", "cand-1": lambda ms: "b"})
    gate = Gate(_cfg())
    tasks = [{"task_id": "p1"}]

    up = lambda name, ts: {"cand-1": 0.75, "v0000": 0.5}[name]
    dec = gate.validate(_candidate(), PARENT, engine, _states(["s"]), tasks, up)
    assert dec.passed is True and dec.probe_delta == 0.25

    down = lambda name, ts: {"cand-1": 0.25, "v0000": 0.5}[name]
    dec = gate.validate(_candidate(), PARENT, engine, _states(["s"]), tasks, down)
    assert dec.passed is False and dec.probe_delta == -0.25
    assert "G3 fail" in dec.reason


def test_validate_needs_both_stages():
    # G3 passes but G2 sees no behavior change -> overall fail.
    engine = MockEngine({"v0000": lambda ms: "x", "cand-1": lambda ms: "x"})
    gate = Gate(_cfg())
    up = lambda name, ts: {"cand-1": 1.0, "v0000": 0.0}[name]
    dec = gate.validate(_candidate(), PARENT, engine, _states(["s"]),
                        [{"task_id": "p"}], up)
    assert dec.passed is False
    assert dec.probe_delta == 1.0 and dec.behavioral_changed == 0.0


# ------------------------------------------------------------------ skips

def test_validate_skips_count_as_pass():
    engine = MockEngine({})
    gate = Gate(_cfg())
    dec = gate.validate(_candidate(), PARENT, engine, [], [], None)
    assert dec.passed is True
    assert dec.behavioral_changed is None and dec.probe_delta is None
    assert engine.calls == []
    assert "G2 skipped" in dec.reason and "G3 skipped" in dec.reason


def test_validate_skips_probe_without_tasks():
    engine = MockEngine({"v0000": lambda ms: "a", "cand-1": lambda ms: "b"})
    gate = Gate(_cfg())
    dec = gate.validate(_candidate(), PARENT, engine, _states(["s"]), [],
                        lambda name, ts: 1.0)
    assert dec.passed is True  # G2 passed (1/1 changed), G3 skipped
    assert dec.probe_delta is None
    assert "G3 skipped (no probe tasks)" in dec.reason


# -------------------------------------------------------------- magnitude

def test_validate_reports_ledger_magnitude():
    gate = Gate(_cfg(gate_recurrence=2))
    gate.propose(_hr(0.1), _traj("toy"))
    gate.propose(_hr(0.3), _traj("toy"))
    dec = gate.validate(_candidate(), PARENT, MockEngine({}), [], [], None)
    assert abs(dec.magnitude - 0.2) < 1e-9
    assert dec.to_dict()["candidate_id"] == "cand-1"  # jsonl-safe
