"""Tests for sediment.buffer: persistence, retrieval ranking, replay states."""
from __future__ import annotations

from sediment.buffer import Buffer
from sediment.types import Message, Trajectory


def make_traj(task_id: str, family: str, user_text: str, n_turns: int = 2) -> Trajectory:
    msgs = [Message("system", "sys"), Message("user", user_text)]
    for i in range(n_turns):
        msgs.append(Message("assistant", f"act-{task_id}-{i}"))
        msgs.append(Message("tool", f"obs-{task_id}-{i}"))
    return Trajectory(
        task_id=task_id,
        env_family=family,
        messages=msgs,
        reward=1.0,
        success=True,
        steps=n_turns,
        meta={"src": task_id},
    )


def test_persistence_round_trip(tmp_path):
    path = tmp_path / "buf.jsonl"
    buf = Buffer(path)
    t1 = make_traj("t1", "toy", "cancel order 42")
    t2 = make_traj("t2", "web", "search flights to NYC")
    t2.reward = None
    t2.success = None
    buf.add(t1)
    buf.add(t2)

    loaded = Buffer.load(path)
    assert len(loaded) == 2
    assert [t.to_dict() for t in loaded._trajs] == [t1.to_dict(), t2.to_dict()]


def test_load_missing_file_is_empty(tmp_path):
    assert len(Buffer.load(tmp_path / "absent.jsonl")) == 0


def test_retrieval_family_beats_overlap(tmp_path):
    buf = Buffer(tmp_path / "b.jsonl")
    buf.add(make_traj("other", "web", "cancel order 42 please now"))  # full overlap, wrong family
    buf.add(make_traj("kin", "toy", "totally unrelated words here"))  # zero overlap, same family
    task = {"task_id": "q", "env_family": "toy", "payload": "cancel order 42 please now"}
    got = buf.retrieve(task, 2)
    assert [t.task_id for t in got] == ["kin", "other"]


def test_retrieval_overlap_orders_within_family_and_excludes_self(tmp_path):
    buf = Buffer(tmp_path / "b.jsonl")
    buf.add(make_traj("low", "toy", "check refund status"))
    buf.add(make_traj("q", "toy", "cancel order 42"))  # same task_id -> excluded
    buf.add(make_traj("high", "toy", "cancel order 42"))
    task = {"task_id": "q", "env_family": "toy", "payload": "cancel order 42"}
    got = buf.retrieve(task, 10)
    assert [t.task_id for t in got] == ["high", "low"]


def test_retrieval_recency_breaks_ties(tmp_path):
    buf = Buffer(tmp_path / "b.jsonl")
    buf.add(make_traj("old", "toy", "same words"))
    buf.add(make_traj("new", "toy", "same words"))
    got = buf.retrieve({"task_id": "q", "env_family": "toy", "payload": "same words"}, 1)
    assert got[0].task_id == "new"


def test_replay_prefix_ends_just_before_assistant_turn(tmp_path):
    buf = Buffer(tmp_path / "b.jsonl")
    trajs = [make_traj(f"t{i}", "toy", f"user text {i}", n_turns=3) for i in range(5)]
    for t in trajs:
        buf.add(t)

    states = buf.replay_states(8, seed=0)  # n > len(trajs) exercises replacement
    assert len(states) == 8
    for prefix in states:
        assert any(
            len(t.messages) > len(prefix)
            and t.messages[: len(prefix)] == prefix
            and t.messages[len(prefix)].role == "assistant"
            for t in trajs
        )


def test_replay_deterministic_under_seed(tmp_path):
    buf = Buffer(tmp_path / "b.jsonl")
    for i in range(6):
        buf.add(make_traj(f"t{i}", "toy", f"user text {i}", n_turns=3))
    assert buf.replay_states(4, seed=7) == buf.replay_states(4, seed=7)
    assert buf.replay_states(20, seed=7) == buf.replay_states(20, seed=7)


def test_replay_skips_trajs_without_assistant(tmp_path):
    buf = Buffer(tmp_path / "b.jsonl")
    buf.add(Trajectory("u", "toy", [Message("user", "hi")]))
    assert buf.replay_states(4, seed=0) == []
