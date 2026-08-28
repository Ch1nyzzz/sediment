"""Tests for sediment.buffer: persistence, retrieval ranking, replay states."""
from __future__ import annotations

import pytest

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


def test_retrieval_ignores_family_and_uses_overlap(tmp_path):
    buf = Buffer(tmp_path / "b.jsonl")
    buf.add(make_traj("other", "web", "cancel order 42 please now"))  # full overlap, wrong family
    buf.add(make_traj("kin", "toy", "totally unrelated words here"))  # zero overlap, same family
    task = {"task_id": "q", "env_family": "toy", "payload": "cancel order 42 please now"}
    got = buf.retrieve(task, 2)
    assert [t.task_id for t in got] == ["other", "kin"]


def test_legacy_retrieval_always_routes_same_family_before_cross_family(tmp_path):
    buf = Buffer(tmp_path / "b.jsonl")
    buf.add(make_traj("cross", "other", "cancel order 42 please now"))
    buf.add(make_traj("same", "toy", "totally unrelated words here"))
    task = {"task_id": "q", "env_family": "toy", "payload": "cancel order 42 please now"}
    got = buf.retrieve(task, 2, score_mode="legacy")
    assert [t.task_id for t in got] == ["same", "cross"]


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


def test_retrieval_prefers_four_distinct_tasks_across_rollout_seeds(tmp_path):
    buf = Buffer(tmp_path / "b.jsonl")
    for seed in range(4):
        buf.add(make_traj("same-source", f"seed-{seed}", "exact matching words"))
    for i in range(3):
        buf.add(make_traj(f"different-{i}", "other", "exact matching words"))
    got = buf.retrieve(
        {"task_id": "q", "env_family": "unseen", "payload": "exact matching words"}, 4
    )
    assert len(got) == 4
    assert len({traj.task_id for traj in got}) == 4


def test_retrieval_offset_exposes_later_distinct_ranks(tmp_path):
    buf = Buffer(tmp_path / "b.jsonl")
    for task_id in ("rank-4", "rank-3", "rank-2", "rank-1"):
        buf.add(make_traj(task_id, "source", "exact matching words"))
    task = {"task_id": "q", "env_family": "target", "payload": "exact matching words"}
    assert [t.task_id for t in buf.retrieve(task, 1, offset=0)] == ["rank-1"]
    assert [t.task_id for t in buf.retrieve(task, 2, offset=1)] == ["rank-2", "rank-3"]
    assert buf.retrieve(task, 2, offset=4) == []


def test_retrieval_family_diversity_prefers_distinct_source_groups(tmp_path):
    buf = Buffer(tmp_path / "b.jsonl")
    buf.add(make_traj("other-family", "source-b", "partial matching words"))
    buf.add(make_traj("same-family-old", "source-a", "exact matching words"))
    buf.add(make_traj("same-family-new", "source-a", "exact matching words"))
    task = {"task_id": "q", "env_family": "target", "payload": "exact matching words"}
    got = buf.retrieve(task, 3, scope="cross_family", diversity="family")
    assert [t.task_id for t in got[:2]] == ["same-family-new", "other-family"]
    assert got[2].task_id == "same-family-old"  # backfill only after coverage


def test_retrieval_rejects_invalid_offset_and_diversity(tmp_path):
    buf = Buffer(tmp_path / "b.jsonl")
    task = {"task_id": "q", "env_family": "target", "payload": "words"}
    with pytest.raises(ValueError, match="offset"):
        buf.retrieve(task, 1, offset=-1)
    with pytest.raises(ValueError, match="diversity"):
        buf.retrieve(task, 1, diversity="unknown")


def test_retrieval_same_family_excludes_cross_family_even_with_better_overlap(tmp_path):
    buf = Buffer(tmp_path / "b.jsonl")
    buf.add(make_traj("cross", "other", "exact matching words"))
    buf.add(make_traj("same", "target", "unrelated content"))
    task = {"task_id": "q", "env_family": "target", "payload": "exact matching words"}
    got = buf.retrieve(task, 4, scope="same_family")
    assert [traj.task_id for traj in got] == ["same"]


def test_retrieval_cross_family_excludes_same_family_even_with_better_overlap(tmp_path):
    buf = Buffer(tmp_path / "b.jsonl")
    buf.add(make_traj("same", "target", "exact matching words"))
    buf.add(make_traj("cross", "other", "unrelated content"))
    task = {"task_id": "q", "env_family": "target", "payload": "exact matching words"}
    got = buf.retrieve(task, 4, scope="cross_family")
    assert [traj.task_id for traj in got] == ["cross"]


def test_retrieval_rejects_unknown_scope(tmp_path):
    buf = Buffer(tmp_path / "b.jsonl")
    try:
        buf.retrieve({"task_id": "q", "env_family": "target", "payload": "words"}, 1,
                     scope="unknown")
    except ValueError as exc:
        assert "unknown retrieval scope" in str(exc)
    else:
        raise AssertionError("unknown retrieval scope should fail")


def test_transfer_score_prefers_matching_workflow_over_raw_task_words(tmp_path):
    buf = Buffer(tmp_path / "b.jsonl")
    lexical = make_traj("lexical", "other-a", "delete duplicate preserve canonical profile")
    lexical.meta["reflection"] = "Convert timestamps before scheduling updates."
    structural = make_traj("structural", "other-b", "unrelated domain vocabulary")
    structural.meta["reflection"] = (
        "Disambiguate duplicate records, preserve the canonical record, then remove the duplicate."
    )
    buf.add(lexical)
    buf.add(structural)
    task = {
        "task_id": "q",
        "env_family": "target",
        "payload": "Disambiguate duplicate users; preserve the canonical user before deletion.",
    }
    assert buf.retrieve(task, 1, scope="cross_family")[0].task_id == "lexical"
    assert buf.retrieve(
        task, 1, scope="cross_family", score_mode="transfer"
    )[0].task_id == "structural"


def test_task_text_score_excludes_state_and_checklist_payload_noise(tmp_path):
    buf = Buffer(tmp_path / "b.jsonl")
    buf.add(make_traj("natural", "source-a", "cancel shipment after validating user"))
    buf.add(make_traj(
        "schema-noise", "source-b",
        "inventory subscription payment product checklist function final state",
    ))
    task = {
        "task_id": "q",
        "env_family": "target",
        "payload": {
            "task": "cancel shipment after validating user",
            "init_config": (
                "inventory subscription payment product checklist function final state"
            ),
            "checklist_with_func": "inventory subscription payment product checklist",
        },
    }
    assert buf.retrieve(task, 1, score_mode="task")[0].task_id == "schema-noise"
    assert buf.retrieve(task, 1, score_mode="task_text")[0].task_id == "natural"


def test_task_text_success_prioritizes_proven_success_then_similarity(tmp_path):
    buf = Buffer(tmp_path / "b.jsonl")
    failure = make_traj("failure", "source-a", "cancel shipment exact match")
    failure.success = False
    failure.reward = 0.9
    success_far = make_traj("success-far", "source-b", "cancel a related order")
    success_near = make_traj("success-near", "source-c", "cancel shipment")
    for traj in (failure, success_far, success_near):
        buf.add(traj)
    task = {
        "task_id": "q",
        "env_family": "target",
        "payload": {"task": "cancel shipment exact match"},
    }
    got = buf.retrieve(task, 3, score_mode="task_text_success")
    assert [traj.task_id for traj in got] == ["success-near", "success-far", "failure"]


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
