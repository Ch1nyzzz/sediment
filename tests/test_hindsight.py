"""experience + spans + hindsight: known deltas flow end to end."""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

import pytest

from sediment.config import StreamConfig
from sediment.experience import EXPERIENCE_CLOSE, EXPERIENCE_OPEN, build_block, inject
from sediment.hindsight import score, to_train_sample
from sediment.spans import align_suffix, message_spans, render_messages, spans_from_lengths
from sediment.types import ExperienceBlock, Message, Trajectory


def mock_tokenize(text: str) -> list[str]:
    """Whitespace tokenizer (same contract as engine.mock.mock_tokenize)."""
    return text.split()


# scorer(token, role, has_block) -> logprob
Scorer = Callable[[str, str, bool], float]


class ScriptedEngine:
    """Engine-protocol stub with an injectable per-token scorer.

    score() returns one logprob list per message, tokenizing each message the
    way sediment.spans.render_messages does (so spans lengths agree).
    """

    def __init__(self, scorer: Scorer):
        self._scorer = scorer

    def generate(self, messages, *, adapter="base", temperature=0.7, max_tokens=256) -> str:
        return ""

    def load_adapter(self, version) -> None:
        pass

    def score(self, messages, *, adapter="base") -> list[list[float]]:
        has_block = any(EXPERIENCE_OPEN in m.content for m in messages if m.role == "user")
        out = []
        for m in messages:
            tokens = mock_tokenize(render_messages([m]))
            out.append([self._scorer(t, m.role, has_block) for t in tokens])
        return out


def scripted_scorer(token: str, role: str, has_block: bool) -> float:
    """Baseline -1.0; with the block, 'shipped'/'locked'/'refuse' shift."""
    if not has_block:
        return -1.0
    if role == "tool":
        return {"shipped": -0.2, "locked": -1.5}.get(token, -1.0)
    if role == "assistant" and token == "refuse":
        return -0.5
    return -1.0


def make_traj(**kw) -> Trajectory:
    msgs = [
        Message("system", "sys"),
        Message("user", "cancel order 7"),
        Message("assistant", "check status"),  # 3 tokens incl. <assistant>
        Message("tool", "status shipped locked"),  # 4 tokens incl. <tool>
        Message("assistant", "refuse"),  # 2 tokens incl. <assistant>
    ]
    base = dict(task_id="t0", env_family="toy_order", messages=msgs, reward=0.0, success=False)
    base.update(kw)
    return Trajectory(**base)


def make_block() -> ExperienceBlock:
    return ExperienceBlock(text=f"{EXPERIENCE_OPEN}\nshipped orders cannot cancel\n{EXPERIENCE_CLOSE}")


# ---------------------------------------------------------------- spans


def test_message_spans_and_lengths():
    msgs = make_traj().messages
    spans = message_spans(mock_tokenize, msgs)
    assert [s.role for s in spans] == ["system", "user", "assistant", "tool", "assistant"]
    assert [s.length for s in spans] == [2, 4, 3, 4, 2]
    assert spans[-1].end == len(mock_tokenize(render_messages(msgs)))
    # per-message score lengths reproduce the same spans
    assert spans_from_lengths(msgs, [s.length for s in spans]) == spans


def test_align_suffix_pairs_and_rejects_mismatch():
    msgs = make_traj().messages
    withm, idx = inject(msgs, "<previous_attempts> x </previous_attempts>")
    assert idx == 1
    sw = message_spans(mock_tokenize, withm)
    so = message_spans(mock_tokenize, msgs)
    pairs = align_suffix(sw, so, idx)
    assert [(a.index, a.role) for a, _ in pairs] == [(2, "assistant"), (3, "tool"), (4, "assistant")]
    assert all(a.length == b.length for a, b in pairs)
    # a diverging suffix message must be rejected
    bad = list(msgs)
    bad[3] = Message("tool", "status shipped locked extra")
    with pytest.raises(ValueError):
        align_suffix(message_spans(mock_tokenize, bad), so, idx)


# ------------------------------------------------------------ experience


def test_build_block_numbers_tags_truncates():
    cfg = StreamConfig(max_result_chars=20)
    long_result = "x" * 50
    peers = [
        make_traj(task_id="a", reward=1.0, success=True),
        make_traj(
            task_id="b",
            reward=0.0,
            success=False,
            messages=[
                Message("system", "sys"),
                Message("user", "task"),
                Message("assistant", "do thing"),
                Message("tool", long_result),
                Message("assistant", "final answer"),
            ],
        ),
    ]
    block = build_block(peers, None, cfg)
    assert block.text.startswith(EXPERIENCE_OPEN) and block.text.rstrip().endswith(EXPERIENCE_CLOSE)
    assert "Attempt 1 — SUCCESS (r=1.00)" in block.text
    assert "Attempt 2 — FAILED (r=0.00)" in block.text
    assert "  1. check status -> status shipped locke…" in block.text  # 21 chars -> cut at 20
    assert "  2. refuse -> (episode end)" in block.text
    assert f"do thing -> {'x' * 20}…" in block.text  # result truncated
    assert long_result not in block.text
    assert block.source_task_ids == ["a", "b"]
    assert block.includes_own_outcome is False


def test_build_block_own_outcome_summary_only():
    cfg = StreamConfig()
    own = make_traj(task_id="me", reward=0.0, success=False)
    block = build_block([], own, cfg)
    assert block.includes_own_outcome is True
    assert block.source_task_ids == []
    assert "ultimately FAILED (reward=0.00)" in block.text
    assert "Final feedback: status shipped locked" in block.text
    # no trajectory body: own actions never appear
    assert "check status" not in block.text
    assert "refuse" not in block.text
    assert build_block([], None, cfg).text == ""


def test_build_block_keeps_all_peer_tasks_and_reflections_before_steps():
    peers = []
    for i in range(4):
        peer = make_traj(task_id=f"p{i}", reward=0.0, success=False)
        peer.messages[1] = Message("user", f"FULL TASK {i} " + "q" * 80)
        peer.meta["reflection"] = f"FULL REFLECTION {i} " + "r" * 80
        peers.append(peer)
    block = build_block(peers, None, StreamConfig(max_block_chars=900, max_result_chars=20))

    assert block.source_task_ids == ["p0", "p1", "p2", "p3"]
    for i in range(4):
        assert f"FULL TASK {i} " + "q" * 80 in block.text
        assert f"FULL REFLECTION {i} " + "r" * 80 in block.text
    last_reflection = block.text.index("FULL REFLECTION 3")
    first_step = block.text.find("selected steps:")
    assert first_step == -1 or last_reflection < first_step


def test_legacy_block_reproduces_first_peer_over_budget_truncation_route():
    peers = [make_traj(task_id=f"p{i}") for i in range(4)]
    for i, peer in enumerate(peers):
        peer.meta["reflection"] = f"reflection-{i}-" + "r" * 300
    cfg = StreamConfig(
        experience_view="legacy", max_block_chars=350, max_result_chars=100,
    )
    block = build_block(peers, None, cfg)
    assert block.source_task_ids == ["p0"]
    assert "cancel order 7" not in block.text  # legacy block omitted source task text
    assert "reflection-0" in block.text
    assert "reflection-1" not in block.text
    assert len(block.text) > cfg.max_block_chars  # first peer was indivisible


# ------------------------------------------------------------- hindsight


def test_score_known_deltas():
    cfg = StreamConfig()
    engine = ScriptedEngine(scripted_scorer)
    hr = score(engine, make_traj(), make_block(), cfg)

    assert hr.task_id == "t0"
    assert [(s.msg_idx, s.role) for s in hr.spans] == [(2, "assistant"), (3, "tool"), (4, "assistant")]
    assert hr.spans[0].deltas == [0.0, 0.0, 0.0]
    assert hr.spans[1].deltas == pytest.approx([0.0, 0.0, 0.8, -0.5])
    assert hr.spans[2].deltas == pytest.approx([0.0, 0.5])
    # obs_surprise = mean(relu) over 4 tool tokens; act_gain = mean over 5 assistant tokens
    assert hr.obs_surprise == pytest.approx(0.8 / 4)
    assert hr.act_gain == pytest.approx(0.5 / 5)


def test_score_empty_safe():
    cfg = StreamConfig()
    engine = ScriptedEngine(scripted_scorer)
    traj = make_traj(messages=[Message("system", "sys"), Message("user", "task")])
    hr = score(engine, traj, make_block(), cfg)
    assert hr.spans == []
    assert hr.obs_surprise == 0.0
    assert hr.act_gain == 0.0


def test_to_train_sample_relu_floor_parallel():
    cfg = StreamConfig(w_floor=0.05)
    traj = make_traj()
    hr = score(ScriptedEngine(scripted_scorer), traj, make_block(), cfg)
    sample = to_train_sample(traj, hr, cfg)

    assert sample.task_id == "t0"
    assert sample.messages == traj.messages
    assert len(sample.token_weights_by_msg) == len(traj.messages)
    assert sample.token_weights_by_msg[0] == []  # system: unsupervised
    assert sample.token_weights_by_msg[1] == []  # prompt: unsupervised
    assert sample.token_weights_by_msg[2] == pytest.approx([0.05, 0.05, 0.05])
    assert sample.token_weights_by_msg[3] == pytest.approx([0.05, 0.05, 0.8, 0.05])  # relu(-0.5) floored
    assert sample.token_weights_by_msg[4] == pytest.approx([0.05, 0.5])

    zero_floor = to_train_sample(traj, hr, replace(cfg, w_floor=0.0))
    assert zero_floor.token_weights_by_msg[3] == pytest.approx([0.0, 0.0, 0.8, 0.0])
