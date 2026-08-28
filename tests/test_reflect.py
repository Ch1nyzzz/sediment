import pytest

from sediment.config import StreamConfig
from sediment.experience import build_block
from sediment.hindsight import to_train_sample
from sediment.reflect import reflect, render_attempt
from sediment.types import HindsightResult, Message, SpanScore, Trajectory


class Eng:
    def __init__(self):
        self.calls = []

    def generate(self, messages, *, adapter, temperature, max_tokens):
        self.calls.append((messages, adapter, temperature, max_tokens))
        return "- verify payment before upgrade\n- ids are strings"


def traj(reflection=None):
    t = Trajectory(task_id="t1", env_family="f", messages=[
        Message("system", "s"), Message("user", "upgrade Alice"),
        Message("assistant", "upgrade(Alice)"), Message("tool", "Error: payment unverified"),
        Message("assistant", "verify(Alice)"), Message("tool", "ok"),
    ], reward=0.5, success=False, adapter="v0000", steps=2)
    if reflection:
        t.meta["reflection"] = reflection
    return t


def test_reflect_prompt_and_storage_bounds():
    cfg = StreamConfig(max_reflection_chars=30)
    eng = Eng()
    out = reflect(eng, traj(), cfg, adapter="v0003")
    msgs, adapter, temp, mt = eng.calls[0]
    assert adapter == "v0003" and temp == 0.0 and mt == cfg.reflect_max_tokens
    assert "1. upgrade(Alice) -> Error: payment unverified" in msgs[1].content
    assert "Outcome: FAILED (reward=0.50)" in render_attempt(traj(), cfg)
    assert len(out) <= 31  # max_reflection_chars + ellipsis


def test_transfer_reflection_is_canonical_and_does_not_call_model():
    cfg = StreamConfig(reflection_mode="transfer")
    eng = Eng()
    out = reflect(eng, traj(), cfg, adapter="v0003")
    assert eng.calls == []
    assert "- [verify]" in out and "- [error_recovery]" in out
    assert "Alice" not in out and "payment" not in out and "upgrade" not in out


def test_block_renders_peer_and_own_reflection():
    cfg = StreamConfig()
    peer = traj("- rule A")
    peer.task_id = "t0"
    own = traj("- rule B")
    text = build_block([peer], own, cfg).text
    assert "Reflection after this attempt:\n    - rule A" in text
    assert "Your reflection on that attempt:\n    - rule B" in text
    assert build_block([traj()], traj(), cfg).text.count("eflection") == 0


def test_reflection_only_view_omits_source_task_and_steps():
    peer = traj("- verify before mutation")
    peer.task_id = "t0"
    text = build_block([peer], None, StreamConfig(experience_view="reflection_only")).text
    assert "- verify before mutation" in text
    assert "upgrade Alice" not in text
    assert "upgrade(Alice)" not in text


def test_matched_reflection_keeps_only_query_supported_canonical_rules():
    peer = traj("- [disambiguate] rule text - [time] other rule")
    peer.task_id = "t0"
    cfg = StreamConfig(experience_view="matched_reflection")
    text = build_block([peer], None, cfg, query_text="remove a duplicate record").text
    assert "[disambiguate]" in text and "[time]" not in text
    assert "upgrade Alice" not in text and "upgrade(Alice)" not in text


def test_matched_reflection_injects_nothing_without_shared_rule():
    peer = traj("- [time] Normalize time first.")
    peer.task_id = "t0"
    block = build_block(
        [peer], None, StreamConfig(experience_view="matched_reflection"),
        query_text="delete a duplicate record",
    )
    assert block.text == "" and block.source_task_ids == []


def test_compiled_reflection_merges_deduplicates_and_bounds_top_n():
    first = traj(
        "- Verify the record before mutation.\n"
        "- Preserve unspecified fields during the update."
    )
    first.task_id = "source-a"
    second = traj(
        "- Verify the record before mutation.\n"
        "- Check permissions before updating protected fields."
    )
    second.task_id = "source-b"
    cfg = StreamConfig(
        experience_view="compiled_reflection",
        memory_compile_max_rules=3,
        memory_compile_max_chars=700,
    )
    block = build_block(
        [first, second], None, cfg,
        query_text="Update a protected record while preserving every other field.",
    )
    assert block.text.count("Verify the record before mutation") == 1
    assert "Preserve unspecified fields" in block.text
    assert "Check permissions" in block.text
    assert "upgrade Alice" not in block.text and "upgrade(Alice)" not in block.text
    assert len(block.text) <= cfg.memory_compile_max_chars
    assert set(block.source_task_ids) == {"source-a", "source-b"}
    assert "do not copy source-specific" in block.text


def test_compiled_reflection_recovers_flattened_bullets_and_covers_each_peer():
    peers = []
    lessons = [
        "Verify the record before mutation",
        "Preserve unspecified fields during update",
        "Check authorization for protected changes",
        "Normalize timestamps before scheduling",
    ]
    for index, lesson in enumerate(lessons):
        peer = traj(
            f"- {lesson}. "
            f"- Inspect unique source evidence {index} before acting."
        )
        peer.task_id = f"source-{index}"
        peers.append(peer)
    cfg = StreamConfig(
        experience_view="compiled_reflection",
        memory_compile_max_rules=4,
        memory_compile_max_chars=1800,
        memory_compile_rule_max_chars=120,
    )
    block = build_block(
        peers, None, cfg,
        query_text="verify a protected update, preserve fields, and normalize time",
    )
    assert len(block.source_task_ids) == 4
    assert set(block.source_task_ids) == {f"source-{index}" for index in range(4)}
    assert sum(line.startswith("- [") for line in block.text.splitlines()) == 4
    assert all(lesson in block.text for lesson in lessons)


def test_compiled_reflection_requires_query_and_rejects_own_outcome():
    cfg = StreamConfig(experience_view="compiled_reflection")
    with pytest.raises(ValueError, match="requires query_text"):
        build_block([traj("- verify first")], None, cfg)
    with pytest.raises(ValueError, match="serving-only"):
        build_block([traj("- verify first")], traj(), cfg, query_text="verify")


def test_error_action_gate_zeroes_bad_step_only():
    t = traj()
    hr = HindsightResult(task_id="t1", spans=[
        SpanScore(2, "assistant", [0.5, 0.5]), SpanScore(3, "tool", [1.0]),
        SpanScore(4, "assistant", [0.3]), SpanScore(5, "tool", [0.2])],
        obs_surprise=0.6, act_gain=0.4)
    on = to_train_sample(t, hr, StreamConfig(gate_error_actions=True)).token_weights_by_msg
    off = to_train_sample(t, hr, StreamConfig()).token_weights_by_msg
    assert on[2] == [0.0, 0.0] and on[3] == [1.0] and on[4] == [0.3]
    assert off[2] == [0.5, 0.5]


def test_train_channels_ablation():
    t = traj()
    hr = HindsightResult(task_id="t1", spans=[
        SpanScore(2, "assistant", [0.5]), SpanScore(3, "tool", [1.0]),
        SpanScore(4, "assistant", [0.3]), SpanScore(5, "tool", [0.2])],
        obs_surprise=0.6, act_gain=0.4)
    act = to_train_sample(t, hr, StreamConfig(train_channels="act")).token_weights_by_msg
    obs = to_train_sample(t, hr, StreamConfig(train_channels="obs")).token_weights_by_msg
    assert act[2:] == [[0.5], [0.0], [0.3], [0.0]]
    assert obs[2:] == [[0.0], [1.0], [0.0], [0.2]]


def test_signed_weights_dead_band_and_caps_on_all_actions():
    t = traj()  # msg2 action -> Error, msg4 action -> ok (status irrelevant here)
    hr = HindsightResult(task_id="t1", spans=[
        SpanScore(2, "assistant", [-9.0, 0.4, 0.8]), SpanScore(3, "tool", [1.0]),
        SpanScore(4, "assistant", [3.0, -0.2, -0.7]), SpanScore(5, "tool", [0.2])],
        obs_surprise=0.6, act_gain=0.4)
    cfg = StreamConfig(signed=True, train_channels="act", pos_thr=0.5, neg_thr=0.5,
                       pos_cap=1.55, neg_cap=4.51)
    w = to_train_sample(t, hr, cfg).token_weights_by_msg
    assert w[2] == [-4.51, 0.0, 0.8]   # capped negative / dead band / positive
    assert w[3] == [0.0] and w[5] == [0.0]  # obs channel off
    assert w[4] == [1.55, 0.0, -0.7]   # capped positive / dead band / negative


def test_weighted_ce_unlikelihood_pushes_down():
    torch = __import__("pytest").importorskip("torch")
    from sediment.trainer import weighted_ce
    logits = torch.zeros(1, 2, 5, requires_grad=True)
    targets = torch.tensor([[1, 3]])
    w = torch.tensor([[1.0, -1.0]])
    loss = weighted_ce(logits, targets, w, ul_lambda=0.5)
    loss.backward()
    g = logits.grad[0]
    assert g[0, 1] < 0            # positive weight: raise p(target)
    assert g[1, 3] > 0            # negative weight: lower p(target)
    plain = weighted_ce(logits.detach(), targets, w.clamp_min(0), ul_lambda=0.0)
    assert float(loss) > float(plain)


def test_norm_floor_scales_dose_with_evidence_mass():
    torch = __import__("pytest").importorskip("torch")
    from sediment.trainer import weighted_ce
    logits = torch.zeros(1, 3, 5)
    targets = torch.tensor([[1, 2, 3]])
    small = torch.tensor([[0.5, 0.0, 0.0]])
    big = torch.tensor([[10.0, 10.0, 10.0]])
    a = weighted_ce(logits, targets, small, norm_floor=20.0)
    b = weighted_ce(logits, targets, big, norm_floor=20.0)
    c = weighted_ce(logits, targets, small)  # mean: mass-independent
    assert float(a) < float(b) and abs(float(c) - float(b)) < 1e-6


def test_binary_weight_mode_selects_without_magnitude():
    t = traj()
    hr = HindsightResult(task_id="t1", spans=[
        SpanScore(2, "assistant", [0.05, 2.0]), SpanScore(3, "tool", [1.0]),
        SpanScore(4, "assistant", [0.3, -1.0]), SpanScore(5, "tool", [0.2])],
        obs_surprise=0.6, act_gain=0.4)
    w = to_train_sample(t, hr, StreamConfig(weight_mode="binary", gate_thr=0.1,
                                            train_channels="act")).token_weights_by_msg
    assert w[2] == [0.0, 1.0] and w[4] == [1.0, 0.0] and w[3] == [0.0]


def test_binary_signed_two_gates():
    t = traj()
    hr = HindsightResult(task_id="t1", spans=[
        SpanScore(2, "assistant", [0.05, 2.0, -0.9]), SpanScore(3, "tool", [1.0]),
        SpanScore(4, "assistant", [0.3, -0.1]), SpanScore(5, "tool", [0.2])],
        obs_surprise=0.6, act_gain=0.4)
    cfg = StreamConfig(weight_mode="binary", signed=True, pos_thr=0.1, neg_thr=0.5,
                       train_channels="act")
    w = to_train_sample(t, hr, cfg).token_weights_by_msg
    assert w[2] == [0.0, 1.0, -1.0] and w[4] == [1.0, 0.0]
