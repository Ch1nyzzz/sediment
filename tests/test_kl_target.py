"""Context distillation (cfg.kl_target): teacher plumbing, masking, KL numerics."""
from __future__ import annotations

import pytest

from sediment.config import StreamConfig
from sediment.engine.mock import MockEngine
from sediment.hindsight import score, to_train_sample
from sediment.types import ExperienceBlock, Message, Trajectory


def _traj() -> Trajectory:
    return Trajectory(
        task_id="t1", env_family="toy",
        messages=[
            Message("system", "sys"),
            Message("user", "do it"),
            Message("assistant", "<tool_call> {\"name\": \"get\", \"arguments\": {}} </tool_call>"),
            Message("tool", "ok done"),
        ],
        success=True, adapter="base",
    )


def test_teacher_flows_from_hindsight_to_sample():
    cfg = StreamConfig(kl_target=True, kl_topk=4, weight_mode="binary", gate_thr=-99)
    traj = _traj()
    hr = score(MockEngine(), traj, ExperienceBlock(text="peer stuff"), cfg)
    assert hr.teacher is not None and len(hr.teacher) == len(hr.spans)
    assert "teacher" not in hr.to_dict()  # megabytes: never persisted
    sample = to_train_sample(traj, hr, cfg)
    assert sample.teacher_by_msg is not None
    for i, (ws, tt) in enumerate(zip(sample.token_weights_by_msg, sample.teacher_by_msg)):
        assert len(tt) == len(ws), f"teacher/weights misaligned at message {i}"
        for d in tt:
            assert len(d) == 4 and all(isinstance(t, int) for t in d)


def test_no_teacher_without_the_flag():
    cfg = StreamConfig(kl_target=False)
    traj = _traj()
    hr = score(MockEngine(), traj, ExperienceBlock(text="peer stuff"), cfg)
    assert hr.teacher is None
    assert to_train_sample(traj, hr, cfg).teacher_by_msg is None


def test_kl_mask_unions_both_gates_inside_the_call_region():
    """+1 and -1 tokens both become KL positions, restricted to the semantic
    <tool_call> region (KL self-masks unpredictable tokens, not narration)."""
    pytest.importorskip("transformers")
    from sediment.chat_template import load_tokenizer
    from sediment.semantic import tool_call_region_mask
    from sediment.trainer import _encode
    from sediment.types import TrainSample

    cfg = StreamConfig(kl_target=True, kl_topk=4, max_seq_len=4096)
    tokenizer = load_tokenizer(cfg.model)
    traj = _traj()
    # per-message spans of the REAL tokenizer, so weights/teacher align 1:1
    prev, spans = [], []
    for i in range(1, len(traj.messages) + 1):
        toks = list(tokenizer.apply_chat_template(
            [m.to_dict() for m in traj.messages[:i]], tokenize=True, return_dict=False))
        spans.append(toks[len(prev):])
        prev = toks
    # production shape: only scored (assistant/tool) messages carry entries
    scored = [m.role in ("assistant", "tool") for m in traj.messages]
    weights = [[1.0 if j % 2 else -1.0 for j in range(len(sp))] if ok else []
               for sp, ok in zip(spans, scored)]
    teacher = [[{7: -0.1, 8: -2.0}] * len(sp) if ok else [] for sp, ok in zip(spans, scored)]
    sample = TrainSample(task_id="t1", messages=traj.messages,
                         token_weights_by_msg=weights, teacher_by_msg=teacher)

    ids, ws, tch = _encode(tokenizer, sample, cfg)
    assert len(ids) == len(ws) == len(tch)
    assert set(ws) == {0.0, 1.0}  # both signs collapse to one unsigned mask
    act = next(i for i, m in enumerate(traj.messages) if m.role == "assistant")
    region = tool_call_region_mask(tokenizer, traj.messages[act].content, spans[act])
    off = sum(len(sp) for sp in spans[:act])
    kept = [j for j in range(len(spans[act])) if ws[off + j]]
    assert kept, "no KL positions inside the tool call"
    assert all(region[j] for j in kept)  # never outside the call region
    assert all(ws[j] == 0.0 for j in range(off))  # unscored prefix: no target

    # a message whose teacher is missing is dropped: a distribution cannot be
    # broadcast the way a scalar weight can
    sample.teacher_by_msg = [[] for _ in spans]
    assert not any(_encode(tokenizer, sample, cfg)[1])


def test_topk_kl_numerics():
    torch = pytest.importorskip("torch")
    from sediment.trainer import topk_kl

    V, K = 8, 3
    logits = torch.zeros(1, 2, V)  # student uniform: logp = -log 8 everywhere
    t_ids = torch.tensor([[[0, 1, 2], [0, 1, 2]]])
    # teacher puts all its mass on token 0 at position 0
    lq = torch.log(torch.tensor([[[1.0, 1e-30, 1e-30], [0.125, 0.125, 0.125]]]))
    mask = torch.tensor([[1.0, 0.0]])
    targets = torch.tensor([[0, 0]])
    loss, dbg = topk_kl(logits, t_ids, lq, mask, targets=targets)
    # KL(delta_0 || uniform_8) = log 8
    assert loss.item() == pytest.approx(torch.log(torch.tensor(8.0)).item(), abs=1e-4)
    assert dbg is not None and dbg[0].tolist() == [0]
    # CE of the same token under the same uniform student is also log 8: the two
    # diagnostics are measured on one identical token set
    assert dbg[2].tolist() == pytest.approx([torch.log(torch.tensor(8.0)).item()], abs=1e-4)

    # teacher == student -> zero KL: this is what makes unpredictable tokens free
    mask2 = torch.tensor([[0.0, 1.0]])
    loss2, _ = topk_kl(logits, t_ids, lq, mask2)
    assert loss2.item() == pytest.approx(0.0, abs=1e-5)

    empty, dbg2 = topk_kl(logits, t_ids, lq, torch.zeros(1, 2))
    assert empty.item() == 0.0 and dbg2 is None


# -- sft mode (teacher-student distillation of ICL trajectories) -------------

def test_sft_mode_makes_every_action_token_a_kl_position():
    cfg = StreamConfig(weight_mode="sft", kl_target=True, kl_topk=4, train_channels="act")
    traj = _traj()
    hr = score(MockEngine(), traj, ExperienceBlock(text="peer stuff"), cfg)
    s = to_train_sample(traj, hr, cfg)
    by_role = {m.role: w for m, w in zip(traj.messages, s.token_weights_by_msg)}
    assert by_role["assistant"] and all(w == 1.0 for w in by_role["assistant"])
    assert not any(by_role["tool"])  # act channel
    assert s.teacher_by_msg is not None
    for ws, tt in zip(s.token_weights_by_msg, s.teacher_by_msg):
        assert len(ws) == len(tt)


def test_reverse_kl_is_mode_seeking():
    """Reverse KL is an expectation under the STUDENT, so a student that hides
    in the teacher's tail is penalised even where the teacher is flat."""
    torch = pytest.importorskip("torch")
    from sediment.trainer import topk_kl

    V, K = 8, 2
    t_ids = torch.tensor([[[0, 1]]])
    # teacher: all mass on token 0
    lq = torch.log(torch.tensor([[[1.0 - 1e-6, 1e-9]]]))
    mask = torch.tensor([[1.0]])
    on_mode = torch.full((1, 1, V), -20.0); on_mode[0, 0, 0] = 20.0    # student agrees
    off_mode = torch.full((1, 1, V), -20.0); off_mode[0, 0, 7] = 20.0  # student in the tail
    assert topk_kl(on_mode, t_ids, lq, mask, reverse=True)[0].item() == pytest.approx(0, abs=1e-3)
    assert topk_kl(off_mode, t_ids, lq, mask, reverse=True)[0].item() > 10
    # forward KL sees the same disagreement from the other side
    assert topk_kl(off_mode, t_ids, lq, mask, reverse=False)[0].item() > 10


def test_sft_credits_outside_the_tool_call_and_damps_copied_values():
    """The gated modes credit only the tool-call interior; sft must cover the
    whole action (including how a turn ends), and must damp argument values
    that are copyable from context."""
    pytest.importorskip("transformers")
    from sediment.chat_template import load_tokenizer
    from sediment.hindsight import sft_sample
    from sediment.semantic import tool_call_region_mask
    from sediment.trainer import _encode

    cfg = StreamConfig(weight_mode="sft", train_channels="act", grounded_weight=0.2,
                       max_seq_len=4096)
    tokenizer = load_tokenizer(cfg.model)
    traj = _traj()
    traj.messages[1].content = "cancel the order for group GRP-C please"
    traj.messages[2].content = ('I will look it up first.\n<tool_call> '
                                '{"name": "get", "arguments": {"group_id": "GRP-C"}} '
                                '</tool_call>')
    ids, ws, tch = _encode(tokenizer, sft_sample(traj, cfg), cfg)
    assert not any(tch)  # no kl target in sft mode

    prev, spans = [], []
    for i in range(1, len(traj.messages) + 1):
        toks = list(tokenizer.apply_chat_template(
            [m.to_dict() for m in traj.messages[:i]], tokenize=True, return_dict=False))
        spans.append(toks[len(prev):]); prev = toks
    off = sum(len(x) for x in spans[:2])
    region = tool_call_region_mask(tokenizer, traj.messages[2].content, spans[2])
    outside = [ws[off + j] for j in range(len(spans[2])) if not region[j]]
    assert any(w == 1.0 for w in outside), "narration/closing tokens must be supervised"
    damped = [w for w in ws if 0.0 < w < 1.0]
    assert damped and all(abs(w - 0.2) < 1e-9 for w in damped)  # GRP-C came from context
