"""Local (trainer-side) full-vocabulary teacher for context distillation
(cfg.kl_teacher="local"): one context = exact reverse KL to that teacher; two
contexts combine in logit space (product of experts); the conflict rule drops
the donor where it disagrees with the failure teacher on the observed token;
the KL mask keeps weight magnitudes."""
from __future__ import annotations

import contextlib
import math
from types import SimpleNamespace

import pytest

from sediment.config import StreamConfig


def _views(torch, n_pos=6, vocab=11, pad_d=3, pad_f=5, seed=0):
    """Student view of length n_pos+1 and two context views that are longer by
    a prefix of pad tokens; logits are indexed by (view length, position)."""
    torch.manual_seed(seed)
    S = torch.randn(n_pos, vocab)
    D = torch.randn(n_pos + pad_d, vocab)
    F = torch.randn(n_pos + pad_f, vocab)
    return S, D, F


class _Model:
    def __init__(self, torch, S, D, F, n_pos):
        self.torch = torch
        self.shift = torch.nn.Parameter(torch.randn(n_pos, S.shape[1]) * 0.1)
        self.tables = {S.shape[0] + 1: ("S", S), D.shape[0] + 1: ("D", D), F.shape[0] + 1: ("F", F)}
        self.adapter_enabled = True

    def __call__(self, *, input_ids, logits_to_keep=0):
        name, table = self.tables[input_ids.shape[1]]
        logits = table + (self.shift if (name == "S" and self.adapter_enabled) else 0.0) if name == "S" else table
        if isinstance(logits_to_keep, self.torch.Tensor):
            logits = logits.index_select(0, logits_to_keep)
        return SimpleNamespace(logits=logits.unsqueeze(0))

    @contextlib.contextmanager
    def disable_adapter(self):
        old = self.adapter_enabled
        self.adapter_enabled = False
        try:
            yield
        finally:
            self.adapter_enabled = old


def _setup(torch, **cfg_kw):
    n_pos, vocab, pad_d, pad_f = 6, 11, 3, 5
    S, D, F = _views(torch, n_pos, vocab, pad_d, pad_f)
    model = _Model(torch, S, D, F, n_pos)
    mask = torch.tensor([[1.0, 0.0, 0.2, 0.0, 0.0, 1.0]])
    targets = torch.tensor([[2, 4, 6, 8, 10, 1]])
    input_ids = torch.zeros(1, n_pos + 1, dtype=torch.long)
    ctx = {"donor": (torch.zeros(1, n_pos + pad_d + 1, dtype=torch.long), pad_d),
           "failure": (torch.zeros(1, n_pos + pad_f + 1, dtype=torch.long), pad_f)}
    cfg = StreamConfig(kl_target=True, kl_teacher="local", kl_reverse=True, anchor_kl_coef=0.0, **cfg_kw)
    return model, mask, targets, input_ids, ctx, cfg, (S, D, F, pad_d, pad_f)


def _monolithic(torch, model, mask, targets, ctx_names, tables, cfg):
    S, D, F, pad_d, pad_f = tables
    sel = mask[0].nonzero(as_tuple=True)[0]
    logp = (S + model.shift).index_select(0, sel).log_softmax(-1)
    base = logp.detach()
    target = base.clone()
    if "donor" in ctx_names:
        target = target + cfg.poe_alpha * (D.index_select(0, sel + pad_d).log_softmax(-1) - base)
    if "failure" in ctx_names:
        target = target + cfg.poe_beta * (F.index_select(0, sel + pad_f).log_softmax(-1) - base)
    target = target.log_softmax(-1)
    kl = (logp.exp() * (logp - target)).sum(-1)
    w = mask[0].index_select(0, sel)
    return (kl * w).sum() / w.sum()


@pytest.mark.parametrize("names", [("failure",), ("donor",), ("donor", "failure")])
def test_local_kl_matches_monolithic_poe(names):
    torch = pytest.importorskip("torch")
    from sediment.trainer import _backward_local_kl

    model, mask, targets, input_ids, ctx, cfg, tables = _setup(torch, poe_alpha=0.7, poe_beta=1.3)
    ctx = {k: v for k, v in ctx.items() if k in names}
    expected = _monolithic(torch, model, mask, targets, names, tables, cfg)
    expected.backward()
    g = model.shift.grad.detach().clone()
    model.shift.grad = None
    actual, dbg = _backward_local_kl(model, input_ids, ctx, mask, targets, cfg, micro_batch=1, chunk=2)
    assert actual == pytest.approx(expected.item(), abs=1e-6)
    assert torch.allclose(model.shift.grad, g, atol=1e-6, rtol=1e-5)
    assert dbg is not None and dbg[0].tolist() == [0, 2, 5] and dbg[3] == 0


def test_single_context_is_exact_full_vocab_reverse_kl():
    torch = pytest.importorskip("torch")
    from sediment.trainer import _backward_local_kl

    model, mask, targets, input_ids, ctx, cfg, (S, D, F, pad_d, pad_f) = _setup(torch)
    sel = mask[0].nonzero(as_tuple=True)[0]
    p = (S + model.shift).index_select(0, sel).log_softmax(-1)
    q = F.index_select(0, sel + pad_f).log_softmax(-1)
    kl = (p.exp() * (p - q)).sum(-1)
    w = mask[0].index_select(0, sel)
    expected = ((kl * w).sum() / w.sum()).item()
    actual, _ = _backward_local_kl(model, input_ids, {"failure": ctx["failure"]}, mask, targets, cfg, micro_batch=1)
    assert actual == pytest.approx(expected, abs=1e-6)


def test_local_teacher_exact_jsd_matches_monolithic_objective():
    torch = pytest.importorskip("torch")
    from sediment.trainer import _backward_local_kl

    model, mask, targets, input_ids, ctx, cfg, (S, D, F, pad_d, pad_f) = _setup(
        torch, kl_jsd_alpha=0.5
    )
    sel = mask[0].nonzero(as_tuple=True)[0]
    logp = (S + model.shift).index_select(0, sel).log_softmax(-1)
    target = F.index_select(0, sel + pad_f).log_softmax(-1)
    logm = torch.logaddexp(math.log(0.5) + target, math.log(0.5) + logp)
    jsd = (0.5 * (target.exp() * (target - logm)).sum(-1)
           + 0.5 * (logp.exp() * (logp - logm)).sum(-1))
    w = mask[0].index_select(0, sel)
    expected = (jsd * w).sum() / w.sum()
    expected.backward()
    expected_grad = model.shift.grad.detach().clone()
    model.shift.grad = None

    actual, _ = _backward_local_kl(
        model, input_ids, {"failure": ctx["failure"]}, mask, targets,
        cfg, micro_batch=1,
    )

    assert actual == pytest.approx(float(expected.detach()), abs=1e-6)
    assert torch.allclose(model.shift.grad, expected_grad, atol=1e-6, rtol=1e-5)


def test_token_level_is_uses_collection_logprobs_and_clips_per_token():
    torch = pytest.importorskip("torch")
    from sediment.trainer import _backward_local_kl

    model, mask, targets, input_ids, ctx, cfg, (S, D, F, pad_d, pad_f) = _setup(
        torch, replay_is_clip=2.0
    )
    ctx = {"failure": ctx["failure"]}
    sel = mask[0].nonzero(as_tuple=True)[0]
    logp = (S + model.shift).index_select(0, sel).log_softmax(-1)
    q = F.index_select(0, sel + pad_f).log_softmax(-1)
    kl = (logp.exp() * (logp - q)).sum(-1)
    tg = targets[0].index_select(0, sel)
    now = logp.gather(-1, tg.unsqueeze(-1)).squeeze(-1).detach()
    raw_ratio = torch.tensor([0.5, 1.5, 4.0])
    old = torch.full_like(mask, float("nan"))
    old[0, sel] = now - raw_ratio.log()
    ratio = raw_ratio.clamp(max=2.0)
    w = mask[0].index_select(0, sel)
    scale = 0.3
    expected = (kl * w * ratio).sum() / w.sum() * scale
    expected.backward()
    expected_grad = model.shift.grad.detach().clone()
    model.shift.grad = None

    actual, dbg = _backward_local_kl(
        model, input_ids, ctx, mask, targets, cfg, micro_batch=1,
        behavior_logp=old, loss_scale=scale,
    )

    assert actual == pytest.approx(float(expected.detach()), abs=1e-6)
    assert torch.allclose(model.shift.grad, expected_grad, atol=1e-6, rtol=1e-5)
    assert dbg is not None and dbg[4].tolist() == pytest.approx([0.5, 1.5, 2.0])


def test_conflict_rule_drops_donor_where_it_disagrees():
    torch = pytest.importorskip("torch")
    from sediment.trainer import _backward_local_kl

    model, mask, targets, input_ids, ctx, cfg, (S, D, F, pad_d, pad_f) = _setup(torch, poe_conflict="failure_wins")
    # force a clear disagreement on the observed token at the first selected position (0):
    # donor raises target token 2, failure lowers it
    D[pad_d + 0, 2] += 6.0
    F[pad_f + 0, 2] -= 6.0
    actual, dbg = _backward_local_kl(model, input_ids, ctx, mask, targets, cfg, micro_batch=1)
    assert dbg[3] >= 1
    # the objective with the donor dropped at that position equals a mixed monolithic target
    sel = mask[0].nonzero(as_tuple=True)[0]
    logp = (S + model.shift).index_select(0, sel).log_softmax(-1)
    base = logp.detach()
    dD = D.index_select(0, sel + pad_d).log_softmax(-1) - base
    dF = F.index_select(0, sel + pad_f).log_softmax(-1) - base
    tg = targets[0].index_select(0, sel)
    dm = dD.gather(-1, tg.unsqueeze(-1)).squeeze(-1)
    df = dF.gather(-1, tg.unsqueeze(-1)).squeeze(-1)
    conflict = (dm * df < 0) & (dm.abs() > 0.25) & (df.abs() > 0.25)
    coef = torch.where(conflict, torch.zeros_like(dm), torch.ones_like(dm))
    target = (base + coef.unsqueeze(-1) * dD + dF).log_softmax(-1)
    kl = (logp.exp() * (logp - target)).sum(-1)
    w = mask[0].index_select(0, sel)
    assert actual == pytest.approx(((kl * w).sum() / w.sum()).item(), abs=1e-6)
    assert int(conflict.sum()) == dbg[3]


def test_encode_preserves_exact_behavior_logprob_alignment():
    pytest.importorskip("transformers")
    from sediment.chat_template import load_tokenizer
    from sediment.trainer import _encode
    from sediment.types import Message, TrainSample

    cfg = StreamConfig(kl_target=True, kl_teacher="local", weight_mode="sft",
                       max_seq_len=4096)
    tokenizer = load_tokenizer(cfg.model)
    msgs = [Message("system", "sys"), Message("user", "task"),
            Message("assistant", "reason, then act")]
    prev = []
    spans = []
    for i in range(1, len(msgs) + 1):
        ids = list(tokenizer.apply_chat_template(
            [m.to_dict() for m in msgs[:i]], tokenize=True, return_dict=False
        ))
        spans.append(ids[len(prev):])
        prev = ids
    old = [[] for _ in msgs]
    old[-1] = [-0.01 * (j + 1) for j in range(len(spans[-1]))]
    sample = TrainSample(
        task_id="t", messages=msgs, token_weights_by_msg=[[], [], [1.0]],
        behavior_logprobs_by_msg=old,
    )

    _encode(tokenizer, sample, cfg)

    aligned = _encode.last_behavior_logprobs
    start = len(spans[0]) + len(spans[1])
    assert all(x != x for x in aligned[:start])  # NaN: history has no stored behavior logp
    assert aligned[start:start + len(spans[-1])] == pytest.approx(old[-1])


def test_context_ids_offset_and_suffix_alignment():
    pytest.importorskip("transformers")
    from sediment.chat_template import load_tokenizer
    from sediment.trainer import _context_ids
    from sediment.types import Message, TrainSample

    cfg = StreamConfig()
    tokenizer = load_tokenizer(cfg.model)
    msgs = [Message("system", "sys"), Message("user", "do the thing"),
            Message("assistant", '<tool_call>{"name": "get", "arguments": {}}</tool_call>'),
            Message("tool", "ok"), Message("assistant", "done")]
    sample = TrainSample(task_id="t", messages=msgs, token_weights_by_msg=[[] for _ in msgs])
    student = list(tokenizer.apply_chat_template([m.to_dict() for m in msgs], tokenize=True, return_dict=False))
    ctx_ids, off = _context_ids(tokenizer, sample, "<previous_attempts>\nsome block text\n</previous_attempts>", student)
    assert off > 0 and len(ctx_ids) == len(student) + off
    # every student token after the first user message maps to the same token in the context view
    pre = list(tokenizer.apply_chat_template([m.to_dict() for m in msgs[:2]], tokenize=True, return_dict=False))
    for j in range(len(pre), len(student)):
        assert ctx_ids[j + off] == student[j]


def test_anchor_assistant_only_restricts_free_positions():
    torch = pytest.importorskip("torch")
    from sediment.trainer import _backward_anchor

    model, mask, targets, input_ids, ctx, cfg, tables = _setup(torch)
    cfg_all = StreamConfig(kl_target=True, kl_teacher="local", anchor_kl_coef=0.5, kl_topk=3)
    cfg_asst = StreamConfig(kl_target=True, kl_teacher="local", anchor_kl_coef=0.5, kl_topk=3,
                            anchor_assistant_only=True)
    anchorable = torch.tensor([False, True, False, False, True, False])  # only positions 1 and 4 are assistant
    v_all = _backward_anchor(model, input_ids, mask, cfg_all, micro_batch=1, anchorable=anchorable)
    model.shift.grad = None
    v_asst = _backward_anchor(model, input_ids, mask, cfg_asst, micro_batch=1, anchorable=anchorable)
    # free positions: mask==0 -> {1,3,4}; restricted -> {1,4}: a different (smaller-denominator) average
    assert v_all > 0 and v_asst > 0 and v_all != v_asst
    # with the restriction, positions outside {1,4} get no anchor gradient
    g = model.shift.grad
    assert torch.allclose(g[3], torch.zeros_like(g[3]))
    assert not torch.allclose(g[1], torch.zeros_like(g[1]))


def test_sparse_weighted_ce_matches_full_weighted_ce():
    torch = pytest.importorskip("torch")
    from sediment.trainer import sparse_weighted_ce, weighted_ce

    n_pos, vocab = 6, 11
    S, D, F = _views(torch, n_pos, vocab, 3, 5)
    model = _Model(torch, S, D, F, n_pos)
    input_ids = torch.tensor([[0, 2, 4, 6, 8, 10, 1]])  # length n_pos + 1
    w = torch.tensor([[0.0, 1.0, 0.0, 0.2, 0.0, 1.0]])
    full = weighted_ce((S + model.shift).unsqueeze(0), input_ids[:, 1:], w)
    sparse = sparse_weighted_ce(model, input_ids, w)
    assert sparse.item() == pytest.approx(full.item(), abs=1e-6)
    full.backward(); g_full = model.shift.grad.clone(); model.shift.grad = None
    sparse.backward()
    assert torch.allclose(model.shift.grad, g_full, atol=1e-6)
