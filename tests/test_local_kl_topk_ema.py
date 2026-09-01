"""New local-teacher KL paths (08-30): student top-k + grouped tail
(cfg.kl_student_topk) and the SDPO-style EMA self-teacher (ema_state passed
into _backward_local_kl). k=V must reproduce the exact full-vocab loss; k<V
lower-bounds it; the EMA path must build the target under the EMA weights and
restore the student's weights afterwards."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from sediment.trainer import _backward_local_kl  # noqa: E402
from test_local_teacher_kl import _Model, _setup  # noqa: E402


class _EmaModel(_Model):
    def named_parameters(self):
        return [("shift", self.shift)]


def _run(model, mask, targets, input_ids, ctx, cfg, names, ema_state=None):
    use = {n: ctx[n] for n in names}
    model.shift.grad = None
    value, _ = _backward_local_kl(
        model, input_ids, use, mask, targets, cfg, micro_batch=1,
        ema_state=ema_state)
    return value


def test_topk_equals_full_at_k_vocab_and_lower_bounds_below():
    model, mask, targets, input_ids, ctx, cfg, tables = _setup(torch)
    vocab = tables[0].shape[1]
    full = _run(model, mask, targets, input_ids, ctx, cfg, ("donor", "failure"))
    cfg_kv = StreamConfigWith(cfg, kl_student_topk=vocab)
    at_v = _run(model, mask, targets, input_ids, ctx, cfg_kv, ("donor", "failure"))
    assert at_v == pytest.approx(full, abs=1e-4)
    cfg_k4 = StreamConfigWith(cfg, kl_student_topk=4)
    below = _run(model, mask, targets, input_ids, ctx, cfg_k4, ("donor", "failure"))
    assert below <= full + 1e-5
    assert below >= -1e-6
    assert model.shift.grad is not None and float(model.shift.grad.abs().sum()) > 0.0


def StreamConfigWith(cfg, **kw):
    from dataclasses import replace

    return replace(cfg, **kw)


def test_ema_state_equal_to_student_matches_zero_lag():
    model, mask, targets, input_ids, ctx, cfg, _ = _setup(torch)
    model = _EmaModel(torch, *_tables_of(model))
    zero_lag = _run(model, mask, targets, input_ids, ctx, cfg, ("donor", "failure"))
    ema = {"shift": model.shift.detach().clone()}
    same = _run(model, mask, targets, input_ids, ctx, cfg, ("donor", "failure"),
                ema_state=ema)
    assert same == pytest.approx(zero_lag, abs=1e-5)


def test_ema_base_shifts_the_poe_target_and_weights_are_restored():
    model, mask, targets, input_ids, ctx, cfg, tables = _setup(torch)
    model = _EmaModel(torch, *_tables_of(model))
    S, D, F, pad_d, pad_f = tables
    before = model.shift.detach().clone()
    ema = {"shift": torch.zeros_like(model.shift)}
    value = _run(model, mask, targets, input_ids, ctx, cfg, ("donor", "failure"),
                 ema_state=ema)
    # weights restored after the teacher pass; student grad exists
    assert torch.equal(model.shift.detach(), before)
    assert model.shift.grad is not None
    # monolithic reference: base under the EMA weights (shift = 0)
    sel = mask[0].nonzero(as_tuple=True)[0]
    logp = (S + model.shift).index_select(0, sel).log_softmax(-1)
    base_e = S.index_select(0, sel).log_softmax(-1)
    target = base_e + (D.index_select(0, sel + pad_d).log_softmax(-1) - base_e) \
        + (F.index_select(0, sel + pad_f).log_softmax(-1) - base_e)
    target = target.log_softmax(-1)
    kl = (logp.exp() * (logp - target)).sum(-1)
    w = mask[0].index_select(0, sel)
    expected = float((kl * w).sum() / w.sum())
    assert value == pytest.approx(expected, abs=1e-5)
    # and it differs from the zero-lag target (base matters under PoE)
    zero_lag = _run(model, mask, targets, input_ids, ctx, cfg, ("donor", "failure"))
    assert abs(zero_lag - value) > 1e-4


def _tables_of(model):
    by_name = {name: table for name, table in model.tables.values()}
    n_pos = by_name["S"].shape[0]
    return by_name["S"], by_name["D"], by_name["F"], n_pos
