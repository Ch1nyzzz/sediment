from types import SimpleNamespace

import pytest


def test_margin_token_is_first_common_prefix_logit_pair():
    torch = pytest.importorskip("torch")
    from sediment.trainer import _margin_token_loss

    class Model:
        def __init__(self):
            self.rows = torch.nn.Parameter(torch.zeros(4, 16))

        def __call__(self, *, input_ids, logits_to_keep):
            return SimpleNamespace(
                logits=self.rows.index_select(0, logits_to_keep).unsqueeze(0)
            )

    model = Model()
    with torch.no_grad():
        model.rows[2, 7] = -2.0
        model.rows[2, 8] = 1.0
    chosen = ([10, 11, 12, 7], [2], [7])
    rejected = ([10, 11, 12, 8], [2], [8])

    loss = _margin_token_loss(model, chosen, rejected, torch.device("cpu"), 1.0)

    # delta = -2 - 1 - 1 = -4; L = softplus(4)
    assert float(loss.detach()) == pytest.approx(float(torch.nn.functional.softplus(
        torch.tensor(4.0))), abs=1e-6)
    loss.backward()
    assert model.rows.grad[2, 7] < 0
    assert model.rows.grad[2, 8] > 0
    assert int(torch.count_nonzero(model.rows.grad)) == 2


def test_margin_token_rejects_different_causal_prefixes():
    torch = pytest.importorskip("torch")
    from sediment.trainer import _margin_token_loss

    class Model:
        def __call__(self, **kwargs):
            raise AssertionError("must not forward an invalid pair")

    chosen = ([10, 11, 12, 7], [2], [7])
    rejected = ([10, 99, 12, 8], [2], [8])
    assert _margin_token_loss(Model(), chosen, rejected, torch.device("cpu"), 1.0) is None
