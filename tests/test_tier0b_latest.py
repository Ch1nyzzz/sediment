from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sediment.types import Message


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "tier0b_latest_worker", ROOT / "scripts" / "tier0b_latest_worker.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class TinyTokenizer:
    def __init__(self):
        self.vocab: dict[str, int] = {}

    def _id(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self._id(x) for x in text.split()]

    def apply_chat_template(self, messages, tokenize=True, return_dict=False):
        assert tokenize and not return_dict
        toks: list[int] = []
        for msg in messages:
            toks.append(self._id(f"<{msg['role']}>"))
            toks.extend(self.encode(msg["content"]))
            toks.append(self._id("<end>"))
        return toks


class ContextGainEngine:
    def __init__(self, tokenizer: TinyTokenizer):
        self.tokenizer = tokenizer

    def _prompt_logprobs(self, messages, adapter):
        del adapter
        ids = self.tokenizer.apply_chat_template(
            [m.to_dict() for m in messages], tokenize=True, return_dict=False)
        shipped = self.tokenizer._id("shipped")
        stale = self.tokenizer._id("stale")
        contextual = len(messages) > 1
        out = [-1.0] * len(ids)
        if contextual:
            out = [(-0.2 if x == shipped else -1.4 if x == stale else -1.0) for x in ids]
        return out


def test_latest_residual_uses_same_target_and_masks_all_prior_messages():
    tok = TinyTokenizer()
    engine = ContextGainEngine(tok)
    prefix = [
        Message("system", "tools"),
        Message("user", "cancel order"),
        Message("assistant", "check status"),
    ]
    sample, diag = MOD.latest_residual_sample(
        engine, tok, "task-1", prefix, "status shipped stale", "base")

    assert sample.messages[:-1] == prefix
    assert sample.messages[-1] == Message("user", "status shipped stale")
    assert sample.token_weights_by_msg[:-1] == [[], [], []]
    assert sample.token_weights_by_msg[-1] == pytest.approx([0.0, 0.0, 0.8, 0.0, 0.0])
    assert diag == {
        "n_tok": 3,
        "mean_delta": pytest.approx(0.133333, abs=1e-6),
        "pos_frac": pytest.approx(1 / 3, abs=1e-6),
        "mean_pos_w": pytest.approx(0.8 / 3, abs=1e-6),
        "max_pos_w": 0.8,
    }


def test_latest_residual_accepts_environment_text_regardless_of_native_role():
    tok = TinyTokenizer()
    engine = ContextGainEngine(tok)
    prefix = [Message("system", "tools"), Message("assistant", "bad call")]
    sample, diag = MOD.latest_residual_sample(
        engine, tok, "task-2", prefix, "Error malformed", "base")
    assert sample.messages[-1].role == "user"
    assert diag["n_tok"] == 2
    assert sum(sample.token_weights_by_msg[-1]) == 0.0


def test_latest_residual_fails_closed_on_engine_template_mismatch():
    tok = TinyTokenizer()
    engine = ContextGainEngine(tok)
    original = engine._prompt_logprobs

    def short(messages, adapter):
        return original(messages, adapter)[:-1]

    engine._prompt_logprobs = short
    with pytest.raises(ValueError, match="engine/tokenizer"):
        MOD.latest_residual_sample(
            engine, tok, "task-3", [Message("assistant", "act")], "status shipped", "base")
