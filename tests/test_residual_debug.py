from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sediment.types import Message


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "residual_debug_worker", ROOT / "scripts" / "residual_debug_worker.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class TinyTokenizer:
    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.inverse: dict[int, str] = {}

    def _id(self, token: str) -> int:
        if token not in self.vocab:
            token_id = len(self.vocab) + 1
            self.vocab[token] = token_id
            self.inverse[token_id] = token
        return self.vocab[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self._id(token) for token in text.split()]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.inverse[token_id] for token_id in ids)

    def apply_chat_template(self, messages, tokenize=True, return_dict=False):
        assert tokenize and not return_dict
        ids = []
        for message in messages:
            ids.append(self._id(f"<{message['role']}>") )
            ids.extend(self.encode(message["content"]))
            ids.append(self._id("<end>"))
        return ids


class GainEngine:
    def __init__(self, tokenizer: TinyTokenizer):
        self.tokenizer = tokenizer

    def _prompt_logprobs(self, messages, adapter):
        del adapter
        ids = self.tokenizer.apply_chat_template(
            [message.to_dict() for message in messages],
            tokenize=True, return_dict=False)
        gain = self.tokenizer._id("gain")
        contextual = len(messages) > 1
        return [(-0.2 if contextual and token == gain else -1.0) for token in ids]


def test_score_observation_preserves_full_and_solo_token_logps():
    tokenizer = TinyTokenizer()
    result = MOD.score_observation(
        GainEngine(tokenizer), tokenizer, [Message("assistant", "act")],
        "plain gain")

    assert result["n_tok"] == 2
    assert result["pos_frac"] == 0.5
    assert result["mean_delta"] == pytest.approx(0.4)
    assert result["tokens"] == [
        {
            "token_id": tokenizer._id("plain"), "token": "plain",
            "full_logp": -1.0, "solo_logp": -1.0, "delta": 0.0,
        },
        {
            "token_id": tokenizer._id("gain"), "token": "gain",
            "full_logp": -0.2, "solo_logp": -1.0, "delta": 0.8,
        },
    ]
