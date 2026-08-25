from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sediment.types import Message


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "action_hint_control_worker", ROOT / "scripts" / "action_hint_control_worker.py")
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


class HintEngine:
    def __init__(self, tokenizer: TinyTokenizer):
        self.tokenizer = tokenizer

    def _prompt_logprobs(self, messages, adapter):
        del adapter
        ids = self.tokenizer.apply_chat_template(
            [message.to_dict() for message in messages],
            tokenize=True, return_dict=False)
        text = " ".join(message.content for message in messages)
        target = self.tokenizer._id("act")
        target_logp = -1.0
        if MOD.MASKED_RESULT in text:
            target_logp = -0.5
        elif "real observation" in text:
            target_logp = -0.2
        return [target_logp if token == target else -1.0 for token in ids]


def test_matched_action_control_subtracts_shared_hint_effect():
    tokenizer = TinyTokenizer()
    result = MOD.score_action(
        HintEngine(tokenizer), tokenizer, [Message("user", "task")],
        "act", "real observation")

    assert result["hint"] == pytest.approx({"mean": 0.5, "pos_frac": 1.0})
    assert result["polluted"] == pytest.approx({"mean": 0.8, "pos_frac": 1.0})
    assert result["controlled"] == pytest.approx({"mean": 0.3, "pos_frac": 1.0})
    assert result["tokens"][0]["controlled_delta"] == pytest.approx(0.3)
