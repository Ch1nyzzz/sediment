from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sediment.types import Message


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "latest_factorial_worker", ROOT / "scripts" / "latest_factorial_worker.py")
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
        return [self._id(token) for token in text.split()]

    def apply_chat_template(self, messages, tokenize=True, return_dict=False):
        assert tokenize and not return_dict
        ids: list[int] = []
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
            [message.to_dict() for message in messages], tokenize=True,
            return_dict=False)
        gain = self.tokenizer._id("gain")
        contextual = len(messages) > 1
        return [(-0.2 if contextual and token == gain else -1.0) for token in ids]


def test_residual_weights_with_standalone_training_context():
    tokenizer = TinyTokenizer()
    sample, diag = MOD.build_sample(
        GainEngine(tokenizer), tokenizer, "task", [Message("assistant", "act")],
        "plain gain", "base", "residual", "standalone", {})

    assert sample.messages == [Message("user", "plain gain")]
    assert sample.token_weights_by_msg[0] == pytest.approx([0.0, 0.0, 0.8, 0.0])
    assert diag["pos_frac"] == 0.5
    assert diag["mean_w"] == 0.4


def test_ngram_weights_with_full_training_context_and_zero_prefix_loss():
    tokenizer = TinyTokenizer()
    observation = "one two three four"
    raw_ids = tokenizer.encode(observation)
    history: dict[tuple[int, ...], int] = {}
    MOD.atttw.add_ngrams(history, raw_ids)
    prefix = [Message("system", "tools"), Message("assistant", "act")]
    sample, diag = MOD.build_sample(
        GainEngine(tokenizer), tokenizer, "task", prefix, observation, "base",
        "ngram", "full", history)

    assert sample.messages[:-1] == prefix
    assert sample.messages[-1] == Message("user", observation)
    assert sample.token_weights_by_msg[:-1] == [[], []]
    assert sample.token_weights_by_msg[-1] == pytest.approx(
        [0.0, 0.5, 0.5, 0.5, 0.5, 0.0])
    assert diag["mean_w"] == 0.5


def test_residual_ngram_multiplies_weights_in_standalone_context():
    tokenizer = TinyTokenizer()
    observation = "gain gain gain"
    raw_ids = tokenizer.encode(observation)
    history: dict[tuple[int, ...], int] = {}
    MOD.atttw.add_ngrams(history, raw_ids)
    sample, diag = MOD.build_sample(
        GainEngine(tokenizer), tokenizer, "task", [Message("assistant", "act")],
        observation, "base", "residual_ngram", "standalone", history)

    assert sample.messages == [Message("user", observation)]
    assert sample.token_weights_by_msg[0] == pytest.approx(
        [0.0, 0.4, 0.4, 0.4, 0.0])
    assert diag["pos_frac"] == 1.0
    assert diag["mean_residual_w"] == 0.8
    assert diag["mean_ngram_w"] == 0.5
    assert diag["mean_w"] == 0.4


def test_factorial_sample_fails_closed_on_template_mismatch():
    tokenizer = TinyTokenizer()
    engine = GainEngine(tokenizer)
    original = engine._prompt_logprobs

    def short(messages, adapter):
        return original(messages, adapter)[:-1]

    engine._prompt_logprobs = short
    with pytest.raises(ValueError, match="engine/tokenizer"):
        MOD.build_sample(
            engine, tokenizer, "task", [Message("assistant", "act")],
            "plain gain", "base", "residual", "standalone", {})
