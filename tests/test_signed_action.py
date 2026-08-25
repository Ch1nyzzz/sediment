from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from sediment.types import Message


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "signed_action_worker", ROOT / "scripts" / "signed_action_worker.py")
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
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


class ControlEngine:
    def __init__(self, tokenizer: TinyTokenizer):
        self.tokenizer = tokenizer

    def _prompt_logprobs(self, messages, adapter):
        del adapter
        ids = self.tokenizer.apply_chat_template(
            [message.to_dict() for message in messages], tokenize=True,
            return_dict=False)
        text = " ".join(message.content for message in messages)
        good = self.tokenizer._id("good")
        bad = self.tokenizer._id("bad")
        values = []
        for token in ids:
            value = -1.0
            if MOD.ahc.MASKED_RESULT in text:
                value = -0.5
            elif "worked" in text and token == good:
                value = -0.1
            elif "failed" in text and token == bad:
                value = -1.2
            values.append(value)
        return values


def test_semantic_mask_excludes_framing_and_keeps_action_values():
    assert not MOD.is_semantic_token("<tool_call>")
    assert not MOD.is_semantic_token(" arguments")
    assert not MOD.is_semantic_token(" },")
    assert MOD.is_semantic_token(" update_location")
    assert MOD.is_semantic_token(" LOC-7")


def test_action_mask_excludes_tool_name_and_keys_but_keeps_values():
    tokenizer = TinyTokenizer()
    action = '{"name": "update location", "arguments": {"location id": "LOC-7"}}'
    raw_ids = tokenizer.encode(action)
    kept = [
        tokenizer.decode([token_id])
        for token_id, keep in zip(
            raw_ids, MOD.action_semantic_mask(tokenizer, action, raw_ids), strict=True)
        if keep
    ]
    assert '"update' not in kept
    assert 'location",' not in kept
    assert '"location' not in kept
    assert 'id":' not in kept
    assert '"LOC-7"}}' in kept


def test_action_mask_excludes_subtokens_from_underscored_tool_name():
    tokenizer = TinyTokenizer()
    action = '{"name": "add maintenance record", "arguments": {"id": "R-1"}}'
    raw_ids = tokenizer.encode(action)
    kept = [
        tokenizer.decode([token_id])
        for token_id, keep in zip(
            raw_ids, MOD.action_semantic_mask(tokenizer, action, raw_ids), strict=True)
        if keep
    ]
    assert '"add' not in kept
    assert "maintenance" not in kept
    assert 'record",' not in kept
    assert '"R-1"}}' in kept


def test_signed_sample_gates_positive_to_ok_and_negative_to_error():
    tokenizer = TinyTokenizer()
    engine = ControlEngine(tokenizer)
    messages = [
        Message("user", "task"),
        Message("assistant", "good"),
        Message("user", "worked"),
        Message("assistant", "bad"),
    ]
    sample, diagnostics = MOD.build_signed_sample(
        engine, tokenizer, "t", messages,
        [(1, "good", "worked", "ok"), (3, "bad", "failed", "ERROR")],
        "base", positive_cap=1.55, negative_cap=4.51, max_seq_len=100)

    assert sample.positive_denominator == 1
    assert sample.negative_denominator == 1
    assert sum(sample.positive_weights_by_msg[1]) == pytest.approx(0.4)
    assert sum(sample.negative_weights_by_msg[3]) == pytest.approx(0.7)
    assert sum(sample.negative_weights_by_msg[1]) == 0
    assert sum(sample.positive_weights_by_msg[3]) == 0
    assert diagnostics[0]["positive_tok"] == 1
    assert diagnostics[1]["negative_tok"] == 1


def test_caps_preserve_absolute_dose_instead_of_normalizing_weight_sum():
    tokenizer = TinyTokenizer()
    engine = ControlEngine(tokenizer)
    messages = [Message("user", "task"), Message("assistant", "bad")]
    sample, _ = MOD.build_signed_sample(
        engine, tokenizer, "t", messages, [(1, "bad", "failed", "ERROR")],
        "base", positive_cap=1.55, negative_cap=0.2, max_seq_len=100)

    assert sample.negative_denominator == 1
    assert sum(sample.negative_weights_by_msg[1]) == pytest.approx(0.2)


def test_direction_gate_ignores_tiny_dose_and_rejects_wrong_signed_moves():
    assert MOD.direction_violation(1e-8, -1.0, 1e-8, 1.0, 1e-5, 1e-5) is None
    assert MOD.direction_violation(
        1e-3, -0.1, 1e-3, -0.1, 1e-5, 1e-5
    ) == "positive_direction_violation"
    assert MOD.direction_violation(
        1e-3, 0.1, 1e-3, 0.1, 1e-5, 1e-5
    ) == "negative_direction_violation"
    assert MOD.direction_violation(1e-3, 0.1, 1e-3, -0.1, 1e-5, 1e-5) is None
