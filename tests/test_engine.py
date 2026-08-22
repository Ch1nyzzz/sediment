"""Engine module tests: mock determinism, adapter routing, tokenize, and the
vLLM HTTP client against a faked transport (no network)."""
from __future__ import annotations

import pytest

from sediment.engine import Engine, MockEngine, VllmClient, mock_tokenize
from sediment.engine.mock import EXPERIENCE_OPEN
from sediment.types import AdapterVersion, Message


def convo(user: str = "Please cancel order #42.") -> list[Message]:
    return [
        Message("system", "You are a customer-service agent."),
        Message("user", user),
        Message("assistant", "<tool_call> cancel </tool_call>"),
        Message("tool", "Error: cannot cancel shipped order"),
    ]


# -- mock ------------------------------------------------------------------

def test_mock_tokenize_whitespace():
    assert mock_tokenize("a b  c\nd") == ["a", "b", "c", "d"]
    assert mock_tokenize("") == []


def test_mock_satisfies_engine_protocol():
    assert isinstance(MockEngine(), Engine)
    assert isinstance(VllmClient("http://x", "m"), Engine)


def test_mock_generate_deterministic():
    eng = MockEngine()
    msgs = convo()[:2]
    a = eng.generate(msgs, temperature=0.7, max_tokens=64)
    b = eng.generate(msgs, temperature=0.7, max_tokens=64)
    assert a == b
    assert "cancel_order" in a  # student policy cancels blindly

    teacher = eng.generate(
        [Message("user", f"{EXPERIENCE_OPEN} shipped fails </previous_attempts> task")],
        temperature=0.0, max_tokens=64,
    )
    assert "get_order_status" in teacher  # teacher checks status first


def test_mock_score_shape_and_determinism():
    eng = MockEngine()
    msgs = convo()
    s1 = eng.score(msgs)
    s2 = eng.score(msgs)
    assert s1 == s2
    assert len(s1) == len(msgs)
    for msg, logps in zip(msgs, s1):
        assert len(logps) == len(mock_tokenize(msg.content))
        assert all(lp == -1.0 for lp in logps)  # no context -> baseline


def test_mock_score_context_lifts_tool_tokens():
    eng = MockEngine()
    block = f"{EXPERIENCE_OPEN} Error: cannot cancel shipped order </previous_attempts>"
    msgs = convo(user=block + "\nPlease cancel order #42.")
    scores = eng.score(msgs)
    tool_scores = scores[3]
    assert -0.4 in tool_scores  # tokens seen in the block get likelier
    assert all(lp == -1.0 for lp in scores[2])  # assistant role unaffected


def test_mock_adapter_policy_routing():
    eng = MockEngine(
        policy=lambda ms: "base-act",
        adapter_policies={"v0001": lambda ms: "v1-act"},
    )
    msgs = [Message("user", "hi")]
    assert eng.generate(msgs, temperature=0.0, max_tokens=8) == "base-act"
    assert eng.generate(msgs, adapter="base", temperature=0.0, max_tokens=8) == "base-act"
    assert eng.generate(msgs, adapter="v0001", temperature=0.0, max_tokens=8) == "v1-act"
    # unknown adapter falls back to the base policy
    assert eng.generate(msgs, adapter="v0002", temperature=0.0, max_tokens=8) == "base-act"


def test_mock_injected_scorer_and_load_adapter():
    eng = MockEngine(scorer=lambda tok, role, ctx, text: -2.5)
    assert eng.score([Message("user", "a b")]) == [[-2.5, -2.5]]
    v = AdapterVersion(name="v0001", path="/tmp/a", parent="v0000")
    eng.load_adapter(v)
    assert eng.adapters["v0001"] is v


# -- vllm client (faked transport) -----------------------------------------

TOKENS_PER_MSG = 3


def make_client() -> tuple[VllmClient, list[tuple[str, dict]]]:
    client = VllmClient("http://localhost:8000/", "qwen-base")
    calls: list[tuple[str, dict]] = []

    def fake_post(path: str, payload: dict) -> dict:
        calls.append((path, payload))
        if path == "/v1/chat/completions" and "prompt_logprobs" in payload:
            entries = []
            for i in range(len(payload["messages"])):
                for t in range(TOKENS_PER_MSG):
                    pos = i * TOKENS_PER_MSG + t
                    entries.append(
                        None if pos == 0 else
                        {"7": {"logprob": -float(i + 1), "rank": 1, "decoded_token": "x"}}
                    )
            return {"prompt_logprobs": entries}
        if path == "/v1/chat/completions":
            return {"choices": [{"message": {"role": "assistant",
                                             "content": f"gen:{payload['model']}"}}]}
        if path == "/v1/load_lora_adapter":
            return {"raw": "Success"}
        raise AssertionError(f"unexpected path {path}")

    client._post = fake_post  # type: ignore[method-assign]
    return client, calls


def test_vllm_generate_model_routing():
    client, calls = make_client()
    msgs = [Message("user", "hi")]
    assert client.generate(msgs, temperature=0.2, max_tokens=16) == "gen:qwen-base"
    assert client.generate(msgs, adapter="v0000", temperature=0.2, max_tokens=16) == "gen:qwen-base"
    assert client.generate(msgs, adapter="v0003", temperature=0.2, max_tokens=16) == "gen:v0003"
    payload = calls[-1][1]
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["temperature"] == 0.2 and payload["max_tokens"] == 16


def test_vllm_score_per_message_slices():
    client, _ = make_client()
    msgs = convo()[:3]
    scores = client.score(msgs)
    assert scores == [
        [0.0, -1.0, -1.0],
        [-2.0, -2.0, -2.0],
        [-3.0, -3.0, -3.0],
    ]


def test_vllm_score_request_shape():
    client, calls = make_client()
    client.score([Message("user", "a"), Message("assistant", "b")])
    assert len(calls) == 2  # one request per message prefix
    for i, (path, payload) in enumerate(calls, 1):
        assert path == "/v1/chat/completions"
        assert len(payload["messages"]) == i
        assert payload["prompt_logprobs"] == 0
        assert payload["add_generation_prompt"] is False
        assert payload["temperature"] == 0.0


def test_vllm_load_adapter():
    client, calls = make_client()
    client.load_adapter(AdapterVersion(name="v0000", path=None, parent=None))
    assert calls == []  # base: no HTTP call
    v1 = AdapterVersion(name="v0001", path="/ckpt/v0001", parent="v0000")
    client.load_adapter(v1)
    assert calls == [("/v1/load_lora_adapter",
                      {"lora_name": "v0001", "lora_path": "/ckpt/v0001"})]
    # registered base-path version routes to the base model
    assert client._model_name("v0000") == "qwen-base"
    assert client._model_name("v0001") == "v0001"


def test_vllm_load_adapter_already_loaded_is_ok():
    client, _ = make_client()

    def failing_post(path: str, payload: dict) -> dict:
        raise RuntimeError("POST /v1/load_lora_adapter failed (400): already loaded")

    client._post = failing_post  # type: ignore[method-assign]
    client.load_adapter(AdapterVersion(name="v0002", path="/ckpt/v0002", parent="v0001"))

    def hard_fail(path: str, payload: dict) -> dict:
        raise RuntimeError("POST /v1/load_lora_adapter failed (500): boom")

    client._post = hard_fail  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="boom"):
        client.load_adapter(AdapterVersion(name="v0003", path="/ckpt/v0003", parent="v0002"))
