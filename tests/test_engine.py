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
    assert "seed" not in payload


def test_vllm_generation_never_adds_a_request_seed():
    client, calls = make_client()
    client.generate([Message("user", "solve query")], max_tokens=16)
    assert "seed" not in calls[-1][1]


def test_vllm_score_per_message_slices():
    client, _ = make_client()
    msgs = convo()[:3]
    scores = client.score(msgs)
    assert scores == [
        [0.0, -1.0, -1.0],
        [-2.0, -2.0, -2.0],
        [-3.0, -3.0, -3.0],
    ]


class FakeTok:
    """Chat template that agrees with the faked server: TOKENS_PER_MSG each."""

    def apply_chat_template(self, msgs, tokenize=True, return_dict=False):
        return list(range(len(msgs) * TOKENS_PER_MSG))


def test_vllm_score_request_shape():
    """No local tokenizer -> one request for the whole rendering, then the
    per-prefix fallback (the alignment could not be verified)."""
    client, calls = make_client()
    client.score([Message("user", "a"), Message("assistant", "b")])
    assert len(calls) == 1 + 2
    assert len(calls[0][1]["messages"]) == 2  # fast path tried first
    for path, payload in calls:
        assert path == "/v1/chat/completions"
        assert payload["prompt_logprobs"] == 0
        assert payload["add_generation_prompt"] is False
        assert payload["temperature"] == 0.0


def test_vllm_score_single_request_when_aligned():
    client, calls = make_client()
    client._tok = FakeTok()
    msgs = convo()[:3]
    assert client.score(msgs) == [
        [0.0, -1.0, -1.0], [-2.0, -2.0, -2.0], [-3.0, -3.0, -3.0]]
    assert len(calls) == 1  # the whole trajectory in one call


def test_vllm_score_topk():
    client, calls = make_client()
    client._tok = FakeTok()
    ids, dists = client.score_topk(convo()[:2], k=4)
    assert [len(x) for x in ids] == [TOKENS_PER_MSG, TOKENS_PER_MSG]
    assert ids[1] == [3, 4, 5]  # local token ids, per message
    assert calls[0][1]["prompt_logprobs"] == 4
    assert dists[0][0] == {}  # position 0 carries no logprob
    assert dists[1][0] == {7: -2.0}


def test_vllm_score_topk_refuses_misaligned():
    client, _ = make_client()  # no local tokenizer -> cannot verify
    with pytest.raises(RuntimeError, match="cannot align top-k teacher"):
        client.score_topk(convo()[:2], k=4)


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


def test_lora_prefix_namespaces_served_adapters(monkeypatch):
    from sediment.types import AdapterVersion
    client = VllmClient("http://localhost:8000/", "qwen-base", lora_prefix="runA-")
    posted = []
    monkeypatch.setattr(client, "_post", lambda path, body: posted.append((path, body)))
    client.load_adapter(AdapterVersion("v0001", "/tmp/a", "v0000"))
    assert posted == [("/v1/load_lora_adapter", {"lora_name": "runA-v0001", "lora_path": "/tmp/a"})]
    assert client._model_name("v0001") == "runA-v0001"
    assert client._model_name("base") == "qwen-base"
    client.unload_adapter("v0001")
    assert posted[-1] == ("/v1/unload_lora_adapter", {"lora_name": "runA-v0001"})


def test_post_retries_transient_timeouts(monkeypatch):
    import urllib.request
    client = VllmClient("http://localhost:8000/", "m")
    calls = {"n": 0}

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"ok": true}'

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("timed out")
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("sediment.engine.vllm_client.time.sleep", lambda s: None)
    assert client._post("/v1/x", {}) == {"ok": True}
    assert calls["n"] == 3


def test_post_retries_5xx_but_not_4xx(monkeypatch):
    import io
    import urllib.error
    import urllib.request
    client = VllmClient("http://localhost:8000/", "m")
    monkeypatch.setattr("sediment.engine.vllm_client.time.sleep", lambda s: None)
    codes = iter([500, 503])

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"ok": true}'

    def fake(req, timeout=None):
        try:
            code = next(codes)
        except StopIteration:
            return Resp()
        raise urllib.error.HTTPError(req.full_url, code, "err", {}, io.BytesIO(b"x"))

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    assert client._post("/v1/x", {}) == {"ok": True}

    def fake404(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, io.BytesIO(b"x"))
    monkeypatch.setattr(urllib.request, "urlopen", fake404)
    import pytest
    with pytest.raises(RuntimeError):
        client._post("/v1/x", {})


def test_vllm_external_adapter_keeps_its_exact_name():
    """Explicitly external adapters do not take a run-scoped prefix."""
    client, calls = make_client()
    client.lora_prefix = "run-a-"
    client.load_external("shared_adapter", "/data/adapters/shared_adapter")
    assert calls == [("/v1/load_lora_adapter",
                      {"lora_name": "shared_adapter",
                       "lora_path": "/data/adapters/shared_adapter"})]
    assert client._model_name("shared_adapter") == "shared_adapter"
    assert client._model_name("v0003") == "run-a-v0003"
