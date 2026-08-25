from sediment.chat_template import (
    NOTHINK_TEMPLATE,
    is_hybrid,
    override_template,
    template_kwargs,
)


def test_hybrid_bases_get_the_nothink_override():
    for m in ["Qwen/Qwen3-8B", "Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-30B-A3B"]:
        assert is_hybrid(m), m
        assert override_template(m) == NOTHINK_TEMPLATE.read_text(encoding="utf-8")
        assert template_kwargs(m) == {"enable_thinking": False}, m


def test_single_mode_bases_keep_their_shipped_template():
    # 2507 refreshes are Instruct-only / Thinking-only: no think branch to strip
    for m in ["Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-4B-Thinking-2507",
              "Qwen/Qwen2.5-7B-Instruct", "", None]:
        assert not is_hybrid(m), m
        assert override_template(m) is None, m
        assert template_kwargs(m) == {}, m


def test_nothink_template_has_no_thinking_machinery():
    src = NOTHINK_TEMPLATE.read_text(encoding="utf-8")
    assert "<think>" not in src and "enable_thinking" not in src
    assert "last_query_index" not in src  # the backward scan that breaks prefixes
    assert "<|im_start|>" in src  # still a Qwen chat template


def test_engine_derives_the_same_contract():
    from sediment.engine.vllm_client import VllmClient

    c = VllmClient("http://127.0.0.1:9/v1", "Qwen/Qwen3-8B")
    assert c.template_kwargs == {"enable_thinking": False}
    assert VllmClient("http://127.0.0.1:9/v1", "Qwen/Qwen3-4B-Instruct-2507").template_kwargs == {}


def test_self_initiated_think_is_stripped_and_counted():
    from sediment.engine.vllm_client import VllmClient

    c = VllmClient("http://127.0.0.1:9/v1", "Qwen/Qwen3-8B")
    assert c._strip_think('<think>\nhmm\n</think>\n\n{"tool":"a"}') == '{"tool":"a"}'
    assert c.n_think == 1 and c.n_think_unclosed == 0
    assert c._strip_think('<think>\nran out of budget') == ""  # never closed
    assert c.n_think == 2 and c.n_think_unclosed == 1
    assert c._strip_think('{"tool":"b"}') == '{"tool":"b"}'  # untouched
    assert c.n_think == 2
    # a <think> that is not at the start is content, not reasoning
    assert c._strip_think('ok <think>x</think>') == 'ok <think>x</think>'
    assert c.n_think == 2
