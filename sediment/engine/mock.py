"""MockEngine: zero heavy deps, deterministic, injectable.

Tokenization is whitespace over each message's content (`mock_tokenize`);
`score` returns one logprob list per message with exactly
`len(mock_tokenize(msg.content))` entries, which is the alignment
sediment.spans uses in tests. Adapter names may route to different injected
policies (`adapter_policies`) so gate A/B behavior is testable; unknown
adapters fall back to the base policy.
"""
from __future__ import annotations

import json
from typing import Callable, Optional

from sediment.types import AdapterVersion, Message

# Kept in sync with sediment.experience (resid injection format).
EXPERIENCE_OPEN = "<previous_attempts>"

# policy(messages) -> raw assistant text
Policy = Callable[[list[Message]], str]
# scorer(token, role, has_context, context_text) -> logprob
Scorer = Callable[[str, str, bool, str], float]


def mock_tokenize(text: str) -> list[str]:
    """Whitespace tokenizer shared with sediment.spans tests."""
    return text.split()


def _tool_call_text(name: str, arguments: dict) -> str:
    return "<tool_call> " + json.dumps({"name": name, "arguments": arguments}) + " </tool_call>"


def _first_user_content(messages: list[Message]) -> str:
    return next((m.content for m in messages if m.role == "user"), "")


def toy_order_policy(messages: list[Message]) -> str:
    """Default scripted policy for ToyOrderEnv.

    Student (no experience): cancels blindly, then claims success -> fails on
    shipped orders. Teacher (experience present): checks status first and
    answers correctly.
    """
    has_exp = EXPERIENCE_OPEN in _first_user_content(messages)
    n_assistant = sum(1 for m in messages if m.role == "assistant")
    last_tool = next((m.content for m in reversed(messages) if m.role == "tool"), "")

    if not has_exp:
        if n_assistant == 0:
            return _tool_call_text("cancel_order", {})
        return _tool_call_text("answer", {"text": "The order has been cancelled."})

    if n_assistant == 0:
        return _tool_call_text("get_order_status", {})
    if "pending" in last_tool:
        return _tool_call_text("cancel_order", {})
    if "cancelled" in last_tool:
        return _tool_call_text("answer", {"text": "The order has been cancelled."})
    return _tool_call_text(
        "answer", {"text": "The order cannot be cancelled because it has already shipped."}
    )


def default_scorer(token: str, role: str, has_context: bool, context_text: str) -> float:
    """Baseline logprob -1.0; tool-observation tokens that also appear in the
    injected experience block become more likely under context."""
    if has_context and role == "tool" and token in set(context_text.split()):
        return -0.4
    return -1.0


class MockEngine:
    def __init__(
        self,
        policy: Optional[Policy] = None,
        scorer: Optional[Scorer] = None,
        adapter_policies: Optional[dict[str, Policy]] = None,
    ):
        self._policy = policy or toy_order_policy
        self._scorer = scorer or default_scorer
        self.adapter_policies: dict[str, Policy] = dict(adapter_policies or {})
        self.adapters: dict[str, AdapterVersion] = {}

    def generate(
        self,
        messages: list[Message],
        *,
        adapter: str = "base",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        return self.adapter_policies.get(adapter, self._policy)(messages)

    def score(self, messages: list[Message], *, adapter: str = "base") -> list[list[float]]:
        ctx = _first_user_content(messages)
        has_ctx = EXPERIENCE_OPEN in ctx
        return [
            [self._scorer(tok, m.role, has_ctx, ctx) for tok in mock_tokenize(m.content)]
            for m in messages
        ]

    def load_adapter(self, version: AdapterVersion) -> None:
        self.adapters[version.name] = version
