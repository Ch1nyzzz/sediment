"""ToyOrderEnv: minimal multi-turn tool env with a hidden rule, used by tests.

Hidden rule: cancel_order fails with "Error: cannot cancel <status> order"
unless the order status is "pending". Reward 1.0 iff the final answer is
correct for the actual outcome.
"""
from __future__ import annotations

import random
from typing import Any, Optional

from sediment.envs.base import parse_tool_call, tool_call_count
from sediment.types import Message

_STATUSES = ("pending", "shipped", "delivered")

_SYSTEM_PROMPT = """You are a customer-service agent. Use the tools to handle the request, then finish with the answer tool. Call exactly one tool per turn as:
<tool_call>
{"name": "<tool_name>", "arguments": {...}}
</tool_call>
Tools:
- get_order_status(): get the current status of the order.
- cancel_order(): attempt to cancel the order.
- answer(text: str): give the final answer to the customer.
A reply without a tool call is treated as your final answer."""


def make_toy_tasks(n: int, seed: int = 0) -> list[dict[str, Any]]:
    """n seeded toy task dicts with keys task_id / env_family / payload."""
    rng = random.Random(seed)
    return [
        {
            "task_id": f"toy-{i:04d}",
            "env_family": "toy_order",
            "payload": {"status": rng.choice(_STATUSES), "order_id": 100 + i},
        }
        for i in range(n)
    ]


class ToyOrderEnv:
    env_family = "toy_order"

    def __init__(self) -> None:
        self.task_id = ""
        self.initial_status = "shipped"
        self.status = self.initial_status
        self.order_id = 42
        self.answer_text: Optional[str] = None

    def reset(self, task: dict[str, Any]) -> list[Message]:
        payload = task.get("payload") or {}
        self.task_id = str(task.get("task_id", "toy-0"))
        self.initial_status = str(payload.get("status", "shipped"))
        self.status = self.initial_status
        self.order_id = int(payload.get("order_id", 42))
        self.answer_text = None
        task_text = (
            f"Please cancel order #{self.order_id}. "
            "If it cannot be cancelled, explain why."
        )
        return [Message("system", _SYSTEM_PROMPT), Message("user", task_text)]

    def step(self, action_text: str) -> tuple[list[Message], bool, float]:
        # All genuine observations here are tool results, hence role "tool".
        n_calls = tool_call_count(action_text)
        if n_calls > 1:
            return [Message(
                "tool",
                f"Error: exactly one tool call is allowed per turn; received {n_calls}.",
            )], False, 0.0
        call = parse_tool_call(action_text)
        if call is None:  # plain-text reply = final answer
            return self._finish(action_text)
        name, args = call["name"], call["arguments"]
        if not name:
            return [Message("tool", "Error: could not parse tool call")], False, 0.0
        if name == "get_order_status":
            return [Message("tool", f"Order status: {self.status}")], False, 0.0
        if name == "cancel_order":
            if self.status == "pending":
                self.status = "cancelled"
                return [Message("tool", "Order cancelled.")], False, 0.0
            return [Message("tool", f"Error: cannot cancel {self.status} order")], False, 0.0
        if name in ("answer", "chat_with_user"):
            return self._finish(str(args.get("text") or args.get("content") or ""))
        return [Message("tool", f"Error: unknown tool {name}")], False, 0.0

    def score_current_state(self) -> float:
        """Grade the current toy state without inventing a final answer."""
        return self._grade()

    def _finish(self, answer: str) -> tuple[list[Message], bool, float]:
        self.answer_text = answer
        return [Message("tool", "Answer recorded.")], True, self._grade()

    def _grade(self) -> float:
        answer = (self.answer_text or "").lower()
        if self.initial_status == "pending":
            return 1.0 if self.status == "cancelled" and "cancelled" in answer else 0.0
        # A non-pending order can never be cancelled: correct answer says so.
        return 1.0 if self.status == self.initial_status and "cannot" in answer else 0.0
