"""Env protocol for the streaming agent loop.

Envs speak raw text: `reset(task)` builds the initial chat (tool schemas
inlined in the system prompt — engines take no tools argument) and
`step(action_text)` consumes one raw assistant turn, returning the
observation messages to append, a done flag, and the current reward
(meaningful at done). Task dicts carry {"task_id", "env_family", "payload"}.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional, Protocol, runtime_checkable

from sediment.types import Message

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_TOOL_CALL_OPEN_RE = re.compile(r"<tool_call>", re.IGNORECASE)


@runtime_checkable
class Env(Protocol):
    env_family: str

    def reset(self, task: dict[str, Any]) -> list[Message]:
        """Reset state for a task dict; returns the initial messages
        (system prompt with tools inlined + first user message)."""
        ...

    def step(self, action_text: str) -> tuple[list[Message], bool, float]:
        """Execute one raw assistant turn; returns (observation messages,
        done, reward). Observation role is "tool" for genuine tool results
        and "user" for parse-error / invalid-action observations that
        re-enter the conversation as plain user messages."""
        ...

    def score_current_state(self) -> float:
        """Score the current state without mutating or terminating the env."""
        ...


def parse_tool_call(text: str) -> Optional[dict[str, Any]]:
    """Extract the first <tool_call>{json}</tool_call> from raw assistant text.

    Returns {"name": str, "arguments": dict} — with name "" when the payload
    inside the tags is not valid JSON — or None when no tool-call tag is
    present (a plain-text reply, treated by envs as the final answer).
    """
    m = _TOOL_CALL_RE.search(text)
    if m is None:
        return None
    try:
        call = json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        call = None
    if not isinstance(call, dict):
        return {"name": "", "arguments": {}}
    args = call.get("arguments")
    return {
        "name": str(call.get("name") or ""),
        "arguments": args if isinstance(args, dict) else {},
    }


def tool_call_count(text: str) -> int:
    """Count emitted ``<tool_call>`` openings, including malformed blocks."""
    return len(_TOOL_CALL_OPEN_RE.findall(text))
