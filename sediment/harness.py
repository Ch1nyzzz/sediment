"""Harness-side context edits shared by rollout and storage.

Three tag families mark text the HARNESS (neither the model nor the env) put
into a conversation:

  <previous_attempts>…</previous_attempts>   retrieved block, first user msg
  <experience_hint>…</experience_hint>       call-time memory snippet (calltime)
  <working_state>…</working_state>           verified task state (working_state)

An intercepted draft (Recuris call-time invocation) is an assistant turn that
was never executed, followed by a user message that STARTS with
``<experience_hint intercepted>``; both are harness turns. ``bare_view`` removes
all of it and yields the prompt the model would have seen with no harness.
Stored student views and analysis scripts work on that view. Generation is
independently sampled; stripping harness text does not imply RNG pairing.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from sediment.envs.base import parse_tool_call
from sediment.types import Message

EXP_OPEN, EXP_CLOSE = "<previous_attempts>", "</previous_attempts>"
HINT_OPEN, HINT_CLOSE = "<experience_hint>", "</experience_hint>"
HINT_INTERCEPT_OPEN = "<experience_hint intercepted>"
STATE_OPEN, STATE_CLOSE = "<working_state>", "</working_state>"

_HINT_RE = re.compile(r"\s*<experience_hint(?: [^>]*)?>.*?</experience_hint>\s*", re.DOTALL)
_STATE_RE = re.compile(r"\s*<working_state>.*?</working_state>\s*", re.DOTALL)
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Tool-name verbs that only read state. Everything else is treated as a
# modification ("write") tool: the EnvScaler system prompt itself frames tools
# as "query tools" vs "modification tools", and names follow verb_object.
READ_VERBS = frozenset({
    "get", "list", "check", "search", "find", "view", "query", "fetch", "show",
    "retrieve", "lookup", "count", "verify", "validate", "calculate", "compute",
    "is", "has", "describe", "read", "filter", "compare", "summarize", "preview",
    "estimate", "export", "print", "inspect", "lists",
    # terminal / conversational pseudo-tools are not modifications either
    "answer", "chat", "reply", "respond",
})
STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "into", "all", "any",
    "are", "not", "but", "its", "has", "have", "been", "was", "were", "will",
    "then", "than", "only", "also", "each", "must", "should", "true", "false",
    "none", "null", "success", "data", "error", "step", "tool", "call",
})


def tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 3 and t not in STOPWORDS}


def tool_verb(name: str) -> str:
    return name.split("_", 1)[0].lower() if name else ""


def is_write_tool(name: str) -> bool:
    """Modification tool by verb; unknown / empty names are not writes."""
    return bool(name) and tool_verb(name) not in READ_VERBS


def is_error_obs(text: str) -> bool:
    """Env observation reporting failure (LOPD 'Error: …' or {'success': False})."""
    head = text.lstrip()[:400]
    return head.startswith("Error") or "'success': False" in head or '"success": false' in head.lower()


def first_call(action_text: str) -> Optional[dict[str, Any]]:
    """First <tool_call> of an assistant turn (the only one the env executes)."""
    return parse_tool_call(action_text)


def strip_harness(text: str) -> str:
    """Remove hint / working-state segments inside one message."""
    if HINT_OPEN[:-1] in text:
        text = _HINT_RE.sub("\n", text)
    if STATE_OPEN in text:
        text = _STATE_RE.sub("\n", text)
    return text.strip("\n") if text.strip() else ""


def drop_state(messages: list[Message]) -> None:
    """In place: remove working-state segments from every message, so that a
    freshly rendered state on the newest observation is the only one in
    context (the model reads one current state, not a history of them)."""
    for i, m in enumerate(messages):
        if m.role in ("user", "tool") and STATE_OPEN in m.content:
            messages[i] = Message(m.role, _STATE_RE.sub("\n", m.content).strip("\n"))


def bare_view(messages: list[Message]) -> list[Message]:
    """Messages as the model would have seen them with no harness at all.

    Drops the retrieved block from the first user message, drops intercepted
    (draft, hint) pairs entirely, and strips inline hint / state segments.
    Idempotent; a list without harness content comes back equal.
    """
    out: list[Message] = []
    stripped_block = False
    for m in messages:
        content = m.content
        if m.role == "user" and content.startswith(HINT_INTERCEPT_OPEN):
            if out and out[-1].role == "assistant":
                out.pop()  # the draft that was never executed
            continue
        if (not stripped_block and m.role == "user"
                and content.startswith(EXP_OPEN) and EXP_CLOSE in content):
            content = content.split(EXP_CLOSE, 1)[1].lstrip("\n")
            stripped_block = True
        if m.role in ("user", "tool"):
            content = strip_harness(content)
        out.append(Message(m.role, content))
    return out


def has_harness(messages: list[Message]) -> bool:
    return any(HINT_OPEN[:-1] in m.content or STATE_OPEN in m.content for m in messages)
