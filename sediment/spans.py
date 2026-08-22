"""Token-position -> message-role mapping via incremental chat-template
tokenization, plus suffix alignment for with/without-context comparisons.

Parameterized by a tokenizer callable ``tokenize(text) -> list`` (e.g.
``sediment.engine.mock.mock_tokenize``) so the same machinery serves the
whitespace mock tokenizer and real HF tokenizers alike.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

Tokenize = Callable[[str], list]


@dataclass(frozen=True)
class MsgSpan:
    index: int  # message index in the conversation
    role: str
    start: int  # token start (inclusive) in the full rendering
    end: int  # token end (exclusive)

    @property
    def length(self) -> int:
        return self.end - self.start


def _role(m) -> str:
    return m["role"] if isinstance(m, dict) else m.role


def _content(m) -> str:
    return (m["content"] if isinstance(m, dict) else m.content) or ""


def render_messages(messages: Sequence) -> str:
    """Canonical mock chat rendering: one ``<role> content`` line per message.

    Whitespace-tokenizing this rendering preserves the token prefix property
    required by :func:`message_spans`.
    """
    return "\n".join(f"<{_role(m)}> {_content(m)}".rstrip() for m in messages) + "\n"


def message_spans(
    tokenize: Tokenize,
    messages: Sequence,
    render: Callable[[Sequence], str] = render_messages,
) -> list[MsgSpan]:
    """Per-message token spans of the full rendering.

    Tokenizes the rendered prefix after each message; span = length diff.
    Asserts the prefix property (each prefix tokenization is a token-level
    prefix of the next) so the final total provably matches the full render.
    """
    prev: list = []
    spans: list[MsgSpan] = []
    for i in range(1, len(messages) + 1):
        toks = tokenize(render(messages[:i]))
        if toks[: len(prev)] != prev:
            raise AssertionError(
                f"chat template broke the token prefix property at message {i - 1} "
                f"(role={_role(messages[i - 1])}); span mapping would be invalid"
            )
        spans.append(MsgSpan(index=i - 1, role=_role(messages[i - 1]), start=len(prev), end=len(toks)))
        prev = toks
    return spans


def spans_from_lengths(messages: Sequence, lengths: Sequence[int]) -> list[MsgSpan]:
    """MsgSpans from per-message token counts (e.g. ``Engine.score`` output)."""
    if len(messages) != len(lengths):
        raise ValueError(f"messages/lengths mismatch: {len(messages)} vs {len(lengths)}")
    spans: list[MsgSpan] = []
    pos = 0
    for i, (m, n) in enumerate(zip(messages, lengths)):
        spans.append(MsgSpan(index=i, role=_role(m), start=pos, end=pos + n))
        pos += n
    return spans


def align_suffix(
    spans_with: list[MsgSpan],
    spans_without: list[MsgSpan],
    injected_index: int = 1,
) -> list[tuple[MsgSpan, MsgSpan]]:
    """Pair up spans of the messages AFTER the injected block.

    The with-context rendering has extra prefix tokens (the experience block
    inside message ``injected_index``); all later messages are textually
    identical, so their spans must match role and length. Returns
    [(span_with, span_without), ...] for messages with index > injected_index.
    """
    a = [s for s in spans_with if s.index > injected_index]
    b = [s for s in spans_without if s.index > injected_index]
    if len(a) != len(b):
        raise ValueError(f"suffix message count mismatch: {len(a)} vs {len(b)}")
    pairs: list[tuple[MsgSpan, MsgSpan]] = []
    for sw, so in zip(a, b):
        if sw.role != so.role:
            raise ValueError(f"role mismatch at message {sw.index}: {sw.role} vs {so.role}")
        if sw.length != so.length:
            raise ValueError(
                f"token length mismatch at message {sw.index} (role={sw.role}): "
                f"{sw.length} vs {so.length}; cannot align suffix"
            )
        pairs.append((sw, so))
    return pairs
