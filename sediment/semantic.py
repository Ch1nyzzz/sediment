"""Semantic-token mask for negative (unlikelihood) action credit.

Ported from scripts/signed_action_worker.action_semantic_mask: a failed tool
call usually has the right function and framing but a bad VALUE, so penalising
`<tool_call>`, JSON keys, or the tool name generalises the error in the wrong
direction (the 08-25 no-floor signed arm collapsed into narrating without
ever calling a tool). Only value/content tokens stay eligible for -δ.
"""
from __future__ import annotations

import re

SHARED_TEMPLATE_WORDS = {"name", "arguments", "tool_call", "json"}
WORD_RE = re.compile(r"[A-Za-z0-9]")
JSON_KEY_RE = re.compile(r'["\u0027]([^"\u0027]+)["\u0027]\s*:')
TOOL_NAME_RE = re.compile(r'["\u0027]name["\u0027]\s*:\s*["\u0027]([^"\u0027]+)["\u0027]', re.I)


def _norm(piece: str) -> str:
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", piece.strip()).lower()


def is_semantic_token(piece: str) -> bool:
    stripped = piece.strip()
    if not stripped or stripped in {"<tool_call>", "</tool_call>", "```"}:
        return False
    normalized = stripped.strip("\"'`{}[]():,.").lower()
    if normalized in SHARED_TEMPLATE_WORDS:
        return False
    return bool(WORD_RE.search(normalized))


def action_semantic_mask(tokenizer, action: str, token_ids: list[int]) -> list[bool]:
    """True for tokens that may carry negative credit (values/content only)."""
    excluded = set(SHARED_TEMPLATE_WORDS)
    fields = JSON_KEY_RE.findall(action)
    m = TOOL_NAME_RE.search(action)
    if m:
        fields.append(m.group(1))
    for field in fields:
        excluded.update(p.lower() for p in re.split(r"[^A-Za-z0-9]+", field) if p)
        for variant in (field, f" {field}", f'"{field}"'):
            for tid in tokenizer.encode(variant, add_special_tokens=False):
                n = _norm(tokenizer.decode([tid]))
                if n:
                    excluded.add(n)
    out = []
    for tid in token_ids:
        piece = tokenizer.decode([tid])
        out.append(is_semantic_token(piece) and _norm(piece) not in excluded)
    return out
