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
# chat-template framing: deterministic given the template, and the spot where
# prompt-logprob artefacts land (08-25: |δ| up to 20-28 on `assistant`/`<|im_end|>`)
FRAMING_TOKENS = {"<tool_call>", "</tool_call>", "```", "<|im_start|>", "<|im_end|>",
                  "assistant", "user", "system"}
WORD_RE = re.compile(r"[A-Za-z0-9]")
JSON_KEY_RE = re.compile(r'["\u0027]([^"\u0027]+)["\u0027]\s*:')
TOOL_NAME_RE = re.compile(r'["\u0027]name["\u0027]\s*:\s*["\u0027]([^"\u0027]+)["\u0027]', re.I)


def _norm(piece: str) -> str:
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", piece.strip()).lower()


def is_semantic_token(piece: str) -> bool:
    stripped = piece.strip()
    if not stripped or stripped in FRAMING_TOKENS:
        return False
    normalized = stripped.strip("\"'`{}[]():,.").lower()
    if normalized in SHARED_TEMPLATE_WORDS:
        return False
    return bool(WORD_RE.search(normalized))


def framing_mask(tokenizer, token_ids: list[int]) -> list[bool]:
    """True for non-framing tokens (eligible for ANY credit, + or -)."""
    return [tokenizer.decode([tid]).strip() not in FRAMING_TOKENS and bool(tokenizer.decode([tid]).strip())
            for tid in token_ids]


def action_semantic_mask(tokenizer, action: str, token_ids: list[int]) -> list[bool]:
    """True for tokens that may carry negative credit: values, content AND the
    tool name (08-25 delta_tokens: half of the strongest -δ sit on the tool
    name -- "don't call this tool here" is exactly what the evidence knows);
    JSON keys and framing stay protected."""
    excluded = set(SHARED_TEMPLATE_WORDS)
    fields = JSON_KEY_RE.findall(action)
    m = TOOL_NAME_RE.search(action)
    if m:  # "name" is a key -> excluded; its VALUE (the tool) stays eligible
        fields = [f for f in fields if f != m.group(1)]
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


def tool_call_region_mask(tokenizer, content: str, token_ids: list[int]) -> list[bool]:
    """True for tokens INSIDE a <tool_call>...</tool_call> span. Positive credit is
    confined to this region (08-25 collapses: the +gate reinforced "Task Completed"
    -> premature termination, and reflection echoes ("I will now proceed...") ->
    narration instead of action). Spans are located on the concatenation of the
    decoded pieces themselves, so offsets stay consistent even though the token
    span carries chat-template framing that `content` does not."""
    pieces = [tokenizer.decode([tid]) for tid in token_ids]
    text = "".join(pieces)
    spans = []
    pos = 0
    while True:
        o = text.find("<tool_call>", pos)
        if o < 0:
            break
        c = text.find("</tool_call>", o)
        spans.append((o + len("<tool_call>"), c if c >= 0 else len(text)))
        pos = (c + len("</tool_call>")) if c >= 0 else len(text)
    out = []
    off = 0
    for piece in pieces:
        start = off
        off += len(piece)
        out.append(any(a <= start < b for a, b in spans))
    return out


def grounded_value_mask(tokenizer, content: str, token_ids: list[int],
                        context: str) -> list[bool]:
    """True for tokens inside a tool-call ARGUMENT VALUE that is copyable verbatim
    from `context` (the conversation so far).

    Forensics 08-26: two thirds of the SFT gradient on memory-conditioned
    trajectories lands on such tokens, and the measured cost of strengthening
    that copy channel is the anti-repetition prior -- 50 merges moved
    logp(re-issue my previous call) by +15 nats. Downweighting (not zeroing:
    copying ids out of context is a real capability) keeps the channel without
    letting it dominate the update.
    """
    values = re.findall(r'"[^"]*"\s*:\s*"([^"]+)"', content)
    values += [m for m in re.findall(r'"[^"]*"\s*:\s*([0-9][0-9.\-]*)', content)]
    hits = [v for v in values if v and v in context]
    if not hits:
        return [False] * len(token_ids)
    pieces = [tokenizer.decode([tid]) for tid in token_ids]
    text = "".join(pieces)
    spans = []
    for v in hits:
        pos = 0
        while True:
            o = text.find(v, pos)
            if o < 0:
                break
            spans.append((o, o + len(v)))
            pos = o + len(v)
    out, off = [], 0
    for piece in pieces:
        start = off
        off += len(piece)
        out.append(any(a <= start < b for a, b in spans))
    return out
