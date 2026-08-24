"""Chat-template handling per base model. Engine and trainer MUST agree.

Qwen3's hybrid-reasoning checkpoints (Qwen3-8B, Qwen3-14B, ...) ship a template
that is unusable for us, for a reason unrelated to whether thinking is on:

  it computes ``last_query_index`` by scanning the WHOLE message list backwards,
  then injects ``<think>...</think>`` into assistant turns after that index.
  Appending a message moves the index, so a think block that was rendered for
  message k disappears once k is no longer last -- render(msgs[:k+1]) is then
  NOT a token prefix of render(msgs[:k+2]).

Our per-token weights (relu(delta), hindsight w_t) are aligned by rendering
message prefixes incrementally, so a broken prefix property misplaces every
weight (``trainer._encode`` and ``VllmClient.score`` both hard-fail on it).
Measured on Qwen3-8B: breaks at the first assistant->user transition, with
enable_thinking both False and True.

Fix: hybrid checkpoints render with ``qwen3_nothink.jinja`` -- the
Qwen3-4B-Instruct-2507 template, which is byte-identical apart from the
thinking machinery (verified by diff: only the backward scan, the assistant
think-injection branch, and the generation-prompt think block differ). It has
the prefix property and emits no think block anywhere, so "thinking off" and
"weights align" are the same change. Both sides load it from this one file.

``template_kwargs`` stays as a fallback for ``generate()`` alone: if a vLLM
server was started without ``--chat-template``, enable_thinking=False at least
keeps the generation budget from being eaten by reasoning. It cannot fix
scoring -- that path asserts instead of degrading silently.
"""
from __future__ import annotations

import re
from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
NOTHINK_TEMPLATE = TEMPLATE_DIR / "qwen3_nothink.jinja"

# hybrid checkpoints: bare "Qwen3-<size>" with no mode suffix
_HYBRID = re.compile(r"^Qwen3-(0\.6B|1\.7B|4B|8B|14B|32B|30B-A3B|235B-A22B)$")


def _base_name(model: str) -> str:
    return str(model or "").rstrip("/").rsplit("/", 1)[-1]


def is_hybrid(model: str) -> bool:
    """True if this base ships the thinking template (needs the override)."""
    return bool(_HYBRID.match(_base_name(model)))


def override_template(model: str) -> str | None:
    """Jinja source both sides must render with, or None to use the shipped one."""
    if not is_hybrid(model):
        return None
    return NOTHINK_TEMPLATE.read_text(encoding="utf-8")


def template_kwargs(model: str) -> dict:
    """Fallback kwargs for generate() when the server lacks the override."""
    return {"enable_thinking": False} if is_hybrid(model) else {}


def load_tokenizer(model: str, **kwargs):
    """AutoTokenizer with the override applied. Use this everywhere, so the
    tokenizer object itself carries the contract instead of every call site."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, **kwargs)
    tmpl = override_template(model)
    if tmpl:
        tok.chat_template = tmpl
    return tok
