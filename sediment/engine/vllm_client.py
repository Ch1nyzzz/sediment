"""VllmClient: OpenAI-compatible HTTP client for a vLLM server (no vllm import).

Generation uses /v1/chat/completions with `model=<adapter name or base>`.
Teacher-forced scoring uses vLLM's `prompt_logprobs` extra body with
`add_generation_prompt=False`. Message boundaries come from the LOCAL chat
template (free, no HTTP) and are verified against the server's own token
count, so the whole trajectory is scored in ONE request; on any mismatch we
fall back to the historical one-request-per-message-prefix loop (the
chat-template prefix property, same approach as resid's vllm_engine). The
loop was the dominant streaming cost (60 s mean per trajectory vs 23 s to
generate it). `score_topk` additionally returns the per-position top-k
distribution, the teacher target of context distillation (cfg.kl_target).
LoRA adapters are registered at runtime via POST /v1/load_lora_adapter
(server must allow runtime LoRA updating).
"""
from __future__ import annotations

import hashlib
import json
import socket
import time
import re
import urllib.error
import urllib.request
from typing import Optional

from sediment.chat_template import template_kwargs
from sediment.types import AdapterVersion, Message

# Documented launch command (out of scope to run); runtime LoRA loading
# additionally requires VLLM_ALLOW_RUNTIME_LORA_UPDATING=True.
SERVER_CMD = (
    "VLLM_ALLOW_RUNTIME_LORA_UPDATING=True vllm serve Qwen/Qwen3-4B-Instruct-2507 "
    "--host 0.0.0.0 --port 8000 "
    "--enable-lora --max-lora-rank 32 --max-loras 8 --enable-prefix-caching "
    "--max-model-len 12288 --gpu-memory-utilization 0.85"
)

_BASE_NAMES = ("base", "v0000")
_CTX_OVERFLOW_RE = re.compile(r"contains at least (\d+) input tokens")
# We render hybrid bases with the no-think template, whose generation prompt
# ends at "<|im_start|>assistant\n" -- nothing forecloses thinking, so the model
# may still open a <think> block on its own. Strip it: the env must see the tool
# call, and the stripped text is what gets stored, scored and trained on (so
# engine and trainer stay consistent). self.n_think counts occurrences -- a high
# rate means switching to a generation prompt that carries the empty block.
_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_THINK_OPEN = re.compile(r"^\s*<think>", re.DOTALL)
# vLLM caps prompt_logprobs at --max-logprobs (default 20); rather than kill a
# 12-hour stream at the first window, clamp once to whatever it reports.
_MAX_LOGPROBS_RE = re.compile(r"greater than max allowed: (\d+)")


def _without_experience(messages: list[Message]) -> list[dict[str, str]]:
    """Canonical prompt view shared by memory/no-memory paired generations:
    no retrieved block, no call-time hints, no working state (harness.bare_view)."""
    from sediment.harness import bare_view

    return [m.to_dict() for m in bare_view(messages)]


def _bare_prompt_seed(messages: list[Message], salt: int = 0) -> int:
    payload = json.dumps(
        _without_experience(messages), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if salt:
        payload += b"\x00salt=" + str(salt).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


class VllmClient:
    def __init__(
        self, base_url: str, model: str, *, timeout: float = 600.0, max_context: int = 12288,
        lora_prefix: str = "", generation_seed_mode: str = "none",
        generation_seed_salt: int = 0,
    ):
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/v1"):  # paths below carry /v1 already
            self.base_url = self.base_url[: -len("/v1")].rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_context = max_context
        if generation_seed_mode not in ("none", "bare_prompt_hash"):
            raise ValueError(f"unknown generation seed mode: {generation_seed_mode!r}")
        self.generation_seed_mode = generation_seed_mode
        self.generation_seed_salt = generation_seed_salt
        # hybrid-reasoning bases need enable_thinking=False; the trainer derives
        # the same kwargs from its tokenizer so both renderings agree
        self.template_kwargs = template_kwargs(model)
        self.n_think = 0  # generations that opened a <think> block anyway
        self.n_think_unclosed = 0  # ... and spent the whole budget inside it
        self._adapters: dict[str, AdapterVersion] = {}
        # served LoRA name = prefix + version name, so concurrent runs sharing
        # one server (each publishing v0001, v0002, ...) never collide
        self.lora_prefix = lora_prefix
        # Explicitly external adapters are served under their exact name rather
        # than the run-scoped prefix used for session versions.
        self._external: set[str] = set()
        self.max_retries = 3

    # -- http -------------------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # transient failures (server queue timeouts under shared load, brief
        # restarts) are retried with backoff; HTTP errors are surfaced at once
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")
                # 5xx = engine crashed / restarting (watchdog brings it back):
                # wait it out like a timeout; 4xx is a real request error
                if e.code >= 500 and attempt < self.max_retries:
                    time.sleep(min(60 * (attempt + 1), 180))
                    continue
                raise RuntimeError(f"POST {path} failed ({e.code}): {detail[:500]}") from e
            except (TimeoutError, socket.timeout, urllib.error.URLError, ConnectionError):
                if attempt >= self.max_retries:
                    raise
                time.sleep(min(60 * (attempt + 1), 180))
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"raw": body}

    def _model_name(self, adapter: str) -> str:
        if adapter in _BASE_NAMES:
            return self.model
        version = self._adapters.get(adapter)
        if version is not None and version.path is None:
            return self.model
        if adapter in self._external:
            return adapter
        return self.lora_prefix + adapter  # served LoRA name

    # -- Engine protocol ---------------------------------------------------
    def generate(
        self,
        messages: list[Message],
        *,
        adapter: str = "base",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        payload = {
            "model": self._model_name(adapter),
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.template_kwargs:
            payload["chat_template_kwargs"] = self.template_kwargs
        if self.generation_seed_mode == "bare_prompt_hash":
            payload["seed"] = _bare_prompt_seed(messages, self.generation_seed_salt)
        try:
            data = self._post("/v1/chat/completions", payload)
        except RuntimeError as e:
            # vLLM's reported input_tokens is a LOWER bound ("at least N"):
            # retry with an escalating safety margin, then yield the turn.
            data = None
            margin = 64
            for _ in range(3):
                m = _CTX_OVERFLOW_RE.search(str(e))
                if m is None or "maximum context length" not in str(e):
                    raise
                budget = self.max_context - int(m.group(1)) - margin
                if budget < 32:  # context exhausted: end the turn, env settles
                    return ""
                try:
                    data = self._post(
                        "/v1/chat/completions", {**payload, "max_tokens": budget}
                    )
                    break
                except RuntimeError as e2:
                    e = e2
                    margin *= 4
            if data is None:
                return ""
        return self._strip_think(data["choices"][0]["message"]["content"] or "")

    def _strip_think(self, text: str) -> str:
        """Drop a leading self-initiated <think>...</think> segment (see above)."""
        if not _THINK_OPEN.match(text):
            return text
        self.n_think += 1
        if "</think>" not in text:  # never closed: the whole budget went to it
            self.n_think_unclosed += 1
            return ""  # yield the turn, the env settles it
        return _THINK_RE.sub("", text, count=1)

    def _prompt_logprobs(self, messages: list[Message], adapter: str,
                         topk: int = 0) -> list[dict[int, float]]:
        """Per-position {token_id: logprob} over the rendered prompt.

        topk=0 gives the actual token only; topk=k adds the k most likely
        alternatives at that position. Position 0 has no logprob ({}).
        """
        payload = {
            "model": self._model_name(adapter),
            "messages": [m.to_dict() for m in messages],
            "temperature": 0.0,
            "max_tokens": 1,
            "prompt_logprobs": topk,  # vLLM extra body
            "add_generation_prompt": False,  # score exactly the rendering
        }
        if self.template_kwargs:
            payload["chat_template_kwargs"] = self.template_kwargs
        try:
            data = self._post("/v1/chat/completions", payload)
        except RuntimeError as e:
            m = _MAX_LOGPROBS_RE.search(str(e))
            if m is None or topk == 0:
                raise
            payload["prompt_logprobs"] = int(m.group(1))
            print(f"[vllm] prompt_logprobs {topk} -> {m.group(1)} (server --max-logprobs)",
                  flush=True)
            data = self._post("/v1/chat/completions", payload)
        out: list[dict[int, float]] = []
        for entry in data.get("prompt_logprobs") or []:
            out.append({int(t): float(v["logprob"]) for t, v in (entry or {}).items()})
        return out

    @staticmethod
    def _actual(entry: dict[int, float]) -> float:
        """The scored token's logprob. Only valid for topk=0 responses, where
        vLLM returns exactly one entry; with topk>0 index by the token id."""
        if not entry:
            return 0.0  # position 0
        vals = list(entry.values())
        return vals[0] if len(vals) == 1 else min(vals)

    def _tokenizer(self):
        tok = getattr(self, "_tok", None)
        if tok is None:
            from sediment.chat_template import load_tokenizer

            tok = self._tok = load_tokenizer(self.model)
        return tok

    def _message_token_ids(self, messages: list[Message]) -> list[list[int]]:
        """Per-message token ids from the LOCAL chat template (unverified)."""
        tok = self._tokenizer()
        dicts = [m.to_dict() for m in messages]
        prev: list[int] = []
        out: list[list[int]] = []
        for i in range(1, len(dicts) + 1):
            toks = list(tok.apply_chat_template(dicts[:i], tokenize=True, return_dict=False))
            out.append(toks[len(prev):])
            prev = toks
        return out

    def _local_ids(self, messages: list[Message], n_scored: int) -> Optional[list[list[int]]]:
        """Local per-message token ids, or None if they disagree with what the
        server actually scored (different template revision, tokenizer absent)."""
        try:
            ids = self._message_token_ids(messages)
        except Exception:
            return None
        return ids if sum(len(x) for x in ids) == n_scored else None

    def _score_by_prefix(self, messages: list[Message], adapter: str,
                         topk: int) -> list[list[dict[int, float]]]:
        """Fallback: one request per message prefix (O(n) requests)."""
        out: list[list[dict[int, float]]] = []
        prev = 0
        for i in range(1, len(messages) + 1):
            entries = self._prompt_logprobs(messages[:i], adapter, topk)
            if len(entries) < prev:
                raise RuntimeError(
                    f"prompt token count shrank at message {i - 1}; "
                    "chat template lacks the prefix property"
                )
            out.append(entries[prev:])
            prev = len(entries)
        return out

    def score(self, messages: list[Message], *, adapter: str = "base") -> list[list[float]]:
        flat = self._prompt_logprobs(messages, adapter, 0)
        ids = self._local_ids(messages, len(flat))
        if ids is None:  # local/server tokenization disagree: pay for the loop
            return [[self._actual(e) for e in msg]
                    for msg in self._score_by_prefix(messages, adapter, 0)]
        out, pos = [], 0
        for span in ids:
            out.append([self._actual(e) for e in flat[pos:pos + len(span)]])
            pos += len(span)
        return out

    def score_topk(self, messages: list[Message], *, adapter: str = "base",
                   k: int = 32) -> tuple[list[list[int]], list[list[dict[int, float]]]]:
        """Per-message (token_ids, top-k {token_id: logprob}) -- the teacher of
        context distillation. Raises if the local tokenization cannot be verified
        against the server's, since the KL target needs exact token alignment.
        """
        flat = self._prompt_logprobs(messages, adapter, k)
        ids = self._local_ids(messages, len(flat))
        if ids is None:
            raise RuntimeError(
                f"cannot align top-k teacher: server scored {len(flat)} tokens, "
                "local chat template disagrees"
            )
        dists, pos = [], 0
        for span in ids:
            dists.append(flat[pos:pos + len(span)])
            pos += len(span)
        return ids, dists

    def load_adapter(self, version: AdapterVersion) -> None:
        self._adapters[version.name] = version
        if version.path is None:  # base model: nothing to load
            return
        try:
            self._post(
                "/v1/load_lora_adapter",
                {"lora_name": self.lora_prefix + version.name, "lora_path": version.path},
            )
        except RuntimeError as e:
            if "already" in str(e).lower():  # idempotent re-load
                return
            raise

    def load_external(self, name: str, path: str) -> None:
        """Register a LoRA served under its exact name (no run prefix)."""
        self._external.add(name)
        try:
            self._post("/v1/load_lora_adapter", {"lora_name": name, "lora_path": path})
        except RuntimeError as e:
            if "already" not in str(e).lower():
                raise

    def unload_adapter(self, name: str) -> None:
        """Drop a served LoRA (bounds --max-loras in long runs); idempotent."""
        if name in _BASE_NAMES:
            return
        self._adapters.pop(name, None)
        try:
            self._post("/v1/unload_lora_adapter", {"lora_name": self.lora_prefix + name})
        except RuntimeError:
            pass  # already gone / server without runtime unload
