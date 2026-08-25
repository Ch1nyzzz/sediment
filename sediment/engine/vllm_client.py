"""VllmClient: OpenAI-compatible HTTP client for a vLLM server (no vllm import).

Generation uses /v1/chat/completions with `model=<adapter name or base>`.
Teacher-forced scoring uses vLLM's `prompt_logprobs` extra body with
`add_generation_prompt=False`: one request per message prefix gives the
prefix token count and its logprobs (the chat-template prefix property, same
approach as resid's vllm_engine), so per-message logprob lists are the tail
slices between consecutive prefix lengths. --enable-prefix-caching makes the
repeated prefixes cheap. LoRA adapters are registered at runtime via
POST /v1/load_lora_adapter (server must allow runtime LoRA updating).
"""
from __future__ import annotations

import json
import socket
import time
import re
import urllib.error
import urllib.request

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


class VllmClient:
    def __init__(
        self, base_url: str, model: str, *, timeout: float = 600.0, max_context: int = 12288,
        lora_prefix: str = "",
    ):
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/v1"):  # paths below carry /v1 already
            self.base_url = self.base_url[: -len("/v1")].rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_context = max_context
        # hybrid-reasoning bases need enable_thinking=False; the trainer derives
        # the same kwargs from its tokenizer so both renderings agree
        self.template_kwargs = template_kwargs(model)
        self.n_think = 0  # generations that opened a <think> block anyway
        self.n_think_unclosed = 0  # ... and spent the whole budget inside it
        self._adapters: dict[str, AdapterVersion] = {}
        # served LoRA name = prefix + version name, so concurrent runs sharing
        # one server (each publishing v0001, v0002, ...) never collide
        self.lora_prefix = lora_prefix
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
            except (TimeoutError, socket.timeout, urllib.error.URLError, ConnectionError) as e:
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

    def _prompt_logprobs(self, messages: list[Message], adapter: str) -> list[float]:
        """Per-token logprobs of the rendered prompt (0.0 at position 0)."""
        payload = {
            "model": self._model_name(adapter),
            "messages": [m.to_dict() for m in messages],
            "temperature": 0.0,
            "max_tokens": 1,
            "prompt_logprobs": 0,  # vLLM extra body: actual token only
            "add_generation_prompt": False,  # score exactly the rendering
        }
        if self.template_kwargs:
            payload["chat_template_kwargs"] = self.template_kwargs
        data = self._post("/v1/chat/completions", payload)
        logps: list[float] = []
        for entry in data.get("prompt_logprobs") or []:
            if not entry:  # first position has no logprob
                logps.append(0.0)
                continue
            vals = [float(v["logprob"]) for v in entry.values()]
            logps.append(vals[0] if len(vals) == 1 else min(vals))
        return logps

    def score(self, messages: list[Message], *, adapter: str = "base") -> list[list[float]]:
        out: list[list[float]] = []
        prev = 0
        for i in range(1, len(messages) + 1):
            logps = self._prompt_logprobs(messages[:i], adapter)
            if len(logps) < prev:
                raise RuntimeError(
                    f"prompt token count shrank at message {i - 1}; "
                    "chat template lacks the prefix property"
                )
            out.append(logps[prev:])
            prev = len(logps)
        return out

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

    def unload_adapter(self, name: str) -> None:
        """Drop a served LoRA (bounds --max-loras in long runs); idempotent."""
        if name in _BASE_NAMES:
            return
        self._adapters.pop(name, None)
        try:
            self._post("/v1/unload_lora_adapter", {"lora_name": self.lora_prefix + name})
        except RuntimeError:
            pass  # already gone / server without runtime unload
