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
import urllib.error
import urllib.request

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


class VllmClient:
    def __init__(self, base_url: str, model: str, *, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._adapters: dict[str, AdapterVersion] = {}

    # -- http -------------------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"POST {path} failed ({e.code}): {detail[:500]}") from e
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
        return adapter  # served LoRA name

    # -- Engine protocol ---------------------------------------------------
    def generate(
        self,
        messages: list[Message],
        *,
        adapter: str = "base",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        data = self._post(
            "/v1/chat/completions",
            {
                "model": self._model_name(adapter),
                "messages": [m.to_dict() for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        return data["choices"][0]["message"]["content"] or ""

    def _prompt_logprobs(self, messages: list[Message], adapter: str) -> list[float]:
        """Per-token logprobs of the rendered prompt (0.0 at position 0)."""
        data = self._post(
            "/v1/chat/completions",
            {
                "model": self._model_name(adapter),
                "messages": [m.to_dict() for m in messages],
                "temperature": 0.0,
                "max_tokens": 1,
                "prompt_logprobs": 0,  # vLLM extra body: actual token only
                "add_generation_prompt": False,  # score exactly the rendering
            },
        )
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
                {"lora_name": version.name, "lora_path": version.path},
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
            self._post("/v1/unload_lora_adapter", {"lora_name": name})
        except RuntimeError:
            pass  # already gone / server without runtime unload
