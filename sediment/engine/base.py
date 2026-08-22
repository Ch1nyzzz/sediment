"""Engine protocol: generation, teacher-forced scoring, adapter loading.

Engines are addressed with adapter version names ("base"/"v0000" = no
adapter). `score` returns per-message per-token logprobs aligned with the
tokenization used by sediment.spans, so hindsight can diff with/without an
injected experience block message-by-message.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from sediment.types import AdapterVersion, Message


@runtime_checkable
class Engine(Protocol):
    def generate(
        self,
        messages: list[Message],
        *,
        adapter: str = "base",
        temperature: float,
        max_tokens: int,
    ) -> str:
        """One assistant turn: returns the raw generated text."""
        ...

    def score(self, messages: list[Message], *, adapter: str = "base") -> list[list[float]]:
        """Teacher-forced per-token logprobs, one list per message,
        aligned with the tokenization used by sediment.spans."""
        ...

    def load_adapter(self, version: AdapterVersion) -> None:
        """Make an adapter version addressable by name in generate/score."""
        ...
