"""Router: stable task -> serving-engine assignment over a fixed pool."""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - engine module lands separately
    from sediment.engine.base import Engine


class Router:
    """Deterministic task_id -> engine mapping (sha1(task_id) mod pool size).

    The mapping depends only on the task_id and the pool size, so a task is
    always served (and hindsight-scored) by the same engine within a run.
    """

    def __init__(self, engines: list["Engine"]):
        if not engines:
            raise ValueError("Router requires at least one engine")
        self._engines = list(engines)

    def for_task(self, task_id: str) -> "Engine":
        """Return the engine that owns `task_id` (stable across calls)."""
        digest = hashlib.sha1(task_id.encode("utf-8")).hexdigest()
        return self._engines[int(digest, 16) % len(self._engines)]

    def all(self) -> list["Engine"]:
        """All engines in the pool (copy)."""
        return list(self._engines)
