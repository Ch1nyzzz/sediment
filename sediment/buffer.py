"""Streaming trajectory buffer: jsonl persistence, retrieval, replay states."""
from __future__ import annotations

import json
import random
from pathlib import Path

from sediment.types import Message, Trajectory


def _tokens(text: str) -> set[str]:
    """Lowercased whitespace tokens of a text."""
    return set(text.lower().split())


def _first_user_text(traj: Trajectory) -> str:
    for m in traj.messages:
        if m.role == "user":
            return m.content
    return ""


class Buffer:
    """Append-only trajectory store backed by a jsonl file.

    ``add`` persists immediately (one ``Trajectory.to_dict`` json line) and
    mirrors the trajectory in memory. ``retrieve`` scores stored trajectories
    by env-family match (strong, +2.0) plus Jaccard token overlap between the
    task payload text and the trajectory's first user message (weak, in
    [0, 1], so a family match always outranks overlap alone).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._trajs: list[Trajectory] = []

    def __len__(self) -> int:
        return len(self._trajs)

    def add(self, traj: Trajectory) -> None:
        """Append the trajectory to the jsonl file and the in-memory list."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
        self._trajs.append(traj)

    @classmethod
    def load(cls, path: str | Path) -> "Buffer":
        """Rebuild a buffer by re-reading its jsonl file (missing file = empty)."""
        buf = cls(path)
        if buf.path.exists():
            with buf.path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        buf._trajs.append(Trajectory.from_dict(json.loads(line)))
        return buf

    def retrieve(self, task: dict, k: int) -> list[Trajectory]:
        """Top-k stored trajectories relevant to a task, excluding the task itself.

        score = 2.0 * [same env_family] + token-overlap(payload text,
        first user message); ties broken by recency (later additions win).
        """
        query = _tokens(str(task.get("payload", "")))
        family = task.get("env_family")
        task_id = task.get("task_id")
        scored: list[tuple[float, int, Trajectory]] = []
        for i, traj in enumerate(self._trajs):
            if traj.task_id == task_id:
                continue
            score = 2.0 if traj.env_family == family else 0.0
            stored = _tokens(_first_user_text(traj))
            union = query | stored
            if union:
                score += len(query & stored) / len(union)
            scored.append((score, i, traj))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [traj for _, _, traj in scored[:k]]

    def replay_states(self, n: int, seed: int) -> list[list[Message]]:
        """Sample n decision-point prefixes for gate G2, deterministic under seed.

        Each prefix is a stored trajectory's message list cut just before a
        randomly chosen assistant turn. Trajectories with no assistant turn
        are skipped; returns [] when none qualify. Samples without
        replacement when enough trajectories exist, with replacement
        otherwise.
        """
        eligible = [
            (t, idxs)
            for t in self._trajs
            if (idxs := [i for i, m in enumerate(t.messages) if m.role == "assistant"])
        ]
        if not eligible:
            return []
        rng = random.Random(seed)
        if n <= len(eligible):
            chosen = rng.sample(eligible, n)
        else:
            chosen = [rng.choice(eligible) for _ in range(n)]
        prefixes: list[list[Message]] = []
        for traj, idxs in chosen:
            cut = rng.choice(idxs)
            prefixes.append([Message(m.role, m.content) for m in traj.messages[:cut]])
        return prefixes
