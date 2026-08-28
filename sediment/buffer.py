"""Streaming trajectory buffer: jsonl persistence, retrieval, replay states."""
from __future__ import annotations

import json
import random
from pathlib import Path

from sediment.transfer import transfer_features
from sediment.types import Message, Trajectory


def _tokens(text: str) -> set[str]:
    """Lowercased whitespace tokens of a text."""
    return set(text.lower().split())


def _task_text(task: dict) -> str:
    """Natural-language instruction, excluding state/checklist payload noise."""
    payload = task.get("payload", "")
    if isinstance(payload, dict):
        for key in ("task", "instruction", "query", "prompt"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return str(payload)


def _first_user_text(traj: Trajectory) -> str:
    for m in traj.messages:
        if m.role == "user":
            return m.content
    return ""


class Buffer:
    """Append-only trajectory store backed by a jsonl file.

    ``add`` persists immediately (one ``Trajectory.to_dict`` json line) and
    mirrors the trajectory in memory. ``retrieve`` ranks the entire memory
    bank by Jaccard token overlap between the task payload and each stored
    trajectory's first user message. Environment family is deliberately not
    used as either a filter or a bonus.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._trajs: list[Trajectory] = []
        self._token_cache: dict[int, set[str]] = {}
        self._transfer_cache: dict[int, set[str]] = {}

    def __len__(self) -> int:
        return len(self._trajs)

    def add(self, traj: Trajectory) -> None:
        """Append the trajectory to the jsonl file and the in-memory list."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
        self._trajs.append(traj)
        self._token_cache[id(traj)] = _tokens(_first_user_text(traj))
        self._transfer_cache[id(traj)] = transfer_features(
            f"{_first_user_text(traj)}\n{traj.meta.get('reflection', '')}"
        )

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

    def retrieve(
        self, task: dict, k: int, *, scope: str = "all", score_mode: str = "task",
        offset: int = 0, diversity: str = "task",
    ) -> list[Trajectory]:
        """Top-k stored trajectories relevant to a task, excluding the task itself.

        Score is token overlap only; ties are broken by recency. Results use
        distinct task ids when possible, so four rollout seeds of one source
        instance cannot silently consume all four retrieval slots. With
        ``diversity="family"``, distinct source families are preferred instead.
        ``offset`` exposes later ranks without concatenating earlier examples.
        ``scope`` may restrict the candidate pool to the query's environment
        family or to trajectories from other families only.
        """
        if k < 0:
            raise ValueError("retrieval k must be non-negative")
        if offset < 0:
            raise ValueError("retrieval offset must be non-negative")
        if scope not in ("all", "same_family", "cross_family"):
            raise ValueError(f"unknown retrieval scope: {scope!r}")
        if score_mode not in (
            "task", "task_text", "task_text_success", "transfer", "legacy"
        ):
            raise ValueError(f"unknown retrieval score mode: {score_mode!r}")
        if diversity not in ("task", "family"):
            raise ValueError(f"unknown retrieval diversity: {diversity!r}")
        if k == 0:
            return []
        payload = str(task.get("payload", ""))
        natural_task = _task_text(task)
        query = _tokens(
            natural_task
            if score_mode in ("task_text", "task_text_success")
            else payload
        )
        query_transfer = transfer_features(natural_task)
        task_id = task.get("task_id")
        family = task.get("env_family")
        scored: list[tuple[float, int, Trajectory]] = []
        for i, traj in enumerate(self._trajs):
            if traj.task_id == task_id:
                continue
            if scope == "same_family" and traj.env_family != family:
                continue
            if scope == "cross_family" and traj.env_family == family:
                continue
            cache_key = id(traj)
            stored = self._token_cache.get(cache_key)
            if stored is None:
                stored = _tokens(_first_user_text(traj))
                self._token_cache[cache_key] = stored
            union = query | stored
            raw_score = len(query & stored) / len(union) if union else 0.0
            if score_mode == "transfer":
                transfer = self._transfer_cache.get(cache_key)
                if transfer is None:
                    transfer = transfer_features(
                        f"{_first_user_text(traj)}\n{traj.meta.get('reflection', '')}"
                    )
                    self._transfer_cache[cache_key] = transfer
                transfer_union = query_transfer | transfer
                transfer_score = (
                    len(query_transfer & transfer) / len(transfer_union)
                    if transfer_union else 0.0
                )
                # Raw overlap is a deterministic secondary tie-breaker only.
                score = transfer_score + 1e-3 * raw_score
            elif score_mode == "legacy":
                # Historical route: any same-family peer outranks every
                # cross-family peer because raw Jaccard is bounded by one.
                score = (2.0 if traj.env_family == family else 0.0) + raw_score
            elif score_mode == "task_text_success":
                # Source outcome is available before target inference.  A two-
                # point margin makes every successful trajectory outrank every
                # failure (Jaccard is at most one), while natural task overlap
                # still orders donors within each outcome class.
                succeeded = traj.success if traj.success is not None else (
                    traj.reward is not None and traj.reward >= 0.999
                )
                score = (2.0 if succeeded else 0.0) + raw_score
            else:
                score = raw_score
            scored.append((score, i, traj))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        needed = k + offset
        selected: list[Trajectory] = []
        seen_keys: set[str] = set()
        duplicates: list[Trajectory] = []
        for _, _, traj in scored:
            key = traj.task_id if diversity == "task" else traj.env_family
            if key in seen_keys:
                duplicates.append(traj)
                continue
            selected.append(traj)
            seen_keys.add(key)
            if len(selected) == needed:
                return selected[offset:needed]
        if len(selected) < needed:
            selected.extend(duplicates[: needed - len(selected)])
        return selected[offset:needed]

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
