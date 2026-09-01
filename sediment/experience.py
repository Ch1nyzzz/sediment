"""Experience block: outcome-tagged evidence from retrieved trajectories.

Every peer's task and stored reflection are mandatory and rendered before any
intermediate trajectory steps. Steps are optional evidence chosen from the
remaining soft character budget. The scored trajectory's OWN evidence remains
outcome-only, so its body cannot be copied from context -- unless
cfg.own_view="full" asks for the self-feedback teacher (the complete failed
attempt), which is only safe to score on the states of a redo (kl_states).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from sediment.config import StreamConfig
from sediment.transfer import CANONICAL_RULES, transfer_features
from sediment.types import ExperienceBlock, Message, Trajectory

EXPERIENCE_OPEN = "<previous_attempts>"
EXPERIENCE_CLOSE = "</previous_attempts>"
HEADER = "Below are labeled attempts on related tasks. Use their lessons, then solve the current task."
LEGACY_HEADER = "Below are labeled previous attempts on this task. Learn from them, then solve the task."
SUCCESS_THRESHOLD = 0.999
COMPILED_HEADER = (
    "Consolidated cross-task memory from related source attempts. Transfer only "
    "workflow lessons: do not copy source-specific identifiers, tool names, or "
    "arguments. Resolve any conflict using the current task and observed state."
)

_COMPILE_STOPWORDS = {
    "about", "after", "again", "also", "before", "being", "current", "from",
    "have", "into", "must", "only", "other", "should", "that", "their", "then",
    "there", "these", "this", "those", "through", "using", "when", "where", "which",
    "with", "would", "your",
}


def _truncate(text: str, max_chars: int) -> str:
    text = text.replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def _indent(text: str, pad: str = "    ") -> str:
    return "\n".join(pad + ln for ln in text.splitlines())


def _tail(text: str, max_chars: int) -> str:
    text = text.replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return "…" + text[-max_chars:]


def _outcome(traj: Trajectory) -> tuple[bool, str]:
    """(succeeded, reward string) from success flag or reward threshold."""
    ok = traj.success if traj.success is not None else (
        traj.reward is not None and traj.reward >= SUCCESS_THRESHOLD
    )
    r = "n/a" if traj.reward is None else f"{traj.reward:.2f}"
    return bool(ok), r


def _steps(traj: Trajectory) -> list[tuple[str, str]]:
    """(action, result) pairs: assistant content -> next tool/user content."""
    out: list[tuple[str, str]] = []
    msgs = traj.messages
    for i, m in enumerate(msgs):
        if m.role != "assistant":
            continue
        nxt = msgs[i + 1] if i + 1 < len(msgs) else None
        result = nxt.content if nxt is not None and nxt.role in ("tool", "user") else "(episode end)"
        out.append((m.content, result))
    return out


def _first_user_text(traj: Trajectory) -> str:
    return next((message.content for message in traj.messages if message.role == "user"), "")


_ERROR_MARKERS = ("error", "failed", "failure", "invalid", "exception", "denied", "forbidden")


def _prioritized_step_indices(traj: Trajectory) -> list[int]:
    """Step indices ordered by value: errors, recovery, terminal, then rest."""
    steps = _steps(traj)
    error = [any(marker in f"{a}\n{r}".lower() for marker in _ERROR_MARKERS)
             for a, r in steps]
    ranked: list[tuple[int, int]] = []
    for idx, _ in enumerate(steps):
        if error[idx]:
            priority = 0
        elif idx > 0 and error[idx - 1]:
            priority = 1
        elif idx == len(steps) - 1:
            priority = 2
        else:
            priority = 3
        ranked.append((priority, idx))
    ranked.sort()
    return [idx for _, idx in ranked]


def _final_feedback(traj: Trajectory) -> str:
    """Content of the last env feedback (tool/user message after an action)."""
    for i in range(len(traj.messages) - 1, -1, -1):
        m = traj.messages[i]
        if m.role in ("tool", "user") and any(x.role == "assistant" for x in traj.messages[:i]):
            return m.content
    return ""


def _compile_tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9_]+", text.lower())
        if len(token) >= 3 and token not in _COMPILE_STOPWORDS
    }


def _reflection_units(text: str) -> list[str]:
    """Split a free-form reflection into bounded, independently ranked lessons."""
    # Reflections are flattened by `_truncate` at storage time, so restore
    # inline bullet boundaries such as "- rule one - rule two" first.
    text = re.sub(r"\s+(?=[-*•]\s+)", "\n", text.strip())
    text = re.sub(r"\s+(?=\d+[.)]\s+)", "\n", text)
    units: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if not line:
            continue
        pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z])", line)
        units.extend(piece.strip() for piece in pieces if len(piece.strip()) >= 8)
    return units


def _build_compiled_reflection_block(
    retrieved: list[Trajectory], cfg: StreamConfig, query_text: str,
) -> ExperienceBlock:
    """Merge top-n reflections into one short, deduplicated pre-rollout memory."""
    if not retrieved:
        return ExperienceBlock(text="")
    query_tokens = _compile_tokens(query_text)
    query_features = transfer_features(query_text)
    candidates: list[dict[str, object]] = []
    for peer_index, traj in enumerate(retrieved):
        ok, reward_text = _outcome(traj)
        reward = float(traj.reward or 0.0)
        for line_index, line in enumerate(
            _reflection_units(str(traj.meta.get("reflection", "") or ""))
        ):
            tokens = _compile_tokens(line)
            features = transfer_features(line)
            candidates.append({
                "peer_index": peer_index,
                "line_index": line_index,
                "task_id": traj.task_id,
                "line": _truncate(line, int(cfg.memory_compile_rule_max_chars)),
                "tokens": tokens,
                "features": features,
                "success": ok,
                "reward": reward,
                "reward_text": reward_text,
            })
    if not candidates:
        return ExperienceBlock(text="")

    feature_support = Counter(
        feature
        for candidate in candidates
        for feature in candidate["features"]  # type: ignore[union-attr]
    )
    for candidate in candidates:
        tokens = candidate["tokens"]  # type: ignore[assignment]
        features = candidate["features"]  # type: ignore[assignment]
        union = query_tokens | tokens
        lexical = len(query_tokens & tokens) / len(union) if union else 0.0
        matched = query_features & features
        recurrence = sum(feature_support[feature] for feature in features)
        candidate["score"] = (
            int(bool(matched)),
            len(matched),
            int(bool(candidate["success"])),
            recurrence,
            float(candidate["reward"]),
            lexical,
            -int(candidate["peer_index"]),
            -int(candidate["line_index"]),
        )
    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)

    selected: list[dict[str, object]] = []
    per_peer: Counter[int] = Counter()
    max_rules = max(1, int(cfg.memory_compile_max_rules))
    max_chars = min(int(cfg.max_block_chars), int(cfg.memory_compile_max_chars))
    size = len(EXPERIENCE_OPEN) + len(EXPERIENCE_CLOSE) + len(COMPILED_HEADER) + 4
    def consider(candidate: dict[str, object]) -> bool:
        nonlocal size
        tokens = candidate["tokens"]  # type: ignore[assignment]
        if any(
            len(tokens & prior["tokens"]) / max(1, len(tokens | prior["tokens"])) >= 0.65
            for prior in selected
        ):
            return False
        peer_index = int(candidate["peer_index"])
        if per_peer[peer_index] >= 2:
            return False
        provenance = "successful source" if candidate["success"] else "failed-source lesson"
        rendered = f"- [{provenance}] {candidate['line']}"
        if size + len(rendered) + 1 > max_chars:
            return False
        candidate["rendered"] = rendered
        selected.append(candidate)
        per_peer[peer_index] += 1
        size += len(rendered) + 1
        return True

    # Coverage pass: give each retrieved source at most one slot before any
    # source gets a second. This prevents one redundant memory from consuming
    # the entire short block.
    for peer_index in range(len(retrieved)):
        for candidate in candidates:
            if int(candidate["peer_index"]) == peer_index and consider(candidate):
                break
        if len(selected) == max_rules:
            break
    # Utility pass: spend remaining slots on the strongest nonredundant rules.
    if len(selected) < max_rules:
        for candidate in candidates:
            if candidate in selected:
                continue
            consider(candidate)
            if len(selected) == max_rules:
                break
    if not selected:
        return ExperienceBlock(text="")
    lines = [EXPERIENCE_OPEN, COMPILED_HEADER, ""]
    lines.extend(str(candidate["rendered"]) for candidate in selected)
    lines.append(EXPERIENCE_CLOSE)
    used_ids = []
    for candidate in selected:
        task_id = str(candidate["task_id"])
        if task_id not in used_ids:
            used_ids.append(task_id)
    return ExperienceBlock(text="\n".join(lines), source_task_ids=used_ids)


def _build_legacy_block(
    retrieved: list[Trajectory], own: Optional[Trajectory], cfg: StreamConfig
) -> ExperienceBlock:
    """Exact pre-08-26 compact renderer for a matched historical control."""
    if not retrieved and own is None:
        return ExperienceBlock(text="")
    budget = getattr(cfg, "max_block_chars", 4000)
    lines = [EXPERIENCE_OPEN, LEGACY_HEADER, ""]
    size = sum(len(line) + 1 for line in lines)
    kept: list[Trajectory] = []
    for index, traj in enumerate(retrieved, 1):
        ok, reward = _outcome(traj)
        tag = "SUCCESS" if ok else "FAILED"
        segment = [f"Attempt {index} — {tag} (r={reward})"]
        for step, (action, result) in enumerate(_steps(traj), 1):
            segment.append(
                f"  {step}. {_truncate(action, cfg.max_result_chars)} -> "
                f"{_truncate(result, cfg.max_result_chars)}"
            )
        reflection = str(traj.meta.get("reflection", "") or "")
        if reflection:
            segment.append(f"  Reflection after this attempt:\n{_indent(reflection)}")
        segment.append("")
        segment_size = sum(len(line) + 1 for line in segment)
        if kept and size + segment_size > budget:
            break
        lines.extend(segment)
        size += segment_size
        kept.append(traj)
    if own is not None:
        ok, reward = _outcome(own)
        tag = "SUCCEEDED" if ok else "FAILED"
        lines.append(
            f"--- Your own attempt below ultimately {tag} (reward={reward}). "
            "Judge each of its steps with that in mind. ---"
        )
        feedback = _final_feedback(own)
        if feedback:
            lines.append(f"Final feedback: {_tail(feedback, cfg.max_result_chars)}")
        reflection = str(own.meta.get("reflection", "") or "")
        if reflection:
            lines.append(f"Your reflection on that attempt:\n{_indent(reflection)}")
    lines.append(EXPERIENCE_CLOSE)
    return ExperienceBlock(
        text="\n".join(lines),
        source_task_ids=[traj.task_id for traj in kept],
        includes_own_outcome=own is not None,
    )


def build_block(
    retrieved: list[Trajectory], own: Optional[Trajectory], cfg: StreamConfig, *,
    query_text: str = "",
) -> ExperienceBlock:
    """Render retrieved trajectories (+ own outcome summary) as a tagged block.

    Each retrieved trajectory first contributes its full task, outcome, and
    full stored reflection. Numbered "action -> result" lines are added only
    from the remaining budget. `own` contributes ONLY its outcome summary,
    except under cfg.own_view="full", where its complete step list competes
    for the budget like a peer's (task text omitted: it is the current task).
    Returns an empty-text block when there is nothing to render.
    """
    budget = getattr(cfg, "max_block_chars", 16000)
    view = getattr(cfg, "experience_view", "full")
    if view not in (
        "full", "reflection_only", "matched_reflection", "compiled_reflection", "legacy"
    ):
        raise ValueError(f"unknown experience view: {view!r}")
    if view == "legacy":
        return _build_legacy_block(retrieved, own, cfg)
    if view == "compiled_reflection":
        if own is not None:
            raise ValueError("compiled_reflection is serving-only and cannot include own outcome")
        if not query_text:
            raise ValueError("compiled_reflection requires query_text")
        return _build_compiled_reflection_block(retrieved, cfg, query_text)
    reflection_overrides: dict[int, str] = {}
    if view == "matched_reflection":
        if not query_text:
            raise ValueError("matched_reflection requires query_text")
        query_features = transfer_features(query_text)
        matched = []
        for traj in retrieved:
            reflection = str(traj.meta.get("reflection", "") or "")
            tags = [tag for tag in re.findall(r"- \[([a-z_]+)\]", reflection)
                    if tag in query_features and tag in CANONICAL_RULES]
            if not tags:
                continue
            matched.append(traj)
            reflection_overrides[id(traj)] = " ".join(
                f"- [{tag}] {CANONICAL_RULES[tag]}" for tag in tags
            )
        retrieved = matched
    if not retrieved and own is None:
        return ExperienceBlock(text="")

    own_view = getattr(cfg, "own_view", "outcome")
    if own_view not in ("outcome", "full"):
        raise ValueError(f"unknown own view: {own_view!r}")
    # (label, trajectory, is_peer): own is rendered like a peer but without its
    # task text, and its steps are always the full list regardless of `view`
    entries = [(f"Attempt {k}", traj, True) for k, traj in enumerate(retrieved, 1)]
    if own is not None and own_view == "full":
        entries.append(("Own attempt", own, False))

    lines = [EXPERIENCE_OPEN, HEADER, ""]
    step_pools: list[list[tuple[int, str]]] = []
    for label, traj, is_peer in entries:
        ok, r = _outcome(traj)
        tag = "SUCCESS" if ok else "FAILED"
        lines.append(f"{label} — {tag} (r={r})" if is_peer else
                     f"{label} (your previous try on THIS task) — {tag} (r={r})")
        task_text = _first_user_text(traj)
        if view == "full" and is_peer and task_text:
            lines.append(f"  Task from this attempt:\n{_indent(task_text)}")
        refl = reflection_overrides.get(
            id(traj), str(traj.meta.get("reflection", "") or "")
        )
        if refl:
            lines.append(f"  Reflection after this attempt:\n{_indent(refl)}")
        lines.append("")
        with_steps = view == "full" or not is_peer
        steps = _steps(traj) if with_steps else []
        pool = []
        prioritized = _prioritized_step_indices(traj) if with_steps else []
        for idx in prioritized:
            # Apply the configured per-side truncation at render time.
            action, result = steps[idx]
            pool.append((idx, f"  {idx + 1}. {_truncate(action, cfg.max_result_chars)} -> "
                              f"{_truncate(result, cfg.max_result_chars)}"))
        step_pools.append(pool)

    # Task/outcome/reflection are an indivisible core. Only intermediate steps
    # compete for the soft budget, with one peer considered per round so a long
    # trajectory cannot starve the other retrieved memories.
    size = sum(len(x) + 1 for x in lines) + len(EXPERIENCE_CLOSE) + 1
    selected: list[list[tuple[int, str]]] = [[] for _ in entries]
    cursors = [0] * len(step_pools)
    section_cost = len("Selected intermediate steps (highest-value evidence first):") + 2
    while step_pools and any(c < len(pool) for c, pool in zip(cursors, step_pools)):
        for peer_idx, pool in enumerate(step_pools):
            cursor = cursors[peer_idx]
            if cursor >= len(pool):
                continue
            idx, line = pool[cursor]
            cursors[peer_idx] += 1
            heading_cost = len(f"{entries[peer_idx][0]} selected steps:") + 1 \
                if not selected[peer_idx] else 0
            if not any(selected):
                heading_cost += section_cost
            line_cost = len(line) + 1
            if size + heading_cost + line_cost <= budget:
                selected[peer_idx].append((idx, line))
                size += heading_cost + line_cost

    if any(selected):
        lines.extend(["Selected intermediate steps (highest-value evidence first):", ""])
        for (label, _, _), chosen in zip(entries, selected):
            if not chosen:
                continue
            lines.append(f"{label} selected steps:")
            # Once selected by priority, show steps chronologically.
            lines.extend(line for _, line in sorted(chosen))
            lines.append("")
    if own is not None:
        ok, r = _outcome(own)
        tag = "SUCCEEDED" if ok else "FAILED"
        lines.append(
            f"--- Your own attempt below ultimately {tag} (reward={r}). "
            "Judge each of its steps with that in mind. ---"
        )
        feedback = _final_feedback(own)
        if feedback:
            lines.append(f"Final feedback: {_tail(feedback, cfg.max_result_chars)}")
        refl = str(own.meta.get("reflection", "") or "")
        if refl:
            lines.append(f"Your reflection on that attempt:\n{_indent(refl)}")
    lines.append(EXPERIENCE_CLOSE)
    return ExperienceBlock(
        text="\n".join(lines),
        source_task_ids=[t.task_id for t in retrieved],
        includes_own_outcome=own is not None,
    )


def inject(messages: list[Message], text: str) -> tuple[list[Message], int]:
    """Copy of messages with `text` prepended to the first user message.

    First user message becomes "{text}\\n\\n{original}" (the rollout / hindsight
    injection convention). Returns (new_messages, injected_index); empty text
    leaves the copy unchanged.
    """
    idx = next((i for i, m in enumerate(messages) if m.role == "user"), None)
    if idx is None:
        raise ValueError("no user message to inject experience into")
    out = list(messages)
    if text:
        out[idx] = Message(role="user", content=f"{text}\n\n{messages[idx].content}")
    return out, idx
