"""Experience block: outcome-tagged evidence from retrieved trajectories.

Copy-safe design: retrieved (peer) trajectories contribute full numbered
"action -> result" lines, while the scored trajectory's OWN evidence is only
an outcome summary (final reward + tail of the final feedback, never its
trajectory body) — so no token of the scored sequence can be copied from
context and hindsight deltas measure genuine belief shift.
"""
from __future__ import annotations

from typing import Optional

from sediment.config import StreamConfig
from sediment.types import ExperienceBlock, Message, Trajectory

EXPERIENCE_OPEN = "<previous_attempts>"
EXPERIENCE_CLOSE = "</previous_attempts>"
HEADER = "Below are labeled previous attempts on this task. Learn from them, then solve the task."
SUCCESS_THRESHOLD = 0.999


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


def _final_feedback(traj: Trajectory) -> str:
    """Content of the last env feedback (tool/user message after an action)."""
    for i in range(len(traj.messages) - 1, -1, -1):
        m = traj.messages[i]
        if m.role in ("tool", "user") and any(x.role == "assistant" for x in traj.messages[:i]):
            return m.content
    return ""


def build_block(
    retrieved: list[Trajectory], own: Optional[Trajectory], cfg: StreamConfig
) -> ExperienceBlock:
    """Render retrieved trajectories (+ own outcome summary) as a tagged block.

    Each retrieved trajectory becomes an outcome tag line plus numbered
    "action -> result" lines with both sides truncated to
    cfg.max_result_chars. `own` contributes ONLY its outcome summary.
    Returns an empty-text block when there is nothing to render.
    """
    if not retrieved and own is None:
        return ExperienceBlock(text="")

    budget = getattr(cfg, "max_block_chars", 4000)
    lines = [EXPERIENCE_OPEN, HEADER, ""]
    size = sum(len(x) + 1 for x in lines)
    kept: list[Trajectory] = []
    for k, traj in enumerate(retrieved, 1):
        ok, r = _outcome(traj)
        tag = "SUCCESS" if ok else "FAILED"
        seg = [f"Attempt {k} — {tag} (r={r})"]
        for n, (action, result) in enumerate(_steps(traj), 1):
            seg.append(
                f"  {n}. {_truncate(action, cfg.max_result_chars)} -> "
                f"{_truncate(result, cfg.max_result_chars)}"
            )
        refl = str(traj.meta.get("reflection", "") or "")
        if refl:
            seg.append(f"  Reflection after this attempt:\n{_indent(refl)}")
        seg.append("")
        seg_size = sum(len(x) + 1 for x in seg)
        if kept and size + seg_size > budget:  # keep at least one peer
            break
        lines.extend(seg)
        size += seg_size
        kept.append(traj)
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
        source_task_ids=[t.task_id for t in kept],
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
