"""Reflection: the actor distills its own finished attempt into a few rules.

Generated once per first attempt (temperature 0, own adapter), stored in
``traj.meta["reflection"]`` and rendered next to the trajectory by
experience.build_block. Input is the settled step summary (verbatim actions,
truncated results, outcome, final feedback) — the same evidence the block
shows — so the reflection adds interpretation, not new observations.
"""
from __future__ import annotations

from sediment.config import StreamConfig
from sediment.experience import _final_feedback, _outcome, _steps, _tail, _truncate
from sediment.transfer import canonical_transfer_reflection
from sediment.types import Message, Trajectory

PROMPT = (
    "You just finished an attempt at a task in a tool-using environment. Below is a "
    "summary of your attempt. Write 3-5 short bullet rules that would help on similar "
    "tasks: environment rules you discovered (preconditions, argument formats, error "
    "causes), what went wrong, and what to do instead. Be concrete and specific to "
    "this environment. Output only the bullets."
)

def render_attempt(traj: Trajectory, cfg: StreamConfig) -> str:
    ok, r = _outcome(traj)
    lines = []
    for n, (action, result) in enumerate(_steps(traj), 1):
        lines.append(f"{n}. {_truncate(action, cfg.max_result_chars)} -> "
                     f"{_truncate(result, cfg.max_result_chars)}")
    lines.append(f"Outcome: {'SUCCEEDED' if ok else 'FAILED'} (reward={r})")
    fb = str(traj.meta.get("final_obs", "") or "") or _final_feedback(traj)
    if fb:
        lines.append(f"Final feedback: {_tail(fb, cfg.max_result_chars * 2)}")
    return "\n".join(lines)


def reflect(engine, traj: Trajectory, cfg: StreamConfig, *, adapter: str) -> str:
    """Return the reflection text (bounded to cfg.max_reflection_chars); '' on error."""
    task_text = next((m.content for m in traj.messages if m.role == "user"), "")
    if cfg.reflection_mode not in ("local", "transfer"):
        raise ValueError(f"unknown reflection mode: {cfg.reflection_mode!r}")
    if cfg.reflection_mode == "transfer":
        return _truncate(
            canonical_transfer_reflection(task_text, render_attempt(traj, cfg)),
            cfg.max_reflection_chars,
        )
    messages = [
        Message("system", PROMPT),
        Message("user", f"Task:\n{_truncate(task_text, 1500)}\n\nYour attempt:\n"
                        f"{render_attempt(traj, cfg)}"),
    ]
    try:
        out = engine.generate(messages, adapter=adapter, temperature=0.0,
                              max_tokens=cfg.reflect_max_tokens)
    except Exception:
        return ""
    return _truncate(str(out).strip(), cfg.max_reflection_chars)
