"""Multi-turn agent loop: engine <-> env, producing a types.Trajectory (G=1)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from sediment.config import StreamConfig
from sediment.types import ExperienceBlock, Message, Trajectory

if TYPE_CHECKING:  # engine is duck-typed at runtime (see sediment.engine.base)
    from sediment.engine.base import Engine

# Injection tags, identical to resid's experience/context.py format.
EXPERIENCE_OPEN = "<previous_attempts>"
EXPERIENCE_CLOSE = "</previous_attempts>"


def inject_experience(task_text: str, experience: Optional[ExperienceBlock]) -> str:
    """First user message = tagged experience block + blank line + task text.

    The block text is wrapped in <previous_attempts> tags unless the builder
    already included them, so the tag appears exactly once either way.
    """
    if experience is None or not experience.text:
        return task_text
    block = experience.text
    if EXPERIENCE_OPEN not in block:
        block = f"{EXPERIENCE_OPEN}\n{block}\n{EXPERIENCE_CLOSE}"
    return f"{block}\n\n{task_text}"


def run_episode(
    engine: "Engine",
    env,
    task: dict[str, Any],
    cfg: StreamConfig,
    *,
    adapter: str,
    experience: Optional[ExperienceBlock] = None,
) -> Trajectory:
    """Single rollout (G=1) of engine on env for one task dict.

    When `experience` is given, its text is injected into the first user
    message. The final observation is not appended to messages (resid
    convention: the trajectory ends on the assistant's answer turn); it is
    kept in meta["final_obs"].
    """
    messages = env.reset(task)
    if experience is not None and experience.text:
        for i, m in enumerate(messages):
            if m.role == "user":
                messages[i] = Message("user", inject_experience(m.content, experience))
                break

    done = False
    reward = 0.0
    steps = 0
    final_obs = ""
    for _ in range(cfg.max_steps):
        action = engine.generate(
            messages,
            adapter=adapter,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        messages.append(Message("assistant", action))
        obs_msgs, done, reward = env.step(action)
        steps += 1
        if done:
            final_obs = "\n".join(m.content for m in obs_msgs)
            break
        messages.extend(obs_msgs)

    reward = float(reward)
    meta: dict[str, Any] = {"done": done, "final_obs": final_obs}
    if experience is not None:
        meta["experience_source_task_ids"] = list(experience.source_task_ids)
    return Trajectory(
        task_id=str(task.get("task_id", "")),
        env_family=str(task.get("env_family", getattr(env, "env_family", ""))),
        messages=messages,
        reward=reward,
        success=reward >= 0.999,
        adapter=adapter,
        steps=steps,
        meta=meta,
    )
