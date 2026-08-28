"""Multi-turn agent loop: engine <-> env, producing a types.Trajectory (G=1)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from sediment.config import StreamConfig
from sediment.harness import drop_state, first_call, is_error_obs
from sediment.types import ExperienceBlock, Message, Trajectory

if TYPE_CHECKING:  # engine is duck-typed at runtime (see sediment.engine.base)
    from sediment.calltime import CallTimeHinter
    from sediment.engine.base import Engine
    from sediment.working_state import WorkingState

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


from sediment.scheduler import strip_experience  # noqa: E402,F401  (shared helper)


def run_episode(
    engine: "Engine",
    env,
    task: dict[str, Any],
    cfg: StreamConfig,
    *,
    adapter: str,
    experience: Optional[ExperienceBlock] = None,
    hinter: Optional["CallTimeHinter"] = None,
    working_state: Optional["WorkingState"] = None,
) -> Trajectory:
    """Single rollout (G=1) of engine on env for one task dict.

    When `experience` is given, its text is injected into the first user
    message. `hinter` adds call-time memory: a drafted action it hints on is
    NOT executed (a synthetic user turn carries the hint, the model re-drafts,
    and that draft does not count as a step); an error observation may get a
    hint appended. `working_state` appends the harness-verified goal ledger
    to the newest observation (earlier copies are removed). `steps` counts env steps only. The final observation
    is not appended to messages (resid convention: the trajectory ends on the
    assistant's answer turn); it is kept in meta["final_obs"].
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
    last_obs = ""
    max_gens = cfg.max_steps + (hinter.max_hints * len(hinter.triggers) if hinter is not None else 0)
    for _ in range(max_gens):
        if steps >= cfg.max_steps:
            break
        action = engine.generate(
            messages,
            adapter=adapter,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        messages.append(Message("assistant", action))
        if hinter is not None:
            hint = hinter.on_draft(action, last_obs)
            if hint is not None:  # intercepted: not executed, re-draft next turn
                messages.append(Message("user", hint))
                continue
        obs_msgs, done, reward = env.step(action)
        steps += 1
        if done:
            final_obs = "\n".join(m.content for m in obs_msgs)
            break
        obs = obs_msgs[-1]
        last_obs = obs.content
        extra: list[str] = []
        if hinter is not None:
            hint = hinter.on_observation(first_call(action), obs.content)
            if hint is not None:
                extra.append(hint)
        if working_state is not None:  # only the newest state stays in context
            working_state.record(steps, first_call(action), obs.content, not is_error_obs(obs.content))
            drop_state(messages)
            extra.append(working_state.render())
        if extra:
            obs_msgs[-1] = Message(obs.role, "\n\n".join([obs.content, *extra]))
        messages.extend(obs_msgs)

    reward = float(reward)
    meta: dict[str, Any] = {"done": done, "final_obs": final_obs}
    if experience is not None:
        meta["experience_source_task_ids"] = list(experience.source_task_ids)
    if hinter is not None:
        meta["calltime"] = {"donors": hinter.donor_ids, "events": hinter.events}
    if working_state is not None:
        meta["working_state"] = working_state.summary()
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
