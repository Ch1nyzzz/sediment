"""Multi-turn agent loop: engine <-> env, producing a types.Trajectory (G=1)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from sediment.config import StreamConfig
from sediment.envs.base import tool_call_count
from sediment.harness import drop_state, first_call, is_error_obs
from sediment.types import ExperienceBlock, Message, Trajectory

if TYPE_CHECKING:  # engine is duck-typed at runtime (see sediment.engine.base)
    from sediment.calltime import CallTimeHinter
    from sediment.engine.base import Engine
    from sediment.stepwise import StepwiseHinter
    from sediment.working_state import WorkingState

# Injection tags, identical to resid's experience/context.py format.
EXPERIENCE_OPEN = "<previous_attempts>"
EXPERIENCE_CLOSE = "</previous_attempts>"


def _count_tokens(engine: "Engine", messages: list[Message]) -> int:
    counter = getattr(engine, "count_tokens", None)
    if counter is None:
        raise ValueError("episode_token_budget requires an engine.count_tokens method")
    return int(counter(messages))


def _fit_message_to_budget(
    engine: "Engine",
    prefix: list[Message],
    message: Message,
    budget: int,
) -> tuple[Optional[Message], bool]:
    if _count_tokens(engine, prefix + [message]) <= budget:
        return message, False
    empty = Message(message.role, "")
    if _count_tokens(engine, prefix + [empty]) > budget:
        return None, True
    low, high = 0, len(message.content)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = Message(message.role, message.content[:middle])
        if _count_tokens(engine, prefix + [candidate]) <= budget:
            low = middle
        else:
            high = middle - 1
    fitted = Message(message.role, message.content[:low])
    while fitted.content and _count_tokens(engine, prefix + [fitted]) > budget:
        fitted = Message(message.role, fitted.content[:-1])
    if _count_tokens(engine, prefix + [fitted]) > budget:
        return None, True
    return fitted, True


def _completion_capacity(
    engine: "Engine", messages: list[Message], budget: int, per_turn_limit: int
) -> int:
    prompt_tokens = _count_tokens(engine, messages)
    with_empty_assistant = _count_tokens(engine, messages + [Message("assistant", "")])
    role_overhead = max(0, with_empty_assistant - prompt_tokens)
    return max(0, min(per_turn_limit, budget - prompt_tokens - role_overhead))


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
    stepwise: Optional["StepwiseHinter"] = None,
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
    checkpoint_steps = sorted({int(step) for step in cfg.reward_checkpoint_steps})
    invalid_checkpoints = [
        step for step in checkpoint_steps if step <= 0 or step > cfg.max_steps
    ]
    if invalid_checkpoints:
        raise ValueError(
            "reward_checkpoint_steps must be within [1, max_steps]; "
            f"got {invalid_checkpoints} with max_steps={cfg.max_steps}"
        )

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
    checkpoint_rewards: dict[str, float] = {}
    tool_calls_emitted = 0
    multi_call_turns = 0
    error_observations = 0
    assistant_turns = 0
    termination_reason = "max_steps"
    episode_tokens = (
        _count_tokens(engine, messages) if cfg.episode_token_budget > 0 else None
    )
    max_gens = cfg.max_steps + (hinter.max_hints * len(hinter.triggers) if hinter is not None else 0)
    for _ in range(max_gens):
        if steps >= cfg.max_steps:
            break
        if stepwise is not None:  # rolling per-turn selection; count unchanged
            stepwise.pre_generate(messages)
        generation_limit = cfg.max_tokens
        if cfg.episode_token_budget > 0:
            generation_limit = _completion_capacity(
                engine, messages, cfg.episode_token_budget, cfg.max_tokens
            )
            if generation_limit <= 0:
                termination_reason = "token_budget"
                break
        action = engine.generate(
            messages,
            adapter=adapter,
            temperature=cfg.temperature,
            max_tokens=generation_limit,
        )
        assistant_message = Message("assistant", action)
        if cfg.episode_token_budget > 0:
            assistant_message, truncated = _fit_message_to_budget(
                engine, messages, assistant_message, cfg.episode_token_budget
            )
            if assistant_message is not None:
                messages.append(assistant_message)
                episode_tokens = _count_tokens(engine, messages)
            if truncated:
                assistant_turns += 1
                termination_reason = "token_budget"
                break
        else:
            messages.append(assistant_message)
        assistant_turns += 1
        emitted = tool_call_count(action)
        tool_calls_emitted += emitted
        multi_call_turns += int(emitted > 1)
        if hinter is not None:
            hint = hinter.on_draft(action, last_obs)
            if hint is not None:  # intercepted: not executed, re-draft next turn
                hint_message = Message("user", hint)
                if cfg.episode_token_budget > 0:
                    hint_message, truncated = _fit_message_to_budget(
                        engine, messages, hint_message, cfg.episode_token_budget
                    )
                    if hint_message is not None:
                        messages.append(hint_message)
                        episode_tokens = _count_tokens(engine, messages)
                    if truncated:
                        termination_reason = "token_budget"
                        break
                else:
                    messages.append(hint_message)
                continue
        obs_msgs, done, reward = env.step(action)
        steps += 1
        error_observations += int(any(is_error_obs(m.content) for m in obs_msgs))
        if steps in checkpoint_steps:
            checkpoint_rewards[str(steps)] = float(
                reward if done else env.score_current_state()
            )
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
        budget_exhausted = False
        for obs_message in obs_msgs:
            if cfg.episode_token_budget > 0:
                fitted, truncated = _fit_message_to_budget(
                    engine, messages, obs_message, cfg.episode_token_budget
                )
                if fitted is not None:
                    messages.append(fitted)
                    episode_tokens = _count_tokens(engine, messages)
                if truncated:
                    termination_reason = "token_budget"
                    budget_exhausted = True
                    break
            else:
                messages.append(obs_message)
        if budget_exhausted:
            break

    natural_finish = bool(done)
    if natural_finish:
        termination_reason = "natural_finish"
        for checkpoint in checkpoint_steps:
            if checkpoint >= steps:
                checkpoint_rewards.setdefault(str(checkpoint), float(reward))
    else:
        reward = float(env.score_current_state())
        done = True
        if cfg.max_steps in checkpoint_steps:
            checkpoint_rewards[str(cfg.max_steps)] = reward

    reward = float(reward)
    meta: dict[str, Any] = {
        "done": done,
        "natural_finish": natural_finish,
        "forced_settle": not natural_finish,
        "termination_reason": termination_reason,
        "termination_step": steps,
        "episode_tokens": episode_tokens,
        "final_obs": final_obs,
        "tool_protocol": {
            "assistant_turns": assistant_turns,
            "env_steps": steps,
            "tool_calls_emitted": tool_calls_emitted,
            "multi_call_turns": multi_call_turns,
            "error_observations": error_observations,
        },
    }
    if checkpoint_steps:
        meta["reward_checkpoints"] = checkpoint_rewards
    if experience is not None:
        meta["experience_source_task_ids"] = list(experience.source_task_ids)
    if hinter is not None:
        meta["calltime"] = {"donors": hinter.donor_ids, "events": hinter.events}
    if working_state is not None:
        meta["working_state"] = working_state.summary()
    if stepwise is not None:
        meta["stepwise_events"] = stepwise.events
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
