"""Tests for sediment.envs + sediment.rollout: toy env, agent loop, injection."""
from __future__ import annotations

import json

import pytest

from sediment.config import StreamConfig
from sediment.engine.mock import MockEngine
from sediment.envs import (
    Env,
    EnvScalerAdapter,
    ToyOrderEnv,
    lopd_available,
    make_env,
    make_toy_tasks,
    parse_tool_call,
    tool_call_count,
)
from sediment.rollout import EXPERIENCE_CLOSE, EXPERIENCE_OPEN, run_episode
from sediment.types import ExperienceBlock, Message


def tool_call(name: str, arguments: dict | None = None) -> str:
    payload = json.dumps({"name": name, "arguments": arguments or {}})
    return f"<tool_call>\n{payload}\n</tool_call>"


def teacher_policy(messages: list[Message]) -> str:
    """Scripted policy: check status, cancel only if pending, answer correctly."""
    n_assistant = sum(1 for m in messages if m.role == "assistant")
    if n_assistant == 0:
        return tool_call("get_order_status")
    last_tool = next((m.content for m in reversed(messages) if m.role == "tool"), "")
    if "pending" in last_tool:
        return tool_call("cancel_order")
    if "cancelled" in last_tool:
        return tool_call("answer", {"text": "The order has been cancelled."})
    return tool_call("answer", {"text": "It cannot be cancelled: already shipped."})


def student_policy(messages: list[Message]) -> str:
    """Scripted policy: cancel blindly, then claim success (fails on shipped)."""
    n_assistant = sum(1 for m in messages if m.role == "assistant")
    if n_assistant == 0:
        return tool_call("cancel_order")
    return tool_call("answer", {"text": "The order has been cancelled."})


def toy_task(status: str = "shipped", task_id: str = "toy-x") -> dict:
    return {"task_id": task_id, "env_family": "toy_order",
            "payload": {"status": status, "order_id": 7}}


def test_make_toy_tasks_deterministic():
    a, b = make_toy_tasks(20, seed=3), make_toy_tasks(20, seed=3)
    assert a == b
    assert len(a) == 20
    for t in a:
        assert set(t) == {"task_id", "env_family", "payload"}
        assert t["env_family"] == "toy_order"
        assert t["payload"]["status"] in ("pending", "shipped", "delivered")
    assert len({t["payload"]["status"] for t in a}) > 1
    assert len({t["task_id"] for t in a}) == 20


def test_env_protocol_and_make_env():
    env = make_env(toy_task())
    assert isinstance(env, ToyOrderEnv)
    assert isinstance(env, Env)
    with pytest.raises(ValueError):
        make_env({"task_id": "x", "env_family": "nope", "payload": {}})


def test_hidden_cancel_rule():
    env = ToyOrderEnv()
    env.reset(toy_task("shipped"))
    obs, done, reward = env.step(tool_call("cancel_order"))
    assert obs[0].role == "tool"
    assert obs[0].content == "Error: cannot cancel shipped order"
    assert not done and reward == 0.0

    env.reset(toy_task("pending"))
    obs, done, _ = env.step(tool_call("cancel_order"))
    assert obs[0].content == "Order cancelled."
    obs, done, reward = env.step(tool_call("answer", {"text": "Order cancelled."}))
    assert done and reward == 1.0


def test_plain_text_reply_is_final_answer():
    env = ToyOrderEnv()
    env.reset(toy_task("shipped"))
    _, done, reward = env.step("Sorry, it cannot be cancelled.")
    assert done and reward == 1.0


def test_parse_tool_call():
    assert parse_tool_call("just some text") is None
    assert parse_tool_call('<tool_call> {"name": "answer", "arguments": {"text": "hi"}} </tool_call>') == \
        {"name": "answer", "arguments": {"text": "hi"}}
    assert parse_tool_call("<tool_call>not json</tool_call>")["name"] == ""
    env = ToyOrderEnv()
    env.reset(toy_task())
    obs, done, _ = env.step("<tool_call>not json</tool_call>")
    assert not done and "could not parse" in obs[0].content
    obs, done, _ = env.step(tool_call("frobnicate"))
    assert not done and "unknown tool" in obs[0].content


def test_multiple_tool_calls_are_rejected_instead_of_silently_dropped():
    action = tool_call("get_order_status") + tool_call("cancel_order")
    assert tool_call_count(action) == 2
    env = ToyOrderEnv()
    env.reset(toy_task("pending"))
    obs, done, reward = env.step(action)
    assert not done and reward == 0.0
    assert "exactly one tool call" in obs[0].content
    assert env.status == "pending"

    malformed_second = tool_call("cancel_order") + "<tool_call>"
    assert tool_call_count(malformed_second) == 2
    obs, done, reward = env.step(malformed_second)
    assert not done and reward == 0.0
    assert "exactly one tool call" in obs[0].content
    assert env.status == "pending"


def test_toy_episode_success_end_to_end():
    engine = MockEngine(policy=teacher_policy)
    cfg = StreamConfig(max_steps=6)
    traj = run_episode(engine, ToyOrderEnv(), toy_task("shipped"), cfg, adapter="v0001")
    assert traj.success is True and traj.reward == 1.0
    assert traj.steps == 2
    assert traj.adapter == "v0001"
    assert traj.task_id == "toy-x" and traj.env_family == "toy_order"
    # system, user, assistant, tool, assistant — final obs held in meta only.
    assert [m.role for m in traj.messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert traj.meta["done"] is True and traj.meta["final_obs"] == "Answer recorded."
    assert traj.is_retry is False


def test_toy_episode_student_fails_on_shipped():
    engine = MockEngine(policy=student_policy)
    cfg = StreamConfig(max_steps=6)
    traj = run_episode(engine, ToyOrderEnv(), toy_task("shipped"), cfg, adapter="base")
    assert traj.success is False and traj.reward == 0.0
    assert traj.steps == 2


def test_max_steps_truncation():
    engine = MockEngine(policy=lambda messages: tool_call("get_order_status"))
    cfg = StreamConfig(max_steps=3)
    traj = run_episode(engine, ToyOrderEnv(), toy_task(), cfg, adapter="base")
    assert traj.steps == 3
    assert traj.success is False and traj.reward == 0.0
    assert traj.meta["done"] is True
    assert traj.meta["natural_finish"] is False
    assert traj.meta["forced_settle"] is True
    assert traj.meta["termination_reason"] == "max_steps"
    assert len(traj.messages) == 2 + 3 * 2  # system+user, then (assistant, tool) x3


class PartialProgressEnv:
    env_family = "partial"

    def __init__(self, goal: int = 4):
        self.goal = goal
        self.progress = 0

    def reset(self, task):
        self.progress = 0
        return [Message("system", "sys"), Message("user", "do work")]

    def step(self, action_text):
        self.progress += 1
        return [Message("tool", f"progress={self.progress}")], False, 0.0

    def score_current_state(self):
        return min(1.0, self.progress / self.goal)


def test_max_steps_force_scores_partial_state_and_records_checkpoints():
    engine = MockEngine(policy=lambda messages: tool_call("work"))
    cfg = StreamConfig(max_steps=3, reward_checkpoint_steps=[1, 2, 3])
    traj = run_episode(engine, PartialProgressEnv(goal=4), toy_task(), cfg, adapter="base")
    assert traj.steps == 3
    assert traj.reward == pytest.approx(0.75)
    assert traj.success is False
    assert traj.meta["reward_checkpoints"] == {"1": 0.25, "2": 0.5, "3": 0.75}
    assert traj.meta["termination_step"] == 3


def test_episode_token_budget_stops_before_max_steps_without_overflow():
    engine = MockEngine(policy=lambda messages: tool_call("work"))
    traj = run_episode(
        engine,
        PartialProgressEnv(goal=4),
        toy_task(),
        StreamConfig(max_steps=50, episode_token_budget=12),
        adapter="base",
    )
    assert traj.meta["termination_reason"] == "token_budget"
    assert traj.meta["episode_tokens"] <= 12
    assert traj.steps < 50


def test_max_steps_force_score_can_turn_completed_state_into_success():
    engine = MockEngine(policy=lambda messages: tool_call("work"))
    traj = run_episode(
        engine,
        PartialProgressEnv(goal=3),
        toy_task(),
        StreamConfig(max_steps=3),
        adapter="base",
    )
    assert traj.reward == 1.0 and traj.success is True
    assert traj.meta["forced_settle"] is True


def test_natural_finish_carries_reward_to_later_checkpoints():
    traj = run_episode(
        MockEngine(policy=teacher_policy),
        ToyOrderEnv(),
        toy_task("shipped"),
        StreamConfig(max_steps=6, reward_checkpoint_steps=[1, 2, 4, 6]),
        adapter="base",
    )
    assert traj.steps == 2 and traj.meta["natural_finish"] is True
    assert traj.meta["reward_checkpoints"] == {"1": 0.0, "2": 1.0, "4": 1.0, "6": 1.0}


def test_checkpoint_steps_must_fit_episode_budget():
    with pytest.raises(ValueError, match="reward_checkpoint_steps"):
        run_episode(
            MockEngine(policy=teacher_policy),
            ToyOrderEnv(),
            toy_task(),
            StreamConfig(max_steps=3, reward_checkpoint_steps=[0, 4]),
            adapter="base",
        )


def test_envscaler_rejects_multiple_calls_and_scores_without_mutation(monkeypatch):
    import importlib
    from types import SimpleNamespace

    module = importlib.import_module("sediment.envs.envscaler")

    class FakeRuntime:
        @staticmethod
        def get_state_info(instance):
            return dict(instance)

        @staticmethod
        def calculate_reward(checklist, initial, final):
            del checklist, initial
            return final["progress"] / 2

    class FakeInner:
        def __init__(self, samples, *, max_steps, tool_protocol):
            del samples, max_steps, tool_protocol
            self.env_instance = {"progress": 0}
            self.init_state = {"progress": 0}
            self.checklist_with_func = [1, 2]
            self.step_calls = 0

        def reset(self, index):
            del index
            return {"text": "task"}

        def get_info(self):
            return SimpleNamespace(extra={"system_prompt": "system"})

        def step(self, action):
            del action
            self.step_calls += 1
            self.env_instance["progress"] += 1
            return {"text": "ok", "_obs_type": "tool"}, 0.0, False, {
                "action_executed": True,
            }

    monkeypatch.setattr(
        module,
        "_import_lopd",
        lambda third_party_dir: (FakeInner, lambda *args: [], FakeRuntime),
    )
    adapter = module.EnvScalerAdapter(third_party_dir="unused")
    initial = adapter.reset({"payload": {}})
    assert "exactly one" in initial[0].content

    obs, done, reward = adapter.step(tool_call("a") + tool_call("b"))
    assert not done and reward == 0.0
    assert "exactly one tool call" in obs[0].content
    assert adapter._env.step_calls == 0
    assert adapter.score_current_state() == 0.0

    adapter.step(tool_call("a"))
    before = dict(adapter._env.env_instance)
    assert adapter.score_current_state() == 0.5
    assert adapter._env.env_instance == before


@pytest.mark.parametrize("text", [
    "1. cancel_order({}) -> Error: cannot cancel shipped order",
    f"{EXPERIENCE_OPEN}\npre-tagged block\n{EXPERIENCE_CLOSE}",
])
def test_experience_injected_exactly_once_in_first_user(text: str):
    engine = MockEngine(policy=teacher_policy)
    cfg = StreamConfig(max_steps=6)
    block = ExperienceBlock(text=text, source_task_ids=["toy-0001"])
    traj = run_episode(engine, ToyOrderEnv(), toy_task("shipped"), cfg,
                       adapter="base", experience=block)
    first_user = next(m for m in traj.messages if m.role == "user")
    assert first_user.content.count(EXPERIENCE_OPEN) == 1
    assert first_user.content.count(EXPERIENCE_CLOSE) == 1
    assert first_user.content.endswith("If it cannot be cancelled, explain why.")
    assert first_user.content.index(EXPERIENCE_OPEN) < first_user.content.index("Please cancel")
    total = sum(m.content.count(EXPERIENCE_OPEN) for m in traj.messages)
    assert total == 1
    assert traj.messages[0].content.count(EXPERIENCE_OPEN) == 0  # system untouched
    assert traj.meta["experience_source_task_ids"] == ["toy-0001"]
    assert traj.success is True  # block flips the mock default too, but policy is scripted


def test_no_experience_leaves_user_message_clean():
    engine = MockEngine(policy=teacher_policy)
    traj = run_episode(engine, ToyOrderEnv(), toy_task(), StreamConfig(max_steps=6),
                       adapter="base", experience=None)
    first_user = next(m for m in traj.messages if m.role == "user")
    assert EXPERIENCE_OPEN not in first_user.content
    assert "experience_source_task_ids" not in traj.meta


@pytest.mark.skipif(not lopd_available(), reason="LOPD checkout absent")
def test_envscaler_adapter_imports_and_constructs():
    adapter = EnvScalerAdapter(max_steps=5)
    assert adapter.env_family == "envscaler"
    assert isinstance(adapter, Env)
    with pytest.raises(RuntimeError):
        adapter.step("<tool_call>{}</tool_call>")  # reset() required first


def test_envscaler_adapter_missing_dir_raises(tmp_path):
    assert not lopd_available(tmp_path)
    with pytest.raises(RuntimeError):
        EnvScalerAdapter(third_party_dir=tmp_path)


def test_strip_experience_inverts_inject():
    from sediment.rollout.agent_loop import inject_experience, strip_experience
    from sediment.types import ExperienceBlock, Message
    block = ExperienceBlock(text="<previous_attempts>\nAttempt 1 — FAILED\n</previous_attempts>")
    msgs = [Message("system", "s"), Message("user", inject_experience("do the task", block)),
            Message("assistant", "act"), Message("tool", "ok")]
    bare = strip_experience(msgs)
    assert [m.content for m in bare] == ["s", "do the task", "act", "ok"]
    assert strip_experience(bare) == bare  # idempotent / no-op without a block
