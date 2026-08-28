"""Harness arms: bare view, working state, call-time hints (08-27)."""
from __future__ import annotations

import json

from sediment.calltime import CallTimeHinter
from sediment.config import StreamConfig
from sediment.engine.mock import MockEngine
from sediment.engine.vllm_client import _bare_prompt_seed
from sediment.envs import ToyOrderEnv
from sediment.harness import (
    HINT_INTERCEPT_OPEN,
    HINT_OPEN,
    STATE_OPEN,
    bare_view,
    is_error_obs,
    is_write_tool,
)
from sediment.rollout import run_episode
from sediment.types import Message, Trajectory
from sediment.working_state import WorkingState, extract_goals


def tool_call(name: str, arguments: dict | None = None) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments or {}}) + "\n</tool_call>"


def toy_task(status: str = "shipped", task_id: str = "toy-x") -> dict:
    return {"task_id": task_id, "env_family": "toy_order",
            "payload": {"status": status, "order_id": 7}}


def donor(success: bool = True) -> Trajectory:
    msgs = [Message("system", "s"), Message("user", "Please cancel order #3."),
            Message("assistant", tool_call("cancel_order")),
            Message("tool", "Error: cannot cancel shipped order"),
            Message("assistant", tool_call("get_order_status")),
            Message("tool", "Order status: shipped"),
            Message("assistant", tool_call("answer", {"text": "Cannot cancel: shipped."}))]
    return Trajectory(task_id="toy-donor", env_family="toy_order", messages=msgs,
                      reward=1.0 if success else 0.0, success=success, meta={
                          "reflection": "- Check the order status before cancelling. "
                                        "- Shipped orders cannot be cancelled; explain why."})


# -- harness.bare_view -------------------------------------------------------

def test_classifiers():
    assert is_write_tool("update_meal_record") and is_write_tool("cancel_order")
    assert not is_write_tool("get_user_by_id") and not is_write_tool("list_all") \
        and not is_write_tool("answer") and not is_write_tool("")
    assert is_error_obs("Error: 'foo' is not a valid tool")
    assert is_error_obs("{'success': False, 'error': 'Subscription not found'}")
    assert not is_error_obs("{'success': True, 'data': {'error_count': 0}}")


def test_bare_view_removes_every_harness_edit_and_is_idempotent():
    msgs = [
        Message("system", "s"),
        Message("user", "<previous_attempts>\nblock\n</previous_attempts>\n\ndo the task"),
        Message("assistant", tool_call("cancel_order")),  # intercepted draft
        Message("user", f"{HINT_INTERCEPT_OPEN}\nhint\n</experience_hint>"),
        Message("assistant", tool_call("get_order_status")),
        Message("tool", f"Order status: shipped\n\n{HINT_OPEN}\nerr hint\n</experience_hint>\n\n"
                        f"{STATE_OPEN}\n1. [pending] x\n</working_state>"),
        Message("assistant", "Task Completed"),
    ]
    bare = bare_view(msgs)
    assert [(m.role, m.content) for m in bare] == [
        ("system", "s"), ("user", "do the task"),
        ("assistant", tool_call("get_order_status")), ("tool", "Order status: shipped"),
        ("assistant", "Task Completed")]
    assert bare_view(bare) == bare
    # the seed the engine uses ignores every harness edit
    assert _bare_prompt_seed(msgs) == _bare_prompt_seed(bare)


# -- working state -----------------------------------------------------------

TASK = ("Handle Alice's account:\n- Cancel the pending order #7 for USR001.\n"
        "- After cancelling:\n  - Confirm the cancellation to the customer.\n- Do not touch other orders.")


def test_goals_from_bullets_skip_headers():
    goals = extract_goals(TASK)
    assert goals == ["Cancel the pending order #7 for USR001.",
                     "Confirm the cancellation to the customer.", "Do not touch other orders."]
    assert extract_goals("Cancel order seven. Then confirm it to the customer.") == [
        "Cancel order seven.", "Then confirm it to the customer."]


def test_working_state_flips_only_on_successful_write():
    ws = WorkingState(TASK)
    ws.record(1, {"name": "get_order_status", "arguments": {}}, "Order status: pending", True)
    assert ws.done == {}  # reads never complete a goal
    ws.record(2, {"name": "cancel_order", "arguments": {"order_id": "USR001"}},
              "Error: cannot cancel shipped order", False)
    assert ws.done == {}  # failed writes never complete a goal
    ws.record(3, {"name": "cancel_order", "arguments": {}}, "Order cancelled.", True)
    assert ws.done == {0: "step 3 cancel_order"}
    text = ws.render()
    assert text.startswith(STATE_OPEN) and "[done: step 3 cancel_order] Cancel" in text
    assert "[pending] Confirm" in text and "step 2 cancel_order -> Error" in text
    assert "Goals without a supporting receipt yet: 2, 3." in text
    assert ws.summary() == {"goals": 3, "done": 1, "calls": 3, "failed": 1}


def test_working_state_is_appended_to_observations_only():
    engine = MockEngine(policy=lambda ms: tool_call("get_order_status"))
    ws = WorkingState("Please cancel order #7. If it cannot be cancelled, explain why.")
    traj = run_episode(engine, ToyOrderEnv(), toy_task(), StreamConfig(max_steps=2),
                       adapter="base", working_state=ws)
    assert traj.steps == 2
    assert STATE_OPEN not in traj.messages[1].content  # first user message untouched
    tools = [m for m in traj.messages if m.role == "tool"]
    assert STATE_OPEN not in tools[0].content and STATE_OPEN in tools[1].content  # newest only
    assert traj.meta["working_state"]["calls"] == 2
    assert bare_view(traj.messages)[3].content == "Order status: shipped"


def test_value_match_beats_verb_match():
    ws = WorkingState("- Rename “Greek Yogurt Breakfast” to “Greek Yogurt with Honey” and set calories to 210.6.\n"
                      "- Update the calories of “Baked Tofu Wrap” to 495.\n- Delete the “Pancake Stack”.")
    ws.record(1, {"name": "update_meal_record", "arguments": {
        "record_id": "MR1", "meal_name": "Greek Yogurt with Honey", "calories": 210.6}}, "ok", True)
    assert ws.done == {0: "step 1 update_meal_record"}
    ws.record(2, {"name": "update_meal_record", "arguments": {"record_id": "MR2", "calories": 495}}, "ok", True)
    assert ws.done == {0: "step 1 update_meal_record", 1: "step 2 update_meal_record"}
    ws.record(3, {"name": "delete_meal_record", "arguments": {"record_id": "MR3"}}, "ok", True)
    assert 2 in ws.done  # verb stem: delete ~ Delete
    ws2 = WorkingState("\n".join(f"- Goal number {i} is a fairly long sentence about records." for i in range(12)))
    assert len(ws2.goals) == 12 and all(f"{i + 1}. [pending]" in ws2.render() for i in range(12))


# -- call-time hints ---------------------------------------------------------

def hinted_policy(messages: list[Message]) -> str:
    """Cancels blindly; after an intercept checks status; then answers."""
    last = messages[-1]
    if last.role == "user" and last.content.startswith(HINT_INTERCEPT_OPEN):
        return tool_call("get_order_status")
    if not any(m.role == "assistant" for m in messages):
        return tool_call("cancel_order")
    last_tool = next((m.content for m in reversed(messages) if m.role == "tool"), "")
    if "pending" in last_tool:
        return tool_call("cancel_order")
    if "cancelled" in last_tool:
        return tool_call("answer", {"text": "The order has been cancelled."})
    return tool_call("answer", {"text": "It cannot be cancelled: already shipped."})


def test_write_intercept_holds_the_draft_and_redraft_is_logged():
    hinter = CallTimeHinter([donor()], max_hints=4)
    traj = run_episode(MockEngine(policy=hinted_policy), ToyOrderEnv(), toy_task("shipped"),
                       StreamConfig(max_steps=6), adapter="base", hinter=hinter)
    assert traj.success is True and traj.steps == 2  # the intercepted draft is not a step
    roles = [m.role for m in traj.messages]
    assert roles == ["system", "user", "assistant", "user", "assistant", "tool", "assistant"]
    hint = traj.messages[3].content
    assert hint.startswith(HINT_INTERCEPT_OPEN) and "`cancel_order` was NOT executed" in hint
    assert "cancel_order({}) -> Error: cannot cancel shipped order" in hint
    assert "(recovery): get_order_status" in hint and "Lesson from that task" in hint
    assert "do not copy identifiers" in hint
    ev = traj.meta["calltime"]["events"]
    assert len(ev) == 1 and ev[0]["trigger"] == "write" and ev[0]["tool"] == "cancel_order"
    assert ev[0]["changed"] is True and ev[0]["redraft_name"] == "get_order_status"
    assert ev[0]["donor"] == "toy-donor" and ev[0]["donor_success"] is True
    assert [m.role for m in bare_view(traj.messages)] == ["system", "user", "assistant", "tool", "assistant"]


def test_error_trigger_appends_to_observation_without_intercept():
    hinter = CallTimeHinter([donor()], triggers="error", max_hints=4)
    policy = lambda ms: tool_call("cancel_order") if sum(  # noqa: E731
        m.role == "assistant" for m in ms) < 2 else tool_call("answer", {"text": "no"})
    traj = run_episode(MockEngine(policy=policy), ToyOrderEnv(), toy_task("shipped"),
                       StreamConfig(max_steps=6), adapter="base", hinter=hinter)
    assert traj.steps == 3
    obs = traj.messages[3]
    assert obs.role == "tool" and obs.content.startswith("Error: cannot cancel shipped order")
    assert f"\n\n{HINT_OPEN}\nA related task hit a similar failure (outcome SUCCESS" in obs.content
    # the same error on the same tool is hinted once; budget is per episode
    assert HINT_OPEN not in traj.messages[5].content
    assert [e["trigger"] for e in traj.meta["calltime"]["events"]] == ["error"]
    assert bare_view(traj.messages)[3].content == "Error: cannot cancel shipped order"


def test_hint_budget_and_render_cap():
    hinter = CallTimeHinter([donor()], triggers="write", max_hints=1, max_chars=400)
    text = hinter.on_draft(tool_call("cancel_order"), "")
    assert text is not None and len(text) <= 400 and "Lesson" not in text
    assert hinter.exhausted
    assert hinter.on_draft(tool_call("cancel_order"), "") is None  # the re-draft passes through
    assert hinter.on_draft(tool_call("update_x"), "") is None  # budget spent


def test_confirm_style_asks_for_the_write_back():
    hinter = CallTimeHinter([donor()], style="confirm")
    text = hinter.on_draft(tool_call("cancel_order"), "")
    assert text.startswith(HINT_INTERCEPT_OPEN) and "is queued, not yet executed" in text
    assert text.rstrip().endswith("Re-issue `cancel_order` now, with the same arguments or corrected ones.\n</experience_hint>")


def test_no_donors_means_no_hints():
    hinter = CallTimeHinter([])
    assert hinter.on_draft(tool_call("cancel_order"), "") is None
    assert hinter.on_observation({"name": "x"}, "Error: nope") is None
