from sediment.scheduler import _stepwise_feedback_samples
from sediment.types import Message, Trajectory


def test_stepwise_feedback_trains_reasoning_and_action_for_success_and_decay():
    first = "Reason carefully, then <tool_call>{\"name\":\"search\",\"arguments\":{\"q\":\"x\"}}</tool_call>"
    final = "The evidence supports the verified answer."
    messages = [
        Message("system", "sys"),
        Message("user", "task"),
        Message("assistant", first),
        Message("tool", f"Executed {first}; found evidence"),
        Message("assistant", final),
    ]
    behavior = [[-0.1] for _ in messages]
    behavior[2] = [-0.2, -0.3, -0.4]
    behavior[4] = [-0.5, -0.6]
    traj = Trajectory(
        task_id="t", env_family="e", messages=messages, reward=1.0, success=True,
        meta={"final_obs": "Answer accepted by verifier."},
    )

    samples, diag = _stepwise_feedback_samples(traj, behavior, 0.5)

    assert len(samples) == 2
    assert diag["supervised_turns"] == 2
    assert diag["action_echo_redactions"] == 1
    assert samples[0].task_id == "t@turn00"
    assert samples[0].messages[-1].content == first
    assert samples[0].token_weights_by_msg[-1] == [1.0]  # whole assistant turn
    assert all(not ws for ws in samples[0].token_weights_by_msg[:-1])
    assert samples[0].behavior_logprobs_by_msg[-1] == behavior[2]
    assert first not in samples[0].teacher_contexts["feedback"]
    assert "found evidence" in samples[0].teacher_contexts["feedback"]
    assert "Verified trajectory outcome" not in samples[0].teacher_contexts["feedback"]
    assert samples[0].loss_scale == 1.0

    assert samples[1].messages[-1].content == final
    assert samples[1].messages[2].content == first  # previous action is causal history
    assert final not in samples[1].teacher_contexts["feedback"]
    assert "SUCCEEDED" in samples[1].teacher_contexts["feedback"]
    assert "Answer accepted" in samples[1].teacher_contexts["feedback"]
    assert samples[1].loss_scale == 0.5


def test_stepwise_feedback_skips_unexecuted_assistant_draft():
    messages = [
        Message("system", "sys"), Message("user", "task"),
        Message("assistant", "draft"), Message("assistant", "executed"),
        Message("tool", "observation"),
    ]
    behavior = [[-0.1] for _ in messages]
    traj = Trajectory(task_id="t", env_family="e", messages=messages, success=False)

    samples, diag = _stepwise_feedback_samples(traj, behavior, 0.9)

    assert [s.messages[-1].content for s in samples] == ["executed"]
    # Decay follows the original assistant-turn index even when an unexecuted
    # draft is not a training pair.
    assert samples[0].loss_scale == 0.9
    assert diag["assistant_turns"] == 2
