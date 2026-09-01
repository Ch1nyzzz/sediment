from sediment.scheduler import _apply_turn_decay
from sediment.types import Message, TrainSample


def test_turn_decay_scales_supervised_assistant_turns_only():
    msgs = [Message(role="system", content="s"), Message(role="user", content="u"),
            Message(role="assistant", content="a0"), Message(role="tool", content="o0"),
            Message(role="assistant", content="a1"), Message(role="tool", content="o1"),
            Message(role="assistant", content="a2")]
    ws = [[], [], [1.0, 1.0], [0.0], [1.0], [0.0], [1.0, 1.0]]
    s = _apply_turn_decay(TrainSample(task_id="t", messages=msgs, token_weights_by_msg=ws), 0.5)
    assert s.token_weights_by_msg[2] == [1.0, 1.0]
    assert s.token_weights_by_msg[4] == [0.5]
    assert s.token_weights_by_msg[6] == [0.25, 0.25]
    assert s.token_weights_by_msg[3] == [0.0]


def test_turn_decay_one_is_identity():
    msgs = [Message(role="assistant", content="a"), Message(role="assistant", content="b")]
    s = _apply_turn_decay(TrainSample(task_id="t", messages=msgs, token_weights_by_msg=[[1.0], [1.0]]), 1.0)
    assert s.token_weights_by_msg == [[1.0], [1.0]]
