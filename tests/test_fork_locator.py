from sediment.scheduler import _divergence_index, _divergence_pair
from sediment.types import Message, Trajectory


def _traj(task_id: str, assistant: list[str]) -> Trajectory:
    messages = [Message("user", "task")]
    for i, content in enumerate(assistant):
        messages.append(Message("assistant", content))
        messages.append(Message("tool", f"observation {i}"))
    return Trajectory(task_id=task_id, env_family="toy", messages=messages)


def _call(name: str, arguments: str, reasoning: str = "") -> str:
    return (reasoning + "<tool_call>" +
            '{"name":"' + name + '","arguments":' + arguments + "}" +
            "</tool_call>")


def test_default_fork_ignores_reasoning_and_json_key_order():
    first = _traj("x", [
        'old wording <tool_call>{"name":"read","arguments":{"b":2,"a":1}}</tool_call>'
    ])
    redo = _traj("x", [
        'new wording <tool_call>{"arguments":{"a":1,"b":2},"name":"read"}</tool_call>'
    ])
    assert _divergence_index(first, redo) is None


def test_default_fork_is_first_parsed_tool_difference():
    first = _traj("x", [_call("read", '{"path":"a"}'),
                         _call("write", '{"path":"b"}')])
    redo = _traj("x", [_call("read", '{"path":"a"}', "different reasoning "),
                        _call("write", '{"path":"c"}')])
    # user=0, assistant/tool pairs put the second redo assistant at message 3
    assert _divergence_index(first, redo) == 3


def test_narration_only_turn_does_not_shift_tool_call_ordinal():
    first = _traj("x", [_call("read", '{"path":"a"}')])
    redo = _traj("x", ["I should inspect first.", _call("read", '{"path":"a"}')])
    assert _divergence_index(first, redo) is None


def test_legacy_text_locator_remains_explicit_only():
    first = _traj("x", [_call("read", '{"path":"a"}', "old ")])
    redo = _traj("x", [_call("read", '{"path":"a"}', "new ")])
    assert _divergence_index(first, redo, "text") == 1


def test_sql_locator_compares_executable_sql_not_fence_or_narration():
    first = _traj("x", [
        "I will inspect.\n```sql\nSHOW TABLES;\n```",
        "```SQL\nSELECT wrong FROM singer;\n```",
    ])
    redo = _traj("x", [
        "```SQL\nSHOW TABLES;\n```",
        "Use the discovered schema.\n```sql\nSELECT name FROM singer;\n```",
    ])
    assert _divergence_index(first, redo, "sql") == 3


def test_sql_locator_pairs_first_answer_with_final_teacher_correction():
    first = _traj("x", [
        "```SQL\nSHOW TABLES;\n```",
        "```SQL\nSELECT wrong FROM singer;\n```",
        "```SQL\nSELECT still_wrong FROM singer;\n```",
    ])
    redo = _traj("x", [
        "```SQL\nSHOW TABLES;\n```",
        "```SQL\nSELECT teacher_wrong FROM singer;\n```",
        "```SQL\nDESC singer;\n```",
        "```SQL\nSELECT name FROM singer;\n```",
    ])
    # user=0, assistant/tool pairs put the first student answer at message 3
    # and the final teacher SQL at message 7.
    assert _divergence_pair(first, redo, "sql") == (3, 7)
