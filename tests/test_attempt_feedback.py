from sediment.harness import bare_view
from sediment.stepwise import AttemptFeedbackHinter, FEEDBACK_OPEN
from sediment.types import Message, Trajectory


def _first_attempt():
    return Trajectory(
        task_id="t",
        env_family="intercode_sql",
        messages=[
            Message("system", "sys"),
            Message("user", "question"),
            Message("assistant", "SELECT bad_one"),
            Message("tool", "Error: missing table"),
            Message("assistant", "SELECT bad_two"),
            Message("tool", "wrong rows"),
        ],
        success=False,
    )


def test_feedback_hinter_branches_once_then_follows_redo_state():
    hinter = AttemptFeedbackHinter(_first_attempt())
    redo = [Message("system", "sys"), Message("user", "question")]

    hinter.pre_generate(redo)
    assert FEEDBACK_OPEN in redo[-1].content
    assert "SELECT bad_one" in redo[-1].content
    assert "Error: missing table" in redo[-1].content
    assert "SELECT bad_two" not in redo[-1].content

    redo += [Message("assistant", "SHOW TABLES"), Message("tool", "tables")]
    hinter.pre_generate(redo)
    joined = "\n".join(m.content for m in redo)
    assert "SELECT bad_one" not in joined  # privileged hint was one-turn only
    assert "SELECT bad_two" not in joined  # divergent states are never aligned
    assert "wrong rows" not in joined
    assert "SHOW TABLES" in joined and "tables" in joined  # redo keeps own history
    assert len(hinter.events) == 1


def test_feedback_continues_only_while_redo_prefix_exactly_matches():
    first = _first_attempt()
    hinter = AttemptFeedbackHinter(first)
    redo = [Message("system", "sys"), Message("user", "question")]
    hinter.pre_generate(redo)
    redo += [
        Message("assistant", "SELECT bad_one"),
        Message("tool", "Error: missing table"),
    ]
    hinter.pre_generate(redo)
    assert "SELECT bad_two" in redo[-1].content
    assert "wrong rows" in redo[-1].content
    assert len(hinter.events) == 2


def test_feedback_is_privileged_teacher_context_not_student_input():
    hinter = AttemptFeedbackHinter(_first_attempt())
    redo = [Message("system", "sys"), Message("user", "question")]
    hinter.pre_generate(redo)
    stripped = bare_view(redo)
    assert stripped == [Message("system", "sys"), Message("user", "question")]
