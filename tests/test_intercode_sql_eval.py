import importlib.util
from pathlib import Path
import sys
from collections import UserDict


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_intercode_sql_base.py"
SPEC = importlib.util.spec_from_file_location("eval_intercode_sql_base", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_sql_action_matches_intercode_code_fence() -> None:
    action, valid = MODULE.parse_sql_action(
        "I will inspect it.\n```sql\nSHOW TABLES;\n```"
    )
    assert valid is True
    assert action == "SHOW TABLES"


def test_parse_sql_action_rejects_non_sql() -> None:
    action, valid = MODULE.parse_sql_action("Please tell me which tables exist")
    assert valid is False
    assert action == "Please tell me which tables exist"


def test_binary_success_is_order_sensitive() -> None:
    gold = [(1,), (2,)]
    assert MODULE.exact_output_success([(1,), (2,)], gold) is True
    assert MODULE.exact_output_success([(2,), (1,)], gold) is False
    assert MODULE.output_iou([(2,), (1,)], gold) == 1.0


def test_output_iou_counts_duplicate_rows() -> None:
    assert MODULE.output_iou([(1,), (1,)], [(1,), (2,)]) == 1 / 3


def test_count_tokens_handles_transformers_batch_encoding() -> None:
    class FakeTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return UserDict(
                {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]}
            )

    assert MODULE.count_tokens(FakeTokenizer(), []) == 4


def test_normalize_gold_sql_repairs_mysql_conversion_gaps() -> None:
    assert (
        MODULE.normalize_gold_sql("SELECT avg(Attendance) FROM SHOW")
        == "SELECT avg(Attendance) FROM `SHOW`"
    )
    source = "SELECT COUNT(*) FROM (SELECT x FROM a INTERSECT SELECT x FROM b)"
    assert MODULE.normalize_gold_sql(source) == source + " AS derived_table"


def test_fit_message_to_budget_truncates_environment_output() -> None:
    class CharacterTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            size = sum(len(message["content"]) for message in messages)
            return UserDict({"input_ids": list(range(size))})

    prefix = [{"role": "user", "content": "12345"}]
    message = {"role": "user", "content": "abcdefghij"}
    fitted, truncated = MODULE.fit_message_to_budget(
        CharacterTokenizer(), prefix, message, 9
    )
    assert truncated is True
    assert fitted is not None
    assert fitted["content"] == "abcd"


def test_fit_message_to_budget_omits_unaffordable_role_boundary() -> None:
    class RoleTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            size = sum(len(message["content"]) + 3 for message in messages)
            return UserDict({"input_ids": list(range(size))})

    prefix = [{"role": "assistant", "content": "123456789"}]
    fitted, truncated = MODULE.fit_message_to_budget(
        RoleTokenizer(), prefix, {"role": "user", "content": "result"}, 12
    )
    assert truncated is True
    assert fitted is None


def test_primary_protocol_uses_token_budget_not_ten_turn_cap() -> None:
    assert MODULE.MAX_TURNS_DEFAULT == 50
    assert MODULE.DIALOGUE_LIMIT_DEFAULT == 0


def test_completion_capacity_reserves_assistant_role_overhead() -> None:
    class RoleTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            size = sum(len(message["content"]) + 3 for message in messages)
            return UserDict({"input_ids": list(range(size))})

    messages = [{"role": "user", "content": "12345"}]
    assert MODULE.completion_capacity(RoleTokenizer(), messages, 12, 10) == 1


def test_aggregate_retains_official_sr_at_10() -> None:
    records = [
        {
            "hardness": "easy",
            "success": True,
            "success_turn": 9,
            "turns": [{}] * 9,
            "transcript_tokens": 100,
            "terminal_reason": "success",
        },
        {
            "hardness": "easy",
            "success": True,
            "success_turn": 11,
            "turns": [{}] * 11,
            "transcript_tokens": 200,
            "terminal_reason": "success",
        },
    ]

    summary = MODULE.aggregate(records, max_turns=100)

    easy = summary["by_hardness"]["easy"]
    assert easy["official_sr_at_10"] == 0.5
    assert easy["sr_curve"]["100"] == 1.0
