import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_intercode_sql_split.py"
SPEC = importlib.util.spec_from_file_location("score_intercode_sql_split", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_score_uses_frozen_source_indices_and_reports_sr10() -> None:
    records = [
        {
            "task_id": 2,
            "hardness": "hard",
            "success": True,
            "success_turn": 11,
            "terminal_reason": "success",
            "transcript_tokens": 500,
            "error": None,
        },
        {
            "task_id": 7,
            "hardness": "easy",
            "success": True,
            "success_turn": 3,
            "terminal_reason": "success",
            "transcript_tokens": 300,
            "error": None,
        },
    ]
    result = MODULE.score(records, [{"source_index": 7, "split": "test"}])
    assert result["by_hardness"]["all"]["success_rate"] == 1.0
    assert result["by_hardness"]["all"]["sr_at_10"] == 1.0
    assert "hard" not in result["by_hardness"]
