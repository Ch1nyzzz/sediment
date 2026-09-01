import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "prepare_intercode_sql_splits.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_intercode_sql_splits", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _row(index: int, db: str, hardness: str) -> dict:
    return {
        "db": db,
        "gold": f"SELECT {index}",
        "query": f"question {index}",
        "hardness": hardness,
        "db_tables": {},
    }


def test_prepare_is_exact_disjoint_and_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    rows = [
        _row(index, "a" if index < 6 else "b", "easy" if index % 2 else "hard")
        for index in range(10)
    ]
    source.write_text(json.dumps(rows))

    first = MODULE.prepare(source, tmp_path / "one", 4)
    second = MODULE.prepare(source, tmp_path / "two", 4)

    first_train = Path(first["train"]["path"]).read_text()
    second_train = Path(second["train"]["path"]).read_text()
    assert first_train == second_train
    assert first["train"]["rows"] == 4
    assert first["test"]["rows"] == 6
    assert first["db_question_overlap"] == 0
    assert first["train"]["sha256"] == second["train"]["sha256"]


def test_largest_remainder_is_exact() -> None:
    quotas = MODULE.largest_remainder_quotas(
        {("a", "easy"): 6, ("b", "hard"): 4}, 5
    )
    assert quotas == {("a", "easy"): 3, ("b", "hard"): 2}
