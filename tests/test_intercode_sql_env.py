from pathlib import Path

from sediment.envs.intercode_sql import (
    InterCodeSQLEnv,
    exact_output_success,
    output_iou,
    parse_sql_action,
    read_manifest,
)


class FakeCursor:
    def __init__(self):
        self.description = None
        self.rows = []

    def execute(self, sql):
        if sql.startswith("SET ") or sql.startswith("USE "):
            self.description = None
            self.rows = []
        elif "gold" in sql:
            self.description = True
            self.rows = [(1,), (2,)]
        elif "right" in sql:
            self.description = True
            self.rows = [(1,), (2,)]
        else:
            self.description = True
            self.rows = [(1,)]

    def fetchall(self):
        return self.rows

    def nextset(self):
        return False

    def close(self):
        pass


class FakeConnection:
    def __init__(self):
        self.value = FakeCursor()

    def cursor(self):
        return self.value

    def close(self):
        pass


def test_parser_and_order_sensitive_reward() -> None:
    action, valid = parse_sql_action("```sql\nSELECT right;\n```")
    assert valid is True
    assert action == "SELECT right"
    assert exact_output_success([(1,), (2,)], [(1,), (2,)]) is True
    assert exact_output_success([(2,), (1,)], [(1,), (2,)]) is False
    assert output_iou([(2,), (1,)], [(1,), (2,)]) == 1.0


def test_environment_executes_multi_turn_sql() -> None:
    env = InterCodeSQLEnv(connection_factory=lambda **kwargs: FakeConnection())
    messages = env.reset(
        {
            "task_id": "x",
            "payload": {"db": "db", "query": "question", "gold": "SELECT gold"},
        }
    )
    assert messages[-1].role == "user"
    observation, done, reward = env.step("```SQL\nSELECT wrong;\n```")
    assert done is False
    assert 0 < reward < 1
    assert "Reward:" in observation[0].content
    _, done, reward = env.step("```SQL\nSELECT right;\n```")
    assert done is True
    assert reward == 1.0


def test_manifest_loader_wraps_task_payload(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        '{"task_id":"t","db":"db","query":"q","gold":"SELECT gold"}\n'
    )
    tasks = read_manifest(path)
    assert tasks[0]["env_family"] == "intercode_sql"
    assert tasks[0]["payload"]["query"] == "q"
