"""InterCode-SQL adapter for Sediment's multi-turn environment protocol."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Callable

from sediment.types import Message


SQL_KEYWORDS = ("SHOW", "SELECT", "DESCRIBE", "DESC")

GAME_SQL_SYSTEM = """`SQLEnv` is a multi-turn game that tests your ability to write
a SQL command that produces an output corresponding to a natural language query.

At the start, you receive only a natural-language query and no schema. In each
turn submit exactly one SQL command, then use the returned database output and
reward to improve the next command. The episode succeeds when reward reaches 1.

Format every action as:
```SQL
Your SQL code here
```

Use SHOW TABLES and DESC <table> when necessary. Do not ask questions and do not
invent the environment's output or reward.
"""

RETRY_NO_CODE = """No SQL code was found in your last response.

Your response should be a SQL command formatted as:
```SQL
Your SQL code here
```
"""


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(
                {
                    "task_id": str(row["task_id"]),
                    "env_family": "intercode_sql",
                    "payload": row,
                }
            )
    return rows


def parse_sql_action(response: str) -> tuple[str, bool]:
    action = re.sub(r"\\_", "_", response)

    def before_semicolon(value: str) -> str:
        return value[: value.index(";")] if ";" in value else value

    if "```" not in action:
        if action.startswith("SQL: "):
            return before_semicolon(action[len("SQL: ") :]), True
        for keyword in SQL_KEYWORDS:
            if keyword in action:
                return before_semicolon(action[action.index(keyword) :]), True
    matches = re.findall(r"```(?:sql|SQL)?([\S\s]+?)```", action, re.DOTALL)
    matches += re.findall(r"```([\S\s]+?)```", action, re.DOTALL)
    if not matches:
        return action, False
    action = before_semicolon(" ".join(matches[0].split()))
    words = action.split()
    return action, bool(words and words[0].upper() in SQL_KEYWORDS)


def normalize_gold_sql(sql: str) -> str:
    normalized = re.sub(r"\bFROM\s+SHOW\b", "FROM `SHOW`", sql, flags=re.IGNORECASE)
    stripped = normalized.strip()
    if stripped.upper().startswith("SELECT COUNT(*) FROM (") and stripped.endswith(")"):
        normalized = stripped + " AS derived_table"
    return normalized


def output_iou(agent_rows: list[Any] | None, gold_rows: list[Any]) -> float:
    if agent_rows is None:
        return 0.0
    agent = Counter(str(row) for row in agent_rows)
    gold = Counter(str(row) for row in gold_rows)
    keys = set(agent) | set(gold)
    if not keys:
        return 1.0
    return sum(min(agent[key], gold[key]) for key in keys) / sum(
        max(agent[key], gold[key]) for key in keys
    )


def exact_output_success(agent_rows: list[Any] | None, gold_rows: list[Any]) -> bool:
    return agent_rows is not None and [str(row) for row in agent_rows] == [
        str(row) for row in gold_rows
    ]


def observation_message(query: str, observation: Any, reward: float) -> str:
    if observation == "" or observation == []:
        observation = "No output"
    return (
        f"MySQL Database Output: {observation}\n"
        f"Reward: {reward}\n"
        f'Here is the query again: "{query}"\n'
        "Try something different to generate a SQL command with reward 1.\n"
        "Do not generate any output or reward.\n"
    )


class InterCodeSQLEnv:
    env_family = "intercode_sql"

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 33307,
        user: str = "admin",
        password: str = "admin",
        sql_mode: str = "IGNORE_SPACE",
        connection_factory: Callable[..., Any] | None = None,
    ):
        self.connection_kwargs = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "connection_timeout": 20,
            "autocommit": True,
        }
        self.sql_mode = sql_mode
        self.connection_factory = connection_factory
        self.connection: Any = None
        self.cursor: Any = None
        self.query = ""
        self.gold_rows: list[Any] = []
        self.last_reward = 0.0
        self.last_step_info: dict[str, Any] = {}

    def _connect(self) -> Any:
        if self.connection_factory is not None:
            return self.connection_factory(**self.connection_kwargs)
        import mysql.connector

        return mysql.connector.connect(**self.connection_kwargs)

    def _execute(self, sql: str) -> tuple[list[Any] | None, str | None]:
        try:
            self.cursor.execute(sql)
            rows = self.cursor.fetchall() if self.cursor.description is not None else None
            return rows, None
        except Exception as error:
            try:
                while self.cursor.nextset():
                    pass
            except Exception:
                pass
            return None, f"Error executing query: {error}"

    def reset(self, task: dict[str, Any]) -> list[Message]:
        self.close()
        payload = task.get("payload") or {}
        self.query = str(payload["query"])
        database = str(payload["db"])
        if not re.fullmatch(r"[A-Za-z0-9_]+", database):
            raise ValueError(f"unsafe database identifier: {database!r}")
        self.connection = self._connect()
        self.cursor = self.connection.cursor()
        self.cursor.execute(f"SET SESSION sql_mode = '{self.sql_mode}'")
        self.cursor.execute(f"USE `{database}`")
        gold_rows, gold_error = self._execute(normalize_gold_sql(str(payload["gold"])))
        if gold_error is not None or gold_rows is None:
            raise RuntimeError(f"gold SQL failed for {task.get('task_id')}: {gold_error}")
        self.gold_rows = gold_rows
        self.last_reward = 0.0
        return [
            Message("system", GAME_SQL_SYSTEM),
            Message(
                "user",
                f'Query: "{self.query}".\nDo not generate any output or reward.\n',
            ),
        ]

    def step(self, action_text: str) -> tuple[list[Message], bool, float]:
        action, is_code = parse_sql_action(action_text)
        if not is_code:
            self.last_step_info = {"action_executed": False, "sql": action}
            return [Message("user", RETRY_NO_CODE)], False, self.last_reward
        agent_rows, sql_error = self._execute(action)
        reward = (
            1.0
            if exact_output_success(agent_rows, self.gold_rows)
            else output_iou(agent_rows, self.gold_rows)
        )
        self.last_reward = float(reward)
        self.last_step_info = {
            "action_executed": sql_error is None,
            "sql": action,
            "sql_error": sql_error,
        }
        observation: Any = sql_error if sql_error else agent_rows
        if isinstance(observation, str) and len(observation) > 1000:
            observation = observation[:1000]
        elif isinstance(observation, list) and len(observation) > 50:
            observation = observation[:50]
        done = reward == 1.0
        return [Message("user", observation_message(self.query, observation, reward))], done, reward

    def score_current_state(self) -> float:
        return self.last_reward

    def close(self) -> None:
        if self.cursor is not None:
            try:
                self.cursor.close()
            except Exception:
                pass
        if self.connection is not None:
            try:
                self.connection.close()
            except Exception:
                pass
        self.cursor = None
        self.connection = None

    def __del__(self) -> None:
        self.close()
