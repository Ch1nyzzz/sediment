#!/usr/bin/env python3
"""Evaluate an OpenAI-compatible base model on InterCode-SQL.

The primary protocol treats ``game_sql`` as the multi-turn task it is:

* Spider dev, without the schema handicap.
* Greedy decoding with the full action/observation history.
* An 8192-token episode budget; ``max_turns`` is only a safety valve.
* Execution-output success against the gold SQL result.

The summary retains SR@10 for comparison with the public InterCode horizon.
Results are appended as JSONL so a stopped run can be resumed without repeating
completed tasks.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import statistics
import time
from typing import Any


MODEL_DEFAULT = "Qwen/Qwen3-4B-Instruct-2507"
ENDPOINTS_DEFAULT = (
    "http://127.0.0.1:8124/v1",
    "http://127.0.0.1:8125/v1",
    "http://127.0.0.1:8126/v1",
    "http://127.0.0.1:8127/v1",
)
SQL_KEYWORDS = ("SHOW", "SELECT", "DESCRIBE", "DESC")
CURVE_TURNS = (1, 2, 4, 6, 8, 10)
MAX_TURNS_DEFAULT = 50
DIALOGUE_LIMIT_DEFAULT = 0


GAME_SQL_SYSTEM = """`SQLEnv` is a multi-turn game that tests your ability to write
a SQL command that produces an output corresponding to a natural language query.

## GAME DESCRIPTION
At the start of this game, you are given a natural language query describing some
desired output (i.e. "Find the first name of a student who have both cat and dog pets").
Aside from the natural language query, you have no information about the tables you have access to.

The game will be played in a series of turns. Each turn, you can submit a SQL command.
You will then get a response detailing the output of your SQL query along with a reward
that tells you how close your SQL command is to the correct answer.

The goal of this game is to write a SQL command that gets a reward of 1. The game will automatically
terminate once you get a reward of 1.

## INPUT DESCRIPTION
Each turn, you can submit a SQL command. Your SQL command should be formatted as follows:

```SQL
Your SQL code here
```

Your SQL command can help you do one of two things:
1. Learn more about the tables you have access to
2. Execute SQL commands based on these tables to generate the correct output.

## OUTPUT DESCRIPTION
Given your SQL command input, `SQLEnv` will then give back output formatted as follows:

Output: <string>
Reward: <decimal value between 0 and 1>

The output is a string displaying the result from executing your SQL query.
The reward is a decimal value between 0 and 1.

## REWARD DESCRIPTION
The reward should be interpreted as a ratio. It tells you how many rows your SQL
command outputted correctly compared to the correct answer.

## RULES
1. Do NOT ask questions. Your commands are fed directly into a SQL compiler.

## STRATEGY
You are free to play as many turns of the game as you'd like to inspect tables
and develop your SQL command.

The best strategy for this game is to first write SQL commands that help you learn
about the tables that you have access to. For instance, in a SQL environment, you might use `SHOW TABLES`
and `DESC <table name>` to learn more about the tables you have access to.

Once you have a good understanding of the tables, you should then write SQL commands
that would answer the natural language query using the tables you have access to.
"""


RETRY_NO_CODE = """No SQL code was found in your last response.

Your response should be a SQL command. Format your SQL command as follows:

```SQL
Your SQL code here
```
"""


def query_message(query: str) -> str:
    return f'Query: "{query}".\nDo not generate any output or reward.\n'


def observation_message(query: str, observation: Any, reward: float) -> str:
    if observation == "" or observation == []:
        observation = "No output"
    return (
        f"MySQL Database Output: {observation}\n"
        f"Reward: {reward}\n"
        f'Here is the query again: "{query}"\n'
        "Try something different to generate SQL command to get a reward of 1.\n"
        "Do not generate any output or reward.\n"
    )


def parse_sql_action(response: str) -> tuple[str, bool]:
    """Match the public InterCode SQL parser, including its guardrails."""
    action = re.sub(r"\\_", "_", response)

    def before_semicolon(value: str) -> str:
        return value[: value.index(";")] if ";" in value else value

    if "```" not in action:
        if action.startswith("SQL: "):
            return before_semicolon(action[len("SQL: ") :]), True
        for keyword in SQL_KEYWORDS:
            if keyword in action:
                return before_semicolon(action[action.index(keyword) :]), True

    pattern1 = r"```(?:sql|SQL)?([\S\s]+?)```"
    pattern2 = r"```([\S\s]+?)```"
    matches = re.findall(pattern1, action, re.DOTALL) + re.findall(
        pattern2, action, re.DOTALL
    )
    if not matches:
        return action, False
    action = " ".join(matches[0].split())
    action = before_semicolon(action)
    words = action.split()
    if not words or words[0].upper() not in SQL_KEYWORDS:
        return action, False
    return action, True


def output_iou(agent_rows: list[Any] | None, gold_rows: list[Any]) -> float:
    """InterCode's multiset row IoU, without the order scaling."""
    if agent_rows is None:
        return 0.0
    agent = Counter(str(row) for row in agent_rows)
    gold = Counter(str(row) for row in gold_rows)
    keys = set(agent) | set(gold)
    if not keys:
        return 1.0
    intersection = sum(min(agent[key], gold[key]) for key in keys)
    union = sum(max(agent[key], gold[key]) for key in keys)
    return intersection / union


def exact_output_success(agent_rows: list[Any] | None, gold_rows: list[Any]) -> bool:
    """Binary condition for InterCode reward 1, including output ordering."""
    if agent_rows is None:
        return False
    return [str(row) for row in agent_rows] == [str(row) for row in gold_rows]


def normalize_gold_sql(sql: str) -> str:
    """Repair deterministic SQLite-to-MySQL gaps in the released Spider dump."""
    normalized = re.sub(r"\bFROM\s+SHOW\b", "FROM `SHOW`", sql, flags=re.IGNORECASE)
    stripped = normalized.strip()
    if stripped.upper().startswith("SELECT COUNT(*) FROM (") and stripped.endswith(")"):
        normalized = stripped + " AS derived_table"
    return normalized


def percentile(values: list[int], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True)
class Config:
    model: str
    endpoints: tuple[str, ...]
    max_turns: int
    dialogue_limit: int
    max_tokens: int
    token_budget: int
    concurrency: int
    request_timeout: float
    mysql_host: str
    mysql_port: int
    mysql_user: str
    mysql_password: str
    mysql_sql_mode: str


def count_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if hasattr(ids, "shape"):
        return int(ids.shape[-1])
    if ids and isinstance(ids[0], list):
        return len(ids[0])
    return len(ids)


def fit_message_to_budget(
    tokenizer: Any,
    prefix: list[dict[str, str]],
    message: dict[str, str],
    token_budget: int,
) -> tuple[dict[str, str] | None, bool]:
    """Truncate an environment message so the canonical transcript fits."""
    if count_tokens(tokenizer, prefix + [message]) <= token_budget:
        return message, False

    empty_message = {"role": message["role"], "content": ""}
    if count_tokens(tokenizer, prefix + [empty_message]) > token_budget:
        # The role boundary alone does not fit.  Ending the episode without an
        # observation is preferable to silently exceeding the strict budget.
        return None, True

    content = message["content"]
    low, high = 0, len(content)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = {"role": message["role"], "content": content[:middle]}
        if count_tokens(tokenizer, prefix + [candidate]) <= token_budget:
            low = middle
        else:
            high = middle - 1
    fitted = {"role": message["role"], "content": content[:low]}
    # Token counts are not guaranteed to be monotone over character prefixes
    # for every BPE tokenizer.  Back off defensively if a merge changed at the
    # selected boundary.
    while fitted["content"] and count_tokens(tokenizer, prefix + [fitted]) > token_budget:
        fitted = {"role": message["role"], "content": fitted["content"][:-1]}
    if count_tokens(tokenizer, prefix + [fitted]) > token_budget:
        return None, True
    return fitted, True


def completion_capacity(
    tokenizer: Any,
    messages: list[dict[str, str]],
    token_budget: int,
    per_turn_limit: int,
) -> int:
    """Reserve the assistant-role boundary before setting max new tokens."""
    prompt_tokens = count_tokens(tokenizer, messages)
    with_empty_assistant = count_tokens(
        tokenizer, messages + [{"role": "assistant", "content": ""}]
    )
    role_overhead = max(0, with_empty_assistant - prompt_tokens)
    return max(0, min(per_turn_limit, token_budget - prompt_tokens - role_overhead))


def trim_dialogue(messages: list[dict[str, str]], dialogue_limit: int) -> None:
    # Mirrors: dialogue[:2] + dialogue[-dialogue_limit:] in InterCode.
    if dialogue_limit and len(messages) - 2 > dialogue_limit:
        messages[:] = messages[:2] + messages[-dialogue_limit:]


async def chat_completion(
    session: Any,
    endpoint: str,
    config: Config,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> tuple[str, dict[str, int]]:
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            async with session.post(
                f"{endpoint}/chat/completions",
                json=payload,
                timeout=config.request_timeout,
            ) as response:
                body = await response.text()
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {body[:500]}")
                data = json.loads(body)
                content = data["choices"][0]["message"]["content"] or ""
                usage = data.get("usage") or {}
                return content.strip(), {
                    "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(usage.get("completion_tokens", 0)),
                }
        except Exception as error:  # network/server errors are retryable here
            last_error = error
            await asyncio.sleep(min(2**attempt, 8))
    raise RuntimeError(f"chat request failed after retries: {last_error}")


def open_db(config: Config) -> Any:
    import mysql.connector

    return mysql.connector.connect(
        host=config.mysql_host,
        port=config.mysql_port,
        user=config.mysql_user,
        password=config.mysql_password,
        connection_timeout=20,
        autocommit=True,
    )


def execute(cursor: Any, sql: str) -> tuple[list[Any] | None, str | None]:
    try:
        cursor.execute(sql)
        rows = cursor.fetchall() if cursor.description is not None else None
        return rows, None
    except Exception as error:
        message = getattr(error, "msg", str(error))
        return None, f"Error executing query: {message}"


async def evaluate_record(
    task_id: int,
    record: dict[str, Any],
    endpoint: str,
    session: Any,
    tokenizer: Any,
    config: Config,
) -> dict[str, Any]:
    started = time.monotonic()
    policy_messages = [
        {"role": "system", "content": GAME_SQL_SYSTEM},
        {"role": "user", "content": query_message(record["query"])},
    ]
    full_messages = [dict(message) for message in policy_messages]
    turns: list[dict[str, Any]] = []
    success_turn: int | None = None
    terminal_reason = "max_turns"
    api_error: str | None = None
    peak_request_tokens = 0
    final_transcript_tokens = count_tokens(tokenizer, full_messages)

    connection = open_db(config)
    cursor = connection.cursor(buffered=True)
    try:
        cursor.execute("SET SESSION sql_mode = %s", (config.mysql_sql_mode,))
        cursor.execute(f"USE `{record['db'].replace('`', '``')}`")
        normalized_gold = normalize_gold_sql(record["gold"])
        gold_rows, gold_error = execute(cursor, normalized_gold)
        if gold_error or gold_rows is None:
            return {
                "task_id": task_id,
                "hardness": record["hardness"],
                "db": record["db"],
                "query": record["query"],
                "success": False,
                "success_turn": None,
                "terminal_reason": "gold_error",
                "error": gold_error or "gold query returned no tabular output",
                "turns": [],
                "transcript_tokens": final_transcript_tokens,
                "peak_request_tokens": 0,
                "elapsed_seconds": time.monotonic() - started,
                "endpoint": endpoint,
                "gold_normalized": normalized_gold != record["gold"],
            }

        for turn in range(1, config.max_turns + 1):
            generation_limit = completion_capacity(
                tokenizer,
                full_messages,
                config.token_budget,
                config.max_tokens,
            )
            if generation_limit <= 0:
                terminal_reason = "token_budget"
                break
            try:
                response, usage = await chat_completion(
                    session,
                    endpoint,
                    config,
                    policy_messages,
                    generation_limit,
                )
            except Exception as error:
                api_error = str(error)
                terminal_reason = "api_error"
                break

            peak_request_tokens = max(
                peak_request_tokens,
                usage["prompt_tokens"] + usage["completion_tokens"],
            )
            assistant_message, response_budget_truncated = fit_message_to_budget(
                tokenizer,
                full_messages,
                {"role": "assistant", "content": response},
                config.token_budget,
            )
            if response_budget_truncated:
                if assistant_message is not None:
                    full_messages.append(assistant_message)
                final_transcript_tokens = count_tokens(tokenizer, full_messages)
                turns.append(
                    {
                        "turn": turn,
                        "response": (
                            assistant_message["content"]
                            if assistant_message is not None
                            else ""
                        ),
                        "action": "",
                        "is_code": False,
                        "valid_action": False,
                        "observation": "",
                        "reward": 0.0,
                        "row_iou": 0.0,
                        "request_usage": usage,
                        "transcript_tokens": final_transcript_tokens,
                        "sql_error": None,
                        "response_truncated_for_budget": True,
                        "observation_truncated_for_budget": False,
                    }
                )
                terminal_reason = "token_budget"
                break
            policy_messages.append(assistant_message)
            full_messages.append(assistant_message)
            action, is_code = parse_sql_action(response)

            if not is_code:
                observation: Any = RETRY_NO_CODE
                reward = 0.0
                row_iou = 0.0
                valid_action = False
                sql_error = None
            else:
                agent_rows, sql_error = execute(cursor, action)
                observation = sql_error if sql_error else agent_rows
                valid_action = sql_error is None
                row_iou = output_iou(agent_rows, gold_rows)
                reward = 1.0 if exact_output_success(agent_rows, gold_rows) else row_iou

            prompt_observation = observation
            if isinstance(prompt_observation, str) and len(prompt_observation) > 1000:
                prompt_observation = prompt_observation[:1000]
            elif isinstance(prompt_observation, list) and len(prompt_observation) > 50:
                prompt_observation = prompt_observation[:50]
            obs_message = observation_message(
                record["query"], prompt_observation, reward
            )
            fitted_obs_message, budget_truncated = fit_message_to_budget(
                tokenizer,
                full_messages,
                {"role": "user", "content": obs_message},
                config.token_budget,
            )
            if fitted_obs_message is not None:
                full_messages.append(fitted_obs_message)
            final_transcript_tokens = count_tokens(tokenizer, full_messages)
            turns.append(
                {
                    "turn": turn,
                    "response": response,
                    "action": action,
                    "is_code": is_code,
                    "valid_action": valid_action,
                    "observation": str(observation),
                    "reward": reward,
                    "row_iou": row_iou,
                    "request_usage": usage,
                    "transcript_tokens": final_transcript_tokens,
                    "sql_error": sql_error,
                    "observation_truncated_for_budget": budget_truncated,
                }
            )

            if reward == 1.0:
                success_turn = turn
                terminal_reason = "success"
                break
            if fitted_obs_message is None:
                terminal_reason = "token_budget"
                break
            if final_transcript_tokens >= config.token_budget:
                terminal_reason = "token_budget"
                break

            policy_messages.append(fitted_obs_message)
            trim_dialogue(policy_messages, config.dialogue_limit)
    finally:
        cursor.close()
        connection.close()

    return {
        "task_id": task_id,
        "hardness": record["hardness"],
        "db": record["db"],
        "query": record["query"],
        "success": success_turn is not None,
        "success_turn": success_turn,
        "terminal_reason": terminal_reason,
        "error": api_error,
        "turns": turns,
        "transcript_tokens": final_transcript_tokens,
        "peak_request_tokens": peak_request_tokens,
        "elapsed_seconds": time.monotonic() - started,
        "endpoint": endpoint,
        "gold_normalized": normalized_gold != record["gold"],
    }


def aggregate(records: list[dict[str, Any]], max_turns: int) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["hardness"]].append(record)
        groups["all"].append(record)

    turn_points = sorted(set(turn for turn in CURVE_TURNS if turn <= max_turns) | {max_turns})
    summary: dict[str, Any] = {"by_hardness": {}}
    for hardness in ("easy", "medium", "hard", "extra", "all"):
        rows = groups.get(hardness, [])
        successes = [row for row in rows if row["success"]]
        tokens = [int(row["transcript_tokens"]) for row in rows]
        turns_taken = [len(row["turns"]) for row in rows]
        summary["by_hardness"][hardness] = {
            "tasks": len(rows),
            "successes": len(successes),
            "success_rate": len(successes) / len(rows) if rows else None,
            "sr_curve": {
                str(turn): (
                    sum(
                        row["success_turn"] is not None
                        and row["success_turn"] <= turn
                        for row in rows
                    )
                    / len(rows)
                    if rows
                    else None
                )
                for turn in turn_points
            },
            "mean_turns": statistics.fmean(turns_taken) if turns_taken else None,
            "transcript_tokens_p50": percentile(tokens, 0.50),
            "transcript_tokens_p95": percentile(tokens, 0.95),
            "transcript_tokens_max": max(tokens) if tokens else None,
            "token_budget_terminations": sum(
                str(row["terminal_reason"]).startswith("token_budget") for row in rows
            ),
            "api_errors": sum(row["terminal_reason"] == "api_error" for row in rows),
            "gold_errors": sum(row["terminal_reason"] == "gold_error" for row in rows),
        }
        if max_turns >= 10:
            summary["by_hardness"][hardness]["official_sr_at_10"] = summary[
                "by_hardness"
            ][hardness]["sr_curve"]["10"]
    return summary


def load_completed(path: Path) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open() as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                completed[int(record["task_id"])] = record
    return completed


async def run(args: argparse.Namespace) -> None:
    import aiohttp
    from transformers import AutoTokenizer

    data = json.loads(Path(args.data).read_text())
    selected: list[tuple[int, dict[str, Any]]] = list(enumerate(data))
    if args.per_hardness_limit is not None:
        seen: Counter[str] = Counter()
        limited = []
        for item in selected:
            hardness = item[1]["hardness"]
            if seen[hardness] < args.per_hardness_limit:
                limited.append(item)
                seen[hardness] += 1
        selected = limited

    endpoints = tuple(value.rstrip("/") for value in args.endpoints.split(","))
    config = Config(
        model=args.model,
        endpoints=endpoints,
        max_turns=args.max_turns,
        dialogue_limit=args.dialogue_limit,
        max_tokens=args.max_tokens,
        token_budget=args.token_budget,
        concurrency=args.concurrency,
        request_timeout=args.request_timeout,
        mysql_host=args.mysql_host,
        mysql_port=args.mysql_port,
        mysql_user=args.mysql_user,
        mysql_password=args.mysql_password,
        mysql_sql_mode=args.mysql_sql_mode,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output)
    pending = [(idx, row) for idx, row in selected if idx not in completed]
    semaphore = asyncio.Semaphore(config.concurrency)
    connector = aiohttp.TCPConnector(limit=config.concurrency)

    async with aiohttp.ClientSession(connector=connector) as session:
        async def bounded(idx: int, row: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                endpoint = config.endpoints[idx % len(config.endpoints)]
                return await evaluate_record(
                    idx, row, endpoint, session, tokenizer, config
                )

        tasks = [asyncio.create_task(bounded(idx, row)) for idx, row in pending]
        with output.open("a") as handle:
            for completed_count, future in enumerate(asyncio.as_completed(tasks), 1):
                result = await future
                completed[int(result["task_id"])] = result
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                if completed_count % 16 == 0 or completed_count == len(tasks):
                    print(
                        json.dumps(
                            {
                                "newly_completed": completed_count,
                                "pending_total": len(tasks),
                                "successes_so_far": sum(
                                    row["success"] for row in completed.values()
                                ),
                            }
                        ),
                        flush=True,
                    )

    selected_ids = {idx for idx, _ in selected}
    final_records = [
        completed[idx] for idx in sorted(selected_ids) if idx in completed
    ]
    summary = aggregate(final_records, config.max_turns)
    summary["protocol"] = {
        "dataset": str(Path(args.data).resolve()),
        "model": config.model,
        "endpoints": config.endpoints,
        "temperature": 0,
        "max_turns": config.max_turns,
        "dialogue_limit": config.dialogue_limit,
        "max_new_tokens_per_turn": config.max_tokens,
        "token_budget": config.token_budget,
        "schema_handicap": False,
        "concurrency": config.concurrency,
        "mysql_sql_mode": config.mysql_sql_mode,
        "gold_normalization": "quote SHOW table; alias top-level COUNT derived table",
    }
    summary_path = output.with_name(output.stem + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--endpoints", default=",".join(ENDPOINTS_DEFAULT))
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS_DEFAULT)
    parser.add_argument("--dialogue-limit", type=int, default=DIALOGUE_LIMIT_DEFAULT)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--mysql-host", default="127.0.0.1")
    parser.add_argument("--mysql-port", type=int, default=33307)
    parser.add_argument("--mysql-user", default="admin")
    parser.add_argument("--mysql-password", default="admin")
    parser.add_argument("--mysql-sql-mode", default="IGNORE_SPACE")
    parser.add_argument("--per-hardness-limit", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
