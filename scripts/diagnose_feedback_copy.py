#!/usr/bin/env python3
"""Paired diagnostic: does previous-action feedback make a redo copy it?

Uses frozen failed trajectories as immutable source evidence. For each task and
request seed, generate one clean initial action and one action with the
prefix-match gate's turn-1 privileged action+feedback hint. No environment is
mutated; an exact repeat of the original action has the same deterministic SQL
outcome that already failed to finish the frozen attempt.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sediment.envs.intercode_sql import parse_sql_action
from sediment.stepwise import AttemptFeedbackHinter
from sediment.types import Message, Trajectory


def _post(url: str, payload: dict, timeout: float) -> str:
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read())
            return data["choices"][0]["message"]["content"] or ""
        except Exception as exc:  # bounded retry; failures remain explicit
            last = exc
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"request failed after retries: {last!r}")


def _norm_sql(text: str) -> tuple[str, bool]:
    sql, valid = parse_sql_action(text)
    if not valid:
        return "", False
    return " ".join(sql.rstrip(";").split()).casefold(), True


def _verb(sql: str) -> str:
    return sql.split(None, 1)[0].upper() if sql else ""


def _seed(task_id: str, replica: int) -> int:
    raw = hashlib.sha256(f"{task_id}:{replica}".encode()).digest()
    return int.from_bytes(raw[:4], "big") & 0x7FFFFFFF


def _binom_two_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def _bootstrap_ci(rows: list[dict], rounds: int = 10000) -> list[float]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    tasks = sorted(by_task)
    rng = random.Random(20260901)
    vals = []
    for _ in range(rounds):
        sampled = [rng.choice(tasks) for _ in tasks]
        picked = [row for task in sampled for row in by_task[task]]
        vals.append(
            sum(r["treatment_repeat"] - r["control_repeat"] for r in picked)
            / len(picked)
        )
    vals.sort()
    return [vals[int(0.025 * rounds)], vals[int(0.975 * rounds)]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8124/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--replicas", type=int, default=4)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    trajectories = []
    with open(args.buffer) as handle:
        for line in handle:
            traj = Trajectory.from_dict(json.loads(line))
            if not traj.success and not traj.is_retry:
                trajectories.append(traj)
    if args.max_tasks:
        trajectories = trajectories[: args.max_tasks]

    jobs = []
    original: dict[str, dict] = {}
    for traj in trajectories:
        first_i = next(i for i, m in enumerate(traj.messages) if m.role == "assistant")
        prefix = list(traj.messages[:first_i])
        action = traj.messages[first_i].content
        feedback = next(
            (m.content for m in traj.messages[first_i + 1:] if m.role in ("tool", "user")),
            "",
        )
        old_sql, old_valid = _norm_sql(action)
        original[traj.task_id] = {
            "action": action,
            "feedback": feedback,
            "sql": old_sql,
            "valid": old_valid,
            "verb": _verb(old_sql),
            "explicit_error": bool(re.search(r"error|invalid|unknown|doesn't exist", feedback, re.I)),
        }
        treatment = list(prefix)
        AttemptFeedbackHinter(traj).pre_generate(treatment)
        for replica in range(args.replicas):
            seed = _seed(traj.task_id, replica)
            jobs.extend([
                (traj.task_id, replica, "control", prefix, seed),
                (traj.task_id, replica, "treatment", treatment, seed),
            ])
    random.Random(20260901).shuffle(jobs)

    def run(job):
        task_id, replica, arm, messages, seed = job
        payload = {
            "model": args.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "seed": seed,
        }
        output = _post(args.base_url, payload, args.timeout)
        sql, valid = _norm_sql(output)
        return task_id, replica, arm, output, sql, valid

    generated = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        future_to_job = {pool.submit(run, job): job for job in jobs}
        done = 0
        for future in concurrent.futures.as_completed(future_to_job):
            task_id, replica, arm, output, sql, valid = future.result()
            generated[(task_id, replica, arm)] = (output, sql, valid)
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"[copy-diagnostic] {done}/{len(jobs)}", flush=True)

    rows = []
    for traj in trajectories:
        old = original[traj.task_id]
        for replica in range(args.replicas):
            co, cs, cv = generated[(traj.task_id, replica, "control")]
            to, ts, tv = generated[(traj.task_id, replica, "treatment")]
            rows.append({
                "task_id": traj.task_id,
                "replica": replica,
                "seed": _seed(traj.task_id, replica),
                "original_action": old["action"],
                "original_feedback": old["feedback"],
                "original_sql": old["sql"],
                "original_verb": old["verb"],
                "explicit_error": old["explicit_error"],
                "control_output": co,
                "control_sql": cs,
                "control_valid": cv,
                "control_repeat": bool(old["valid"] and cv and cs == old["sql"]),
                "control_same_verb": bool(old["verb"] and _verb(cs) == old["verb"]),
                "treatment_output": to,
                "treatment_sql": ts,
                "treatment_valid": tv,
                "treatment_repeat": bool(old["valid"] and tv and ts == old["sql"]),
                "treatment_same_verb": bool(old["verb"] and _verb(ts) == old["verb"]),
            })

    def rates(items: list[dict]) -> dict:
        n = len(items)
        return {
            "n": n,
            "control_valid": sum(r["control_valid"] for r in items) / max(n, 1),
            "treatment_valid": sum(r["treatment_valid"] for r in items) / max(n, 1),
            "control_repeat": sum(r["control_repeat"] for r in items) / max(n, 1),
            "treatment_repeat": sum(r["treatment_repeat"] for r in items) / max(n, 1),
            "repeat_delta": sum(r["treatment_repeat"] - r["control_repeat"] for r in items) / max(n, 1),
            "control_same_verb": sum(r["control_same_verb"] for r in items) / max(n, 1),
            "treatment_same_verb": sum(r["treatment_same_verb"] for r in items) / max(n, 1),
        }

    b = sum(r["control_repeat"] and not r["treatment_repeat"] for r in rows)
    c = sum(r["treatment_repeat"] and not r["control_repeat"] for r in rows)
    strata = {
        "all": rates(rows),
        "explicit_error": rates([r for r in rows if r["explicit_error"]]),
        "non_error_feedback": rates([r for r in rows if not r["explicit_error"]]),
    }
    for verb in sorted({r["original_verb"] for r in rows}):
        strata[f"verb_{verb or 'NONE'}"] = rates(
            [r for r in rows if r["original_verb"] == verb]
        )
    summary = {
        "protocol": vars(args),
        "n_failed_tasks": len(trajectories),
        "n_pairs": len(rows),
        "strata": strata,
        "paired_discordant": {"control_only": b, "treatment_only": c},
        "mcnemar_exact_two_sided_p": _binom_two_sided(b, c),
        "task_cluster_bootstrap_repeat_delta_95ci": _bootstrap_ci(rows),
        "original_verb_counts": dict(Counter(r["original_verb"] for r in rows)),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out.with_suffix(".summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
