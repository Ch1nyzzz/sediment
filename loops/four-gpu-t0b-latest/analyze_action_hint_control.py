#!/usr/bin/env python3
"""Analyze matched action-hint scoring traces by action and by token."""

from __future__ import annotations

import glob
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "artifacts" / "action_hint_control_debug40"
VARIANTS = ("hint", "polluted", "controlled")
DELTA_KEYS = {
    "hint": "hint_delta",
    "polluted": "polluted_delta",
    "controlled": "controlled_delta",
}


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def auc(ok: list[float], error: list[float]) -> float:
    wins = sum(a > b for a in ok for b in error)
    ties = sum(a == b for a in ok for b in error)
    return (wins + 0.5 * ties) / (len(ok) * len(error))


task_rows = []
for path in sorted(glob.glob(str(ROOT / "debug_shard*.jsonl"))):
    for line in Path(path).read_text().splitlines():
        if line.strip():
            task_rows.append(json.loads(line))
assert len(task_rows) == 40
assert len({row["task_id"] for row in task_rows}) == 40
assert all("error" not in row for row in task_rows)

actions = [
    {**diag, "task_id": row["task_id"]}
    for row in task_rows
    for diag in row["debug"]["diagnostics"]
]


def summarize(items: list[dict], variant: str) -> dict:
    key = DELTA_KEYS[variant]
    action_means = [
        statistics.fmean(float(token[key]) for token in item["tokens"])
        for item in items
    ]
    action_pos_fracs = [
        statistics.fmean(float(token[key]) > 0 for token in item["tokens"])
        for item in items
    ]
    token_values = [float(token[key]) for item in items for token in item["tokens"]]
    positive = [max(value, 0.0) for value in token_values]
    positive_values = [value for value in token_values if value > 0]
    return {
        "actions": len(items),
        "tokens": len(token_values),
        "action_mean_delta_positive_fraction": statistics.fmean(
            value > 0 for value in action_means),
        "actions_with_any_positive_token_fraction": statistics.fmean(
            value > 0 for value in action_pos_fracs),
        "actions_with_zero_positive_tokens": sum(value == 0 for value in action_pos_fracs),
        "action_mean_positive_token_fraction": statistics.fmean(action_pos_fracs),
        "action_median_positive_token_fraction": statistics.median(action_pos_fracs),
        "action_p10_positive_token_fraction": quantile(action_pos_fracs, 0.10),
        "action_p90_positive_token_fraction": quantile(action_pos_fracs, 0.90),
        "token_positive_fraction": statistics.fmean(value > 0 for value in token_values),
        "token_mean_delta": statistics.fmean(token_values),
        "positive_weight_per_token": statistics.fmean(positive),
        "mean_weight_conditional_on_positive": (
            statistics.fmean(positive_values) if positive_values else 0.0),
        "near_zero_token_fraction_abs_lt_1e_4": statistics.fmean(
            abs(value) < 1e-4 for value in token_values),
    }


by_status = {}
for status in ("all", "ok", "ERROR"):
    subset = actions if status == "all" else [
        action for action in actions if action["status"] == status
    ]
    by_status[status] = {
        variant: summarize(subset, variant) for variant in VARIANTS
    }

ok_actions = [action for action in actions if action["status"] == "ok"]
error_actions = [action for action in actions if action["status"] == "ERROR"]
separation = {}
for variant in VARIANTS:
    key = DELTA_KEYS[variant]
    ok_means = [statistics.fmean(float(t[key]) for t in a["tokens"]) for a in ok_actions]
    error_means = [
        statistics.fmean(float(t[key]) for t in a["tokens"]) for a in error_actions
    ]
    separation[variant] = {
        "ok_minus_error_action_mean_delta": (
            statistics.fmean(ok_means) - statistics.fmean(error_means)
        ),
        "auc_ok_above_error": auc(ok_means, error_means),
    }

all_tokens = [token for action in actions for token in action["tokens"]]
saturation = {
    name: statistics.fmean(float(token[key]) > math.log(0.99) for token in all_tokens)
    for name, key in (
        ("base_probability_over_99pct", "base_logp"),
        ("masked_probability_over_99pct", "masked_logp"),
        ("actual_probability_over_99pct", "actual_logp"),
    )
}

top_controlled = sorted(
    (
        {
            "task_id": action["task_id"],
            "status": action["status"],
            "token": token["token"],
            "controlled_delta": token["controlled_delta"],
            "action": action["action"][:160],
            "observation": action["observation"][:180],
        }
        for action in actions
        for token in action["tokens"]
    ),
    key=lambda row: float(row["controlled_delta"]),
    reverse=True,
)[:30]

print(json.dumps({
    "integrity": {
        "tasks": len(task_rows),
        "actions": len(actions),
        "ok_actions": len(ok_actions),
        "error_actions": len(error_actions),
        "top_level_errors": 0,
        "training": False,
    },
    "by_status": by_status,
    "status_separation": separation,
    "saturation": saturation,
    "top_controlled_tokens": top_controlled,
}, indent=2, ensure_ascii=False))
