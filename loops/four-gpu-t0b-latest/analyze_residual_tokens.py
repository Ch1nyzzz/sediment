#!/usr/bin/env python3
"""Analyze scoring-only token traces for the residual's actual semantics."""

from __future__ import annotations

import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "artifacts" / "residual_token_debug20"


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )


task_rows = []
for path in sorted(glob.glob(str(ROOT / "debug_shard*.jsonl"))):
    for line in Path(path).read_text().splitlines():
        if line.strip():
            task_rows.append(json.loads(line))

observations = [
    (row["task_id"], diag)
    for row in task_rows
    for diag in row["debug"]["diagnostics"]
]
tokens = [
    {**token, "task_id": task_id, "observation": diag["observation"]}
    for task_id, diag in observations
    for token in diag["tokens"]
]
full = [float(row["full_logp"]) for row in tokens]
solo = [float(row["solo_logp"]) for row in tokens]
delta = [float(row["delta"]) for row in tokens]
positive = [max(value, 0.0) for value in delta]
positive_mass = sum(positive)

by_text: dict[str, dict[str, float]] = defaultdict(
    lambda: {"count": 0, "positive_mass": 0.0, "delta_sum": 0.0})
for row in tokens:
    item = by_text[row["token"]]
    item["count"] += 1
    item["positive_mass"] += max(float(row["delta"]), 0.0)
    item["delta_sum"] += float(row["delta"])

top_token_types = sorted(
    (
        {
            "token": token,
            "count": int(values["count"]),
            "positive_mass": values["positive_mass"],
            "mean_delta": values["delta_sum"] / values["count"],
        }
        for token, values in by_text.items()
    ),
    key=lambda item: item["positive_mass"],
    reverse=True,
)[:30]

top_instances = sorted(tokens, key=lambda row: float(row["delta"]), reverse=True)[:30]
for row in top_instances:
    row["observation"] = row["observation"][:180]

ordered_delta = sorted(delta)
result = {
    "tasks": len(task_rows),
    "tasks_with_selection": len({task_id for task_id, _ in observations}),
    "observations": len(observations),
    "tokens": len(tokens),
    "positive_fraction": sum(value > 0 for value in delta) / len(delta),
    "delta_quantiles": {
        "p10": ordered_delta[round((len(delta) - 1) * 0.10)],
        "p50": ordered_delta[round((len(delta) - 1) * 0.50)],
        "p90": ordered_delta[round((len(delta) - 1) * 0.90)],
        "p99": ordered_delta[round((len(delta) - 1) * 0.99)],
    },
    "mean_full_logp": statistics.fmean(full),
    "mean_solo_logp": statistics.fmean(solo),
    "mean_delta": statistics.fmean(delta),
    "corr_delta_full_logp": pearson(delta, full),
    "corr_delta_negative_solo_logp": pearson(delta, [-value for value in solo]),
    "full_probability_over_99pct_fraction": sum(
        value > math.log(0.99) for value in full
    ) / len(full),
    "positive_mass_from_full_probability_over_99pct": sum(
        weight for weight, value in zip(positive, full) if value > math.log(0.99)
    ) / positive_mass,
    "positive_mass_from_solo_probability_under_1pct": sum(
        weight for weight, value in zip(positive, solo) if value < math.log(0.01)
    ) / positive_mass,
    "top_token_types_by_positive_mass": top_token_types,
    "top_token_instances": top_instances,
}
print(json.dumps(result, indent=2, ensure_ascii=False))
