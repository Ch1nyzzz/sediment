#!/usr/bin/env python3
"""Post-hoc diagnostics for whether residual statistics predict useful updates."""

from __future__ import annotations

import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "artifacts"


def load(run: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted(glob.glob(str(ROOT / run / "t0_shard*.jsonl"))):
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                assert row["task_id"] not in rows
                rows[row["task_id"]] = row
    assert len(rows) == 200, (run, len(rows))
    return rows


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sx * sy)


baseline_rows = load("tier0")
resid_rows = load("latest_resid_standalone_paper")
combo_rows = load("latest_resid_ngram_standalone_paper")
attt_rows = load("attt_paper")

records: list[dict] = []
for task_id, row in resid_rows.items():
    value = row["tier0"]
    diags = value["update_diag"]
    std = baseline_rows[task_id]["std"]
    attt = attt_rows[task_id]["tier0"]
    combo = combo_rows[task_id]["tier0"]
    rec = {
        "task_id": task_id,
        "n_updates": value["n_updates"],
        "success": float(value["success"]),
        "reward": float(value["reward"]),
        "std_success": float(std["success"]),
        "std_reward": float(std["reward"]),
        "attt_success": float(attt["success"]),
        "attt_reward": float(attt["reward"]),
        "steps": float(value["steps"]),
        "reward_lift_std": float(value["reward"]) - float(std["reward"]),
        "success_lift_std": float(value["success"]) - float(std["success"]),
        "success_lift_attt": float(value["success"]) - float(attt["success"]),
        "success_lift_combo": float(value["success"]) - float(combo["success"]),
    }
    if diags:
        n_tok = sum(int(diag["n_tok"]) for diag in diags)
        rec.update({
            "mean_pos_frac": statistics.fmean(
                float(diag["pos_frac"]) for diag in diags),
            "token_pos_frac": sum(
                float(diag["pos_frac"]) * int(diag["n_tok"]) for diag in diags
            ) / n_tok,
            "mean_delta": statistics.fmean(
                float(diag["mean_delta"]) for diag in diags),
            "mean_weight": statistics.fmean(
                float(diag["mean_w"]) for diag in diags),
            "mean_peak_ratio": statistics.fmean(
                float(diag["max_w"]) / max(float(diag["mean_w"]), 1e-12)
                for diag in diags),
            "first_pos_frac": float(diags[0]["pos_frac"]),
            "first_delta": float(diags[0]["mean_delta"]),
            "first_weight": float(diags[0]["mean_w"]),
            "first_peak_ratio": (
                float(diags[0]["max_w"]) / max(float(diags[0]["mean_w"]), 1e-12)
            ),
        })
    records.append(rec)

features = [
    "n_updates", "mean_pos_frac", "token_pos_frac", "mean_delta",
    "mean_weight", "mean_peak_ratio", "first_pos_frac", "first_delta",
    "first_weight", "first_peak_ratio",
]
targets = [
    "success", "reward", "std_success", "std_reward", "attt_success",
    "attt_reward", "steps", "reward_lift_std", "success_lift_std",
    "success_lift_attt", "success_lift_combo",
]

correlations: dict[str, dict[str, float | None]] = {}
for feature in features:
    subset = [record for record in records if feature in record]
    correlations[feature] = {
        target: pearson(
            [float(record[feature]) for record in subset],
            [float(record[target]) for record in subset],
        )
        for target in targets
    }

groups: dict[str, list[dict]] = defaultdict(list)
for record in records:
    if "mean_pos_frac" not in record:
        groups["no_update"].append(record)
    elif record["success_lift_attt"] > 0:
        groups["residual_beats_attt"].append(record)
    elif record["success_lift_attt"] < 0:
        groups["residual_loses_attt"].append(record)
    else:
        groups["residual_ties_attt"].append(record)

group_summary = {}
for name, values in groups.items():
    group_summary[name] = {
        "n": len(values),
        "success_rate": statistics.fmean(v["success"] for v in values),
        "mean_reward_lift_std": statistics.fmean(v["reward_lift_std"] for v in values),
    }
    for feature in (
        "n_updates", "mean_pos_frac", "token_pos_frac", "mean_delta",
        "mean_weight", "mean_peak_ratio", "first_pos_frac", "first_delta",
        "first_weight", "first_peak_ratio",
    ):
        available = [float(v[feature]) for v in values if feature in v]
        group_summary[name][feature] = (
            statistics.fmean(available) if available else None
        )

by_updates = {}
for count in range(6):
    values = [record for record in records if record["n_updates"] == count]
    by_updates[str(count)] = {
        "n": len(values),
        "residual_success_rate": statistics.fmean(v["success"] for v in values),
        "mean_reward_lift_std": statistics.fmean(v["reward_lift_std"] for v in values),
        "mean_success_lift_std": statistics.fmean(v["success_lift_std"] for v in values),
    }

print(json.dumps({
    "updated_tasks": sum(record["n_updates"] > 0 for record in records),
    "correlations": correlations,
    "groups": group_summary,
    "by_updates": by_updates,
}, indent=2, sort_keys=True))
