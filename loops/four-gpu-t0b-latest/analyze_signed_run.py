#!/usr/bin/env python3
"""Validate and compare the completed signed-action 200-task arm."""
from __future__ import annotations

import glob
import json
import math
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "artifacts"


def load(run: str) -> dict[str, dict]:
    rows = {}
    paths = sorted(glob.glob(str(ROOT / run / "t0_shard*.jsonl")))
    assert len(paths) == 4, (run, paths)
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            assert row["task_id"] not in rows, (run, row["task_id"])
            rows[row["task_id"]] = row
    assert len(rows) == 200, (run, len(rows))
    return rows


def exact_mcnemar(rescues: int, harms: int) -> float:
    n = rescues + harms
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(rescues, harms) + 1)) / 2**n
    return min(1.0, 2 * tail)


def arm(rows: dict[str, dict], field: str) -> dict[str, dict]:
    return {task_id: row[field] for task_id, row in rows.items()}


def paired(candidate: dict[str, dict], baseline: dict[str, dict]) -> dict:
    assert candidate.keys() == baseline.keys()
    rescues = sum(
        candidate[key]["success"] and not baseline[key]["success"]
        for key in candidate)
    harms = sum(
        not candidate[key]["success"] and baseline[key]["success"]
        for key in candidate)
    return {
        "rescues": rescues,
        "harms": harms,
        "net": rescues - harms,
        "mcnemar_p": exact_mcnemar(rescues, harms),
        "mean_reward_delta": statistics.fmean(
            float(candidate[key]["reward"]) - float(baseline[key]["reward"])
            for key in candidate),
    }


signed_raw = load("signed_action_ul002_paper")
baseline_raw = {
    name: load(name)
    for name in (
        "tier0", "tier0c", "attt", "attt_paper", "tier0c_paper",
        "t0b_latest_k5_paper", "latest_resid_standalone_paper",
        "latest_resid_ngram_standalone_paper",
    )
}
task_ids = set(signed_raw)
assert all(set(rows) == task_ids for rows in baseline_raw.values())
assert all("error" not in row for row in signed_raw.values())

signed = arm(signed_raw, "tier0")
baselines = {
    "std": arm(baseline_raw["tier0"], "std"),
    "retry": arm(baseline_raw["tier0"], "retry"),
    "t0b": arm(baseline_raw["tier0"], "tier0"),
    "t0c": arm(baseline_raw["tier0c"], "tier0"),
    "attt": arm(baseline_raw["attt"], "tier0"),
    "attt_paper": arm(baseline_raw["attt_paper"], "tier0"),
    "t0c_paper": arm(baseline_raw["tier0c_paper"], "tier0"),
    "t0b_latest_paper": arm(baseline_raw["t0b_latest_k5_paper"], "tier0"),
    "residual_standalone": arm(
        baseline_raw["latest_resid_standalone_paper"], "tier0"),
    "residual_ngram_standalone": arm(
        baseline_raw["latest_resid_ngram_standalone_paper"], "tier0"),
}

with (ROOT / "signed_action_ul002_paper" / "config.json").open() as handle:
    config = json.load(handle)
assert config["lora_r"] == 8
assert config["lora_alpha"] == 16
assert config["lr"] == 5e-4
expected = {
    "method": "matched_action_signed_fixed_denominator",
    "cadence": 5,
    "max_updates": 5,
    "lambda_ul": 0.02,
    "beta_kl": 0.01,
    "post_kl_limit": 0.02,
    "positive_cap": 1.55,
    "negative_cap": 4.51,
    "grad_clip": 0.5,
    "max_backtracks": 3,
    "direction_min_dose": 1e-5,
    "direction_tolerance": 1e-5,
    "steps_per_update": 2,
}
assert all(config["extra"][key] == value for key, value in expected.items())

updates = []
selected = []
safe_skip_reasons = Counter()
unexpected_skip_errors = []
for task_id, result in signed.items():
    assert result["n_selected"] == len(result["update_diag"])
    assert result["n_selected"] <= 5
    for diagnostic in result["update_diag"]:
        selected.append((task_id, diagnostic))
        if diagnostic.get("error"):
            error_text = diagnostic.get("trace", "") + " " + diagnostic["error"]
            reasons = [
                reason for reason in (
                    "post_kl_limit", "positive_direction_violation",
                    "negative_direction_violation")
                if reason in error_text
            ]
            if reasons:
                safe_skip_reasons[reasons[-1]] += 1
            else:
                unexpected_skip_errors.append(
                    (task_id, diagnostic.get("step"), diagnostic["error"]))
        train = diagnostic.get("train", {})
        if train.get("trained"):
            updates.append((task_id, diagnostic, train))

violations = []
rejections = Counter()
learning_rates = Counter()
for task_id, diagnostic, train in updates:
    accepted = train["attempts"][-1]
    assert accepted["accepted"]
    assert float(train["post_kl"]) <= 0.02
    learning_rates[float(train["accepted_lr"])] += 1
    for attempt in train["attempts"][:-1]:
        assert not attempt["accepted"]
        rejections[attempt["reason"]] += 1
    if (
        accepted["positive_dose"] >= 1e-5
        and accepted["positive_weighted_logp_change"] < -1e-5
    ):
        violations.append((task_id, diagnostic["step"], "positive"))
    if (
        accepted["negative_dose"] >= 1e-5
        and accepted["negative_weighted_logp_change"] > 1e-5
    ):
        violations.append((task_id, diagnostic["step"], "negative"))
assert not violations, violations
assert not unexpected_skip_errors, unexpected_skip_errors

accepted_grad_norms = [
    float(step["preclip_grad_norm"])
    for _, _, train in updates
    for step in train["attempts"][-1]["steps"]
]

env191 = [value for key, value in signed.items() if key.startswith("env_191_")]
negative_updates = [
    train for _, _, train in updates
    if train["attempts"][-1]["negative_dose"] >= 1e-5]
positive_updates = [
    train for _, _, train in updates
    if train["attempts"][-1]["positive_dose"] >= 1e-5]
summary = {
    "outcome": {
        "successes": sum(bool(value["success"]) for value in signed.values()),
        "mean_reward": statistics.fmean(float(value["reward"]) for value in signed.values()),
        "env_191_successes": sum(bool(value["success"]) for value in env191),
        "env_191_n": len(env191),
        "mean_steps": statistics.fmean(int(value["steps"]) for value in signed.values()),
    },
    "integrity": {
        "rows": len(signed),
        "top_level_errors": sum("error" in row for row in signed_raw.values()),
        "selected_windows": len(selected),
        "accepted_updates": len(updates),
        "inactive_windows": len(selected) - len(updates) - sum(safe_skip_reasons.values()),
        "safe_skipped_windows": sum(safe_skip_reasons.values()),
        "safe_skip_reasons": safe_skip_reasons,
        "unexpected_update_errors": unexpected_skip_errors,
        "adapter_load_rollbacks": sum(int(value["rollbacks"]) for value in signed.values()),
        "max_post_kl": max(float(train["post_kl"]) for _, _, train in updates),
        "accepted_learning_rates": learning_rates,
        "rejected_attempt_reasons": rejections,
        "significant_direction_violations": violations,
        "negative_branch_updates": len(negative_updates),
        "positive_branch_updates": len(positive_updates),
        "mean_negative_logp_change": (
            statistics.fmean(
                float(train["negative_weighted_logp_change"])
                for train in negative_updates) if negative_updates else None),
        "mean_positive_logp_change": (
            statistics.fmean(
                float(train["positive_weighted_logp_change"])
                for train in positive_updates) if positive_updates else None),
        "accepted_optimizer_steps": len(accepted_grad_norms),
        "max_preclip_grad_norm": max(accepted_grad_norms),
        "accepted_steps_clipped_at_0_5": sum(
            value > 0.5 for value in accepted_grad_norms),
    },
    "comparisons": {
        name: paired(signed, baseline) for name, baseline in baselines.items()
    },
    "baseline_outcomes": {
        name: {
            "successes": sum(bool(value["success"]) for value in baseline.values()),
            "mean_reward": statistics.fmean(
                float(value["reward"]) for value in baseline.values()),
        }
        for name, baseline in baselines.items()
    },
}
print(json.dumps(summary, indent=2, default=dict))
