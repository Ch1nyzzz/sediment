#!/usr/bin/env python3
"""Validate the completed run and recompute paired metrics from raw JSONL."""

from __future__ import annotations

import glob
import json
import math
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "artifacts"


def load(run: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    paths = sorted(glob.glob(str(ROOT / run / "t0_shard*.jsonl")))
    assert len(paths) == 4, (run, paths)
    for path in paths:
        with open(path) as handle:
            for line in handle:
                row = json.loads(line)
                task_id = row["task_id"]
                assert task_id not in rows, (run, task_id)
                rows[task_id] = row
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
        bool(candidate[k]["success"]) and not bool(baseline[k]["success"])
        for k in candidate
    )
    harms = sum(
        not bool(candidate[k]["success"]) and bool(baseline[k]["success"])
        for k in candidate
    )
    reward_delta = sum(
        float(candidate[k]["reward"]) - float(baseline[k]["reward"])
        for k in candidate
    ) / len(candidate)
    return {
        "rescues": rescues,
        "harms": harms,
        "net": rescues - harms,
        "mcnemar_p": exact_mcnemar(rescues, harms),
        "mean_reward_delta": reward_delta,
        "rescue_task_ids": sorted(
            k
            for k in candidate
            if bool(candidate[k]["success"]) and not bool(baseline[k]["success"])
        ),
        "harm_task_ids": sorted(
            k
            for k in candidate
            if not bool(candidate[k]["success"]) and bool(baseline[k]["success"])
        ),
    }


raw = {
    run: load(run)
    for run in (
        "tier0", "tier0c", "tier0c_paper", "attt", "attt_paper",
        "t0b_latest_k5_paper", "latest_resid_standalone_paper",
        "latest_ngram_fullctx_paper", "latest_resid_ngram_standalone_paper",
    )
}
ids = set(raw["tier0"])
assert all(set(rows) == ids for rows in raw.values())

arms = {
    "std": arm(raw["tier0"], "std"),
    "retry": arm(raw["tier0"], "retry"),
    "t0b": arm(raw["tier0"], "tier0"),
    "t0c": arm(raw["tier0c"], "tier0"),
    "attt": arm(raw["attt"], "tier0"),
    "t0c_paper": arm(raw["tier0c_paper"], "tier0"),
    "attt_paper": arm(raw["attt_paper"], "tier0"),
    "t0b_latest_paper": arm(raw["t0b_latest_k5_paper"], "tier0"),
    "latest_resid_standalone_paper": arm(
        raw["latest_resid_standalone_paper"], "tier0"),
    "latest_ngram_fullctx_paper": arm(
        raw["latest_ngram_fullctx_paper"], "tier0"),
    "latest_resid_ngram_standalone_paper": arm(
        raw["latest_resid_ngram_standalone_paper"], "tier0"),
}

summary = {}
for name, values in arms.items():
    env191 = [v for k, v in values.items() if k.startswith("env_191_")]
    summary[name] = {
        "successes": sum(bool(v["success"]) for v in values.values()),
        "mean_reward": sum(float(v["reward"]) for v in values.values()) / len(values),
        "env_191_n": len(env191),
        "env_191_successes": sum(bool(v["success"]) for v in env191),
        "mean_steps": sum(int(v["steps"]) for v in values.values()) / len(values),
        "total_updates": sum(int(v.get("n_updates", 0)) for v in values.values()),
        "vs_std": None if name == "std" else paired(values, arms["std"]),
    }

new_raw = raw["t0b_latest_k5_paper"]
assert all("error" not in row for row in new_raw.values())
new = arms["t0b_latest_paper"]
diags = [diag for value in new.values() for diag in value["update_diag"]]
assert all(value["n_selected"] == len(value["update_diag"]) for value in new.values())
assert all(value["n_selected"] <= 5 for value in new.values())
assert all(diag["step"] in (5, 10, 15, 20, 25) for diag in diags)
assert all(diag["n_tok"] > 0 for diag in diags)
assert all(0 <= diag["pos_frac"] <= 1 for diag in diags)
trained = [diag for diag in diags if diag["trained"]]
assert all(len(diag["losses"]) == 2 for diag in trained)
assert sum(value["n_updates"] for value in new.values()) == len(trained)

with open(ROOT / "t0b_latest_k5_paper" / "config.json") as handle:
    config = json.load(handle)
assert config["lora_r"] == 8
assert config["lora_alpha"] == 16
assert config["lr"] == 5e-4
assert config["micro_batch"] == 1
assert config["extra"]["cadence"] == 5
assert config["extra"]["max_selections"] == 5
assert config["extra"]["steps_per_update"] == 2
assert config["extra"]["candidate"] == "latest_env"
assert config["extra"]["pricing"] == "relu(logp_full_natural_prefix-logp_standalone)"


def percentile(values: list[float], fraction: float) -> float:
    assert values
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]

integrity = {
    "rows": len(new),
    "unique_task_ids": len(set(new)),
    "top_level_errors": sum("error" in row for row in new_raw.values()),
    "update_errors": sum(int(value["upd_errors"]) for value in new.values()),
    "selected_candidates": sum(int(value["n_selected"]) for value in new.values()),
    "successful_updates": sum(int(value["n_updates"]) for value in new.values()),
    "trained_diagnostics": len(trained),
    "all_trained_have_two_losses": all(len(diag["losses"]) == 2 for diag in trained),
    "source_roles": Counter(diag["source_role"] for diag in diags),
    "mean_positive_token_fraction": (
        sum(float(diag["pos_frac"]) for diag in diags) / len(diags) if diags else 0.0
    ),
    "zero_positive_candidates": sum(float(diag["pos_frac"]) == 0 for diag in diags),
    "mean_of_mean_positive_weight": statistics.fmean(
        float(diag["mean_pos_w"]) for diag in diags
    ),
    "median_of_mean_positive_weight": statistics.median(
        float(diag["mean_pos_w"]) for diag in diags
    ),
    "p90_of_mean_positive_weight": percentile(
        [float(diag["mean_pos_w"]) for diag in diags], 0.9
    ),
    "maximum_token_weight": max(float(diag["max_pos_w"]) for diag in diags),
    "rounded_zero_loss_updates": sum(
        bool(diag["losses"]) and all(float(loss) == 0 for loss in diag["losses"])
        for diag in trained
    ),
    "selection_count_distribution": Counter(
        int(value["n_selected"]) for value in new.values()
    ),
}

comparisons = {
    "vs_retry": paired(new, arms["retry"]),
    "vs_t0b": paired(new, arms["t0b"]),
    "vs_attt_paper": paired(new, arms["attt_paper"]),
    "vs_t0c_paper": paired(new, arms["t0c_paper"]),
}

resid_standalone = arms["latest_resid_standalone_paper"]
resid_standalone_raw = raw["latest_resid_standalone_paper"]
factorial_comparisons = {
    "integrity": {
        "rows": len(resid_standalone),
        "top_level_errors": sum(
            "error" in row for row in resid_standalone_raw.values()),
        "update_errors": sum(
            int(value["upd_errors"]) for value in resid_standalone.values()),
        "total_updates": sum(
            int(value["n_updates"]) for value in resid_standalone.values()),
    },
    "vs_attt_paper": paired(resid_standalone, arms["attt_paper"]),
    "vs_residual_full": paired(resid_standalone, new),
    "vs_std": paired(resid_standalone, arms["std"]),
}

ngram_full = arms["latest_ngram_fullctx_paper"]
ngram_full_raw = raw["latest_ngram_fullctx_paper"]
factorial_2x2 = {
    "ngram_full_integrity": {
        "rows": len(ngram_full),
        "top_level_errors": sum("error" in row for row in ngram_full_raw.values()),
        "update_errors": sum(int(value["upd_errors"]) for value in ngram_full.values()),
        "total_updates": sum(int(value["n_updates"]) for value in ngram_full.values()),
    },
    "context_effect_with_ngram_full_vs_standalone": paired(
        ngram_full, arms["attt_paper"]),
    "context_effect_with_residual_full_vs_standalone": paired(
        new, resid_standalone),
    "weight_effect_with_standalone_residual_vs_ngram": paired(
        resid_standalone, arms["attt_paper"]),
    "weight_effect_with_full_residual_vs_ngram": paired(new, ngram_full),
    "ngram_full_vs_std": paired(ngram_full, arms["std"]),
}

resid_ngram = arms["latest_resid_ngram_standalone_paper"]
resid_ngram_raw = raw["latest_resid_ngram_standalone_paper"]
resid_ngram_diags = [
    diag for value in resid_ngram.values() for diag in value["update_diag"]
]
with open(
    ROOT / "latest_resid_ngram_standalone_paper" / "config.json"
) as handle:
    resid_ngram_config = json.load(handle)
assert resid_ngram_config["lora_r"] == 8
assert resid_ngram_config["lora_alpha"] == 16
assert resid_ngram_config["lr"] == 5e-4
assert resid_ngram_config["extra"]["cadence"] == 5
assert resid_ngram_config["extra"]["max_selections"] == 5
assert resid_ngram_config["extra"]["steps_per_update"] == 2
assert resid_ngram_config["extra"]["weighting"] == "residual_ngram"
assert resid_ngram_config["extra"]["train_context"] == "standalone"
assert all("error" not in row for row in resid_ngram_raw.values())
assert all(value["upd_errors"] == 0 for value in resid_ngram.values())
assert all(value["n_selected"] == len(value["update_diag"])
           for value in resid_ngram.values())
assert all(diag["trained"] and len(diag["losses"]) == 2
           for diag in resid_ngram_diags)

residual_ngram_followup = {
    "integrity": {
        "rows": len(resid_ngram),
        "top_level_errors": sum(
            "error" in row for row in resid_ngram_raw.values()),
        "update_errors": sum(
            int(value["upd_errors"]) for value in resid_ngram.values()),
        "selected_candidates": len(resid_ngram_diags),
        "successful_updates": sum(
            int(value["n_updates"]) for value in resid_ngram.values()),
        "all_updates_have_two_losses": all(
            len(diag["losses"]) == 2 for diag in resid_ngram_diags),
        "mean_positive_fraction": statistics.fmean(
            float(diag["pos_frac"]) for diag in resid_ngram_diags),
        "mean_ngram_weight": statistics.fmean(
            float(diag["mean_ngram_w"]) for diag in resid_ngram_diags),
        "discounted_candidates": sum(
            float(diag["mean_ngram_w"]) < 1 for diag in resid_ngram_diags),
    },
    "vs_residual_standalone": paired(resid_ngram, resid_standalone),
    "vs_attt_paper": paired(resid_ngram, arms["attt_paper"]),
    "vs_std": paired(resid_ngram, arms["std"]),
    "vs_retry": paired(resid_ngram, arms["retry"]),
    "vs_original_t0b": paired(resid_ngram, arms["t0b"]),
}

result = {
    "integrity": integrity,
    "config": {
        "lora_r": config["lora_r"],
        "lora_alpha": config["lora_alpha"],
        "lr": config["lr"],
        "cadence": config["extra"]["cadence"],
        "max_selections": config["extra"]["max_selections"],
        "steps_per_update": config["extra"]["steps_per_update"],
        "candidate": config["extra"]["candidate"],
        "pricing": config["extra"]["pricing"],
    },
    "summary": summary,
    "new_comparisons": comparisons,
    "residual_standalone": factorial_comparisons,
    "factorial_2x2": factorial_2x2,
    "residual_ngram_followup": residual_ngram_followup,
}
print(json.dumps(result, indent=2, sort_keys=True))
