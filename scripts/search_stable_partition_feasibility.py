#!/usr/bin/env python3
"""Choose family role capacities using only memory/probe feasibility evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plan_stable_signal_windows import build_plan  # noqa: E402
from sediment.buffer import Buffer  # noqa: E402
from sediment.compiler.stable_data import (  # noqa: E402
    member_task_cluster_id,
    partition_families,
)


def _family_stats(buffers: list[str]) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for path in buffers:
        buffer = Buffer.load(path)
        for trajectory in buffer._trajs:
            if trajectory.is_retry:
                continue
            row = stats.setdefault(
                trajectory.env_family, {"completed": 0, "successful": 0}
            )
            row["completed"] += 1
            row["successful"] += int(bool(trajectory.success))
    return stats


def _plan_cluster_stats(plan: dict) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for partition in ("train", "test"):
        groups: dict[str, list[str]] = {}
        for row in plan["windows"]:
            if row["partition"] != partition:
                continue
            cluster = member_task_cluster_id(row["member_task_ids"])
            groups.setdefault(cluster, []).append(row["window_id"])
        result[partition] = {
            "windows": sum(len(values) for values in groups.values()),
            "clusters": len(groups),
            "cluster_sizes": {
                key: len(values) for key, values in sorted(groups.items())
            },
        }
    return result


def _freeze_ragged_plan(plan: dict, *, family_partition: Path) -> dict:
    """Convert a reward-blind capacity candidate into an explicit ragged plan."""

    frozen = json.loads(json.dumps(plan))
    original_shortfalls = list(frozen.get("shortfalls", []))
    per_stream: dict[str, dict[str, int]] = {}
    for stream in ("s0", "s1", "s2", "s3"):
        per_stream[stream] = {
            partition: sum(
                row["stream_id"] == stream and row["partition"] == partition
                for row in frozen["windows"]
            )
            for partition in ("train", "test")
        }
    frozen["family_partition"] = str(family_partition.resolve())
    frozen["requested_uniform_train_per_stream"] = frozen.get("train_per_stream")
    frozen["requested_uniform_test_per_stream"] = frozen.get("test_per_stream")
    frozen["per_stream_targets"] = per_stream
    frozen["ragged_seed_replication"] = True
    frozen["ragged_reason"] = (
        "strict_four_cross_family_targets_per_donor_capacity"
    )
    frozen["pre_normalization_shortfalls"] = original_shortfalls
    frozen["shortfalls"] = []
    digest_payload = dict(frozen)
    digest_payload.pop("digest", None)
    digest_payload.pop("family_partition", None)
    frozen["digest"] = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return frozen


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer", action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--min-train-members", type=int, default=8)
    parser.add_argument("--max-train-members", type=int, default=14)
    parser.add_argument("--probe-families-per-role", type=int, default=4)
    parser.add_argument("--train-per-stream", type=int, default=4)
    parser.add_argument("--test-per-stream", type=int, default=3)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--min-successes", type=int, default=2)
    parser.add_argument("--allow-ragged", action="store_true")
    parser.add_argument("--min-train-total", type=int, default=0)
    parser.add_argument("--min-test-total", type=int, default=0)
    parser.add_argument("--min-train-clusters", type=int, default=0)
    parser.add_argument("--min-test-clusters", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    family_stats = _family_stats(args.buffer)
    total = len(family_stats)
    probe_total = 4 * args.probe_families_per_role
    reports: list[dict] = []
    feasible: list[tuple[tuple[int, ...], Path, Path, dict]] = []
    for train_members in range(args.min_train_members, args.max_train_members + 1):
        test_members = total - probe_total - train_members
        if test_members <= 0:
            continue
        counts = (
            train_members,
            args.probe_families_per_role,
            args.probe_families_per_role,
            test_members,
            args.probe_families_per_role,
            args.probe_families_per_role,
        )
        partition = partition_families(
            family_stats,
            seed=args.seed,
            counts=counts,
            success_counts={
                family: stats["successful"] for family, stats in family_stats.items()
            },
        )
        partition_path = output / f"family_partition_tm{train_members}.json"
        partition_payload = {
            "format": "sediment-stable-family-partition-v1",
            **partition.__dict__,
            "family_stats": family_stats,
            "source_buffers": [str(Path(value).resolve()) for value in args.buffer],
            "candidate_rewards_inspected": False,
            "stratification": "completed_success_count_greedy_balance_v1",
            "role_counts": list(counts),
        }
        with partition_path.open("w", encoding="utf-8") as handle:
            json.dump(partition_payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        plan = build_plan(
            args.buffer,
            partition_path,
            train_per_stream=args.train_per_stream,
            test_per_stream=args.test_per_stream,
            window_size=args.window_size,
            min_successes=args.min_successes,
        )
        plan_path = output / f"window_plan_tm{train_members}.json"
        with plan_path.open("w", encoding="utf-8") as handle:
            json.dump(plan, handle, indent=2, sort_keys=True)
            handle.write("\n")
        per_stream = {
            f"{stream}:{role}": sum(
                row["stream_id"] == stream and row["partition"] == role
                for row in plan["windows"]
            )
            for stream in ("s0", "s1", "s2", "s3")
            for role in ("train", "test")
        }
        report = {
            "train_member_families": train_members,
            "test_member_families": test_members,
            "role_counts": list(counts),
            "partition_digest": partition.digest,
            "plan_digest": plan["digest"],
            "train_windows": sum(row["partition"] == "train" for row in plan["windows"]),
            "test_windows": sum(row["partition"] == "test" for row in plan["windows"]),
            "per_stream": per_stream,
            "shortfalls": plan["shortfalls"],
            "cluster_stats": _plan_cluster_stats(plan),
        }
        reports.append(report)
        cluster_stats = report["cluster_stats"]
        ragged_ok = (
            args.allow_ragged
            and report["train_windows"] >= args.min_train_total
            and report["test_windows"] >= args.min_test_total
            and cluster_stats["train"]["clusters"] >= args.min_train_clusters
            and cluster_stats["test"]["clusters"] >= args.min_test_clusters
        )
        if not plan["shortfalls"] or ragged_ok:
            score = (
                -min(
                    int(cluster_stats["train"]["clusters"]),
                    int(cluster_stats["test"]["clusters"]),
                ),
                -int(cluster_stats["test"]["clusters"]),
                -int(cluster_stats["train"]["clusters"]),
                -report["test_windows"],
                -report["train_windows"],
                abs(train_members - test_members),
                -train_members,
            )
            feasible.append(
                (
                    score,
                    partition_path,
                    plan_path,
                    report,
                )
            )
    if not feasible:
        raise SystemExit(f"no feasible member-family capacity found: {reports}")
    _, selected_partition, selected_plan, selected_report = min(feasible)
    frozen_partition = output / "family_partition_frozen.json"
    frozen_plan = output / "window_plan_frozen.json"
    shutil.copy2(selected_partition, frozen_partition)
    with selected_plan.open(encoding="utf-8") as handle:
        selected_plan_payload = json.load(handle)
    if selected_plan_payload.get("shortfalls"):
        selected_plan_payload = _freeze_ragged_plan(
            selected_plan_payload,
            family_partition=frozen_partition,
        )
        with frozen_plan.open("w", encoding="utf-8") as handle:
            json.dump(selected_plan_payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    else:
        shutil.copy2(selected_plan, frozen_plan)
    result = {
        "format": "sediment-stable-partition-feasibility-v1",
        "candidate_rewards_inspected": False,
        "search": reports,
        "selected": selected_report,
        "frozen_plan_digest": selected_plan_payload["digest"],
        "frozen_partition": str(frozen_partition),
        "frozen_plan": str(frozen_plan),
    }
    with (output / "feasibility_search.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
