#!/usr/bin/env python3
"""Freeze eligible B=16 windows and disjoint future probes before GPU labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sediment.buffer import Buffer  # noqa: E402


@dataclass(frozen=True)
class IndexedTrajectory:
    index: int
    trajectory: Any


def _probe_set(
    indexed: list[IndexedTrajectory],
    *,
    after: int,
    families: list[str],
    used_task_ids: set[str],
) -> list[IndexedTrajectory] | None:
    """Take the first unused later task from every preregistered probe family."""

    selected: list[IndexedTrajectory] = []
    for family in families:
        match = next(
            (
                item
                for item in indexed
                if item.index > after
                and item.trajectory.env_family == family
                and item.trajectory.task_id not in used_task_ids
            ),
            None,
        )
        if match is None:
            return None
        selected.append(match)
    return selected


def _has_assistant(trajectory: Any) -> bool:
    return any(
        getattr(message, "role", None) == "assistant"
        for message in getattr(trajectory, "messages", ())
    )


def _structurally_eligible_donors(
    window: list[IndexedTrajectory],
    *,
    max_success_candidates: int,
    min_targets_per_donor: int,
) -> list[IndexedTrajectory]:
    """Mirror the reward-free eligibility part of donor-KL collection.

    The collector only considers the first ``max_success_candidates`` successful
    members.  A strict single proposal then needs ``min_targets_per_donor``
    other members with assistant states and a different source family.  This
    preflight deliberately does not score a teacher, train an adapter, or read
    any future reward.
    """

    successes = [item for item in window if bool(item.trajectory.success)][
        :max_success_candidates
    ]
    eligible: list[IndexedTrajectory] = []
    for donor in successes:
        targets = [
            target
            for target in window
            if target.trajectory.task_id != donor.trajectory.task_id
            and target.trajectory.env_family != donor.trajectory.env_family
            and _has_assistant(target.trajectory)
        ]
        if len(targets) >= min_targets_per_donor:
            eligible.append(donor)
    return eligible


def _plan_stream_partition(
    indexed: list[IndexedTrajectory],
    *,
    stream_id: str,
    partition: str,
    member_families: list[str],
    discovery_families: list[str],
    confirmation_families: list[str],
    windows: int,
    window_size: int,
    min_successes: int,
    used_task_ids: set[str],
    max_success_candidates: int = 4,
    min_targets_per_donor: int = 4,
    min_strict_singles: int = 2,
) -> list[dict]:
    members = [item for item in indexed if item.trajectory.env_family in member_families]
    planned: list[dict] = []
    chunks = len(members) // window_size
    for chunk in range(chunks):
        window = members[chunk * window_size : (chunk + 1) * window_size]
        successes = [item for item in window if bool(item.trajectory.success)]
        if len(successes) < min_successes:
            continue
        eligible_donors = _structurally_eligible_donors(
            window,
            max_success_candidates=max_success_candidates,
            min_targets_per_donor=min_targets_per_donor,
        )
        if len(eligible_donors) < min_strict_singles:
            continue
        end = max(item.index for item in window)
        discovery = _probe_set(
            indexed,
            after=end,
            families=discovery_families,
            used_task_ids=used_task_ids,
        )
        confirmation = _probe_set(
            indexed,
            after=end,
            families=confirmation_families,
            used_task_ids=used_task_ids,
        )
        if discovery is None or confirmation is None:
            continue
        probes = discovery + confirmation
        probe_ids = {item.trajectory.task_id for item in probes}
        member_ids = {item.trajectory.task_id for item in window}
        if member_ids & probe_ids or len(probe_ids) != len(probes):
            raise RuntimeError("member/probe selection was not task-disjoint")
        used_task_ids.update(probe_ids)
        planned.append(
            {
                "window_id": f"stable-{partition}-{stream_id}-w{chunk:04d}",
                "stream_id": stream_id,
                "partition": partition,
                "member_indices": [item.index for item in window],
                "member_task_ids": [item.trajectory.task_id for item in window],
                "member_families": [item.trajectory.env_family for item in window],
                "successful_member_task_ids": [
                    item.trajectory.task_id for item in successes
                ],
                "successful_member_families": [
                    item.trajectory.env_family for item in successes
                ],
                "structurally_eligible_donor_task_ids": [
                    item.trajectory.task_id for item in eligible_donors
                ],
                "structurally_eligible_donor_families": [
                    item.trajectory.env_family for item in eligible_donors
                ],
                "discovery_task_ids": [item.trajectory.task_id for item in discovery],
                "discovery_families": [item.trajectory.env_family for item in discovery],
                "confirmation_task_ids": [
                    item.trajectory.task_id for item in confirmation
                ],
                "confirmation_families": [
                    item.trajectory.env_family for item in confirmation
                ],
                "future_tasks_used_for_candidate_update": False,
            }
        )
        if len(planned) == windows:
            return planned
    return planned


def build_plan(
    buffers: list[str | Path],
    partition_path: str | Path,
    *,
    train_per_stream: int = 4,
    test_per_stream: int = 3,
    window_size: int = 16,
    min_successes: int = 2,
    max_success_candidates: int = 4,
    min_targets_per_donor: int = 4,
    min_strict_singles: int = 2,
) -> dict:
    with Path(partition_path).open(encoding="utf-8") as handle:
        partition = json.load(handle)
    if partition.get("format") != "sediment-stable-family-partition-v1":
        raise ValueError("unsupported family partition manifest")
    used_task_ids: set[str] = set()
    windows: list[dict] = []
    shortfalls: list[dict] = []
    for buffer_index, buffer_path in enumerate(buffers):
        buffer = Buffer.load(buffer_path)
        indexed = [
            IndexedTrajectory(index, trajectory)
            for index, trajectory in enumerate(buffer._trajs)
            if not trajectory.is_retry
        ]
        stream_id = f"s{buffer_index}"
        for role, wanted in (("train", train_per_stream), ("test", test_per_stream)):
            prefix = "train" if role == "train" else "test"
            planned = _plan_stream_partition(
                indexed,
                stream_id=stream_id,
                partition=role,
                member_families=partition[f"{prefix}_members"],
                discovery_families=partition[f"{prefix}_discovery"],
                confirmation_families=partition[f"{prefix}_confirmation"],
                windows=wanted,
                window_size=window_size,
                min_successes=min_successes,
                max_success_candidates=max_success_candidates,
                min_targets_per_donor=min_targets_per_donor,
                min_strict_singles=min_strict_singles,
                used_task_ids=used_task_ids,
            )
            windows.extend(planned)
            if len(planned) != wanted:
                shortfalls.append(
                    {
                        "stream_id": stream_id,
                        "partition": role,
                        "wanted": wanted,
                        "planned": len(planned),
                    }
                )
    payload = {
        "format": "sediment-stable-window-plan-v1",
        "family_partition": str(Path(partition_path).resolve()),
        "family_partition_digest": partition["digest"],
        "candidate_rewards_inspected": False,
        "window_size": window_size,
        "min_successes": min_successes,
        "max_success_candidates": max_success_candidates,
        "min_targets_per_donor": min_targets_per_donor,
        "min_strict_singles": min_strict_singles,
        "train_per_stream": train_per_stream,
        "test_per_stream": test_per_stream,
        "source_buffers": [str(Path(value).resolve()) for value in buffers],
        "windows": windows,
        "shortfalls": shortfalls,
    }
    digest_payload = dict(payload)
    digest_payload.pop("family_partition")
    payload["digest"] = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer", action="append", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--train-per-stream", type=int, default=4)
    parser.add_argument("--test-per-stream", type=int, default=3)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--min-successes", type=int, default=2)
    parser.add_argument("--allow-shortfalls", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if min(
        args.train_per_stream,
        args.test_per_stream,
        args.window_size,
        args.min_successes,
    ) <= 0:
        raise SystemExit("window counts, size, and minimum successes must be positive")
    report = build_plan(
        args.buffer,
        args.partition,
        train_per_stream=args.train_per_stream,
        test_per_stream=args.test_per_stream,
        window_size=args.window_size,
        min_successes=args.min_successes,
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    summary = {
        "format": report["format"],
        "digest": report["digest"],
        "family_partition_digest": report["family_partition_digest"],
        "windows": len(report["windows"]),
        "train_windows": sum(row["partition"] == "train" for row in report["windows"]),
        "test_windows": sum(row["partition"] == "test" for row in report["windows"]),
        "unique_probe_tasks": len(
            {
                task_id
                for row in report["windows"]
                for task_id in row["discovery_task_ids"] + row["confirmation_task_ids"]
            }
        ),
        "candidate_rewards_inspected": False,
        "shortfalls": report["shortfalls"],
        "output": str(target.resolve()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if report["shortfalls"] and not args.allow_shortfalls:
        raise SystemExit(f"window plan has shortfalls: {report['shortfalls']}")
    return report


if __name__ == "__main__":
    main()
