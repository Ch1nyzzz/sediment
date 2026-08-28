#!/usr/bin/env python3
"""Audit stable-signal proposal manifests against the frozen window plan."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                row["__manifest_dir"] = str(path.resolve().parent)
                rows.append(row)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def _gain_shape(candidate: dict, discovery_families: list[str], confirmation_families: list[str], repeats: int):
    discovery = candidate.get("discovery_family_gains", {})
    confirmation = candidate.get("confirmation_family_gains", {})
    return {
        "discovery_families_exact": set(discovery) == set(discovery_families),
        "discovery_finite": all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in discovery.values()
        ),
        "confirmation_families_exact": set(confirmation) == set(confirmation_families),
        "confirmation_repeats_exact": all(
            isinstance(values, list) and len(values) == repeats
            for values in confirmation.values()
        ),
        "confirmation_finite": all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for values in confirmation.values()
            if isinstance(values, list)
            for value in values
        ),
    }


def audit_manifests(manifests: list[str | Path], plan_path: str | Path) -> dict:
    with Path(plan_path).open(encoding="utf-8") as handle:
        plan = json.load(handle)
    if plan.get("format") != "sediment-stable-window-plan-v1":
        raise ValueError("unsupported frozen window plan")
    planned = {row["window_id"]: row for row in plan["windows"]}
    rows = [row for manifest in manifests for row in _read_jsonl(Path(manifest))]
    ids = [row.get("window_id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("proposal manifests contain duplicate window ids")
    audits = []
    for row in rows:
        window_id = str(row.get("window_id"))
        expected = planned.get(window_id)
        errors: list[str] = []
        if expected is None:
            errors.append("window_not_in_frozen_plan")
            expected = {}
        if row.get("format") != "sediment-stable-proposal-window-v1":
            errors.append("format")
        if row.get("stream_id") != expected.get("stream_id"):
            errors.append("stream_id")
        if row.get("partition") != expected.get("partition"):
            errors.append("partition")
        member_ids = list(row.get("member_ids", []))
        member_families = list(row.get("member_families", []))
        if member_ids != expected.get("member_task_ids") or len(member_ids) != 16:
            errors.append("members")
        if member_families != expected.get("member_families"):
            errors.append("member_families")
        metadata = row.get("metadata", {})
        if metadata.get("family_partition_digest") != plan.get("family_partition_digest"):
            errors.append("family_partition_digest")
        if metadata.get("window_plan_digest") != plan.get("digest"):
            errors.append("window_plan_digest")
        if metadata.get("future_tasks_used_for_candidate_update") is not False:
            errors.append("future_update_leak_flag")
        if metadata.get("candidate_reward_used_for_window_selection") is not False:
            errors.append("reward_selection_leak_flag")
        discovery_ids = list(metadata.get("discovery_task_ids", []))
        confirmation_ids = list(metadata.get("confirmation_task_ids", []))
        discovery_families = list(metadata.get("discovery_families", []))
        confirmation_families = list(metadata.get("confirmation_families", []))
        repeats = int(metadata.get("confirmation_repeats", 0))
        if discovery_ids != expected.get("discovery_task_ids"):
            errors.append("discovery_tasks")
        if confirmation_ids != expected.get("confirmation_task_ids"):
            errors.append("confirmation_tasks")
        if discovery_families != expected.get("discovery_families"):
            errors.append("discovery_families")
        if confirmation_families != expected.get("confirmation_families"):
            errors.append("confirmation_families")
        if repeats != 3:
            errors.append("confirmation_repeats")
        if set(member_ids) & set(discovery_ids + confirmation_ids):
            errors.append("member_probe_task_overlap")
        if set(member_families) & set(discovery_families + confirmation_families):
            errors.append("member_probe_family_overlap")

        candidates = list(row.get("candidates", []))
        by_kind: dict[str, list[dict]] = {}
        for candidate in candidates:
            by_kind.setdefault(str(candidate.get("kind")), []).append(candidate)
        singles = by_kind.get("single", [])
        if len(by_kind.get("noop", [])) != 1:
            errors.append("noop_count")
        if len(singles) < 2 or len(singles) > 4:
            errors.append("single_count")
        if len(by_kind.get("heuristic_joint", [])) != 1:
            errors.append("joint_count")
        if len(by_kind.get("mean", [])) != 1:
            errors.append("mean_count")
        if set(by_kind) != {"noop", "single", "heuristic_joint", "mean"}:
            errors.append("candidate_kinds")

        proposal_ids = {str(candidate.get("proposal_id")) for candidate in singles}
        for candidate in candidates:
            kind = candidate.get("kind")
            raw_update_path = candidate.get("update_path")
            if not isinstance(raw_update_path, str) or not raw_update_path:
                errors.append(f"{kind}_update_path")
            else:
                update_path = Path(raw_update_path)
                if not update_path.is_absolute():
                    update_path = Path(row["__manifest_dir"]) / update_path
                if not (
                    (update_path / "adapter_model.safetensors").is_file()
                    and (update_path / "adapter_config.json").is_file()
                ):
                    errors.append(f"{kind}_adapter_artifacts")
            if kind == "noop":
                if candidate.get("optimizer_steps") != 0:
                    errors.append("noop_steps")
                continue
            shape = _gain_shape(
                candidate, discovery_families, confirmation_families, repeats
            )
            errors.extend(
                f"{kind}_{name}" for name, passed in shape.items() if not passed
            )
            audit = candidate.get("parameter_audit", {})
            if audit.get("finite") is not True or int(audit.get("changed_tensors", 0)) <= 0:
                errors.append(f"{kind}_parameter_delta")
            if kind in {"single", "heuristic_joint"} and candidate.get("optimizer_steps") != 2:
                errors.append(f"{kind}_steps")
            if kind in {"single", "heuristic_joint"}:
                selected = list(candidate.get("selected_member_ids", []))
                if kind == "single":
                    selected = [candidate.get("member_id")]
                if not selected or any(member_id not in member_ids for member_id in selected):
                    errors.append(f"{kind}_members")
                pairs = candidate.get("pair_diagnostics", [])
                if (
                    (kind == "single" and len(pairs) != 4)
                    or (kind == "heuristic_joint" and len(pairs) < 4)
                    or any(pair.get("error") is not None for pair in pairs)
                ):
                    errors.append(f"{kind}_pair_count")
                if any(
                    pair.get("donor_id") not in member_ids
                    or pair.get("target_id") not in member_ids
                    for pair in pairs
                ):
                    errors.append(f"{kind}_pair_scope")
            if kind == "mean":
                if candidate.get("optimizer_steps") != 0:
                    errors.append("mean_steps")
                if set(candidate.get("derived_from", [])) != proposal_ids:
                    errors.append("mean_sources")
        audits.append(
            {
                "window_id": window_id,
                "partition": row.get("partition"),
                "candidates": len(candidates),
                "singles": len(singles),
                "passed": not errors,
                "errors": sorted(set(errors)),
            }
        )
    collected = set(ids)
    return {
        "format": "sediment-stable-proposal-audit-v1",
        "manifests": [str(Path(path).resolve()) for path in manifests],
        "frozen_plan": str(Path(plan_path).resolve()),
        "windows_collected": len(rows),
        "train_windows": sum(row.get("partition") == "train" for row in rows),
        "test_windows": sum(row.get("partition") == "test" for row in rows),
        "planned_windows_missing": sorted(set(planned) - collected),
        "unknown_windows": sorted(collected - set(planned)),
        "all_passed": all(audit["passed"] for audit in audits),
        "windows": audits,
    }


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = audit_manifests(args.manifest, args.plan)
    if args.require_complete and report["planned_windows_missing"]:
        report["all_passed"] = False
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_passed"]:
        raise SystemExit("stable proposal audit failed")
    return report


if __name__ == "__main__":
    main()
