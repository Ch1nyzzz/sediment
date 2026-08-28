#!/usr/bin/env python3
"""Partition, project, audit, and train the stable memory-signal experiment."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sediment.buffer import Buffer  # noqa: E402
from sediment.compiler.basis import UpdateBasis  # noqa: E402
from sediment.compiler.stable_data import (  # noqa: E402
    DonorProposal,
    StableWindowRecord,
    load_stable_windows,
    member_task_cluster_id,
    partition_families,
    select_validation_memory_clusters,
    stable_feature_tensors,
    stable_target,
    write_stable_windows,
)
from sediment.compiler.stable_model import StableSignalConfig  # noqa: E402
from sediment.compiler.stable_train import (  # noqa: E402
    StableTrainingConfig,
    train_stable_signal,
)
from sediment.compiler.state import load_tensor_state, load_update  # noqa: E402


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: str | Path) -> list[dict]:
    source = Path(path).resolve()
    rows: list[dict] = []
    with source.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{lineno}: invalid JSON: {exc}") from exc
            row["__manifest_dir"] = str(source.parent)
            rows.append(row)
    if not rows:
        raise ValueError(f"{source}: no records")
    return rows


def _resolve(path: str, row: dict) -> Path:
    source = Path(path)
    return source if source.is_absolute() else Path(row["__manifest_dir"]) / source


def _load_frozen_plan(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        plan = json.load(handle)
    if (
        plan.get("format") != "sediment-stable-window-plan-v1"
        or plan.get("candidate_rewards_inspected") is not False
        or plan.get("shortfalls")
    ):
        raise ValueError("frozen plan is incomplete or not reward-blind")
    return plan


def _planned_ids(plan: dict, partition: str) -> set[str]:
    return {
        str(row["window_id"])
        for row in plan.get("windows", [])
        if row.get("partition") == partition
    }


def partition_command(args) -> dict:
    family_stats: dict[str, dict[str, int]] = {}
    for buffer_path in args.buffer:
        buffer = Buffer.load(buffer_path)
        for trajectory in buffer._trajs:
            if trajectory.is_retry:
                continue
            stats = family_stats.setdefault(
                trajectory.env_family, {"completed": 0, "successful": 0}
            )
            stats["completed"] += 1
            stats["successful"] += int(bool(trajectory.success))
    partition = partition_families(
        family_stats,
        seed=args.seed,
        counts=tuple(args.counts),
        success_counts={
            family: stats["successful"] for family, stats in family_stats.items()
        },
    )
    payload = {
        "format": "sediment-stable-family-partition-v1",
        **partition.__dict__,
        "family_stats": family_stats,
        "stratification": "completed_success_count_greedy_balance_v1",
        "role_counts": list(args.counts),
        "source_buffers": [str(Path(value).resolve()) for value in args.buffer],
        "candidate_rewards_inspected": False,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def _single_candidates(row: dict) -> list[dict]:
    candidates = [
        candidate
        for candidate in row.get("candidates", [])
        if candidate.get("kind") == "single"
    ]
    if not candidates:
        raise ValueError(f"window {row.get('window_id')!r} has no single donor proposals")
    return candidates


def build_data_command(args) -> dict:
    checkpoint_report_path = None
    checkpoint_report = None
    checkpoint_provenance = None
    if args.mode == "project-test":
        if not args.checkpoint:
            raise ValueError("project-test requires a frozen checkpoint")
        checkpoint_report_path = Path(args.checkpoint).with_suffix(
            Path(args.checkpoint).suffix + ".report.json"
        )
        with checkpoint_report_path.open(encoding="utf-8") as handle:
            checkpoint_report = json.load(handle)
        checkpoint_provenance = checkpoint_report.get("provenance", {})
    frozen_plan = _load_frozen_plan(args.frozen_plan)
    expected_partition = "train" if args.mode == "fit-train" else "test"
    expected_ids = _planned_ids(frozen_plan, expected_partition)
    if not expected_ids:
        raise ValueError(f"frozen plan has no {expected_partition} windows")
    if args.mode == "project-test":
        assert checkpoint_report is not None
        assert checkpoint_provenance is not None
        if (
            checkpoint_report.get("format") != "sediment-stable-signal-v1"
            or checkpoint_report.get("test_windows_unread") != len(expected_ids)
            or checkpoint_provenance.get("outer_test_labels_loaded") is not False
            or checkpoint_provenance.get("basis_sha256") != _sha256(args.basis)
            or checkpoint_provenance.get("frozen_plan_digest")
            != frozen_plan.get("digest")
        ):
            raise ValueError(
                "checkpoint does not prove a frozen basis and all unread outer-test labels"
            )
    rows = [row for manifest in args.manifest for row in _jsonl(manifest)]
    window_ids = [str(row.get("window_id")) for row in rows]
    if len(window_ids) != len(set(window_ids)):
        raise ValueError("stable proposal manifests contain duplicate window ids")
    if set(window_ids) != expected_ids or {row.get("partition") for row in rows} != {
        expected_partition
    }:
        raise ValueError(
            f"{args.mode} manifests do not exactly match the frozen "
            f"{expected_partition} window-id set ({len(expected_ids)} windows)"
        )
    if any(
        row.get("metadata", {}).get("window_plan_digest")
        != frozen_plan.get("digest")
        for row in rows
    ):
        raise ValueError("proposal manifests disagree with the frozen window plan")
    for row in rows:
        row["__memory_task_cluster"] = member_task_cluster_id(row["member_ids"])
    reference = load_tensor_state(args.reference)
    if args.mode == "fit-train":
        if args.checkpoint:
            raise ValueError("fit-train must run before a checkpoint exists")
        train_clusters = {row["__memory_task_cluster"] for row in rows}
        validation_clusters = select_validation_memory_clusters(
            train_clusters,
            fraction=args.validation_cluster_fraction,
            seed=args.seed,
        )
        train_candidates = [
            (row, candidate)
            for row in rows
            if row["__memory_task_cluster"] not in validation_clusters
            for candidate in _single_candidates(row)
        ]
        if not train_candidates:
            raise ValueError(
                "no fitting-cluster single proposals are available for the basis"
            )
        first = load_update(
            _resolve(train_candidates[0][1]["update_path"], train_candidates[0][0]),
            reference,
        )
        parameter_count = sum(value.size for value in first.values())
        dense_bytes = len(train_candidates) * parameter_count * 4
        if dense_bytes > args.max_dense_gb * 1024**3:
            raise SystemExit(
                f"proposal basis would allocate about {dense_bytes / 1024**3:.1f} GiB; "
                "raise --max-dense-gb only after checking host memory"
            )
        updates = (
            load_update(_resolve(candidate["update_path"], row), reference)
            for row, candidate in train_candidates[1:]
        )
        basis = UpdateBasis.fit(itertools.chain([first], updates), args.rank)
        basis.save(args.basis)
        checkpoint_report = None
    else:
        assert checkpoint_report is not None
        validation_clusters = list(
            checkpoint_report["validation_memory_task_clusters"]
        )
        train_clusters = set(
            checkpoint_report["train_memory_task_clusters"]
        ) | set(validation_clusters)
        train_candidates = []
        basis = UpdateBasis.load(args.basis)

    projected: list[StableWindowRecord] = []
    reconstruction_errors: list[float] = []
    for row in rows:
        proposals: list[DonorProposal] = []
        for candidate in _single_candidates(row):
            update = load_update(_resolve(candidate["update_path"], row), reference)
            coefficients = basis.project_state(update)
            error = basis.reconstruction_error(update)
            reconstruction_errors.append(error)
            features = {
                str(key): float(value)
                for key, value in dict(candidate["intervention_features"]).items()
            }
            features["basis_reconstruction_error"] = float(error)
            proposals.append(
                DonorProposal(
                    proposal_id=str(candidate["proposal_id"]),
                    member_id=str(candidate["member_id"]),
                    member_family=str(candidate["member_family"]),
                    coefficients=coefficients.tolist(),
                    intervention_features=features,
                    discovery_family_gains={
                        str(key): float(value)
                        for key, value in candidate.get("discovery_family_gains", {}).items()
                    },
                    confirmation_family_gains={
                        str(key): [float(value) for value in values]
                        for key, values in candidate.get(
                            "confirmation_family_gains", {}
                        ).items()
                    },
                    metadata={
                        "update_path": str(_resolve(candidate["update_path"], row).resolve()),
                    },
                )
            )
        metadata = dict(row.get("metadata", {}))
        metadata["memory_task_cluster_id"] = row["__memory_task_cluster"]
        metadata["basis_fit_excluded_memory_task_clusters"] = validation_clusters
        metadata["baseline_candidates"] = [
            {
                **candidate,
                "update_path": str(_resolve(candidate["update_path"], row).resolve()),
            }
            for candidate in row.get("candidates", [])
            if candidate.get("kind") in {"noop", "heuristic_joint", "mean"}
        ]
        projected.append(
            StableWindowRecord(
                window_id=str(row["window_id"]),
                stream_id=str(row["stream_id"]),
                partition=str(row["partition"]),
                member_ids=[str(value) for value in row["member_ids"]],
                member_families=[str(value) for value in row["member_families"]],
                member_features=[dict(value) for value in row["member_features"]],
                proposals=proposals,
                metadata=metadata,
            )
        )
    write_stable_windows(args.output, projected)
    report: dict = {
        "format": "sediment-stable-projection-v1",
        "mode": args.mode,
        "partition": expected_partition,
        "basis_fit_excluded_memory_task_clusters": validation_clusters,
        "basis_rank": basis.rank,
        "basis_sha256": _sha256(args.basis),
        "windows": len(projected),
        "train_windows": sum(record.partition == "train" for record in projected),
        "test_windows": sum(record.partition == "test" for record in projected),
        "mean_relative_reconstruction_error": float(np.mean(reconstruction_errors)),
        "basis": str(Path(args.basis).resolve()),
        "output": str(Path(args.output).resolve()),
        "output_sha256": _sha256(args.output),
        "frozen_plan": str(Path(args.frozen_plan).resolve()),
        "frozen_plan_digest": frozen_plan["digest"],
    }
    if args.mode == "fit-train":
        report.update(
            {
                "basis_fitted_on_partition": "train",
                "basis_fitted_on_memory_task_clusters": sorted(
                    train_clusters - set(validation_clusters)
                ),
                "basis_fit_used_validation_inputs": False,
                "basis_training_proposals": len(train_candidates),
                "captured_energy": float(basis.explained_energy.sum()),
            }
        )
    else:
        report.update(
            {
                "basis_frozen_before_test_projection": True,
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "checkpoint_report": str(checkpoint_report_path.resolve()),
            }
        )
    with Path(args.output).with_suffix(".report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def audit_targets_command(args) -> dict:
    records = load_stable_windows(args.data)
    rows = []
    for record in records:
        target = stable_target(
            record, kappa=args.kappa, catastrophe_floor=args.catastrophe_floor
        )
        rows.append(
            {
                "window_id": record.window_id,
                "partition": record.partition,
                "write": target.write,
                "accepted_proposals": target.proposal_ids,
                "conservative_gain": target.conservative_gain,
                "worst_family_gain": target.worst_family_gain,
                "coefficient_norm": float(np.linalg.norm(target.coefficients)),
                "coefficient_dispersion": target.coefficient_dispersion,
                "proposal_evidence": [item.__dict__ for item in target.evidence],
            }
        )
    report = {
        "format": "sediment-stable-target-audit-v1",
        "windows": len(rows),
        "train_windows": sum(row["partition"] == "train" for row in rows),
        "test_windows": sum(row["partition"] == "test" for row in rows),
        "train_write_rate": (
            float(np.mean([row["write"] for row in rows if row["partition"] == "train"]))
            if any(row["partition"] == "train" for row in rows)
            else None
        ),
        "test_write_rate": (
            float(np.mean([row["write"] for row in rows if row["partition"] == "test"]))
            if any(row["partition"] == "test" for row in rows)
            else None
        ),
        "kappa": args.kappa,
        "catastrophe_floor": args.catastrophe_floor,
        "targets": rows,
    }
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return report


def train_command(args) -> dict:
    records = load_stable_windows(args.data)
    frozen_plan = _load_frozen_plan(args.frozen_plan)
    planned_train = _planned_ids(frozen_plan, "train")
    planned_test = _planned_ids(frozen_plan, "test")
    if (
        {record.window_id for record in records} != planned_train
        or any(record.partition != "train" for record in records)
    ):
        raise ValueError(
            "stable training data does not exactly match the frozen train window-id set"
        )
    with Path(args.fit_report).open(encoding="utf-8") as handle:
        fit_report = json.load(handle)
    with Path(args.proposal_audit).open(encoding="utf-8") as handle:
        proposal_audit = json.load(handle)
    if (
        fit_report.get("mode") != "fit-train"
        or Path(fit_report.get("output", "")).resolve() != Path(args.data).resolve()
        or fit_report.get("output_sha256") != _sha256(args.data)
        or fit_report.get("basis_sha256") != _sha256(args.basis)
    ):
        raise ValueError("fit report does not match the train data and frozen basis")
    if (
        not planned_test
        or proposal_audit.get("format") != "sediment-stable-proposal-audit-v1"
        or proposal_audit.get("all_passed") is not True
        or proposal_audit.get("train_windows") != len(planned_train)
        or proposal_audit.get("test_windows") != len(planned_test)
        or proposal_audit.get("planned_windows_missing") != []
        or Path(proposal_audit.get("frozen_plan", "")).resolve()
        != Path(args.frozen_plan).resolve()
    ):
        raise ValueError(
            "frozen plan/audit does not prove the exact untouched outer-test windows"
        )
    if any(
        record.metadata.get("window_plan_digest") != frozen_plan.get("digest")
        for record in records
    ):
        raise ValueError("train records disagree with the frozen window plan digest")
    member, coefficients, intervention, *_ = stable_feature_tensors(records)
    model_config = StableSignalConfig(
        member_feature_dim=member.shape[-1],
        proposal_feature_dim=intervention.shape[-1],
        basis_rank=coefficients.shape[-1],
        member_hidden=args.member_hidden,
        window_hidden=args.window_hidden,
        attention_temperature=args.attention_temperature,
        deployment_topk=args.deployment_topk,
        max_step_norm=args.max_step_norm,
        initial_write_probability=args.initial_write_probability,
    )
    train_config = StableTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_cluster_fraction=args.validation_cluster_fraction,
        kappa=args.kappa,
        catastrophe_floor=args.catastrophe_floor,
        write_loss_weight=args.write_loss_weight,
        consistency_loss_weight=args.consistency_loss_weight,
        noop_loss_weight=args.noop_loss_weight,
        seed=args.seed,
        bootstrap_views=args.bootstrap_views,
        bootstrap_size=args.bootstrap_size,
        write_threshold=args.write_threshold,
        uncertainty_quantile=args.uncertainty_quantile,
        device=args.device,
    )
    provenance = {
        "outer_test_labels_loaded": False,
        "outer_test_windows": len(planned_test),
        "frozen_plan": str(Path(args.frozen_plan).resolve()),
        "frozen_plan_digest": frozen_plan["digest"],
        "proposal_audit": str(Path(args.proposal_audit).resolve()),
        "basis": str(Path(args.basis).resolve()),
        "basis_sha256": _sha256(args.basis),
        "train_data": str(Path(args.data).resolve()),
        "train_data_sha256": _sha256(args.data),
    }
    return train_stable_signal(
        records,
        model_config,
        train_config,
        args.output,
        heldout_test_count=len(planned_test),
        provenance=provenance,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    partition = subparsers.add_parser("partition")
    partition.add_argument("--buffer", action="append", required=True)
    partition.add_argument("--seed", type=int, default=20260827)
    partition.add_argument(
        "--counts",
        nargs=6,
        type=int,
        default=(10, 4, 4, 8, 4, 4),
        metavar=("TRAIN_MEMBER", "TRAIN_DISC", "TRAIN_CONF", "TEST_MEMBER", "TEST_DISC", "TEST_CONF"),
    )
    partition.add_argument("--output", required=True)
    partition.set_defaults(func=partition_command)

    build = subparsers.add_parser("build-data")
    build.add_argument("--mode", choices=("fit-train", "project-test"), required=True)
    build.add_argument("--manifest", action="append", required=True)
    build.add_argument("--reference", required=True)
    build.add_argument("--frozen-plan", required=True)
    build.add_argument("--rank", type=int, default=16)
    build.add_argument("--max-dense-gb", type=float, default=8.0)
    build.add_argument("--validation-cluster-fraction", type=float, default=0.25)
    build.add_argument("--seed", type=int, default=20260827)
    build.add_argument("--basis", required=True)
    build.add_argument("--checkpoint")
    build.add_argument("--output", required=True)
    build.set_defaults(func=build_data_command)

    audit = subparsers.add_parser("audit-targets")
    audit.add_argument("--data", required=True)
    audit.add_argument("--kappa", type=float, default=1.0)
    audit.add_argument("--catastrophe-floor", type=float, default=-0.2)
    audit.add_argument("--output")
    audit.set_defaults(func=audit_targets_command)

    train = subparsers.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--fit-report", required=True)
    train.add_argument("--basis", required=True)
    train.add_argument("--frozen-plan", required=True)
    train.add_argument("--proposal-audit", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--validation-cluster-fraction", type=float, default=0.25)
    train.add_argument("--kappa", type=float, default=1.0)
    train.add_argument("--catastrophe-floor", type=float, default=-0.2)
    train.add_argument("--write-loss-weight", type=float, default=0.5)
    train.add_argument("--consistency-loss-weight", type=float, default=0.25)
    train.add_argument("--noop-loss-weight", type=float, default=0.25)
    train.add_argument("--member-hidden", type=int, default=32)
    train.add_argument("--window-hidden", type=int, default=32)
    train.add_argument("--attention-temperature", type=float, default=1.0)
    train.add_argument("--deployment-topk", type=int, default=8)
    train.add_argument("--max-step-norm", type=float, default=12.0)
    train.add_argument("--initial-write-probability", type=float, default=0.01)
    train.add_argument("--bootstrap-views", type=int, default=4)
    train.add_argument("--bootstrap-size", type=int, default=8)
    train.add_argument("--write-threshold", type=float, default=0.5)
    train.add_argument("--uncertainty-quantile", type=float, default=0.75)
    train.add_argument("--seed", type=int, default=20260827)
    train.add_argument("--device", default="auto")
    train.set_defaults(func=train_command)
    return parser


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    report = args.func(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
