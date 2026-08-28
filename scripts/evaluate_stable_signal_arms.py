#!/usr/bin/env python3
"""Unblind and evaluate the preregistered six-arm stable-signal test."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from collect_stable_signal_proposals import (  # noqa: E402
    _family_gains,
    _load_task_map,
    _paired_evidence,
)
from sediment.compiler.basis import UpdateBasis  # noqa: E402
from sediment.compiler.stable_data import (  # noqa: E402
    load_stable_windows,
    memory_task_cluster_id,
)
from sediment.compiler.stable_train import (  # noqa: E402
    load_stable_signal_checkpoint,
    predict_stable_record,
)
from sediment.compiler.state import load_tensor_state  # noqa: E402
from sediment.config import StreamConfig  # noqa: E402
from sediment.engine.vllm_client import VllmClient  # noqa: E402
from sediment.types import AdapterVersion  # noqa: E402


ARMS = ("noop", "single", "heuristic_joint", "mean", "stable", "shuffled")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_gain(gains: dict[str, list[float]]) -> float:
    values = [float(value) for family in gains.values() for value in family]
    return float(np.mean(values)) if values else 0.0


def _materialize_adapter(
    output: Path,
    reference_path: Path,
    reference: dict[str, np.ndarray],
    basis: UpdateBasis,
    coefficients: list[float],
    metadata: dict[str, Any],
) -> Path:
    delta = basis.reconstruct_state(np.asarray(coefficients, dtype=np.float32))
    if set(delta) != set(reference):
        raise ValueError("basis/reference tensor keys differ")
    state = {
        key: np.asarray(reference[key], dtype=np.float32)
        + np.asarray(delta[key], dtype=np.float32)
        for key in sorted(reference)
    }
    if not all(np.isfinite(value).all() for value in state.values()):
        raise ValueError("synthesized stable adapter is non-finite")
    output.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.numpy import save_file
    except ImportError as exc:  # pragma: no cover - remote dependency
        raise RuntimeError("stable adapter materialization requires safetensors") from exc
    save_file(state, str(output / "adapter_model.safetensors"))
    shutil.copy2(reference_path / "adapter_config.json", output / "adapter_config.json")
    with (output / "adapter_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "format": "sediment-stable-synthesized-adapter-v1",
                "reference": str(reference_path),
                "coefficients": [float(value) for value in coefficients],
                **metadata,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return output


def _derangement(records, *, seed: int) -> dict[str, Any]:
    groups: dict[str, list[Any]] = {}
    for record in records:
        groups.setdefault(memory_task_cluster_id(record), []).append(record)
    if len(groups) < 2:
        raise ValueError(
            "shuffled control requires at least two source-memory task clusters"
        )
    ordered_clusters = sorted(
        groups,
        key=lambda cluster: hashlib.sha256(f"{seed}:{cluster}".encode()).hexdigest(),
    )
    shifted_clusters = ordered_clusters[1:] + ordered_clusters[:1]
    mapping: dict[str, Any] = {}
    for source_cluster, shuffled_cluster in zip(
        ordered_clusters, shifted_clusters
    ):
        candidates = sorted(
            groups[shuffled_cluster], key=lambda record: record.window_id
        )
        by_stream = {record.stream_id: record for record in candidates}
        for index, record in enumerate(
            sorted(groups[source_cluster], key=lambda item: item.window_id)
        ):
            mapping[record.window_id] = by_stream.get(
                record.stream_id, candidates[index % len(candidates)]
            )
    if any(
        memory_task_cluster_id(record)
        == memory_task_cluster_id(mapping[record.window_id])
        for record in records
    ):
        raise AssertionError("cluster-level derangement failed")
    return mapping


def _select_baselines(record) -> tuple[dict[str, dict[str, list[float]]], dict]:
    singles = list(record.proposals)
    selected = max(
        singles,
        key=lambda proposal: (
            float(np.mean(list(proposal.discovery_family_gains.values()))),
            proposal.proposal_id,
        ),
    )
    baselines = {
        str(candidate["kind"]): candidate
        for candidate in record.metadata["baseline_candidates"]
    }
    if "heuristic_joint" not in baselines or "mean" not in baselines:
        raise ValueError(f"{record.window_id}: missing joint or mean baseline")
    confirmation_families = record.metadata["confirmation_families"]
    repeats = int(record.metadata["confirmation_repeats"])
    arms = {
        "noop": {family: [0.0] * repeats for family in confirmation_families},
        "single": selected.confirmation_family_gains,
        "heuristic_joint": {
            str(key): [float(value) for value in values]
            for key, values in baselines["heuristic_joint"][
                "confirmation_family_gains"
            ].items()
        },
        "mean": {
            str(key): [float(value) for value in values]
            for key, values in baselines["mean"]["confirmation_family_gains"].items()
        },
    }
    return arms, {
        "selected_single": selected.proposal_id,
        "selected_single_discovery_gain": float(
            np.mean(list(selected.discovery_family_gains.values()))
        ),
    }


def _exact_sign_flip_p(differences: list[float]) -> float:
    values = np.asarray(differences, dtype=np.float64)
    observed = float(values.mean())
    if observed <= 0.0:
        return 1.0
    absolute = np.abs(values)
    total = 2 ** len(values)
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted = float(np.mean(absolute * np.asarray(signs)))
        extreme += int(permuted >= observed - 1e-12)
    return extreme / total


def _bootstrap_interval(
    differences: list[float], *, seed: int, samples: int = 10000
) -> list[float]:
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _paired_comparison(
    left: list[float],
    right: list[float],
    *,
    seed: int,
) -> dict:
    differences = (
        np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    ).tolist()
    return {
        "mean": float(np.mean(differences)),
        "median": float(np.median(differences)),
        "positive_memory_clusters": int(sum(value > 0 for value in differences)),
        "negative_memory_clusters": int(sum(value < 0 for value in differences)),
        "equal_memory_clusters": int(sum(value == 0 for value in differences)),
        "memory_cluster_bootstrap_95": _bootstrap_interval(differences, seed=seed),
        "exact_one_sided_sign_flip_p": _exact_sign_flip_p(differences),
        "per_memory_cluster": differences,
    }


def _cluster_means(per_window: list[dict], values: list[float]):
    grouped: dict[str, list[float]] = {}
    for window, value in zip(per_window, values):
        grouped.setdefault(str(window["source_memory_cluster"]), []).append(
            float(value)
        )
    clusters = sorted(grouped)
    return clusters, [float(np.mean(grouped[cluster])) for cluster in clusters]


def aggregate_six_arms(
    per_window: list[dict],
    *,
    seed: int = 20260827,
    catastrophe_floor: float = -0.20,
) -> dict:
    by_arm_window = {
        arm: [float(window["arm_mean_gains"][arm]) for window in per_window]
        for arm in ARMS
    }
    cluster_ids: list[str] | None = None
    by_arm: dict[str, list[float]] = {}
    for arm in ARMS:
        clusters, values = _cluster_means(per_window, by_arm_window[arm])
        if cluster_ids is None:
            cluster_ids = clusters
        elif clusters != cluster_ids:
            raise ValueError("arm aggregation produced inconsistent memory clusters")
        by_arm[arm] = values
    cluster_ids = cluster_ids or []
    stable = np.asarray(by_arm["stable"], dtype=np.float64)
    comparisons = {}
    for comparator in ("noop", "single", "heuristic_joint", "mean", "shuffled"):
        comparisons[f"stable_minus_{comparator}"] = _paired_comparison(
            stable.tolist(),
            by_arm[comparator],
            seed=seed + len(comparisons),
        )
    comparisons["shuffled_minus_noop"] = _paired_comparison(
        by_arm["shuffled"],
        by_arm["noop"],
        seed=seed + len(comparisons),
    )
    family_cluster_values: dict[str, dict[str, list[float]]] = {}
    for window in per_window:
        for family, values in window["arm_family_gains"]["stable"].items():
            family_cluster_values.setdefault(family, {}).setdefault(
                str(window["source_memory_cluster"]), []
            ).extend(float(value) for value in values)
    family_means = {
        family: float(
            np.mean([np.mean(values) for values in cluster_values.values()])
        )
        for family, cluster_values in sorted(family_cluster_values.items())
    }
    leave_one_family_out: dict[str, float] = {}
    for omitted in sorted(family_cluster_values):
        window_means = []
        for window in per_window:
            retained = [
                float(value)
                for family, values in window["arm_family_gains"]["stable"].items()
                if family != omitted
                for value in values
            ]
            if not retained:
                raise ValueError(f"omitting {omitted!r} leaves a window without stable probes")
            window_means.append(float(np.mean(retained)))
        _, cluster_loo = _cluster_means(per_window, window_means)
        leave_one_family_out[omitted] = float(np.mean(cluster_loo))

    primary_names = [
        "stable_minus_noop",
        "stable_minus_single",
        "stable_minus_heuristic_joint",
        "stable_minus_mean",
    ]
    efficacy_requirements = {
        "stable_beats_noop_and_three_donor_baselines": all(
            comparisons[name]["mean"] > 0.0 for name in primary_names
        ),
        "stable_beats_shuffled": comparisons["stable_minus_shuffled"]["mean"] > 0.0,
        "shuffled_has_no_positive_aggregate_advantage_over_noop": comparisons[
            "shuffled_minus_noop"
        ]["mean"]
        <= 0.0,
        "stable_worst_family_above_catastrophe_floor": bool(family_means)
        and min(family_means.values()) >= catastrophe_floor,
        "stable_gain_positive_after_omitting_each_family": bool(leave_one_family_out)
        and all(value > 0.0 for value in leave_one_family_out.values()),
    }
    precision_requirements = {
        "primary_bootstrap_lower_bounds_positive": all(
            comparisons[name]["memory_cluster_bootstrap_95"][0] > 0.0
            for name in primary_names
        ),
        "primary_one_sided_p_values_below_0_05": all(
            comparisons[name]["exact_one_sided_sign_flip_p"] < 0.05
            for name in primary_names
        ),
        "shuffled_noop_interval_contains_zero": (
            comparisons["shuffled_minus_noop"]["memory_cluster_bootstrap_95"][0]
            <= 0.0
            <= comparisons["shuffled_minus_noop"]["memory_cluster_bootstrap_95"][1]
        ),
    }
    cluster_sizes = {
        cluster: sum(
            str(window["source_memory_cluster"]) == cluster for window in per_window
        )
        for cluster in cluster_ids
    }
    _, write_by_cluster = _cluster_means(
        per_window,
        [float(window["stable_prediction"]["gate_passed"]) for window in per_window],
    )
    result = {
        "windows": len(per_window),
        "source_memory_task_clusters": len(cluster_ids),
        "source_memory_cluster_sizes": cluster_sizes,
        "arm_mean_gains": {arm: float(np.mean(values)) for arm, values in by_arm.items()},
        "window_weighted_arm_mean_gains": {
            arm: float(np.mean(values)) for arm, values in by_arm_window.items()
        },
        "comparisons": comparisons,
        "stable_family_mean_gains": family_means,
        "stable_worst_family_gain": min(family_means.values()) if family_means else None,
        "stable_leave_one_family_out_mean_gains": leave_one_family_out,
        "stable_write_rate": float(np.mean(write_by_cluster)),
        "efficacy_claim_gate": {
            "catastrophe_floor": catastrophe_floor,
            "requirements": efficacy_requirements,
            "passed": all(efficacy_requirements.values()),
        },
        "sampling_precision_diagnostics": {
            "requirements": precision_requirements,
            "all_passed": all(precision_requirements.values()),
            "interpretation": (
                "Descriptive only: failure does not negate a directional pilot, and "
                "a null shuffled effect is not established by non-significance alone."
            ),
        },
    }
    return result


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--projection-report", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--basis", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--url", action="append", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--data-dir", default="/data/erv1n/resid/data")
    parser.add_argument("--lopd-dir", default="/data/erv1n/resid/third_party/LOPD")
    parser.add_argument("--max-seq-len", type=int, default=12288)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    with Path(args.projection_report).open(encoding="utf-8") as handle:
        projection_report = json.load(handle)
    model, checkpoint = load_stable_signal_checkpoint(args.checkpoint, device=args.device)
    provenance = checkpoint["report"].get("provenance", {})
    frozen_plan_path = Path(provenance.get("frozen_plan", ""))
    with frozen_plan_path.open(encoding="utf-8") as handle:
        frozen_plan = json.load(handle)
    expected_test_ids = {
        str(row["window_id"])
        for row in frozen_plan.get("windows", [])
        if row.get("partition") == "test"
    }
    if (
        frozen_plan.get("format") != "sediment-stable-window-plan-v1"
        or frozen_plan.get("shortfalls")
        or not expected_test_ids
    ):
        raise ValueError("checkpoint references an invalid frozen test plan")
    if (
        projection_report.get("mode") != "project-test"
        or projection_report.get("test_windows") != len(expected_test_ids)
        or Path(projection_report.get("output", "")).resolve()
        != Path(args.data).resolve()
        or projection_report.get("output_sha256") != _sha256(args.data)
        or projection_report.get("basis_sha256") != _sha256(args.basis)
        or Path(projection_report.get("checkpoint", "")).resolve()
        != Path(args.checkpoint).resolve()
    ):
        raise ValueError("test projection report does not match data/basis/checkpoint")
    records = [record for record in load_stable_windows(args.data) if record.partition == "test"]
    if {record.window_id for record in records} != expected_test_ids:
        raise ValueError("test data does not exactly match the frozen outer window-id set")
    if checkpoint["report"].get("test_windows_unread") != len(expected_test_ids):
        raise ValueError("checkpoint does not prove all outer test labels stayed unread")
    if (
        provenance.get("outer_test_labels_loaded") is not False
        or provenance.get("outer_test_windows") != len(expected_test_ids)
        or provenance.get("basis_sha256") != _sha256(args.basis)
        or provenance.get("frozen_plan_digest") != frozen_plan.get("digest")
    ):
        raise ValueError("checkpoint provenance does not match the blinded outer test")
    gate = checkpoint.get("deployment_gate", {})
    if gate.get("test_labels_used") is not False:
        raise ValueError("deployment gate was not frozen without test labels")
    basis = UpdateBasis.load(args.basis)
    reference_path = Path(args.reference).resolve()
    reference = load_tensor_state(reference_path)
    if basis.layout.keys != tuple(sorted(reference)):
        raise ValueError("basis layout differs from the reference adapter")
    shuffled = _derangement(records, seed=args.seed)
    task_map = _load_task_map(args.data_dir, args.lopd_dir)
    prefix = f"stable-eval-{args.seed}-"
    engines = [
        VllmClient(
            url,
            args.model,
            max_context=args.max_seq_len,
            lora_prefix=prefix,
            generation_seed_mode="bare_prompt_hash",
        )
        for url in args.url
    ]
    cfg = StreamConfig(
        model=args.model,
        engine="vllm",
        temperature=0.0,
        max_tokens=args.max_tokens,
        max_model_len=args.max_seq_len,
        max_steps=args.max_steps,
        data_dir=args.data_dir,
        split="rl",
        generation_seed_mode="bare_prompt_hash",
    )
    reference_name = "reference"
    reference_version = AdapterVersion(reference_name, str(reference_path), None)
    for engine in engines:
        engine.load_adapter(reference_version)
    output = Path(args.output).resolve()
    adapter_root = output.parent / "synthesized_adapters"
    per_window: list[dict] = []
    try:
        for record in records:
            baseline_arms, baseline_meta = _select_baselines(record)
            stable_prediction = predict_stable_record(model, checkpoint, record)
            shuffled_record = shuffled[record.window_id]
            shuffled_prediction = predict_stable_record(model, checkpoint, shuffled_record)
            arm_gains = dict(baseline_arms)
            for arm, prediction, source_window in (
                ("stable", stable_prediction, record.window_id),
                ("shuffled", shuffled_prediction, shuffled_record.window_id),
            ):
                adapter_path = _materialize_adapter(
                    adapter_root / record.window_id / arm,
                    reference_path,
                    reference,
                    basis,
                    prediction["coefficients"],
                    {
                        "arm": arm,
                        "evaluation_window": record.window_id,
                        "source_window": source_window,
                        "prediction": prediction,
                    },
                )
                adapter_name = f"{arm}-{hashlib.sha256(record.window_id.encode()).hexdigest()[:10]}"
                version = AdapterVersion(adapter_name, str(adapter_path), None)
                for engine in engines:
                    engine.load_adapter(version)
                try:
                    tasks = [
                        task_map[task_id]
                        for task_id in record.metadata["confirmation_task_ids"]
                    ]
                    rows = _paired_evidence(
                        engines,
                        tasks,
                        cfg,
                        reference_adapter=reference_name,
                        candidate_adapter=adapter_name,
                        repeats=int(record.metadata["confirmation_repeats"]),
                    )
                finally:
                    for engine in engines:
                        engine.unload_adapter(adapter_name)
                arm_gains[arm] = _family_gains(rows)
            per_window.append(
                {
                    "window_id": record.window_id,
                    "source_memory_cluster": memory_task_cluster_id(record),
                    **baseline_meta,
                    "shuffled_source_window": shuffled_record.window_id,
                    "stable_prediction": stable_prediction,
                    "shuffled_prediction": shuffled_prediction,
                    "arm_family_gains": arm_gains,
                    "arm_mean_gains": {
                        arm: _mean_gain(arm_gains[arm]) for arm in ARMS
                    },
                }
            )
            print(
                f"[stable-eval] {record.window_id} "
                f"write={stable_prediction['gate_passed']} "
                f"stable={per_window[-1]['arm_mean_gains']['stable']:+.4f}",
                flush=True,
            )
    finally:
        for engine in engines:
            engine.unload_adapter(reference_name)

    aggregate = aggregate_six_arms(per_window, seed=args.seed)
    report = {
        "format": "sediment-stable-six-arm-evaluation-v1",
        "data": str(Path(args.data).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "projection_report": str(Path(args.projection_report).resolve()),
        "basis": str(Path(args.basis).resolve()),
        "test_labels_used_for_training": False,
        "single_selected_on_discovery_only": True,
        "confirmation_repeats": 3,
        "temperature": 0.0,
        "arms": list(ARMS),
        "deployment_gate": gate,
        "aggregate": aggregate,
        "per_window": per_window,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
