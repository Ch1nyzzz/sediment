#!/usr/bin/env python3
"""Collect preregistered donor proposals and paired future-reward evidence.

The script consumes a frozen window plan. It never chooses member or probe tasks
from observed candidate reward. Each non-noop update starts from one common
reference LoRA. Test gains are written for later blinded evaluation but are not
printed before the stable model and gates are frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from collect_window_oracles import (  # noqa: E402
    _donor_kl_specs,
    _load_task_map,
    _pair_metadata,
)
from sediment.buffer import Buffer  # noqa: E402
from sediment.compiler.oracle import deployment_features  # noqa: E402
from sediment.compiler.state import load_tensor_state  # noqa: E402
from sediment.config import StreamConfig  # noqa: E402
from sediment.engine.vllm_client import VllmClient  # noqa: E402
from sediment.rollout.agent_loop import run_episode  # noqa: E402
from sediment.scheduler import _make_env  # noqa: E402
from sediment.trainer import train_candidate  # noqa: E402
from sediment.types import AdapterVersion  # noqa: E402


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {
            str(json.loads(line)["window_id"])
            for line in handle
            if line.strip()
        }


def _relative(path: str | Path, base: Path) -> str:
    return os.path.relpath(Path(path).resolve(), base.resolve())


def _paired_repeat(
    engines: list[VllmClient],
    tasks: list[dict[str, Any]],
    cfg: StreamConfig,
    *,
    reference_adapter: str,
    candidate_adapter: str,
    repeat: int,
) -> list[dict[str, Any]]:
    shards = [tasks[index :: len(engines)] for index in range(len(engines))]
    order = (
        (reference_adapter, candidate_adapter)
        if repeat % 2 == 0
        else (candidate_adapter, reference_adapter)
    )

    def run_shard(item):
        engine_index, shard = item
        engine = engines[engine_index]
        rows: list[dict[str, Any]] = []
        for task in shard:
            rewards: dict[str, float] = {}
            for adapter in order:
                trajectory = run_episode(
                    engine,
                    _make_env(task),
                    task,
                    cfg,
                    adapter=adapter,
                )
                key = (
                    "reference_reward"
                    if adapter == reference_adapter
                    else "candidate_reward"
                )
                rewards[key] = float(trajectory.reward or 0.0)
            rows.append(
                {
                    "repeat": repeat,
                    "task_id": str(task["task_id"]),
                    "family": str(task["env_family"]),
                    "engine_index": engine_index,
                    "order": [
                        "reference" if adapter == reference_adapter else "candidate"
                        for adapter in order
                    ],
                    **rewards,
                    "delta": rewards["candidate_reward"]
                    - rewards["reference_reward"],
                }
            )
        return rows

    with ThreadPoolExecutor(max_workers=len(engines)) as pool:
        nested = list(pool.map(run_shard, enumerate(shards)))
    return [row for shard in nested for row in shard]


def _paired_evidence(
    engines: list[VllmClient],
    tasks: list[dict[str, Any]],
    cfg: StreamConfig,
    *,
    reference_adapter: str,
    candidate_adapter: str,
    repeats: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        rows.extend(
            _paired_repeat(
                engines,
                tasks,
                cfg,
                reference_adapter=reference_adapter,
                candidate_adapter=candidate_adapter,
                repeat=repeat,
            )
        )
    return rows


def _family_gains(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for row in rows:
        result.setdefault(str(row["family"]), []).append(float(row["delta"]))
    return result


def _intervention_features(pairs: list[dict[str, Any]]) -> dict[str, float]:
    usable = [pair for pair in pairs if pair.get("error") is None]
    if not usable:
        raise ValueError("donor proposal has no usable intervention pairs")

    def values(name: str) -> np.ndarray:
        return np.asarray([float(pair[name]) for pair in usable], dtype=np.float64)

    action_shift = values("action_shift")
    act_gain = values("act_gain")
    obs_surprise = values("obs_surprise")
    teacher_positions = values("teacher_positions")
    similarity = values("task_similarity")
    return {
        "action_shift_mean": float(action_shift.mean()),
        "action_shift_std": float(action_shift.std()),
        "action_shift_max": float(action_shift.max()),
        "act_gain_mean": float(act_gain.mean()),
        "obs_surprise_mean": float(obs_surprise.mean()),
        "teacher_positions_mean": float(teacher_positions.mean()),
        "teacher_positions_total": float(teacher_positions.sum()),
        "task_similarity_mean": float(similarity.mean()),
        "target_count": float(len(usable)),
        "target_family_count": float(len({pair["target_family"] for pair in usable})),
    }


def _delta_audit(path: str | Path, reference: dict[str, np.ndarray]) -> dict[str, Any]:
    state = load_tensor_state(path)
    if set(state) != set(reference):
        raise ValueError("candidate/reference tensor keys differ")
    squared = 0.0
    changed = 0
    finite = True
    for key in sorted(state):
        value = np.asarray(state[key], dtype=np.float32)
        base = np.asarray(reference[key], dtype=np.float32)
        if value.shape != base.shape:
            raise ValueError(f"candidate/reference shape differs for {key}")
        delta = value - base
        finite = finite and bool(np.isfinite(delta).all())
        changed += int(bool(np.any(delta != 0.0)))
        squared += float(np.square(delta.astype(np.float64)).sum())
    return {
        "tensors": len(state),
        "changed_tensors": changed,
        "finite": finite,
        "delta_norm": math.sqrt(squared),
    }


def _write_mean_adapter(
    output: Path,
    reference_path: Path,
    candidate_paths: list[Path],
) -> Path:
    if len(candidate_paths) < 2:
        raise ValueError("mean baseline requires at least two single proposals")
    reference = load_tensor_state(reference_path)
    states = [load_tensor_state(path) for path in candidate_paths]
    if any(set(state) != set(reference) for state in states):
        raise ValueError("mean adapter candidates do not share reference keys")
    mean_state: dict[str, np.ndarray] = {}
    for key in sorted(reference):
        base = np.asarray(reference[key], dtype=np.float32)
        deltas = []
        for state in states:
            value = np.asarray(state[key], dtype=np.float32)
            if value.shape != base.shape:
                raise ValueError(f"mean adapter shape mismatch for {key}")
            deltas.append(value - base)
        mean_state[key] = base + np.mean(deltas, axis=0, dtype=np.float32)
    output.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.numpy import save_file
    except ImportError as exc:  # pragma: no cover - remote training dependency
        raise RuntimeError("writing a mean LoRA requires safetensors") from exc
    save_file(mean_state, str(output / "adapter_model.safetensors"))
    shutil.copy2(reference_path / "adapter_config.json", output / "adapter_config.json")
    with (output / "adapter_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "format": "sediment-mean-donor-adapter-v1",
                "reference": str(reference_path),
                "candidates": [str(path) for path in candidate_paths],
                "coordinate_mean": "reference + mean(candidate - reference)",
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return output


def _candidate_evidence(
    engines: list[VllmClient],
    discovery_tasks: list[dict[str, Any]],
    confirmation_tasks: list[dict[str, Any]],
    cfg: StreamConfig,
    *,
    reference_adapter: str,
    candidate_adapter: str,
    confirmation_repeats: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    discovery = _paired_evidence(
        engines,
        discovery_tasks,
        cfg,
        reference_adapter=reference_adapter,
        candidate_adapter=candidate_adapter,
        repeats=1,
    )
    confirmation = _paired_evidence(
        engines,
        confirmation_tasks,
        cfg,
        reference_adapter=reference_adapter,
        candidate_adapter=candidate_adapter,
        repeats=confirmation_repeats,
    )
    return discovery, confirmation


def _load_plan(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        plan = json.load(handle)
    if plan.get("format") != "sediment-stable-window-plan-v1":
        raise ValueError("unsupported stable window plan")
    if plan.get("candidate_rewards_inspected") is not False or plan.get("shortfalls"):
        raise ValueError("window plan was not a complete reward-blind plan")
    return plan


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--stream-id", required=True)
    parser.add_argument("--partition", choices=("train", "test"), required=True)
    parser.add_argument("--window-id", action="append")
    parser.add_argument("--limit-windows", type=int)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--url", action="append", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--data-dir", default="/data/erv1n/resid/data")
    parser.add_argument("--lopd-dir", default="/data/erv1n/resid/third_party/LOPD")
    parser.add_argument("--max-success-candidates", type=int, default=4)
    parser.add_argument("--max-targets-per-donor", type=int, default=4)
    parser.add_argument("--min-targets-per-donor", type=int, default=4)
    parser.add_argument("--confirmation-repeats", type=int, default=3)
    parser.add_argument("--kl-topk", type=int, default=20)
    parser.add_argument("--max-block-chars", type=int, default=16000)
    parser.add_argument("--max-result-chars", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--anchor-kl-coef", type=float, default=0.5)
    parser.add_argument("--max-seq-len", type=int, default=12288)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--trainer", choices=("torch", "stub"), default="torch")
    args = parser.parse_args(argv)
    if args.confirmation_repeats < 2:
        raise SystemExit("confirmation-repeats must be at least 2")
    if args.steps != 2:
        raise SystemExit("the frozen pilot protocol requires exactly two optimizer steps")
    if args.min_targets_per_donor != 4 or args.max_targets_per_donor != 4:
        raise SystemExit("the frozen pilot protocol requires exactly four targets per donor")

    plan = _load_plan(args.plan)
    try:
        stream_index = int(args.stream_id.removeprefix("s"))
        buffer_path = plan["source_buffers"][stream_index]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"invalid stream id {args.stream_id!r}") from exc
    planned = [
        row
        for row in plan["windows"]
        if row["stream_id"] == args.stream_id and row["partition"] == args.partition
    ]
    if args.window_id:
        wanted = set(args.window_id)
        planned = [row for row in planned if row["window_id"] in wanted]
        missing = wanted - {row["window_id"] for row in planned}
        if missing:
            raise ValueError(f"planned windows not found: {sorted(missing)}")
    if args.limit_windows is not None:
        planned = planned[: args.limit_windows]
    if not planned:
        raise ValueError("no frozen windows match the requested stream/partition")

    reference_path = Path(args.reference).resolve()
    if args.trainer == "torch" and not (
        reference_path / "adapter_model.safetensors"
    ).exists():
        raise FileNotFoundError(f"missing reference adapter: {reference_path}")
    reference_state = load_tensor_state(reference_path) if args.trainer == "torch" else None
    output = Path(args.output).resolve()
    manifest = output / "stable_proposal_manifest.jsonl"
    done = _completed(manifest)
    buffer = Buffer.load(buffer_path)
    trajectory_by_index = {
        index: trajectory
        for index, trajectory in enumerate(buffer._trajs)
        if not trajectory.is_retry
    }
    task_map = _load_task_map(args.data_dir, args.lopd_dir)
    prefix = f"stable-{args.partition}-{args.stream_id}-{os.getpid()}-"
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
    for engine in engines:
        engine._tokenizer()
    eval_cfg = StreamConfig(
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
    train_cfg = StreamConfig(
        model=args.model,
        trainer=args.trainer,
        weight_mode="sft",
        train_channels="act",
        grounded_weight=0.2,
        anchor_kl_coef=args.anchor_kl_coef,
        kl_target=True,
        kl_reverse=True,
        kl_topk=args.kl_topk,
        max_block_chars=args.max_block_chars,
        max_result_chars=args.max_result_chars,
        lora_r=args.rank,
        lora_alpha=args.alpha,
        lr=args.learning_rate,
        epochs=args.steps,
        steps_per_merge=args.steps,
        max_seq_len=args.max_seq_len,
        retry_on_fail=False,
        reflect=False,
    )
    reference_name = "reference"
    reference_version = AdapterVersion(
        name=reference_name, path=str(reference_path), parent=None
    )
    for engine in engines:
        engine.load_adapter(reference_version)

    written = skipped = 0
    try:
        for window in planned:
            window_id = str(window["window_id"])
            if window_id in done:
                skipped += 1
                continue
            members = [trajectory_by_index[int(index)] for index in window["member_indices"]]
            if [member.task_id for member in members] != window["member_task_ids"]:
                raise RuntimeError(f"{window_id}: source buffer no longer matches frozen plan")
            discovery_tasks = [task_map[task_id] for task_id in window["discovery_task_ids"]]
            confirmation_tasks = [
                task_map[task_id] for task_id in window["confirmation_task_ids"]
            ]
            member_ids = {member.task_id for member in members}
            member_families = {member.env_family for member in members}
            probe_ids = set(window["discovery_task_ids"] + window["confirmation_task_ids"])
            probe_families = set(
                window["discovery_families"] + window["confirmation_families"]
            )
            if member_ids & probe_ids or member_families & probe_families:
                raise RuntimeError(f"{window_id}: member/probe leakage")

            candidate_specs, pair_diagnostics = _donor_kl_specs(
                engines,
                members,
                train_cfg,
                max_success_candidates=args.max_success_candidates,
                max_targets_per_donor=args.max_targets_per_donor,
                min_targets_per_donor=args.min_targets_per_donor,
            )
            single_specs = [spec for spec in candidate_specs if ":single:" in spec["name"]]
            joint_specs = [spec for spec in candidate_specs if spec["name"] == "donor_kl:joint"]
            if len(single_specs) < 2 or len(joint_specs) != 1:
                raise RuntimeError(
                    f"{window_id}: expected >=2 strict singles and one joint, got "
                    f"{len(single_specs)} singles/{len(joint_specs)} joints"
                )
            ordered_specs = single_specs + joint_specs
            candidates: list[dict[str, Any]] = [
                {
                    "kind": "noop",
                    "name": "noop",
                    "update_path": _relative(reference_path, manifest.parent),
                    "selected_member_ids": [],
                    "optimizer_steps": 0,
                }
            ]
            single_paths: list[Path] = []
            for spec in ordered_specs:
                policy = str(spec["name"])
                selected = list(spec["selected"])
                samples = list(spec["samples"])
                parent = AdapterVersion(
                    name=f"ref-{window_id}-{policy}",
                    path=str(reference_path),
                    parent=None,
                )
                trained = train_candidate(
                    samples,
                    parent,
                    train_cfg,
                    str(output / "candidates" / window_id),
                )
                if trained.train_stats.get("optimizer_steps") != 2:
                    raise RuntimeError(f"{window_id}/{policy}: did not complete two steps")
                version = AdapterVersion(
                    name=trained.candidate_id,
                    path=trained.adapter_path,
                    parent=parent.name,
                )
                for engine in engines:
                    engine.load_adapter(version)
                try:
                    discovery, confirmation = _candidate_evidence(
                        engines,
                        discovery_tasks,
                        confirmation_tasks,
                        eval_cfg,
                        reference_adapter=reference_name,
                        candidate_adapter=trained.candidate_id,
                        confirmation_repeats=args.confirmation_repeats,
                    )
                finally:
                    for engine in engines:
                        engine.unload_adapter(trained.candidate_id)
                audit = (
                    _delta_audit(trained.adapter_path, reference_state)
                    if reference_state is not None
                    else {"tensors": 0, "changed_tensors": 0, "finite": True, "delta_norm": 0.0}
                )
                if not audit["finite"] or audit["changed_tensors"] == 0:
                    raise RuntimeError(f"{window_id}/{policy}: invalid parameter delta")
                pairs = [_pair_metadata(pair) for pair in spec["pairs"]]
                kind = "single" if ":single:" in policy else "heuristic_joint"
                row = {
                    "kind": kind,
                    "name": policy,
                    "update_path": _relative(trained.adapter_path, manifest.parent),
                    "selected_member_ids": [item.task_id for item in selected],
                    "optimizer_steps": 2,
                    "losses": [float(value) for value in trained.train_stats.get("losses", [])],
                    "parameter_audit": audit,
                    "intervention_features": _intervention_features(pairs),
                    "discovery_family_gains": {
                        family: values[0]
                        for family, values in _family_gains(discovery).items()
                    },
                    "confirmation_family_gains": _family_gains(confirmation),
                    "task_pairs": {
                        "discovery": discovery,
                        "confirmation": confirmation,
                    },
                    "pair_diagnostics": pairs,
                }
                if kind == "single":
                    donor = selected[0]
                    row.update(
                        {
                            "proposal_id": policy,
                            "member_id": donor.task_id,
                            "member_family": donor.env_family,
                        }
                    )
                    single_paths.append(Path(trained.adapter_path))
                candidates.append(row)

            mean_hash = hashlib.sha256(window_id.encode()).hexdigest()[:10]
            mean_path = _write_mean_adapter(
                output / "candidates" / window_id / f"mean-{mean_hash}",
                reference_path,
                single_paths,
            )
            mean_name = f"mean-{mean_hash}"
            mean_version = AdapterVersion(name=mean_name, path=str(mean_path), parent=None)
            for engine in engines:
                engine.load_adapter(mean_version)
            try:
                discovery, confirmation = _candidate_evidence(
                    engines,
                    discovery_tasks,
                    confirmation_tasks,
                    eval_cfg,
                    reference_adapter=reference_name,
                    candidate_adapter=mean_name,
                    confirmation_repeats=args.confirmation_repeats,
                )
            finally:
                for engine in engines:
                    engine.unload_adapter(mean_name)
            mean_audit = (
                _delta_audit(mean_path, reference_state)
                if reference_state is not None
                else {"tensors": 0, "changed_tensors": 0, "finite": True, "delta_norm": 0.0}
            )
            if not mean_audit["finite"] or mean_audit["changed_tensors"] == 0:
                raise RuntimeError(f"{window_id}/mean: invalid parameter delta")
            candidates.append(
                {
                    "kind": "mean",
                    "name": "mean",
                    "update_path": _relative(mean_path, manifest.parent),
                    "selected_member_ids": [
                        candidate["member_id"]
                        for candidate in candidates
                        if candidate["kind"] == "single"
                    ],
                    "optimizer_steps": 0,
                    "derived_from": [
                        candidate["proposal_id"]
                        for candidate in candidates
                        if candidate["kind"] == "single"
                    ],
                    "parameter_audit": mean_audit,
                    "discovery_family_gains": {
                        family: values[0]
                        for family, values in _family_gains(discovery).items()
                    },
                    "confirmation_family_gains": _family_gains(confirmation),
                    "task_pairs": {
                        "discovery": discovery,
                        "confirmation": confirmation,
                    },
                }
            )
            record = {
                "format": "sediment-stable-proposal-window-v1",
                "window_id": window_id,
                "stream_id": args.stream_id,
                "partition": args.partition,
                "member_ids": [member.task_id for member in members],
                "member_families": [member.env_family for member in members],
                "member_features": [deployment_features(member) for member in members],
                "candidates": candidates,
                "metadata": {
                    "family_partition_digest": plan["family_partition_digest"],
                    "window_plan_digest": plan["digest"],
                    "source_buffer": str(Path(buffer_path).resolve()),
                    "reference": str(reference_path),
                    "discovery_task_ids": window["discovery_task_ids"],
                    "discovery_families": window["discovery_families"],
                    "confirmation_task_ids": window["confirmation_task_ids"],
                    "confirmation_families": window["confirmation_families"],
                    "confirmation_repeats": args.confirmation_repeats,
                    "future_tasks_used_for_candidate_update": False,
                    "candidate_reward_used_for_window_selection": False,
                    "pair_diagnostics": pair_diagnostics,
                },
            }
            _append_jsonl(manifest, record)
            done.add(window_id)
            written += 1
            if args.partition == "train":
                discovery_best = max(
                    (
                        np.mean(list(candidate.get("discovery_family_gains", {}).values()))
                        for candidate in candidates
                        if candidate["kind"] != "noop"
                    ),
                    default=0.0,
                )
                print(
                    f"[stable-proposal] {window_id} candidates={len(candidates)} "
                    f"best_discovery={discovery_best:+.4f}",
                    flush=True,
                )
            else:
                print(
                    f"[stable-proposal] {window_id} candidates={len(candidates)} "
                    "test_gains=BLINDED",
                    flush=True,
                )
    finally:
        for engine in engines:
            engine.unload_adapter(reference_name)

    report = {
        "format": "sediment-stable-proposal-collection-v1",
        "partition": args.partition,
        "stream_id": args.stream_id,
        "planned": len(planned),
        "written": written,
        "skipped": skipped,
        "manifest": str(manifest),
        "family_partition_digest": plan["family_partition_digest"],
        "window_plan_digest": plan["digest"],
        "test_gains_printed": False if args.partition == "test" else None,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
