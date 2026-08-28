#!/usr/bin/env python3
"""Collect real window-level update oracles from complete successful episodes.

For each chronological window, all candidates start from one common reference
LoRA. Candidate actions are: no-op, one update per successful complete episode,
and one joint update over the successful episodes. ``trajectory_sft`` copies
the successful source actions directly. ``donor_kl`` instead lets the frozen
actor read exactly one complete successful donor while scoring other current-
window trajectories, then distils that context-induced policy distribution
into prompt-only parameters. Candidates are selected only by reward on later
tasks; future tasks never enter the update. This is an offline meta-training
oracle, not a deployable selector.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sediment import experience as experience_mod  # noqa: E402
from sediment import hindsight as hindsight_mod  # noqa: E402
from sediment.buffer import Buffer  # noqa: E402
from sediment.compiler.oracle import deployment_features  # noqa: E402
from sediment.config import StreamConfig  # noqa: E402
from sediment.engine.vllm_client import VllmClient  # noqa: E402
from sediment.envs.envscaler import list_tasks  # noqa: E402
from sediment.hindsight import sft_sample  # noqa: E402
from sediment.rollout.agent_loop import run_episode  # noqa: E402
from sediment.scheduler import _make_env  # noqa: E402
from sediment.trainer import train_candidate  # noqa: E402
from sediment.types import AdapterVersion, Trajectory  # noqa: E402


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
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


def _future_slices(
    trajectories: list[Trajectory],
    end: int,
    member_families: set[str],
    *,
    next_size: int,
    cross_size: int,
) -> tuple[list[Trajectory], list[Trajectory]]:
    future = trajectories[end:]
    near = future[:next_size]
    eligible = [
        trajectory
        for trajectory in future
        if trajectory.env_family not in member_families
    ]
    cross: list[Trajectory] = []
    used_families: set[str] = set()
    for trajectory in eligible:
        if trajectory.env_family in used_families:
            continue
        cross.append(trajectory)
        used_families.add(trajectory.env_family)
        if len(cross) == cross_size:
            return near, cross
    for trajectory in eligible:
        if trajectory not in cross:
            cross.append(trajectory)
        if len(cross) == cross_size:
            break
    return near, cross


def _candidate_specs(
    members: list[Trajectory], max_success_candidates: int
) -> list[tuple[str, list[Trajectory]]]:
    successes = [trajectory for trajectory in members if bool(trajectory.success)]
    successes = successes[:max_success_candidates]
    specs = [(f"single:{trajectory.task_id}", [trajectory]) for trajectory in successes]
    if len(successes) > 1:
        specs.append(("joint:successful", successes))
    return specs


def _first_user_text(trajectory: Trajectory) -> str:
    return next(
        (message.content for message in trajectory.messages if message.role == "user"),
        "",
    )


def _lexical_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", text.lower()) if len(token) > 2}


def _task_similarity(left: Trajectory, right: Trajectory) -> float:
    left_tokens = _lexical_tokens(_first_user_text(left))
    right_tokens = _lexical_tokens(_first_user_text(right))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def _trajectory_chars(trajectory: Trajectory) -> int:
    return sum(len(message.content) for message in trajectory.messages)


def _donor_targets(
    members: list[Trajectory], donor: Trajectory, limit: int
) -> list[Trajectory]:
    """Rank diverse cross-family target states for one complete donor.

    The target outcome is intentionally absent from the ranking. Similar task
    text makes the intervention relevant; shorter trajectories win exact ties
    because the teacher must fit donor + target into one context. A coverage
    pass takes at most one target per family before any family is repeated.
    """

    candidates = [
        target
        for target in members
        if target.task_id != donor.task_id
        and target.env_family != donor.env_family
        and any(message.role == "assistant" for message in target.messages)
    ]
    candidates.sort(
        key=lambda target: (
            _task_similarity(donor, target),
            -_trajectory_chars(target),
            target.task_id,
        ),
        reverse=True,
    )
    selected: list[Trajectory] = []
    used_families: set[str] = set()
    for target in candidates:
        if target.env_family in used_families:
            continue
        selected.append(target)
        used_families.add(target.env_family)
        if len(selected) == limit:
            return selected
    for target in candidates:
        if target not in selected:
            selected.append(target)
        if len(selected) == limit:
            break
    return selected


def _action_shift(hindsight) -> float:
    values = [
        abs(delta)
        for span in hindsight.spans
        if span.role == "assistant"
        for delta in span.deltas
    ]
    return sum(values) / len(values) if values else 0.0


def _trainable_teacher_positions(sample) -> int:
    teacher = sample.teacher_by_msg or [[] for _ in sample.messages]
    return sum(
        1
        for weights, distributions in zip(sample.token_weights_by_msg, teacher)
        for weight, distribution in zip(weights, distributions)
        if weight != 0.0 and distribution
    )


def _score_donor_target(
    engine: VllmClient,
    donor: Trajectory,
    target: Trajectory,
    cfg: StreamConfig,
) -> dict[str, Any]:
    """Build one leak-free donor teacher and return an in-memory KL sample."""

    try:
        target_view = dataclasses.replace(target, adapter="base", meta=dict(target.meta))
        block = experience_mod.build_block(
            [donor],
            None,
            cfg,
        )
        if block.source_task_ids != [donor.task_id] or block.includes_own_outcome:
            raise RuntimeError("donor block violated single-source/no-own-outcome contract")
        hindsight = hindsight_mod.score(engine, target_view, block, cfg)
        sample = hindsight_mod.to_train_sample(target_view, hindsight, cfg)
        teacher_positions = _trainable_teacher_positions(sample)
        if teacher_positions == 0:
            raise RuntimeError("no aligned action-token teacher positions")
        return {
            "donor_id": donor.task_id,
            "donor_family": donor.env_family,
            "target_id": target.task_id,
            "target_family": target.env_family,
            "task_similarity": _task_similarity(donor, target),
            "block_chars": len(block.text),
            "teacher_positions": teacher_positions,
            "action_shift": _action_shift(hindsight),
            "act_gain": float(hindsight.act_gain),
            "obs_surprise": float(hindsight.obs_surprise),
            "sample": sample,
            "error": None,
        }
    except Exception as error:  # one overflow/alignment miss must not kill the window
        return {
            "donor_id": donor.task_id,
            "donor_family": donor.env_family,
            "target_id": target.task_id,
            "target_family": target.env_family,
            "task_similarity": _task_similarity(donor, target),
            "sample": None,
            "error": f"{type(error).__name__}: {error}",
        }


def _pair_metadata(pair: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in pair.items() if key != "sample"}


def _donor_kl_specs(
    engines: list[VllmClient],
    members: list[Trajectory],
    cfg: StreamConfig,
    *,
    max_success_candidates: int,
    max_targets_per_donor: int,
    min_targets_per_donor: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score donor-target interventions and construct singles plus one joint.

    The joint arm never concatenates memories. For a target covered by multiple
    donors it keeps exactly one teacher: the donor producing the largest action-
    distribution shift on that target's frozen states.
    """

    donors = [member for member in members if bool(member.success)][
        :max_success_candidates
    ]
    jobs = [
        (donor, target)
        for donor in donors
        for target in _donor_targets(members, donor, max_targets_per_donor)
    ]

    def score_job(item):
        index, (donor, target) = item
        return _score_donor_target(
            engines[index % len(engines)], donor, target, cfg
        )

    with ThreadPoolExecutor(max_workers=max(1, min(len(jobs), len(engines)))) as pool:
        pairs = list(pool.map(score_job, enumerate(jobs)))

    specs: list[dict[str, Any]] = []
    usable = [pair for pair in pairs if pair["sample"] is not None]
    for donor in donors:
        donor_pairs = [pair for pair in usable if pair["donor_id"] == donor.task_id]
        if len(donor_pairs) < min_targets_per_donor:
            continue
        specs.append(
            {
                "name": f"donor_kl:single:{donor.task_id}",
                "selected": [donor],
                "samples": [pair["sample"] for pair in donor_pairs],
                "pairs": donor_pairs,
            }
        )

    eligible_donor_ids = {
        spec["selected"][0].task_id for spec in specs if spec["selected"]
    }
    best_by_target: dict[str, dict[str, Any]] = {}
    for pair in usable:
        if pair["donor_id"] not in eligible_donor_ids:
            continue
        prior = best_by_target.get(pair["target_id"])
        score = (
            float(pair["action_shift"]),
            float(pair["act_gain"]),
            int(pair["teacher_positions"]),
            str(pair["donor_id"]),
        )
        if prior is None:
            best_by_target[pair["target_id"]] = pair
            continue
        prior_score = (
            float(prior["action_shift"]),
            float(prior["act_gain"]),
            int(prior["teacher_positions"]),
            str(prior["donor_id"]),
        )
        if score > prior_score:
            best_by_target[pair["target_id"]] = pair
    joint_pairs = list(best_by_target.values())
    joint_donor_ids = {pair["donor_id"] for pair in joint_pairs}
    if len(eligible_donor_ids) > 1 and joint_pairs:
        specs.append(
            {
                "name": "donor_kl:joint",
                "selected": [
                    donor for donor in donors if donor.task_id in joint_donor_ids
                ],
                "samples": [pair["sample"] for pair in joint_pairs],
                "pairs": joint_pairs,
            }
        )
    return specs, [_pair_metadata(pair) for pair in pairs]


def _mean_reward(outcomes: dict[str, float], task_ids: list[str]) -> float:
    values = [outcomes[task_id] for task_id in task_ids if task_id in outcomes]
    return sum(values) / len(values) if values else 0.0


def _load_task_map(data_dir: str, lopd_dir: str) -> dict[str, dict[str, Any]]:
    tasks = list_tasks("rl", data_dir, third_party_dir=lopd_dir)
    result: dict[str, dict[str, Any]] = {}
    for task in tasks:
        copied = dict(task)
        copied["lopd_dir"] = lopd_dir
        copied["env_family"] = copied["task_id"].rsplit("_rl-task_", 1)[0]
        result[copied["task_id"]] = copied
    return result


def _evaluate(
    engines: list[VllmClient],
    tasks: list[dict[str, Any]],
    cfg: StreamConfig,
    adapter: str,
) -> dict[str, float]:
    def one(item):
        index, task = item
        engine = engines[index % len(engines)]
        trajectory = run_episode(
            engine, _make_env(task), task, cfg, adapter=adapter
        )
        return trajectory.task_id, float(trajectory.reward or 0.0)

    with ThreadPoolExecutor(max_workers=max(1, min(len(tasks), len(engines)))) as pool:
        return dict(pool.map(one, enumerate(tasks)))


def _relative(path: str | Path, base: Path) -> str:
    return os.path.relpath(Path(path).resolve(), base.resolve())


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--url", action="append", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--data-dir", default="/data/erv1n/resid/data")
    parser.add_argument("--lopd-dir", default="/data/erv1n/resid/third_party/LOPD")
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--next-size", type=int, default=8)
    parser.add_argument("--cross-size", type=int, default=8)
    parser.add_argument("--start-window", type=int, default=0)
    parser.add_argument("--limit-windows", type=int)
    parser.add_argument("--max-success-candidates", type=int, default=4)
    parser.add_argument(
        "--candidate-objective",
        choices=("trajectory_sft", "donor_kl"),
        default="trajectory_sft",
        help=(
            "trajectory_sft copies successful source actions; donor_kl distils "
            "the policy shift caused by one complete donor on other window states"
        ),
    )
    parser.add_argument("--max-targets-per-donor", type=int, default=4)
    parser.add_argument("--min-targets-per-donor", type=int, default=1)
    parser.add_argument("--kl-topk", type=int, default=20)
    parser.add_argument("--max-block-chars", type=int, default=16000)
    parser.add_argument("--max-result-chars", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--grounded-weight", type=float, default=0.2)
    parser.add_argument("--anchor-kl-coef", type=float, default=0.5)
    parser.add_argument("--max-seq-len", type=int, default=12288)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="future-label decoding temperature; v0 defaults to paired greedy decoding",
    )
    parser.add_argument("--trainer", choices=("torch", "stub"), default="torch")
    parser.add_argument(
        "--family",
        action="append",
        help=(
            "restrict source and future-probe trajectories to these families; "
            "repeat after splitting families, before constructing windows"
        ),
    )
    args = parser.parse_args(argv)

    if args.window_size < 2 or args.next_size < 1 or args.cross_size < 1:
        raise SystemExit("window-size must be >=2 and probe sizes must be positive")
    if (
        args.steps <= 0
        or args.max_success_candidates <= 0
        or args.max_targets_per_donor <= 0
        or args.min_targets_per_donor <= 0
        or args.min_targets_per_donor > args.max_targets_per_donor
    ):
        raise SystemExit(
            "steps/candidate counts must be positive and min targets cannot exceed max"
        )
    reference = Path(args.reference).resolve()
    if args.trainer == "torch" and not (reference / "adapter_model.safetensors").exists():
        raise SystemExit(f"missing common reference adapter: {reference}")

    output = Path(args.output).resolve()
    manifest = output / "window_oracle_manifest.jsonl"
    done = _completed(manifest)
    buffer = Buffer.load(args.buffer)
    trajectories = [trajectory for trajectory in buffer._trajs if not trajectory.is_retry]
    if args.family:
        allowed_families = set(args.family)
        trajectories = [
            trajectory
            for trajectory in trajectories
            if trajectory.env_family in allowed_families
        ]
    task_map = _load_task_map(args.data_dir, args.lopd_dir)
    engines = [
        VllmClient(
            url,
            args.model,
            max_context=args.max_seq_len,
            lora_prefix=f"{args.run_id}-",
        )
        for url in args.url
    ]
    eval_cfg = StreamConfig(
        model=args.model,
        engine="vllm",
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_model_len=args.max_seq_len,
        max_steps=args.max_steps,
        data_dir=args.data_dir,
        split="rl",
    )
    train_cfg = StreamConfig(
        model=args.model,
        trainer=args.trainer,
        weight_mode="sft",
        train_channels="act",
        grounded_weight=args.grounded_weight,
        anchor_kl_coef=args.anchor_kl_coef,
        kl_target=args.candidate_objective == "donor_kl",
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
    if args.candidate_objective == "donor_kl":
        # AutoTokenizer initialization may touch the same Hugging Face cache.
        # Warm each client sequentially before pair scoring so concurrent first
        # loads cannot be misreported as token-alignment failures.
        for engine in engines:
            engine._tokenizer()

    attempted = written = skipped = 0
    total_windows = max(0, (len(trajectories) - args.window_size) // args.window_size)
    stop_window = total_windows
    if args.limit_windows is not None:
        stop_window = min(stop_window, args.start_window + args.limit_windows)

    for window_index in range(args.start_window, stop_window):
        start = window_index * args.window_size
        end = start + args.window_size
        members = trajectories[start:end]
        window_id = f"{args.run_id}:w{window_index:04d}"
        attempted += 1
        if window_id in done:
            skipped += 1
            continue
        near, cross = _future_slices(
            trajectories,
            end,
            {member.env_family for member in members},
            next_size=args.next_size,
            cross_size=args.cross_size,
        )
        probe_ids = list(dict.fromkeys(
            [trajectory.task_id for trajectory in near + cross]
        ))
        missing = [task_id for task_id in probe_ids if task_id not in task_map]
        if missing:
            raise RuntimeError(f"future probe tasks missing from corpus: {missing[:3]}")
        probe_tasks = [task_map[task_id] for task_id in probe_ids]
        baseline = _evaluate(engines, probe_tasks, eval_cfg, "base")
        next_ids = [trajectory.task_id for trajectory in near]
        cross_ids = [trajectory.task_id for trajectory in cross]
        baseline_next = _mean_reward(baseline, next_ids)
        baseline_cross = _mean_reward(baseline, cross_ids)
        candidates: list[dict[str, Any]] = [
            {
                "name": "noop",
                "selected_member_ids": [],
                "update_path": str(reference),
                "next_reward": baseline_next,
                "cross_reward": baseline_cross,
                "next_gain": 0.0,
                "cross_gain": 0.0,
                "losses": [],
                "optimizer_steps": 0,
                "outcomes": baseline,
            }
        ]

        pair_diagnostics: list[dict[str, Any]] = []
        if args.candidate_objective == "donor_kl":
            candidate_specs, pair_diagnostics = _donor_kl_specs(
                engines,
                members,
                train_cfg,
                max_success_candidates=args.max_success_candidates,
                max_targets_per_donor=args.max_targets_per_donor,
                min_targets_per_donor=args.min_targets_per_donor,
            )
        else:
            candidate_specs = [
                {
                    "name": policy,
                    "selected": selected,
                    "samples": [
                        sft_sample(trajectory, train_cfg) for trajectory in selected
                    ],
                    "pairs": [],
                }
                for policy, selected in _candidate_specs(
                    members, args.max_success_candidates
                )
            ]

        for spec in candidate_specs:
            policy = str(spec["name"])
            selected = list(spec["selected"])
            samples = list(spec["samples"])
            parent = AdapterVersion(
                name=f"ref-{args.run_id}-w{window_index:04d}-{policy}",
                path=str(reference),
                parent=None,
            )
            candidate = train_candidate(
                samples,
                parent,
                train_cfg,
                str(output / "candidates" / f"w{window_index:04d}"),
            )
            version = AdapterVersion(
                name=candidate.candidate_id,
                path=candidate.adapter_path,
                parent=parent.name,
            )
            for engine in engines:
                engine.load_adapter(version)
            try:
                outcomes = _evaluate(
                    engines, probe_tasks, eval_cfg, candidate.candidate_id
                )
            finally:
                for engine in engines:
                    engine.unload_adapter(candidate.candidate_id)
            next_reward = _mean_reward(outcomes, next_ids)
            cross_reward = _mean_reward(outcomes, cross_ids)
            candidates.append(
                {
                    "name": policy,
                    "selected_member_ids": [item.task_id for item in selected],
                    "update_path": candidate.adapter_path,
                    "next_reward": next_reward,
                    "cross_reward": cross_reward,
                    "next_gain": next_reward - baseline_next,
                    "cross_gain": cross_reward - baseline_cross,
                    "losses": candidate.train_stats.get("losses", []),
                    "optimizer_steps": candidate.train_stats.get("optimizer_steps", 0),
                    "outcomes": outcomes,
                    "objective": args.candidate_objective,
                    "training_target_ids": [sample.task_id for sample in samples],
                    "pair_diagnostics": [
                        _pair_metadata(pair) for pair in spec.get("pairs", [])
                    ],
                }
            )

        # Cross-family transfer is primary.  Ties prefer near-future gain and
        # then the smaller/no-op update so zero evidence cannot force a write.
        winner = max(
            candidates,
            key=lambda row: (
                float(row["cross_gain"]),
                float(row["next_gain"]),
                -len(row["selected_member_ids"]),
            ),
        )
        record = {
            "window_id": window_id,
            "stream_id": args.run_id,
            "member_ids": [member.task_id for member in members],
            "member_families": [member.env_family for member in members],
            "member_features": [deployment_features(member) for member in members],
            "selected_member_ids": winner["selected_member_ids"],
            "update_path": _relative(winner["update_path"], manifest.parent),
            "transfer_gain": float(winner["cross_gain"]),
            "next_gain": float(winner["next_gain"]),
            "metadata": {
                "format": "sediment-window-oracle-v1",
                "window_index": window_index,
                "window_size": len(members),
                "future_probe_task_ids": next_ids,
                "cross_probe_task_ids": cross_ids,
                "baseline_next_reward": baseline_next,
                "baseline_cross_reward": baseline_cross,
                "baseline_outcomes": baseline,
                "winner": winner["name"],
                "candidates": [
                    {
                        **candidate,
                        "update_path": _relative(candidate["update_path"], manifest.parent),
                    }
                    for candidate in candidates
                ],
                "reference": str(reference),
                "source_buffer": str(Path(args.buffer).resolve()),
                "family_filter": sorted(args.family or []),
                "future_queries_used_for_update": False,
                "primary_label": "cross_family_future_reward_delta",
                "candidate_objective": args.candidate_objective,
                "max_targets_per_donor": args.max_targets_per_donor,
                "min_targets_per_donor": args.min_targets_per_donor,
                "pair_diagnostics": pair_diagnostics,
            },
        }
        _append_jsonl(manifest, record)
        done.add(window_id)
        written += 1
        print(
            f"[window-oracle] {window_id} winner={winner['name']} "
            f"cross={winner['cross_gain']:+.3f} next={winner['next_gain']:+.3f} "
            f"candidates={len(candidates)}",
            flush=True,
        )

    report = {
        "format": "sediment-window-oracle-collection-v1",
        "run_id": args.run_id,
        "source_trajectories": len(trajectories),
        "window_size": args.window_size,
        "candidate_objective": args.candidate_objective,
        "attempted": attempted,
        "written": written,
        "skipped": skipped,
        "manifest": str(manifest),
        "reference": str(reference),
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
