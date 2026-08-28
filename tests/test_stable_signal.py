from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sediment.compiler.stable_data import (
    DonorProposal,
    StableWindowRecord,
    memory_task_cluster_id,
    memory_views,
    partition_families,
    stable_feature_tensors,
    stable_target,
)
from sediment.compiler.stable_model import StableSignalConfig, build_stable_signal_model
from sediment.compiler.state import load_tensor_state
from sediment.compiler.stable_train import (
    StableTrainingConfig,
    _cluster_balanced_epoch_indices,
    load_stable_signal_checkpoint,
    predict_stable_record,
    train_stable_signal,
)
from sediment.config import StreamConfig


def _proposal(
    proposal_id: str,
    member_id: str,
    family: str,
    coefficients,
    gains,
):
    return DonorProposal(
        proposal_id=proposal_id,
        member_id=member_id,
        member_family=family,
        coefficients=list(coefficients),
        intervention_features={"action_shift": 0.2, "teacher_positions": 16.0},
        discovery_family_gains={"discovery": 0.9},
        confirmation_family_gains={
            name: [float(value), float(value)] for name, value in gains.items()
        },
    )


def _record(
    window_id: str,
    *,
    stream: str = "s0",
    partition: str = "train",
    positive: bool = True,
    size: int = 4,
):
    families = [f"f{index % 3}" for index in range(size)]
    gains = {"c1": 0.4, "c2": 0.4} if positive else {"c1": 0.1, "c2": -0.1}
    return StableWindowRecord(
        window_id=window_id,
        stream_id=stream,
        partition=partition,
        member_ids=[f"{window_id}:m{index}" for index in range(size)],
        member_families=families,
        member_features=[
            {"reward": float(index) / max(1, size - 1), "steps": float(index + 1)}
            for index in range(size)
        ],
        proposals=[
            _proposal(
                f"{window_id}:p0",
                f"{window_id}:m0",
                families[0],
                [1.0, 0.0],
                gains,
            ),
            _proposal(
                f"{window_id}:p1",
                f"{window_id}:m1",
                families[1],
                [0.0, 1.0],
                gains,
            ),
        ],
    )


def test_family_partition_is_frozen_complete_and_disjoint():
    families = [f"env_{index:03d}" for index in range(34)]
    left = partition_families(families)
    right = partition_families(reversed(families))
    assert left == right
    assert len(left.digest) == 64
    assert len(left.all_families) == 34
    assert len(set(left.all_families)) == 34
    assert [
        len(left.train_members),
        len(left.train_discovery),
        len(left.train_confirmation),
        len(left.test_members),
        len(left.test_discovery),
        len(left.test_confirmation),
    ] == [10, 4, 4, 8, 4, 4]

    success = {family: index for index, family in enumerate(families)}
    stratified = partition_families(families, success_counts=success)
    repeated = partition_families(reversed(families), success_counts=success)
    assert stratified == repeated
    assert stratified.digest != left.digest
    role_means = [
        np.mean([success[family] for family in group])
        for group in (
            stratified.train_members,
            stratified.train_discovery,
            stratified.train_confirmation,
            stratified.test_members,
            stratified.test_discovery,
            stratified.test_confirmation,
        )
    ]
    assert max(role_means) - min(role_means) < 5.0


def test_conservative_target_is_reward_weighted_and_rejects_noise_and_catastrophe():
    record = _record("w")
    proposals = [
        _proposal("strong", "w:m0", "f0", [1.0, 0.0], {"a": 0.4, "b": 0.4}),
        _proposal("weak", "w:m1", "f1", [0.0, 1.0], {"a": 0.2, "b": 0.2}),
        _proposal("noisy", "w:m2", "f2", [2.0, 2.0], {"a": 1.0, "b": -1.0}),
        _proposal("catastrophe", "w:m3", "f0", [3.0, 3.0], {"a": 0.9, "b": -0.3}),
    ]
    record = StableWindowRecord(
        window_id=record.window_id,
        stream_id=record.stream_id,
        partition=record.partition,
        member_ids=record.member_ids,
        member_families=record.member_families,
        member_features=record.member_features,
        proposals=proposals,
    )
    target = stable_target(record, kappa=1.0, catastrophe_floor=-0.2)
    assert target.write
    assert target.proposal_ids == ["strong", "weak"]
    assert target.coefficients == pytest.approx([2 / 3, 1 / 3])
    assert sum(target.proposal_weights) == pytest.approx(1.0)
    assert target.worst_family_gain == pytest.approx(0.2)

    negative = stable_target(_record("negative", positive=False))
    assert not negative.write
    assert negative.coefficients == [0.0, 0.0]


def test_memory_views_are_deterministic_unique_and_keep_a_donor():
    record = _record("view", size=16)
    views = memory_views(record, bootstrap_views=4, bootstrap_size=8)
    assert views == memory_views(record, bootstrap_views=4, bootstrap_size=8)
    assert views[0].name == "full"
    assert sum(view.name.startswith("subset:") for view in views) == 4
    assert len({view.member_indices for view in views}) == len(views)
    donor_indices = {0, 1}
    assert all(donor_indices.intersection(view.member_indices) for view in views)


def test_tensorization_excludes_future_reward_labels():
    left = _record("left")
    right = StableWindowRecord(
        window_id="right",
        stream_id=left.stream_id,
        partition=left.partition,
        member_ids=left.member_ids,
        member_families=left.member_families,
        member_features=left.member_features,
        proposals=[
            DonorProposal(
                proposal_id=proposal.proposal_id,
                member_id=proposal.member_id,
                member_family=proposal.member_family,
                coefficients=proposal.coefficients,
                intervention_features=proposal.intervention_features,
                discovery_family_gains={"leak": 999.0},
                confirmation_family_gains={"leak": [-999.0, 999.0]},
            )
            for proposal in left.proposals
        ],
    )
    tensors = stable_feature_tensors([left])
    changed = stable_feature_tensors([right])
    for before, after in zip(tensors[:5], changed[:5]):
        np.testing.assert_array_equal(before, after)


def test_stable_model_is_safe_noop_permutation_invariant_and_uncertain():
    torch = pytest.importorskip("torch")
    record = _record("model", size=4)
    member, coefficients, intervention, proposal_mask, member_mask, _, _ = (
        stable_feature_tensors([record])
    )
    config = StableSignalConfig(
        member_feature_dim=member.shape[-1],
        proposal_feature_dim=intervention.shape[-1],
        basis_rank=coefficients.shape[-1],
        member_hidden=8,
        window_hidden=8,
        deployment_topk=4,
    )
    model = build_stable_signal_model(config)

    def tensor(value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype)

    result = model(
        tensor(member),
        tensor(coefficients),
        tensor(intervention),
        tensor(proposal_mask, torch.bool),
        tensor(member_mask, torch.bool),
    )
    assert torch.equal(result["delta"], torch.zeros_like(result["delta"]))
    assert torch.all(result["coefficient_variance"] > 0)
    assert float(result["write_probability"][0].detach()) == pytest.approx(0.01, abs=1e-6)
    assert float(result["shrinkage"][0].detach()) == pytest.approx(1.0)
    assert float(result["proposal_weights"].sum().detach()) == pytest.approx(1.0)
    assert torch.equal(
        result["proposal_weights"][~tensor(proposal_mask, torch.bool)],
        torch.zeros_like(result["proposal_weights"][~tensor(proposal_mask, torch.bool)]),
    )

    order = torch.tensor([2, 0, 3, 1])
    permuted = model(
        tensor(member)[:, order],
        tensor(coefficients)[:, order],
        tensor(intervention)[:, order],
        tensor(proposal_mask, torch.bool)[:, order],
        tensor(member_mask, torch.bool)[:, order],
    )
    assert torch.allclose(result["coefficient_mean"], permuted["coefficient_mean"])
    assert torch.allclose(result["coefficient_variance"], permuted["coefficient_variance"])
    assert torch.allclose(result["write_probability"], permuted["write_probability"])


def test_stable_training_holds_out_memory_clusters_and_leaves_test_labels_unread(tmp_path):
    pytest.importorskip("torch")
    records = [
        _record(f"w{index}", stream=f"s{index % 4}", positive=index % 3 != 0)
        for index in range(12)
    ]
    # Empty confirmation labels are valid file data but cannot form a target.
    # Training must not inspect them because this record is outer-held-out.
    test = _record("heldout", stream="test", partition="test")
    test = StableWindowRecord(
        window_id=test.window_id,
        stream_id=test.stream_id,
        partition=test.partition,
        member_ids=test.member_ids,
        member_families=test.member_families,
        member_features=test.member_features,
        proposals=[
            DonorProposal(
                proposal_id=proposal.proposal_id,
                member_id=proposal.member_id,
                member_family=proposal.member_family,
                coefficients=proposal.coefficients,
                intervention_features=proposal.intervention_features,
            )
            for proposal in test.proposals
        ],
    )
    records.append(test)
    train_records = records[:-1]
    member, coefficients, intervention, *_ = stable_feature_tensors(train_records)
    output = tmp_path / "stable.pt"
    report = train_stable_signal(
        train_records,
        StableSignalConfig(
            member_feature_dim=member.shape[-1],
            proposal_feature_dim=intervention.shape[-1],
            basis_rank=coefficients.shape[-1],
            member_hidden=8,
            window_hidden=8,
            deployment_topk=4,
        ),
        StableTrainingConfig(
            epochs=2,
            batch_size=4,
            validation_cluster_fraction=0.25,
            bootstrap_views=2,
            bootstrap_size=2,
            device="cpu",
        ),
        output,
        heldout_test_count=1,
        provenance={"outer_test_labels_loaded": False},
    )
    assert report["format"] == "sediment-stable-signal-v1"
    assert report["train"]["windows"] > 0
    assert report["validation"]["windows"] > 0
    assert report["validation"]["memory_task_clusters"] > 0
    validation_clusters = set(report["validation_memory_task_clusters"])
    assert {
        memory_task_cluster_id(record)
        for record in records[:-1]
        if memory_task_cluster_id(record) in validation_clusters
    } == validation_clusters
    assert not set(report["train_memory_task_clusters"]) & validation_clusters
    assert report["test_windows_unread"] == 1
    assert report["provenance"]["outer_test_labels_loaded"] is False
    assert report["deployment_gate"]["test_labels_used"] is False
    assert math.isfinite(report["history"][-1]["train_loss"])
    assert output.exists()
    model, checkpoint = load_stable_signal_checkpoint(output)
    prediction = predict_stable_record(model, checkpoint, records[0])
    assert len(prediction["coefficients"]) == 2
    assert len(prediction["view_names"]) >= 3
    assert prediction["total_uncertainty"] >= 0.0
    assert isinstance(prediction["gate_passed"], bool)
    with pytest.raises(ValueError, match="only partition=train"):
        train_stable_signal(
            records,
            StableSignalConfig(
                member_feature_dim=member.shape[-1],
                proposal_feature_dim=intervention.shape[-1],
                basis_rank=coefficients.shape[-1],
                member_hidden=8,
                window_hidden=8,
                deployment_topk=4,
            ),
            StableTrainingConfig(epochs=1, bootstrap_views=1, bootstrap_size=2),
            tmp_path / "must-not-read-test.pt",
        )


def test_cluster_balanced_epochs_cycle_request_seed_variants():
    first = _record("same-s0", stream="s0")
    variant = StableWindowRecord(
        window_id="same-s1",
        stream_id="s1",
        partition=first.partition,
        member_ids=first.member_ids,
        member_families=first.member_families,
        member_features=first.member_features,
        proposals=first.proposals,
        metadata=first.metadata,
    )
    other = _record("other", stream="s2")
    records = [first, variant, other]
    epoch_zero = _cluster_balanced_epoch_indices(
        records, [0, 1, 2], epoch=0, seed=7
    )
    epoch_one = _cluster_balanced_epoch_indices(
        records, [0, 1, 2], epoch=1, seed=7
    )
    assert len(epoch_zero) == len(epoch_one) == 2
    assert 2 in epoch_zero and 2 in epoch_one
    assert {next(index for index in epoch if index != 2) for epoch in (epoch_zero, epoch_one)} == {
        0,
        1,
    }


def test_window_planner_freezes_later_disjoint_probe_families():
    spec = importlib.util.spec_from_file_location(
        "plan_stable_signal_windows",
        Path(__file__).parents[1] / "scripts" / "plan_stable_signal_windows.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    indexed = []
    for index in range(16):
        indexed.append(
            module.IndexedTrajectory(
                index,
                SimpleNamespace(
                    task_id=f"member-{index}",
                    env_family=f"m{index % 2}",
                    success=index < 2,
                    messages=[SimpleNamespace(role="assistant")],
                ),
            )
        )
    for offset, family in enumerate(["d1", "d2", "d3", "d4", "c1", "c2", "c3", "c4"]):
        indexed.append(
            module.IndexedTrajectory(
                16 + offset,
                SimpleNamespace(task_id=f"probe-{family}", env_family=family, success=False),
            )
        )
    used = set()
    windows = module._plan_stream_partition(
        indexed,
        stream_id="s0",
        partition="train",
        member_families=["m0", "m1"],
        discovery_families=["d1", "d2", "d3", "d4"],
        confirmation_families=["c1", "c2", "c3", "c4"],
        windows=1,
        window_size=16,
        min_successes=2,
        used_task_ids=used,
    )
    assert len(windows) == 1
    assert windows[0]["discovery_families"] == ["d1", "d2", "d3", "d4"]
    assert windows[0]["confirmation_families"] == ["c1", "c2", "c3", "c4"]
    assert min(windows[0]["discovery_task_ids"]) not in windows[0]["member_task_ids"]
    assert len(used) == 8


def test_window_planner_rejects_successes_without_four_cross_family_targets():
    spec = importlib.util.spec_from_file_location(
        "plan_stable_signal_windows_structural",
        Path(__file__).parents[1] / "scripts" / "plan_stable_signal_windows.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    indexed = []
    # Four successes from one family but only three cross-family members exactly
    # reproduces the mechanically invalid frozen smoke window.
    for index in range(16):
        family = "dominant" if index < 13 else "other"
        indexed.append(
            module.IndexedTrajectory(
                index,
                SimpleNamespace(
                    task_id=f"member-{index}",
                    env_family=family,
                    success=index < 4,
                    messages=[SimpleNamespace(role="assistant")],
                ),
            )
        )
    for offset, family in enumerate(
        ["d1", "d2", "d3", "d4", "c1", "c2", "c3", "c4"]
    ):
        indexed.append(
            module.IndexedTrajectory(
                16 + offset,
                SimpleNamespace(
                    task_id=f"probe-{family}",
                    env_family=family,
                    success=False,
                    messages=[SimpleNamespace(role="assistant")],
                ),
            )
        )

    windows = module._plan_stream_partition(
        indexed,
        stream_id="s0",
        partition="test",
        member_families=["dominant", "other"],
        discovery_families=["d1", "d2", "d3", "d4"],
        confirmation_families=["c1", "c2", "c3", "c4"],
        windows=1,
        window_size=16,
        min_successes=2,
        used_task_ids=set(),
    )
    assert windows == []


def test_stable_proposal_collection_helpers_are_paired_and_mean_coordinates(
    tmp_path, monkeypatch
):
    safetensors = pytest.importorskip("safetensors.numpy")
    spec = importlib.util.spec_from_file_location(
        "collect_stable_signal_proposals",
        Path(__file__).parents[1] / "scripts" / "collect_stable_signal_proposals.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    calls = []

    def fake_episode(engine, env, task, cfg, *, adapter):
        del engine, env, cfg
        calls.append(adapter)
        return SimpleNamespace(task_id=task["task_id"], reward=1.0 if adapter == "cand" else 0.0)

    monkeypatch.setattr(module, "run_episode", fake_episode)
    monkeypatch.setattr(module, "_make_env", lambda task: task)
    tasks = [{"task_id": "t", "env_family": "f"}]
    even = module._paired_repeat(
        [object()],
        tasks,
        StreamConfig(),
        reference_adapter="ref",
        candidate_adapter="cand",
        repeat=0,
    )
    odd = module._paired_repeat(
        [object()],
        tasks,
        StreamConfig(),
        reference_adapter="ref",
        candidate_adapter="cand",
        repeat=1,
    )
    assert calls == ["ref", "cand", "cand", "ref"]
    assert even[0]["delta"] == 1.0
    assert odd[0]["order"] == ["candidate", "reference"]
    assert module._family_gains(even + odd) == {"f": [1.0, 1.0]}

    reference = tmp_path / "reference"
    first = tmp_path / "first"
    second = tmp_path / "second"
    for directory, values in (
        (reference, [0.0, 0.0]),
        (first, [2.0, 0.0]),
        (second, [0.0, 4.0]),
    ):
        directory.mkdir()
        safetensors.save_file(
            {"weight": np.asarray(values, dtype=np.float32)},
            str(directory / "adapter_model.safetensors"),
        )
    (reference / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    mean = module._write_mean_adapter(tmp_path / "mean", reference, [first, second])
    np.testing.assert_allclose(load_tensor_state(mean)["weight"], [1.0, 2.0])

    features = module._intervention_features(
        [
            {
                "error": None,
                "action_shift": 0.2,
                "act_gain": 0.1,
                "obs_surprise": 0.3,
                "teacher_positions": 10,
                "task_similarity": 0.4,
                "target_family": "x",
            },
            {
                "error": None,
                "action_shift": 0.4,
                "act_gain": -0.1,
                "obs_surprise": 0.5,
                "teacher_positions": 20,
                "task_similarity": 0.2,
                "target_family": "y",
            },
        ]
    )
    assert features["action_shift_mean"] == pytest.approx(0.3)
    assert features["teacher_positions_total"] == 30.0
    assert features["target_family_count"] == 2.0


def test_six_arm_aggregation_is_window_paired_and_shuffle_is_deranged():
    spec = importlib.util.spec_from_file_location(
        "evaluate_stable_signal_arms",
        Path(__file__).parents[1] / "scripts" / "evaluate_stable_signal_arms.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    records = [
        SimpleNamespace(
            window_id=f"w{index}",
            stream_id=f"s{index}",
            member_ids=[f"member-{index}"],
        )
        for index in range(4)
    ]
    shuffled = module._derangement(records, seed=7)
    assert set(shuffled) == {record.window_id for record in records}
    assert all(
        module.memory_task_cluster_id(record)
        != module.memory_task_cluster_id(shuffled[record.window_id])
        for record in records
    )

    per_window = []
    for index in range(4):
        gains = {
            "noop": 0.0,
            "single": 0.05,
            "heuristic_joint": 0.04,
            "mean": 0.03,
            "stable": 0.2 + index * 0.01,
            "shuffled": 0.0,
        }
        per_window.append(
            {
                "source_memory_cluster": f"cluster-{index // 2}",
                "arm_mean_gains": gains,
                "arm_family_gains": {
                    arm: {"family_a": [value], "family_b": [value]}
                    for arm, value in gains.items()
                },
                "stable_prediction": {"gate_passed": True},
            }
        )
    aggregate = module.aggregate_six_arms(per_window, seed=3)
    comparison = aggregate["comparisons"]["stable_minus_noop"]
    assert comparison["mean"] == pytest.approx(0.215)
    assert comparison["positive_memory_clusters"] == 2
    assert comparison["exact_one_sided_sign_flip_p"] == pytest.approx(1 / 4)
    assert aggregate["comparisons"]["shuffled_minus_noop"]["mean"] == 0.0
    assert aggregate["source_memory_task_clusters"] == 2
    assert aggregate["source_memory_cluster_sizes"] == {
        "cluster-0": 2,
        "cluster-1": 2,
    }
    assert aggregate["efficacy_claim_gate"]["passed"] is True
    assert aggregate["stable_leave_one_family_out_mean_gains"] == {
        "family_a": pytest.approx(0.215),
        "family_b": pytest.approx(0.215),
    }
    assert aggregate["stable_write_rate"] == 1.0


def test_project_test_preflight_runs_before_test_manifest_is_opened(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "stable_signal_compiler",
        Path(__file__).parents[1] / "scripts" / "stable_signal_compiler.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    args = SimpleNamespace(
        mode="project-test",
        checkpoint=str(tmp_path / "missing-checkpoint.pt"),
        basis=str(tmp_path / "missing-basis.npz"),
        manifest=[str(tmp_path / "must-remain-unread-test-manifest.jsonl")],
    )
    with pytest.raises(FileNotFoundError) as error:
        module.build_data_command(args)
    assert "missing-checkpoint.pt.report.json" in str(error.value)


def test_stable_proposal_audit_covers_frozen_protocol(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "audit_stable_signal_proposals",
        Path(__file__).parents[1] / "scripts" / "audit_stable_signal_proposals.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    member_ids = [f"m{index}" for index in range(16)]
    member_families = [f"mf{index % 2}" for index in range(16)]
    discovery_ids = [f"d{index}" for index in range(4)]
    confirmation_ids = [f"c{index}" for index in range(4)]
    discovery_families = [f"df{index}" for index in range(4)]
    confirmation_families = [f"cf{index}" for index in range(4)]
    plan = {
        "format": "sediment-stable-window-plan-v1",
        "digest": "plan",
        "family_partition_digest": "families",
        "windows": [
            {
                "window_id": "w",
                "stream_id": "s0",
                "partition": "train",
                "member_task_ids": member_ids,
                "member_families": member_families,
                "discovery_task_ids": discovery_ids,
                "confirmation_task_ids": confirmation_ids,
                "discovery_families": discovery_families,
                "confirmation_families": confirmation_families,
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    def gains():
        return {
            "discovery_family_gains": {family: 0.1 for family in discovery_families},
            "confirmation_family_gains": {
                family: [0.1, 0.1, 0.1] for family in confirmation_families
            },
        }

    def adapter(name):
        path = tmp_path / name
        path.mkdir()
        (path / "adapter_model.safetensors").write_bytes(b"test")
        (path / "adapter_config.json").write_text("{}", encoding="utf-8")
        return name

    def single(number):
        donor = member_ids[number]
        return {
            "kind": "single",
            "proposal_id": f"p{number}",
            "member_id": donor,
            "selected_member_ids": [donor],
            "update_path": adapter(f"single{number}"),
            "optimizer_steps": 2,
            "parameter_audit": {"finite": True, "changed_tensors": 2},
            "pair_diagnostics": [
                {"error": None, "donor_id": donor, "target_id": member_ids[index + 2]}
                for index in range(4)
            ],
            **gains(),
        }

    row = {
        "format": "sediment-stable-proposal-window-v1",
        "window_id": "w",
        "stream_id": "s0",
        "partition": "train",
        "member_ids": member_ids,
        "member_families": member_families,
        "metadata": {
            "family_partition_digest": "families",
            "window_plan_digest": "plan",
            "future_tasks_used_for_candidate_update": False,
            "candidate_reward_used_for_window_selection": False,
            "discovery_task_ids": discovery_ids,
            "confirmation_task_ids": confirmation_ids,
            "discovery_families": discovery_families,
            "confirmation_families": confirmation_families,
            "confirmation_repeats": 3,
        },
        "candidates": [
            {"kind": "noop", "optimizer_steps": 0, "update_path": adapter("noop")},
            single(0),
            single(1),
            {
                "kind": "heuristic_joint",
                "update_path": adapter("joint"),
                "selected_member_ids": member_ids[:2],
                "optimizer_steps": 2,
                "parameter_audit": {"finite": True, "changed_tensors": 2},
                "pair_diagnostics": [
                    {
                        "error": None,
                        "donor_id": member_ids[index % 2],
                        "target_id": member_ids[index + 2],
                    }
                    for index in range(4)
                ],
                **gains(),
            },
            {
                "kind": "mean",
                "update_path": adapter("mean"),
                "optimizer_steps": 0,
                "derived_from": ["p0", "p1"],
                "parameter_audit": {"finite": True, "changed_tensors": 2},
                **gains(),
            },
        ],
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    report = module.audit_manifests([manifest], plan_path)
    assert report["all_passed"]
    assert report["planned_windows_missing"] == []
    assert report["windows"][0]["singles"] == 2
