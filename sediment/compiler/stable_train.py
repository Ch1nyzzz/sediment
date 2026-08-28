"""Multi-view supervised training for the stable memory-signal compiler."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .stable_data import (
    StableWindowRecord,
    memory_task_cluster_id,
    memory_views,
    select_validation_memory_clusters,
    stable_feature_tensors,
    stable_target,
)
from .stable_model import StableSignalConfig, build_stable_signal_model


@dataclass(frozen=True)
class StableTrainingConfig:
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_cluster_fraction: float = 0.25
    kappa: float = 1.0
    catastrophe_floor: float = -0.20
    write_loss_weight: float = 0.5
    consistency_loss_weight: float = 0.25
    noop_loss_weight: float = 0.25
    seed: int = 20260827
    bootstrap_views: int = 4
    bootstrap_size: int = 8
    write_threshold: float = 0.5
    uncertainty_quantile: float = 0.75
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning rate must be positive and weight decay nonnegative")
        if not 0.0 < self.validation_cluster_fraction < 1.0:
            raise ValueError("validation_cluster_fraction must be in (0, 1)")
        for name in (
            "kappa",
            "write_loss_weight",
            "consistency_loss_weight",
            "noop_loss_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.bootstrap_views < 0 or self.bootstrap_size <= 0:
            raise ValueError("bootstrap view count/size must be nonnegative/positive")
        if not 0.0 < self.write_threshold < 1.0:
            raise ValueError("write_threshold must be in (0, 1)")
        if not 0.0 < self.uncertainty_quantile <= 1.0:
            raise ValueError("uncertainty_quantile must be in (0, 1]")


def _memory_cluster_split(
    records: list[StableWindowRecord], fraction: float, seed: int
):
    clusters = sorted({memory_task_cluster_id(record) for record in records})
    validation_clusters = set(
        select_validation_memory_clusters(clusters, fraction=fraction, seed=seed)
    )
    train = [
        index
        for index, record in enumerate(records)
        if memory_task_cluster_id(record) not in validation_clusters
    ]
    validation = [
        index
        for index, record in enumerate(records)
        if memory_task_cluster_id(record) in validation_clusters
    ]
    if not train or not validation:
        raise ValueError("stream-held-out split produced an empty side")
    return train, validation, sorted(validation_clusters)


def _cluster_balanced_epoch_indices(
    records: list[StableWindowRecord],
    indices: list[int],
    *,
    epoch: int,
    seed: int,
) -> list[int]:
    """Choose one request-seed variant per source-memory cluster each epoch."""

    grouped: dict[str, list[int]] = {}
    for index in indices:
        grouped.setdefault(memory_task_cluster_id(records[index]), []).append(index)
    selected = []
    for cluster, variants in sorted(grouped.items()):
        variants = sorted(variants, key=lambda index: records[index].window_id)
        offset = int.from_bytes(
            hashlib.sha256(f"{seed}:{cluster}".encode()).digest()[:4], "big"
        )
        selected.append(variants[(epoch + offset) % len(variants)])
    return selected


def _normalize_inputs(arrays, member_mask, proposal_mask, train_indices):
    member, coefficients, intervention = arrays

    def normalize(values, valid, *, keep_zero):
        selected = values[train_indices][valid[train_indices]]
        mean = selected.mean(axis=0)
        std = selected.std(axis=0)
        std[std < 1e-6] = 1.0
        normalized = (values - mean.reshape((1, 1, -1))) / std.reshape((1, 1, -1))
        if keep_zero:
            normalized[~valid] = 0.0
        return normalized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)

    member, member_mean, member_std = normalize(
        member, member_mask, keep_zero=True
    )
    intervention, intervention_mean, intervention_std = normalize(
        intervention, proposal_mask, keep_zero=True
    )
    return (
        (member, coefficients, intervention),
        {
            "member_mean": member_mean.tolist(),
            "member_std": member_std.tolist(),
            "proposal_coefficient_mean": np.zeros(
                coefficients.shape[-1], dtype=np.float32
            ).tolist(),
            "proposal_coefficient_std": np.ones(
                coefficients.shape[-1], dtype=np.float32
            ).tolist(),
            "proposal_coefficients_normalized": False,
            "proposal_feature_mean": intervention_mean.tolist(),
            "proposal_feature_std": intervention_std.tolist(),
        },
    )


def _view_mask(
    records: list[StableWindowRecord],
    indices: list[int],
    max_members: int,
    *,
    epoch: int,
    config: StableTrainingConfig,
) -> np.ndarray:
    mask = np.zeros((len(indices), max_members), dtype=np.bool_)
    for row, index in enumerate(indices):
        views = memory_views(
            records[index],
            seed=config.seed,
            bootstrap_views=config.bootstrap_views,
            bootstrap_size=min(config.bootstrap_size, len(records[index].member_ids)),
        )
        # Exclude full when alternatives exist. Cycling is deterministic and
        # exposes every generated view without stochastic label selection.
        choices = views[1:] or views
        view = choices[(epoch + index) % len(choices)]
        mask[row, list(view.member_indices)] = True
    return mask


def train_stable_signal(
    records: list[StableWindowRecord],
    model_config: StableSignalConfig,
    train_config: StableTrainingConfig,
    output: str | Path,
    *,
    heldout_test_count: int = 0,
    provenance: dict | None = None,
) -> dict:
    """Train from a train-only file; outer-test labels never enter this process."""

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - remote dependency
        raise RuntimeError("stable signal training requires PyTorch") from exc

    if any(record.partition != "train" for record in records):
        raise ValueError("stable training accepts only partition=train records")
    if heldout_test_count < 0:
        raise ValueError("heldout_test_count must be nonnegative")
    training_records = list(records)
    heldout_count = int(heldout_test_count)
    if not training_records:
        raise ValueError("no partition=train windows were supplied")
    (
        member,
        coefficients,
        intervention,
        proposal_mask,
        member_mask,
        member_names,
        proposal_names,
    ) = stable_feature_tensors(training_records)
    if member.shape[-1] != model_config.member_feature_dim:
        raise ValueError("member feature dimension differs from model config")
    if intervention.shape[-1] != model_config.proposal_feature_dim:
        raise ValueError("proposal feature dimension differs from model config")
    if coefficients.shape[-1] != model_config.basis_rank:
        raise ValueError("proposal coefficient rank differs from model config")

    targets = [
        stable_target(
            record,
            kappa=train_config.kappa,
            catastrophe_floor=train_config.catastrophe_floor,
        )
        for record in training_records
    ]
    target_coefficients = np.asarray(
        [target.coefficients for target in targets], dtype=np.float32
    )
    target_write = np.asarray([target.write for target in targets], dtype=np.float32)
    target_dispersion = np.asarray(
        [target.coefficient_dispersion for target in targets], dtype=np.float32
    )
    train_idx, validation_idx, validation_clusters = _memory_cluster_split(
        training_records,
        train_config.validation_cluster_fraction,
        train_config.seed,
    )
    declared_records = [
        record
        for record in training_records
        if "basis_fit_excluded_memory_task_clusters" in record.metadata
    ]
    if declared_records and len(declared_records) != len(training_records):
        raise ValueError("basis-fit exclusions are missing from some training records")
    declared_basis_exclusions = {
        tuple(record.metadata["basis_fit_excluded_memory_task_clusters"])
        for record in declared_records
    }
    if declared_basis_exclusions and declared_basis_exclusions != {
        tuple(validation_clusters)
    }:
        raise ValueError(
            "basis-fit exclusions disagree with the training validation clusters"
        )
    (member, coefficients, intervention), normalization = _normalize_inputs(
        (member, coefficients, intervention), member_mask, proposal_mask, train_idx
    )

    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_config.seed)
    device = (
        "cuda"
        if train_config.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if train_config.device == "auto"
        else train_config.device
    )
    model = build_stable_signal_model(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    x_member = torch.as_tensor(member, dtype=torch.float32, device=device)
    x_coefficients = torch.as_tensor(coefficients, dtype=torch.float32, device=device)
    x_intervention = torch.as_tensor(intervention, dtype=torch.float32, device=device)
    x_proposal_mask = torch.as_tensor(proposal_mask, dtype=torch.bool, device=device)
    x_member_mask = torch.as_tensor(member_mask, dtype=torch.bool, device=device)
    y_coefficients = torch.as_tensor(target_coefficients, dtype=torch.float32, device=device)
    y_write = torch.as_tensor(target_write, dtype=torch.float32, device=device)
    y_dispersion = torch.as_tensor(target_dispersion, dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu").manual_seed(train_config.seed)
    history: list[dict[str, float]] = []

    for epoch in range(train_config.epochs):
        model.train()
        balanced = _cluster_balanced_epoch_indices(
            training_records,
            train_idx,
            epoch=epoch,
            seed=train_config.seed,
        )
        order = torch.as_tensor(balanced)[
            torch.randperm(len(balanced), generator=generator)
        ].tolist()
        epoch_losses: list[float] = []
        for start in range(0, len(order), train_config.batch_size):
            indices = order[start : start + train_config.batch_size]
            index = torch.as_tensor(indices, dtype=torch.long, device=device)
            view = torch.as_tensor(
                _view_mask(
                    training_records,
                    indices,
                    member.shape[1],
                    epoch=epoch,
                    config=train_config,
                ),
                dtype=torch.bool,
                device=device,
            )
            full = model(
                x_member[index],
                x_coefficients[index],
                x_intervention[index],
                x_proposal_mask[index],
                x_member_mask[index],
            )
            partial = model(
                x_member[index],
                x_coefficients[index],
                x_intervention[index],
                x_proposal_mask[index],
                x_member_mask[index],
                view_mask=view,
            )

            target = y_coefficients[index]
            extra_variance = y_dispersion[index].square().unsqueeze(-1)

            def coefficient_nll(result):
                variance = result["coefficient_variance"] + extra_variance
                squared = (result["coefficient_mean"] - target).square()
                return 0.5 * (squared / variance + torch.log(variance)).mean()

            coefficient_loss = 0.5 * (coefficient_nll(full) + coefficient_nll(partial))
            write_loss = 0.5 * (
                functional.binary_cross_entropy(full["write_probability"], y_write[index])
                + functional.binary_cross_entropy(
                    partial["write_probability"], y_write[index]
                )
            )
            consistency_loss = (
                full["coefficient_mean"] - partial["coefficient_mean"]
            ).square().mean()
            negative = y_write[index] == 0
            noop_loss = (
                0.5
                * (
                    full["coefficient_mean"][negative].square().mean()
                    + partial["coefficient_mean"][negative].square().mean()
                )
                if bool(negative.any())
                else torch.zeros((), device=device)
            )
            loss = (
                coefficient_loss
                + train_config.write_loss_weight * write_loss
                + train_config.consistency_loss_weight * consistency_loss
                + train_config.noop_loss_weight * noop_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(epoch_losses))})

    model.eval()
    with torch.no_grad():
        prediction = model(
            x_member,
            x_coefficients,
            x_intervention,
            x_proposal_mask,
            x_member_mask,
            hard_selection=True,
        )
    predicted_coefficients = prediction["coefficient_mean"].cpu().numpy()
    predicted_write = prediction["write_probability"].cpu().numpy()
    predicted_variance = prediction["coefficient_variance"].cpu().numpy()

    view_uncertainty: list[float] = []
    with torch.no_grad():
        for index, record in enumerate(training_records):
            views = memory_views(
                record,
                seed=train_config.seed,
                bootstrap_views=train_config.bootstrap_views,
                bootstrap_size=min(train_config.bootstrap_size, len(record.member_ids)),
            )
            masks = np.zeros((len(views), member.shape[1]), dtype=np.bool_)
            for row, view in enumerate(views):
                masks[row, list(view.member_indices)] = True
            count = len(views)
            result = model(
                x_member[index : index + 1].expand(count, -1, -1),
                x_coefficients[index : index + 1].expand(count, -1, -1),
                x_intervention[index : index + 1].expand(count, -1, -1),
                x_proposal_mask[index : index + 1].expand(count, -1),
                x_member_mask[index : index + 1].expand(count, -1),
                view_mask=torch.as_tensor(masks, dtype=torch.bool, device=device),
                hard_selection=True,
            )
            epistemic = float(
                result["coefficient_mean"].var(dim=0, unbiased=False).mean().cpu()
            )
            aleatoric = float(result["coefficient_variance"][0].mean().cpu())
            view_uncertainty.append(epistemic + aleatoric)
    view_uncertainty_array = np.asarray(view_uncertainty, dtype=np.float32)
    positive_validation = np.asarray(validation_idx, dtype=np.int64)
    positive_validation = positive_validation[target_write[positive_validation] > 0]
    uncertainty_threshold = (
        float(
            np.quantile(
                view_uncertainty_array[positive_validation],
                train_config.uncertainty_quantile,
            )
        )
        if len(positive_validation)
        else 0.0
    )
    deployment_gate = {
        "write_threshold": train_config.write_threshold,
        "uncertainty_threshold": uncertainty_threshold,
        "uncertainty_quantile": train_config.uncertainty_quantile,
        "calibration_partition": "train-validation-memory-task-clusters",
        "positive_calibration_windows": int(len(positive_validation)),
        "test_labels_used": False,
    }

    def metrics(indices: list[int]) -> dict:
        idx = np.asarray(indices, dtype=np.int64)
        positive = idx[target_write[idx] > 0]
        negative = idx[target_write[idx] == 0]
        mse = np.square(predicted_coefficients[idx] - target_coefficients[idx]).mean(axis=1)
        cosine: list[float] = []
        for row in positive:
            denominator = np.linalg.norm(predicted_coefficients[row]) * np.linalg.norm(
                target_coefficients[row]
            )
            if denominator > 1e-12:
                cosine.append(
                    float(
                        np.dot(predicted_coefficients[row], target_coefficients[row])
                        / denominator
                    )
                )
        return {
            "windows": int(len(idx)),
            "streams": len({training_records[int(row)].stream_id for row in idx}),
            "memory_task_clusters": len(
                {
                    memory_task_cluster_id(training_records[int(row)])
                    for row in idx
                }
            ),
            "positive_targets": int(len(positive)),
            "coefficient_mse": float(mse.mean()),
            "positive_direction_cosine": float(np.mean(cosine)) if cosine else None,
            "negative_noop_norm": (
                float(np.linalg.norm(predicted_coefficients[negative], axis=1).mean())
                if len(negative)
                else None
            ),
            "write_accuracy": float(
                ((predicted_write[idx] >= 0.5) == (target_write[idx] > 0)).mean()
            ),
            "mean_predicted_variance": float(predicted_variance[idx].mean()),
            "mean_total_uncertainty": float(view_uncertainty_array[idx].mean()),
            "deployment_write_rate": float(
                np.mean(
                    (predicted_write[idx] >= train_config.write_threshold)
                    & (view_uncertainty_array[idx] <= uncertainty_threshold)
                )
            ),
        }

    report = {
        "format": "sediment-stable-signal-v1",
        "train": metrics(train_idx),
        "validation": metrics(validation_idx),
        "train_memory_task_clusters": sorted(
            {memory_task_cluster_id(training_records[index]) for index in train_idx}
        ),
        "validation_memory_task_clusters": validation_clusters,
        "validation_streams": sorted(
            {training_records[index].stream_id for index in validation_idx}
        ),
        "test_windows_unread": heldout_count,
        "provenance": dict(provenance or {}),
        "member_feature_names": list(member_names),
        "proposal_feature_names": list(proposal_names),
        "positive_target_rate": float(
            np.mean(
                [
                    np.mean(
                        [
                            target_write[index]
                            for index in train_idx
                            if memory_task_cluster_id(training_records[index]) == cluster
                        ]
                    )
                    for cluster in sorted(
                        {
                            memory_task_cluster_id(training_records[index])
                            for index in train_idx
                        }
                    )
                ]
            )
        ),
        "positive_target_rate_window_weighted": float(target_write[train_idx].mean()),
        "cluster_balanced_training": True,
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "training_examples_per_epoch": len(
            {memory_task_cluster_id(training_records[index]) for index in train_idx}
        ),
        "deployment_gate": deployment_gate,
        "history": history,
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": report["format"],
            "model_config": model_config.to_dict(),
            "training_config": asdict(train_config),
            "normalization": normalization,
            "deployment_gate": deployment_gate,
            "member_feature_names": list(member_names),
            "proposal_feature_names": list(proposal_names),
            "model_state": model.state_dict(),
            "report": report,
        },
        target,
    )
    with target.with_suffix(target.suffix + ".report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def load_stable_signal_checkpoint(path: str | Path, *, device: str = "cpu"):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("loading a stable signal model requires PyTorch") from exc
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if checkpoint.get("format") != "sediment-stable-signal-v1":
        raise ValueError(f"unsupported stable signal checkpoint: {checkpoint.get('format')!r}")
    config = StableSignalConfig.from_dict(checkpoint["model_config"])
    model = build_stable_signal_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def stable_inference_tensors(
    records: list[StableWindowRecord], checkpoint: dict
):
    """Apply the checkpoint's train-only normalization to deployment inputs."""

    (
        member,
        coefficients,
        intervention,
        proposal_mask,
        member_mask,
        member_names,
        proposal_names,
    ) = stable_feature_tensors(records)
    if list(member_names) != checkpoint["member_feature_names"]:
        raise ValueError("member feature schema differs from the stable checkpoint")
    if list(proposal_names) != checkpoint["proposal_feature_names"]:
        raise ValueError("proposal feature schema differs from the stable checkpoint")
    normalization = checkpoint["normalization"]

    def normalize(values, mean_name, std_name, valid):
        mean = np.asarray(normalization[mean_name], dtype=np.float32)
        std = np.asarray(normalization[std_name], dtype=np.float32)
        if values.shape[-1] != len(mean) or mean.shape != std.shape:
            raise ValueError(f"normalization shape differs for {mean_name}")
        result = (values - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
        result[~valid] = 0.0
        return result.astype(np.float32)

    member = normalize(member, "member_mean", "member_std", member_mask)
    coefficients = normalize(
        coefficients,
        "proposal_coefficient_mean",
        "proposal_coefficient_std",
        proposal_mask,
    )
    intervention = normalize(
        intervention,
        "proposal_feature_mean",
        "proposal_feature_std",
        proposal_mask,
    )
    return member, coefficients, intervention, proposal_mask, member_mask


def predict_stable_record(
    model,
    checkpoint: dict,
    record: StableWindowRecord,
) -> dict:
    """Return frozen full/view prediction, total uncertainty, and gate decision."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("stable signal inference requires PyTorch") from exc
    member, coefficients, intervention, proposal_mask, member_mask = (
        stable_inference_tensors([record], checkpoint)
    )
    training = checkpoint["training_config"]
    views = memory_views(
        record,
        seed=int(training["seed"]),
        bootstrap_views=int(training["bootstrap_views"]),
        bootstrap_size=min(int(training["bootstrap_size"]), len(record.member_ids)),
    )
    view_mask = np.zeros((len(views), member.shape[1]), dtype=np.bool_)
    for row, view in enumerate(views):
        view_mask[row, list(view.member_indices)] = True
    device = next(model.parameters()).device
    count = len(views)

    def tensor(value, *, dtype):
        return torch.as_tensor(value, dtype=dtype, device=device)

    with torch.no_grad():
        result = model(
            tensor(member, dtype=torch.float32).expand(count, -1, -1),
            tensor(coefficients, dtype=torch.float32).expand(count, -1, -1),
            tensor(intervention, dtype=torch.float32).expand(count, -1, -1),
            tensor(proposal_mask, dtype=torch.bool).expand(count, -1),
            tensor(member_mask, dtype=torch.bool).expand(count, -1),
            view_mask=tensor(view_mask, dtype=torch.bool),
            hard_selection=True,
        )
    means = result["coefficient_mean"].cpu().numpy()
    variances = result["coefficient_variance"].cpu().numpy()
    write = result["write_probability"].cpu().numpy()
    epistemic = float(np.var(means, axis=0).mean())
    aleatoric = float(variances[0].mean())
    uncertainty = epistemic + aleatoric
    gate = checkpoint["deployment_gate"]
    passed = bool(
        write[0] >= float(gate["write_threshold"])
        and uncertainty <= float(gate["uncertainty_threshold"])
    )
    coefficients_out = means[0] if passed else np.zeros_like(means[0])
    return {
        "window_id": record.window_id,
        "coefficients": coefficients_out.astype(np.float32).tolist(),
        "ungated_coefficients": means[0].astype(np.float32).tolist(),
        "write_probability": float(write[0]),
        "predicted_variance": aleatoric,
        "view_dispersion": epistemic,
        "total_uncertainty": uncertainty,
        "write_threshold": float(gate["write_threshold"]),
        "uncertainty_threshold": float(gate["uncertainty_threshold"]),
        "gate_passed": passed,
        "view_names": [view.name for view in views],
        "view_coefficients": means.astype(np.float32).tolist(),
        "member_weights": result["member_weights"][0].cpu().numpy().astype(np.float32).tolist(),
        "proposal_weights": result["proposal_weights"][0]
        .cpu()
        .numpy()
        .astype(np.float32)
        .tolist(),
        "shrinkage": float(result["shrinkage"][0].cpu()),
    }
