"""File-backed records for oracle updates and projected compiler targets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class OracleRecord:
    """One experience and its independently generated reverse-KL update.

    ``update_path`` points to either an already-differenced ``.npz`` tensor
    state or a trained LoRA checkpoint. In the latter case callers supply the
    one common reference initialization when loading the update.
    """

    sample_id: str
    family: str
    update_path: str
    features: dict[str, float]
    transfer_gain: float | None = None
    current_reward: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(
        raw: dict[str, Any], *, base_dir: Path | None = None
    ) -> "OracleRecord":
        required = {"sample_id", "family", "update_path", "features"}
        missing = required - raw.keys()
        if missing:
            raise ValueError(f"oracle record missing fields: {sorted(missing)}")
        path = Path(str(raw["update_path"]))
        if base_dir is not None and not path.is_absolute():
            path = (base_dir / path).resolve()
        features = {str(k): float(v) for k, v in dict(raw["features"]).items()}
        if not features:
            raise ValueError(f"oracle record {raw['sample_id']!r} has no features")
        return OracleRecord(
            sample_id=str(raw["sample_id"]),
            family=str(raw["family"]),
            update_path=str(path),
            features=features,
            transfer_gain=(
                None
                if raw.get("transfer_gain") is None
                else float(raw["transfer_gain"])
            ),
            current_reward=(
                None
                if raw.get("current_reward") is None
                else float(raw["current_reward"])
            ),
            metadata=dict(raw.get("metadata", {})),
        )


@dataclass(frozen=True)
class ProjectedRecord:
    """An oracle record after projecting its update into the frozen basis."""

    sample_id: str
    family: str
    features: dict[str, float]
    kl_coefficients: list[float]
    transfer_gain: float
    current_reward: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "ProjectedRecord":
        required = {
            "sample_id",
            "family",
            "features",
            "kl_coefficients",
            "transfer_gain",
        }
        missing = required - raw.keys()
        if missing:
            raise ValueError(f"projected record missing fields: {sorted(missing)}")
        return ProjectedRecord(
            sample_id=str(raw["sample_id"]),
            family=str(raw["family"]),
            features={str(k): float(v) for k, v in dict(raw["features"]).items()},
            kl_coefficients=[float(v) for v in raw["kl_coefficients"]],
            transfer_gain=float(raw["transfer_gain"]),
            current_reward=(
                None
                if raw.get("current_reward") is None
                else float(raw["current_reward"])
            ),
            metadata=dict(raw.get("metadata", {})),
        )


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{lineno}: invalid JSON: {exc}") from exc
    if not rows:
        raise ValueError(f"{source}: no records")
    return rows


def load_oracle_records(path: str | Path) -> list[OracleRecord]:
    source = Path(path).resolve()
    records = [
        OracleRecord.from_dict(row, base_dir=source.parent)
        for row in _read_jsonl(source)
    ]
    _validate_unique_ids(records, source)
    return records


def load_projected_records(path: str | Path) -> list[ProjectedRecord]:
    source = Path(path).resolve()
    records = [ProjectedRecord.from_dict(row) for row in _read_jsonl(source)]
    _validate_unique_ids(records, source)
    return records


def write_projected_records(
    path: str | Path, records: Iterable[ProjectedRecord]
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n"
            )


def feature_matrix(records: list[ProjectedRecord]):
    """Return ``(matrix, feature_names)`` with strict schema validation."""
    import numpy as np

    if not records:
        raise ValueError("at least one projected record is required")
    names = tuple(sorted(records[0].features))
    if not names:
        raise ValueError("feature schema is empty")
    expected = set(names)
    for record in records:
        if set(record.features) != expected:
            raise ValueError(
                f"feature schema mismatch for {record.sample_id}: "
                f"expected {names}, got {tuple(sorted(record.features))}"
            )
    matrix = np.asarray(
        [[record.features[name] for name in names] for record in records],
        dtype=np.float32,
    )
    if not np.isfinite(matrix).all():
        raise ValueError("features contain NaN or infinity")
    return matrix, names


def _validate_unique_ids(records: list[Any], source: Path) -> None:
    ids = [record.sample_id for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{source}: duplicate sample_id values")
