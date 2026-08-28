"""Leak-free records and deterministic views for stable memory-signal learning.

The compiler input is the completed memory window plus donor-intervention
features.  Discovery and confirmation rewards live in the same file-backed
record because they are offline labels, but tensorization deliberately excludes
them.  This module has no Torch dependency.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def _finite_vector(values: Iterable[float], *, where: str) -> list[float]:
    vector = [float(value) for value in values]
    if not vector or not np.isfinite(vector).all():
        raise ValueError(f"{where} must be a non-empty finite vector")
    return vector


def _numeric_features(raw: dict[str, Any], *, where: str) -> dict[str, float]:
    values = {str(key): float(value) for key, value in dict(raw).items()}
    if not values or not np.isfinite(list(values.values())).all():
        raise ValueError(f"{where} must contain finite numeric features")
    return values


@dataclass(frozen=True)
class DonorProposal:
    """One single-donor intervention projected into the frozen update basis."""

    proposal_id: str
    member_id: str
    member_family: str
    coefficients: list[float]
    intervention_features: dict[str, float]
    discovery_family_gains: dict[str, float] = field(default_factory=dict)
    confirmation_family_gains: dict[str, list[float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _finite_vector(self.coefficients, where=f"proposal {self.proposal_id!r} coefficients")
        _numeric_features(
            self.intervention_features,
            where=f"proposal {self.proposal_id!r} intervention_features",
        )
        for family, value in self.discovery_family_gains.items():
            if not family or not math.isfinite(float(value)):
                raise ValueError(f"proposal {self.proposal_id!r} has invalid discovery gain")
        for family, values in self.confirmation_family_gains.items():
            if not family:
                raise ValueError(f"proposal {self.proposal_id!r} has an empty family")
            _finite_vector(
                values,
                where=f"proposal {self.proposal_id!r} confirmation family {family!r}",
            )

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "DonorProposal":
        required = {
            "proposal_id",
            "member_id",
            "member_family",
            "coefficients",
            "intervention_features",
        }
        missing = required - raw.keys()
        if missing:
            raise ValueError(f"donor proposal missing fields: {sorted(missing)}")
        return DonorProposal(
            proposal_id=str(raw["proposal_id"]),
            member_id=str(raw["member_id"]),
            member_family=str(raw["member_family"]),
            coefficients=_finite_vector(
                raw["coefficients"], where=f"proposal {raw['proposal_id']!r} coefficients"
            ),
            intervention_features=_numeric_features(
                raw["intervention_features"],
                where=f"proposal {raw['proposal_id']!r} intervention_features",
            ),
            discovery_family_gains={
                str(key): float(value)
                for key, value in dict(raw.get("discovery_family_gains", {})).items()
            },
            confirmation_family_gains={
                str(key): [float(value) for value in values]
                for key, values in dict(raw.get("confirmation_family_gains", {})).items()
            },
            metadata=dict(raw.get("metadata", {})),
        )


@dataclass(frozen=True)
class StableWindowRecord:
    """One B=16 memory window with offline proposal-reward labels."""

    window_id: str
    stream_id: str
    partition: str
    member_ids: list[str]
    member_families: list[str]
    member_features: list[dict[str, float]]
    proposals: list[DonorProposal]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.partition not in {"train", "test"}:
            raise ValueError("stable window partition must be 'train' or 'test'")
        size = len(self.member_ids)
        if size == 0 or len(self.member_families) != size or len(self.member_features) != size:
            raise ValueError("member ids, families, and features must have equal non-zero length")
        if len(set(self.member_ids)) != size:
            raise ValueError(f"window {self.window_id!r} has duplicate member ids")
        for features in self.member_features:
            _numeric_features(features, where=f"window {self.window_id!r} member features")
        proposal_ids = [proposal.proposal_id for proposal in self.proposals]
        if len(proposal_ids) != len(set(proposal_ids)):
            raise ValueError(f"window {self.window_id!r} has duplicate proposal ids")
        ranks = {len(proposal.coefficients) for proposal in self.proposals}
        if len(ranks) > 1:
            raise ValueError(f"window {self.window_id!r} has inconsistent proposal ranks")
        by_member = dict(zip(self.member_ids, self.member_families))
        seen_members: set[str] = set()
        for proposal in self.proposals:
            if proposal.member_id not in by_member:
                raise ValueError(f"proposal {proposal.proposal_id!r} selects an unknown member")
            if by_member[proposal.member_id] != proposal.member_family:
                raise ValueError(f"proposal {proposal.proposal_id!r} member family disagrees")
            if proposal.member_id in seen_members:
                raise ValueError("stable v1 permits at most one proposal per member")
            seen_members.add(proposal.member_id)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "StableWindowRecord":
        required = {
            "window_id",
            "stream_id",
            "partition",
            "member_ids",
            "member_families",
            "member_features",
            "proposals",
        }
        missing = required - raw.keys()
        if missing:
            raise ValueError(f"stable window record missing fields: {sorted(missing)}")
        return StableWindowRecord(
            window_id=str(raw["window_id"]),
            stream_id=str(raw["stream_id"]),
            partition=str(raw["partition"]),
            member_ids=[str(value) for value in raw["member_ids"]],
            member_families=[str(value) for value in raw["member_families"]],
            member_features=[
                _numeric_features(value, where=f"window {raw['window_id']!r} member")
                for value in raw["member_features"]
            ],
            proposals=[DonorProposal.from_dict(value) for value in raw["proposals"]],
            metadata=dict(raw.get("metadata", {})),
        )


def member_task_cluster_id(member_ids: Iterable[str]) -> str:
    """Stable hash for request-seed variants sharing one source task set."""

    payload = json.dumps(
        sorted(str(value) for value in member_ids),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "memory-" + hashlib.sha256(payload).hexdigest()[:16]


def memory_task_cluster_id(record: StableWindowRecord) -> str:
    return member_task_cluster_id(record.member_ids)


def select_validation_memory_clusters(
    cluster_ids: Iterable[str], *, fraction: float, seed: int
) -> list[str]:
    clusters = sorted(set(cluster_ids))
    if len(clusters) < 2:
        raise ValueError(
            "stable training validation requires at least two source-memory task clusters"
        )
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation cluster fraction must be in (0, 1)")
    random.Random(seed).shuffle(clusters)
    count = min(len(clusters) - 1, max(1, round(len(clusters) * fraction)))
    return sorted(clusters[:count])


@dataclass(frozen=True)
class ProposalEvidence:
    proposal_id: str
    mean_gain: float
    standard_error: float
    lower_bound: float
    worst_family_gain: float
    family_count: int


@dataclass(frozen=True)
class StableTarget:
    coefficients: list[float]
    write: bool
    proposal_ids: list[str]
    proposal_weights: list[float]
    conservative_gain: float
    worst_family_gain: float | None
    coefficient_dispersion: float
    evidence: list[ProposalEvidence]


def proposal_evidence(proposal: DonorProposal, *, kappa: float = 1.0) -> ProposalEvidence:
    """Aggregate repeats within family, then compute uncertainty across families."""

    if kappa < 0:
        raise ValueError("kappa must be nonnegative")
    means = np.asarray(
        [np.mean(values) for values in proposal.confirmation_family_gains.values()],
        dtype=np.float64,
    )
    if len(means) == 0:
        return ProposalEvidence(proposal.proposal_id, 0.0, math.inf, -math.inf, -math.inf, 0)
    mean = float(means.mean())
    standard_error = (
        float(means.std(ddof=1) / math.sqrt(len(means))) if len(means) > 1 else math.inf
    )
    lower = mean - kappa * standard_error
    return ProposalEvidence(
        proposal_id=proposal.proposal_id,
        mean_gain=mean,
        standard_error=standard_error,
        lower_bound=lower,
        worst_family_gain=float(means.min()),
        family_count=int(len(means)),
    )


def stable_target(
    record: StableWindowRecord,
    *,
    kappa: float = 1.0,
    catastrophe_floor: float = -0.20,
) -> StableTarget:
    """Reward-weighted barycenter of conservatively positive donor proposals."""

    if not record.proposals:
        raise ValueError(f"window {record.window_id!r} has no donor proposals")
    evidence = [proposal_evidence(proposal, kappa=kappa) for proposal in record.proposals]
    accepted: list[tuple[DonorProposal, ProposalEvidence]] = [
        (proposal, item)
        for proposal, item in zip(record.proposals, evidence)
        if item.lower_bound > 0.0 and item.worst_family_gain >= catastrophe_floor
    ]
    rank = len(record.proposals[0].coefficients)
    if not accepted:
        return StableTarget(
            coefficients=[0.0] * rank,
            write=False,
            proposal_ids=[],
            proposal_weights=[],
            conservative_gain=0.0,
            worst_family_gain=None,
            coefficient_dispersion=0.0,
            evidence=evidence,
        )
    raw_weights = np.asarray([item.lower_bound for _, item in accepted], dtype=np.float64)
    weights = raw_weights / raw_weights.sum()
    coefficients = np.asarray(
        [proposal.coefficients for proposal, _ in accepted], dtype=np.float64
    )
    target = np.sum(coefficients * weights[:, None], axis=0)
    dispersion = float(
        np.sqrt(np.sum(weights[:, None] * np.square(coefficients - target)).mean())
    )
    return StableTarget(
        coefficients=target.astype(np.float32).tolist(),
        write=True,
        proposal_ids=[proposal.proposal_id for proposal, _ in accepted],
        proposal_weights=weights.astype(np.float32).tolist(),
        conservative_gain=float(np.dot(weights, [item.lower_bound for _, item in accepted])),
        worst_family_gain=min(item.worst_family_gain for _, item in accepted),
        coefficient_dispersion=dispersion,
        evidence=evidence,
    )


@dataclass(frozen=True)
class FamilyPartition:
    train_members: list[str]
    train_discovery: list[str]
    train_confirmation: list[str]
    test_members: list[str]
    test_discovery: list[str]
    test_confirmation: list[str]
    seed: int
    digest: str

    @property
    def all_families(self) -> list[str]:
        return (
            self.train_members
            + self.train_discovery
            + self.train_confirmation
            + self.test_members
            + self.test_discovery
            + self.test_confirmation
        )


def partition_families(
    families: Iterable[str],
    *,
    seed: int = 20260827,
    counts: tuple[int, int, int, int, int, int] = (10, 4, 4, 8, 4, 4),
    success_counts: Mapping[str, int | float] | None = None,
) -> FamilyPartition:
    """Partition families before any candidate reward is observed.

    When completed-memory success counts are supplied, a deterministic greedy
    assignment balances success density per role capacity. This is allowed
    feasibility stratification: it uses facts already present in the memory
    windows, never proposal or future-probe reward.
    """

    unique = sorted({str(family) for family in families})
    if any(count < 1 for count in counts) or sum(counts) != len(unique):
        raise ValueError(f"partition counts {counts} must be positive and sum to {len(unique)}")
    def hash_key(value: str) -> str:
        return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    if success_counts is None:
        ordered = sorted(unique, key=hash_key)
        offsets = np.cumsum((0,) + counts).tolist()
        groups = [ordered[offsets[index] : offsets[index + 1]] for index in range(6)]
    else:
        if set(success_counts) != set(unique):
            raise ValueError("success_counts keys must exactly match the family set")
        weights = {family: float(success_counts[family]) for family in unique}
        if any(not math.isfinite(value) or value < 0 for value in weights.values()):
            raise ValueError("success_counts must be finite and nonnegative")
        ordered = sorted(unique, key=lambda family: (-weights[family], hash_key(family)))
        groups = [[] for _ in counts]
        loads = [0.0 for _ in counts]
        for family in ordered:
            eligible = [index for index, count in enumerate(counts) if len(groups[index]) < count]
            role = min(
                eligible,
                key=lambda index: (
                    loads[index] / counts[index],
                    len(groups[index]) / counts[index],
                    hash_key(f"role:{index}:{family}"),
                ),
            )
            groups[role].append(family)
            loads[role] += weights[family]
        groups = [sorted(group, key=hash_key) for group in groups]
    payload = {
        "seed": seed,
        "train_members": groups[0],
        "train_discovery": groups[1],
        "train_confirmation": groups[2],
        "test_members": groups[3],
        "test_discovery": groups[4],
        "test_confirmation": groups[5],
        "stratification": (
            "completed_success_count_greedy_balance_v1"
            if success_counts is not None
            else "hash_only_v1"
        ),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload.pop("stratification")
    return FamilyPartition(**payload, digest=digest)


@dataclass(frozen=True)
class MemoryView:
    name: str
    member_indices: tuple[int, ...]


def memory_views(
    record: StableWindowRecord,
    *,
    seed: int = 20260827,
    bootstrap_views: int = 4,
    bootstrap_size: int = 8,
) -> list[MemoryView]:
    """Full, leave-one-family-out, and deterministic 8/16 subset views."""

    size = len(record.member_ids)
    if not 1 <= bootstrap_size <= size or bootstrap_views < 0:
        raise ValueError("bootstrap size/views are incompatible with the window")
    donor_indices = {record.member_ids.index(proposal.member_id) for proposal in record.proposals}
    if not donor_indices:
        raise ValueError(f"window {record.window_id!r} has no donor-bearing member")
    views = [MemoryView("full", tuple(range(size)))]
    for family in sorted(set(record.member_families)):
        kept = tuple(
            index for index, value in enumerate(record.member_families) if value != family
        )
        if len(kept) >= 2 and donor_indices.intersection(kept):
            views.append(MemoryView(f"leave-family:{family}", kept))

    rng = random.Random(f"{seed}:{record.window_id}")
    seen = {view.member_indices for view in views}
    attempts = 0
    while sum(view.name.startswith("subset:") for view in views) < bootstrap_views:
        attempts += 1
        if attempts > max(100, bootstrap_views * 50):
            raise ValueError("could not construct enough unique donor-bearing subset views")
        indices = tuple(sorted(rng.sample(range(size), bootstrap_size)))
        if indices in seen or not donor_indices.intersection(indices):
            continue
        seen.add(indices)
        number = sum(view.name.startswith("subset:") for view in views)
        views.append(MemoryView(f"subset:{number}", indices))
    return views


def stable_feature_tensors(records: list[StableWindowRecord]):
    """Tensorize deployment inputs only.

    Returns member features, proposal coefficients, proposal intervention
    features, proposal mask, member mask, and both feature-name tuples. Reward
    labels are intentionally absent from the return value.
    """

    if not records:
        raise ValueError("at least one stable window is required")
    if any(not record.proposals for record in records):
        raise ValueError("every tensorized window needs at least one proposal")
    member_names = tuple(sorted(records[0].member_features[0]))
    proposal_names = tuple(sorted(records[0].proposals[0].intervention_features))
    rank = len(records[0].proposals[0].coefficients)
    max_members = max(len(record.member_ids) for record in records)
    member = np.zeros((len(records), max_members, len(member_names)), dtype=np.float32)
    coefficients = np.zeros((len(records), max_members, rank), dtype=np.float32)
    intervention = np.zeros(
        (len(records), max_members, len(proposal_names)), dtype=np.float32
    )
    proposal_mask = np.zeros((len(records), max_members), dtype=np.bool_)
    member_mask = np.zeros((len(records), max_members), dtype=np.bool_)
    for row, record in enumerate(records):
        if any(set(values) != set(member_names) for values in record.member_features):
            raise ValueError(f"member feature schema mismatch in {record.window_id!r}")
        by_member = {proposal.member_id: proposal for proposal in record.proposals}
        for column, (member_id, values) in enumerate(
            zip(record.member_ids, record.member_features)
        ):
            member[row, column] = [values[name] for name in member_names]
            member_mask[row, column] = True
            proposal = by_member.get(member_id)
            if proposal is None:
                continue
            if len(proposal.coefficients) != rank:
                raise ValueError(f"proposal rank mismatch in {record.window_id!r}")
            if set(proposal.intervention_features) != set(proposal_names):
                raise ValueError(f"proposal feature schema mismatch in {record.window_id!r}")
            coefficients[row, column] = proposal.coefficients
            intervention[row, column] = [
                proposal.intervention_features[name] for name in proposal_names
            ]
            proposal_mask[row, column] = True
    for array in (member, coefficients, intervention):
        if not np.isfinite(array).all():
            raise ValueError("stable compiler inputs contain NaN or infinity")
    return (
        member,
        coefficients,
        intervention,
        proposal_mask,
        member_mask,
        member_names,
        proposal_names,
    )


def load_stable_windows(path: str | Path) -> list[StableWindowRecord]:
    source = Path(path)
    records: list[StableWindowRecord] = []
    with source.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(StableWindowRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{source}:{lineno}: {exc}") from exc
    if not records:
        raise ValueError(f"{source}: no stable window records")
    ids = [record.window_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{source}: duplicate window ids")
    return records


def write_stable_windows(path: str | Path, records: Iterable[StableWindowRecord]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")
