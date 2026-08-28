"""Low-dimensional basis for cross-layer LoRA update states."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class ParameterLayout:
    """Deterministic flattening layout shared by every oracle update."""

    keys: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]

    @staticmethod
    def from_state(state: Mapping[str, np.ndarray]) -> "ParameterLayout":
        if not state:
            raise ValueError("cannot build a layout from an empty state")
        keys = tuple(sorted(state))
        return ParameterLayout(
            keys, tuple(tuple(np.asarray(state[key]).shape) for key in keys)
        )

    @property
    def size(self) -> int:
        return sum(int(np.prod(shape, dtype=np.int64)) for shape in self.shapes)

    def flatten(self, state: Mapping[str, np.ndarray]) -> np.ndarray:
        if set(state) != set(self.keys):
            raise ValueError("tensor keys do not match the basis layout")
        chunks = []
        for key, shape in zip(self.keys, self.shapes):
            value = np.asarray(state[key], dtype=np.float32)
            if value.shape != shape:
                raise ValueError(f"shape mismatch for {key}: {value.shape} != {shape}")
            chunks.append(value.reshape(-1))
        return np.concatenate(chunks)

    def unflatten(self, vector: np.ndarray) -> dict[str, np.ndarray]:
        flat = np.asarray(vector, dtype=np.float32).reshape(-1)
        if flat.size != self.size:
            raise ValueError(
                f"vector has {flat.size} values; layout expects {self.size}"
            )
        out: dict[str, np.ndarray] = {}
        offset = 0
        for key, shape in zip(self.keys, self.shapes):
            count = int(np.prod(shape, dtype=np.int64))
            out[key] = flat[offset : offset + count].reshape(shape).copy()
            offset += count
        return out

    def to_dict(self) -> dict:
        return {
            "keys": list(self.keys),
            "shapes": [list(shape) for shape in self.shapes],
        }

    @staticmethod
    def from_dict(raw: dict) -> "ParameterLayout":
        return ParameterLayout(
            tuple(raw["keys"]),
            tuple(tuple(int(v) for v in shape) for shape in raw["shapes"]),
        )


@dataclass
class UpdateBasis:
    """Orthonormal update directions with dimensionless coefficient scales.

    If ``X = U S V^T`` over positive-transfer updates, the stored atom is
    ``B_k = (S_k / sqrt(n)) V_k``. Consequently projected coefficients are on
    an approximately unit scale instead of inheriting raw adapter norms.
    """

    layout: ParameterLayout
    directions: np.ndarray  # [K, D], orthonormal rows
    scales: np.ndarray  # [K], RMS update magnitude along each direction
    singular_values: np.ndarray
    explained_energy: np.ndarray
    fit_samples: int

    @staticmethod
    def fit(states: Iterable[Mapping[str, np.ndarray]], rank: int) -> "UpdateBasis":
        states = list(states)
        if not states:
            raise ValueError("at least one positive-transfer update is required")
        if rank <= 0:
            raise ValueError("rank must be positive")
        layout = ParameterLayout.from_state(states[0])
        matrix = np.stack([layout.flatten(state) for state in states]).astype(
            np.float32
        )
        if not np.isfinite(matrix).all():
            raise ValueError("oracle updates contain NaN or infinity")
        _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        # Only ``min(n_samples, n_parameters)`` singular values are observable.
        # Scaling the tolerance by the parameter width can exceed ``s_max`` for
        # the very wide, few-sample LoRA matrices used here and mechanically
        # classify every non-zero update as rank zero.  Use the size of the
        # observed spectrum for the relative tolerance instead.
        spectrum_size = max(1, int(singular.size))
        numerical_rank = int(
            np.sum(
                singular
                > np.finfo(np.float32).eps
                * spectrum_size
                * (float(singular[0]) if singular.size else 0.0)
            )
        )
        keep = min(rank, numerical_rank)
        if keep == 0:
            raise ValueError("all oracle updates are numerically zero")
        energy = singular**2
        denom = float(energy.sum())
        explained = (
            energy[:keep] / denom if denom > 0 else np.zeros(keep, dtype=np.float32)
        )
        scales = singular[:keep] / np.sqrt(len(states))
        floor = np.finfo(np.float32).eps
        scales = np.maximum(scales, floor)
        return UpdateBasis(
            layout=layout,
            directions=np.asarray(vh[:keep], dtype=np.float32),
            scales=np.asarray(scales, dtype=np.float32),
            singular_values=np.asarray(singular[:keep], dtype=np.float32),
            explained_energy=np.asarray(explained, dtype=np.float32),
            fit_samples=len(states),
        )

    @property
    def rank(self) -> int:
        return int(self.directions.shape[0])

    @property
    def atoms(self) -> np.ndarray:
        return self.directions * self.scales[:, None]

    def project_vector(self, vector: np.ndarray) -> np.ndarray:
        flat = np.asarray(vector, dtype=np.float32).reshape(-1)
        if flat.size != self.layout.size:
            raise ValueError(
                f"vector has {flat.size} values; expected {self.layout.size}"
            )
        return np.asarray((self.directions @ flat) / self.scales, dtype=np.float32)

    def project_state(self, state: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.project_vector(self.layout.flatten(state))

    def reconstruct_vector(self, coefficients: np.ndarray) -> np.ndarray:
        coeff = np.asarray(coefficients, dtype=np.float32).reshape(-1)
        if coeff.size != self.rank:
            raise ValueError(
                f"got {coeff.size} coefficients; basis rank is {self.rank}"
            )
        return np.asarray((coeff * self.scales) @ self.directions, dtype=np.float32)

    def reconstruct_state(self, coefficients: np.ndarray) -> dict[str, np.ndarray]:
        return self.layout.unflatten(self.reconstruct_vector(coefficients))

    def reconstruction_error(self, state: Mapping[str, np.ndarray]) -> float:
        vector = self.layout.flatten(state)
        residual = vector - self.reconstruct_vector(self.project_vector(vector))
        return float(
            np.linalg.norm(residual) / max(float(np.linalg.norm(vector)), 1e-12)
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format": "sediment-update-basis-v1",
            "layout": self.layout.to_dict(),
            "fit_samples": self.fit_samples,
        }
        encoded = np.frombuffer(
            json.dumps(metadata, sort_keys=True).encode("utf-8"), dtype=np.uint8
        )
        with target.open("wb") as handle:
            np.savez_compressed(
                handle,
                directions=np.asarray(self.directions, dtype=np.float32),
                scales=np.asarray(self.scales, dtype=np.float32),
                singular_values=np.asarray(self.singular_values, dtype=np.float32),
                explained_energy=np.asarray(self.explained_energy, dtype=np.float32),
                metadata=encoded,
            )

    @staticmethod
    def load(path: str | Path) -> "UpdateBasis":
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(bytes(archive["metadata"].tolist()).decode("utf-8"))
            if metadata.get("format") != "sediment-update-basis-v1":
                raise ValueError(
                    f"unsupported basis format: {metadata.get('format')!r}"
                )
            return UpdateBasis(
                layout=ParameterLayout.from_dict(metadata["layout"]),
                directions=np.asarray(archive["directions"], dtype=np.float32),
                scales=np.asarray(archive["scales"], dtype=np.float32),
                singular_values=np.asarray(
                    archive["singular_values"], dtype=np.float32
                ),
                explained_energy=np.asarray(
                    archive["explained_energy"], dtype=np.float32
                ),
                fit_samples=int(metadata["fit_samples"]),
            )
