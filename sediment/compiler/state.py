"""Tensor-state loading for KL oracle adapters.

``safetensors`` is optional locally and imported only when a real adapter is
read. NumPy ``.npz`` states keep the basis pipeline unit-testable without the
training stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

TensorState = dict[str, np.ndarray]


def load_tensor_state(path: str | Path) -> TensorState:
    source = Path(path)
    if source.is_dir():
        source = source / "adapter_model.safetensors"
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            state = {
                key: np.asarray(archive[key], dtype=np.float32)
                for key in archive.files
                if not key.startswith("__")
            }
    elif source.suffix == ".safetensors":
        try:
            from safetensors.numpy import load_file
        except ImportError as exc:  # pragma: no cover - remote training dependency
            raise RuntimeError("reading LoRA checkpoints requires safetensors") from exc
        state = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in load_file(str(source)).items()
        }
    else:
        raise ValueError(
            f"unsupported tensor state: {source}; use .npz or .safetensors"
        )
    if not state:
        raise ValueError(f"tensor state is empty: {source}")
    return state


def delta_state(
    trained: Mapping[str, np.ndarray], reference: Mapping[str, np.ndarray] | None = None
) -> TensorState:
    """Return float32 ``trained - reference`` with exact key/shape checks."""
    if reference is None:
        return {
            key: np.asarray(value, dtype=np.float32) for key, value in trained.items()
        }
    if set(trained) != set(reference):
        missing = sorted(set(reference) - set(trained))
        extra = sorted(set(trained) - set(reference))
        raise ValueError(f"tensor keys differ; missing={missing}, extra={extra}")
    out: TensorState = {}
    for key in sorted(trained):
        left = np.asarray(trained[key], dtype=np.float32)
        right = np.asarray(reference[key], dtype=np.float32)
        if left.shape != right.shape:
            raise ValueError(f"shape mismatch for {key}: {left.shape} != {right.shape}")
        out[key] = left - right
    return out


def load_update(
    path: str | Path, reference: Mapping[str, np.ndarray] | None = None
) -> TensorState:
    return delta_state(load_tensor_state(path), reference)


def save_tensor_state(path: str | Path, state: Mapping[str, np.ndarray]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{
                key: np.asarray(value, dtype=np.float32) for key, value in state.items()
            },
        )
