"""Consolidate an accepted candidate into the session adapter line."""
from __future__ import annotations

import os
import shutil

from .config import StreamConfig
from .registry import Registry
from .types import AdapterVersion, UpdateCandidate

_ADAPTER_FILE = "adapter_model.safetensors"


def _provenance_union(session: AdapterVersion, candidate: UpdateCandidate) -> list[str]:
    """Session provenance + candidate task ids, order-preserving dedup."""
    out = list(session.provenance)
    seen = set(out)
    for t in candidate.task_ids:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def merge(
    session: AdapterVersion,
    candidate: UpdateCandidate,
    cfg: StreamConfig,
    registry: Registry,
) -> AdapterVersion:
    """Merge a validated candidate into the session and publish the result.

    Stub mode publishes the candidate dir as the next version with the
    provenance union. Torch mode EMA-interpolates the LoRA tensors of
    ``session.path`` and the candidate (alpha = cfg.merge_alpha on the
    candidate) and publishes the merged dir; a base session (path None) has
    no tensors, so the candidate is published as-is.
    """
    provenance = _provenance_union(session, candidate)
    if cfg.trainer == "stub" or session.path is None:
        return registry.publish(candidate.adapter_path, parent=session.name, provenance=provenance)
    merged_dir = _ema_merge(session.path, candidate.adapter_path, cfg.merge_alpha)
    return registry.publish(merged_dir, parent=session.name, provenance=provenance)


def _ema_merge(session_path: str, candidate_path: str, alpha: float) -> str:
    """EMA over matching LoRA tensors: out = (1-alpha)*session + alpha*candidate.

    Tensor names present in only one checkpoint are carried over unchanged.
    The merged dir mirrors the candidate dir (adapter_config.json etc.) with
    the interpolated safetensors file; returns its path.
    """
    from safetensors.torch import load_file, save_file

    out_dir = candidate_path.rstrip("/") + "-merged"
    shutil.copytree(candidate_path, out_dir, dirs_exist_ok=True)
    a = load_file(os.path.join(session_path, _ADAPTER_FILE))
    b = load_file(os.path.join(candidate_path, _ADAPTER_FILE))
    merged = {k: (1.0 - alpha) * a[k] + alpha * t if k in a else t for k, t in b.items()}
    merged.update({k: t for k, t in a.items() if k not in merged})
    save_file(merged, os.path.join(out_dir, _ADAPTER_FILE))
    return out_dir
