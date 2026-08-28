"""Permutation-invariant stable memory-signal model with uncertainty output."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StableSignalConfig:
    member_feature_dim: int
    proposal_feature_dim: int
    basis_rank: int
    member_hidden: int = 32
    window_hidden: int = 32
    attention_temperature: float = 1.0
    deployment_topk: int = 8
    max_step_norm: float = 12.0
    variance_floor: float = 1e-4
    initial_write_probability: float = 0.01

    def __post_init__(self) -> None:
        if min(self.member_feature_dim, self.proposal_feature_dim, self.basis_rank) <= 0:
            raise ValueError("feature dimensions and basis rank must be positive")
        if min(self.member_hidden, self.window_hidden, self.deployment_topk) <= 0:
            raise ValueError("hidden dimensions and deployment_topk must be positive")
        if min(self.attention_temperature, self.max_step_norm, self.variance_floor) <= 0:
            raise ValueError("temperature, norm, and variance floor must be positive")
        if not 0.0 < self.initial_write_probability < 1.0:
            raise ValueError("initial_write_probability must be in (0, 1)")

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(raw: dict) -> "StableSignalConfig":
        return StableSignalConfig(**raw)


def build_stable_signal_model(config: StableSignalConfig):
    """Build ``q_phi(z | M)`` over deployment features and donor proposals.

    The coefficient mean is a learned, positively scaled convex combination of
    this window's donor proposals; the model cannot hallucinate an unconstrained
    direction from only a few training clusters. The write head starts near zero
    and the returned deployment delta is hard-noop below 0.5. Diagonal variance
    is positive and inspectable; epistemic view dispersion is computed by the
    caller from repeated forward passes with different masks.
    """

    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - remote dependency
        raise RuntimeError("the stable signal model requires PyTorch") from exc

    input_dim = (
        config.member_feature_dim
        + config.basis_rank
        + config.proposal_feature_dim
        + 1
    )

    class StableSignalModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.member_encoder = nn.Sequential(
                nn.Linear(input_dim, config.member_hidden),
                nn.GELU(),
                nn.Linear(config.member_hidden, config.member_hidden),
                nn.GELU(),
            )
            self.salience_head = nn.Linear(config.member_hidden, 1)
            self.window_network = nn.Sequential(
                nn.Linear(config.member_hidden, config.window_hidden),
                nn.GELU(),
                nn.Linear(config.window_hidden, config.window_hidden),
                nn.GELU(),
            )
            self.write_head = nn.Linear(config.window_hidden, 1)
            self.shrinkage_head = nn.Linear(config.window_hidden, 1)
            self.variance_head = nn.Linear(config.window_hidden, config.basis_rank)
            nn.init.zeros_(self.write_head.weight)
            initial_logit = math.log(
                config.initial_write_probability / (1.0 - config.initial_write_probability)
            )
            nn.init.constant_(self.write_head.bias, initial_logit)
            nn.init.zeros_(self.shrinkage_head.weight)
            nn.init.zeros_(self.shrinkage_head.bias)
            nn.init.zeros_(self.variance_head.weight)
            nn.init.zeros_(self.variance_head.bias)

        def _weights(self, logits, mask, *, hard_selection: bool):
            masked = logits.masked_fill(~mask, float("-inf"))
            if hard_selection and masked.shape[1] > config.deployment_topk:
                count = min(config.deployment_topk, masked.shape[1])
                indices = torch.topk(masked, k=count, dim=1).indices
                keep = torch.zeros_like(mask, dtype=torch.bool)
                keep.scatter_(1, indices, True)
                masked = masked.masked_fill(~(mask & keep), float("-inf"))
            return torch.softmax(masked / config.attention_temperature, dim=1)

        def forward(
            self,
            member_features,
            proposal_coefficients,
            proposal_features,
            proposal_mask,
            member_mask=None,
            *,
            view_mask=None,
            hard_selection=False,
        ):
            tensors = (
                member_features,
                proposal_coefficients,
                proposal_features,
                proposal_mask,
            )
            if member_features.ndim == 2:
                tensors = tuple(value.unsqueeze(0) for value in tensors)
                (
                    member_features,
                    proposal_coefficients,
                    proposal_features,
                    proposal_mask,
                ) = tensors
            if member_features.ndim != 3:
                raise ValueError("stable model inputs must have batch/member dimensions")
            shape = member_features.shape[:2]
            if proposal_coefficients.shape != (*shape, config.basis_rank):
                raise ValueError("proposal coefficient shape differs from config")
            if proposal_features.shape != (*shape, config.proposal_feature_dim):
                raise ValueError("proposal feature shape differs from config")
            if proposal_mask.shape != shape:
                raise ValueError("proposal mask shape differs from members")
            if member_features.shape[-1] != config.member_feature_dim:
                raise ValueError("member feature shape differs from config")
            device = member_features.device
            proposal_mask = proposal_mask.to(dtype=torch.bool, device=device)
            if member_mask is None:
                member_mask = torch.ones(shape, dtype=torch.bool, device=device)
            elif member_mask.ndim == 1:
                member_mask = member_mask.unsqueeze(0)
            member_mask = member_mask.to(dtype=torch.bool, device=device)
            if member_mask.shape != shape:
                raise ValueError("member mask shape differs from inputs")
            if view_mask is not None:
                if view_mask.ndim == 1:
                    view_mask = view_mask.unsqueeze(0)
                if view_mask.shape != shape:
                    raise ValueError("view mask shape differs from inputs")
                member_mask = member_mask & view_mask.to(dtype=torch.bool, device=device)
            if not bool(member_mask.any(dim=1).all()):
                raise ValueError("every stable-signal view must retain at least one member")
            proposal_view_mask = member_mask & proposal_mask
            if not bool(proposal_view_mask.any(dim=1).all()):
                raise ValueError("every stable-signal view must retain at least one proposal")

            encoded_input = torch.cat(
                (
                    member_features,
                    proposal_coefficients,
                    proposal_features,
                    proposal_mask.to(member_features.dtype).unsqueeze(-1),
                ),
                dim=-1,
            )
            encoded = self.member_encoder(encoded_input)
            salience_logits = self.salience_head(encoded).squeeze(-1)
            weights = self._weights(
                salience_logits, member_mask, hard_selection=hard_selection
            )
            proposal_weights = self._weights(
                salience_logits, proposal_view_mask, hard_selection=hard_selection
            )
            pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
            hidden = self.window_network(pooled)
            write_probability = torch.sigmoid(self.write_head(hidden)).squeeze(-1)
            shrinkage = 2.0 * torch.sigmoid(self.shrinkage_head(hidden)).squeeze(-1)
            raw_mean = (
                torch.sum(
                    proposal_coefficients * proposal_weights.unsqueeze(-1), dim=1
                )
                * shrinkage.unsqueeze(-1)
            )
            norm = torch.linalg.vector_norm(raw_mean, dim=-1, keepdim=True).clamp_min(1e-12)
            scale = torch.clamp(config.max_step_norm / norm, max=1.0)
            coefficient_mean = raw_mean * scale
            coefficient_variance = (
                torch.nn.functional.softplus(self.variance_head(hidden))
                + config.variance_floor
            )
            gated_delta = torch.where(
                (write_probability >= 0.5).unsqueeze(-1),
                coefficient_mean,
                torch.zeros_like(coefficient_mean),
            )
            return {
                "coefficient_mean": coefficient_mean,
                "coefficient_variance": coefficient_variance,
                "write_probability": write_probability,
                "delta": gated_delta,
                "member_weights": weights,
                "proposal_weights": proposal_weights,
                "shrinkage": shrinkage,
                "salience_logits": salience_logits,
            }

    return StableSignalModel()
