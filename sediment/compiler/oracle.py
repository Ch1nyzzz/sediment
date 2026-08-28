"""Stage-A helpers for independent reverse-KL oracle updates."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np

from sediment.types import HindsightResult, Trajectory

_ERROR = re.compile(r"error|invalid|fail|exceed|denied|cannot|unable", re.I)


def single_sample_optimizer_schedule(steps: int) -> dict[str, int]:
    """Map requested optimizer steps to the trainer's epoch/accumulation knobs.

    A Stage-A candidate contains exactly one sample. Therefore it needs one
    epoch per optimizer step; setting only ``steps_per_merge`` cannot create
    extra passes over that sample.
    """
    if steps <= 0:
        raise ValueError("optimizer steps must be positive")
    return {"epochs": steps, "steps_per_merge": steps}


def deployment_features(traj: Trajectory) -> dict[str, float]:
    """Cheap features of one *complete* deployment experience.

    These fields are available after the episode finishes and before the next
    window is served.  They deliberately exclude hindsight deltas, retrieved
    peers, and future-query outcomes.  The numeric representation is the v0
    baseline for the window compiler; a trajectory encoder must eventually
    beat it rather than silently relying on leaked oracle information.
    """
    assistant = [message for message in traj.messages if message.role == "assistant"]
    feedback = [
        message for message in traj.messages if message.role in ("tool", "user")
    ]
    reflection = str(traj.meta.get("reflection", "") or "")
    action_chars = sum(len(message.content) for message in assistant)
    feedback_chars = sum(len(message.content) for message in feedback)
    return {
        "reward": float(traj.reward or 0.0),
        "success": float(bool(traj.success)),
        "steps": float(traj.steps),
        "assistant_turns": float(len(assistant)),
        "feedback_turns": float(len(feedback)),
        "error_feedback_count": float(
            sum(bool(_ERROR.search(message.content[:500])) for message in feedback)
        ),
        "trajectory_chars": float(
            sum(len(message.content) for message in traj.messages)
        ),
        "action_chars": float(action_chars),
        "feedback_chars": float(feedback_chars),
        "has_reflection": float(bool(reflection)),
        "reflection_chars": float(len(reflection)),
        "terminated": float(bool(traj.meta.get("done", False))),
    }


def experience_features(
    traj: Trajectory, peers: list[Trajectory], hindsight: HindsightResult
) -> dict[str, float]:
    """Cheap v0 features available at deployment, with no future-query leak."""
    features = deployment_features(traj)
    action_deltas = [
        delta
        for span in hindsight.spans
        if span.role == "assistant"
        for delta in span.deltas
    ]
    all_deltas = [delta for span in hindsight.spans for delta in span.deltas]
    same_family = sum(peer.env_family == traj.env_family for peer in peers)

    def rate(values: Iterable[float], predicate) -> float:
        values = list(values)
        return (
            sum(bool(predicate(value)) for value in values) / len(values)
            if values
            else 0.0
        )

    features.update({
        "peer_count": float(len(peers)),
        "same_family_peer_fraction": float(same_family / len(peers)) if peers else 0.0,
        "obs_surprise": float(hindsight.obs_surprise),
        "act_gain": float(hindsight.act_gain),
        "mean_abs_delta": float(np.mean(np.abs(all_deltas))) if all_deltas else 0.0,
        "positive_action_delta_rate": rate(action_deltas, lambda value: value > 0.0),
        "negative_action_delta_rate": rate(action_deltas, lambda value: value < 0.0),
    })
    return features
