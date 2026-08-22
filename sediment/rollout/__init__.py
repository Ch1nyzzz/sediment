"""Rollout: single-episode agent loop and experience injection."""
from sediment.rollout.agent_loop import (
    EXPERIENCE_CLOSE,
    EXPERIENCE_OPEN,
    inject_experience,
    run_episode,
)

__all__ = ["EXPERIENCE_CLOSE", "EXPERIENCE_OPEN", "inject_experience", "run_episode"]
