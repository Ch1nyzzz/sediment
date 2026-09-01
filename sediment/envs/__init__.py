"""Environments: Env protocol, ToyOrderEnv, EnvScaler adapter, task factories."""
from __future__ import annotations

from typing import Any, Optional

from sediment.config import StreamConfig
from sediment.envs.base import Env, parse_tool_call, tool_call_count
from sediment.envs.envscaler import (
    DEFAULT_LOPD_DIR,
    EnvScalerAdapter,
    lopd_available,
)
from sediment.envs.toy import ToyOrderEnv, make_toy_tasks

__all__ = [
    "DEFAULT_LOPD_DIR",
    "Env",
    "EnvScalerAdapter",
    "ToyOrderEnv",
    "lopd_available",
    "make_env",
    "make_toy_tasks",
    "parse_tool_call",
    "tool_call_count",
]


def make_env(task: dict[str, Any], cfg: Optional[StreamConfig] = None) -> Env:
    """Build a fresh env instance for one task dict, dispatched on env_family."""
    family = str(task.get("env_family", "toy_order"))
    if family == "toy_order":
        return ToyOrderEnv()
    if family == "envscaler":
        return EnvScalerAdapter(max_steps=cfg.max_steps if cfg is not None else 30)
    raise ValueError(f"unknown env_family: {family}")
