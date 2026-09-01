"""Adapter over the vendored LOPD EnvScaler environment.

LOPD is imported lazily via sys.path injection; its envs/ package is pure
stdlib, so the adapter runs without torch/vllm. NOTE: the EnvScaler task
corpora (rl_scenario + env_meta JSON files) are NOT shipped in the LOPD
repo — list_tasks() requires files from the upstream EnvScaler release
(schema documented in LOPD envs/envscaler/data.py); point data_dir at
cfg.data_dir.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, Union

from sediment.envs.base import tool_call_count
from sediment.types import Message

DEFAULT_LOPD_DIR = "/Users/erv1n/res/resid/third_party/LOPD"


def lopd_available(third_party_dir: Union[str, Path] = DEFAULT_LOPD_DIR) -> bool:
    return (Path(third_party_dir) / "envs" / "envscaler" / "env.py").is_file()


def _import_lopd(third_party_dir: Union[str, Path]):
    if not lopd_available(third_party_dir):
        raise RuntimeError(f"LOPD checkout not found at {third_party_dir}")
    root = str(Path(third_party_dir).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from envs.envscaler.data import load_envscaler_samples  # type: ignore
    from envs.envscaler.env import EnvScalerEnv  # type: ignore
    from envs.envscaler import runtime as envscaler_runtime  # type: ignore
    return EnvScalerEnv, load_envscaler_samples, envscaler_runtime


def list_tasks(
    split: str,
    data_dir: Optional[str] = None,
    *,
    third_party_dir: Union[str, Path] = DEFAULT_LOPD_DIR,
    scenario_path: Optional[str] = None,
    env_meta_path: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Load joined EnvScaler samples for a split as sediment task dicts.

    Either pass explicit file paths, or a data_dir (cfg.data_dir) containing
    {split}_scenarios.json and {split}_env_meta.json.
    """
    _, load_samples, _ = _import_lopd(third_party_dir)
    if scenario_path is None or env_meta_path is None:
        if data_dir is None:
            raise ValueError("need data_dir or explicit scenario_path/env_meta_path")
        scenario_path = scenario_path or str(Path(data_dir) / f"{split}_scenarios.json")
        env_meta_path = env_meta_path or str(Path(data_dir) / f"{split}_env_meta.json")
    samples = load_samples(scenario_path, env_meta_path)
    return [
        {"task_id": str(s.get("task_id", i)), "env_family": "envscaler", "payload": s}
        for i, s in enumerate(samples)
    ]


class EnvScalerAdapter:
    """Wraps LOPD's EnvScalerEnv with the sediment Env protocol.

    Tool schemas are inlined into the system prompt by LOPD; raw assistant
    text is passed straight to LOPD's qwen3 <tool_call> parser. Parse-error /
    invalid-action observations re-enter the conversation as plain user
    messages (LOPD protocol), keeping tool-role accounting pure. The outer
    agent loop owns the step budget and force-scores the current state when the
    budget is exhausted; ordinary LOPD tool steps themselves return zero.
    """

    env_family = "envscaler"

    def __init__(self, third_party_dir: Union[str, Path] = DEFAULT_LOPD_DIR,
                 max_steps: int = 30):
        self.third_party_dir = Path(third_party_dir)
        self.max_steps = max_steps
        self._env_cls, _, self._runtime = _import_lopd(third_party_dir)
        self._env = None
        self.last_step_info: dict[str, Any] = {}

    def reset(self, task: dict[str, Any]) -> list[Message]:
        sample = task.get("payload") or {}
        self._env = self._env_cls([sample], max_steps=self.max_steps,
                                  tool_protocol="qwen3")
        obs = self._env.reset(0)
        system = self._env.get_info().extra["system_prompt"]
        system += (
            "\n\n# Tool-call protocol\n"
            "Emit exactly one <tool_call>...</tool_call> block per assistant "
            "turn. Multiple tool calls in one turn are invalid; wait for the "
            "environment observation before choosing the next call."
        )
        return [Message("system", system), Message("user", obs["text"])]

    def step(self, action_text: str) -> tuple[list[Message], bool, float]:
        if self._env is None:
            raise RuntimeError("call reset() before step()")
        n_calls = tool_call_count(action_text)
        if n_calls > 1:
            self.last_step_info = {
                "action_executed": False,
                "protocol_error": "multiple_tool_calls",
                "tool_calls_emitted": n_calls,
            }
            return [Message(
                "user",
                f"Error: exactly one tool call is allowed per turn; received {n_calls}.",
            )], False, 0.0
        obs, reward, done, info = self._env.step({"_raw_text": action_text})
        self.last_step_info = dict(info or {})
        self.last_step_info.setdefault("tool_calls_emitted", n_calls)
        role = "user" if obs.get("_obs_type") == "user" else "tool"
        return [Message(role, obs["text"])], done, float(reward)

    def score_current_state(self) -> float:
        """Return checklist completion for the live state without mutation."""
        if self._env is None:
            raise RuntimeError("call reset() before score_current_state()")
        final_state = self._runtime.get_state_info(self._env.env_instance)
        return float(self._runtime.calculate_reward(
            self._env.checklist_with_func,
            self._env.init_state,
            final_state,
        ))
