#!/usr/bin/env python3
"""Streaming experiment CLI (predict-then-update over a task stream).

Mock run (no GPU):
    python scripts/run_stream.py --engine mock --tasks 8 --window 4
Any StreamConfig field can be overridden with --set key=value (repeatable),
values parsed as json when possible:
    python scripts/run_stream.py --set gate_min_surprise=0.1 --set retry_on_fail=false
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sediment.config import StreamConfig

# argparse attribute -> StreamConfig field
_FLAG_TO_FIELD = {
    "engine": "engine", "tasks": "num_tasks", "window": "window_size",
    "out": "out_dir", "seed": "seed", "trainer": "trainer", "run_id": "run_id",
}


def coerce(text: str) -> Any:
    """Parse a --set value: json first (ints, floats, bools, lists), else str."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


def parse_args(argv: Optional[list[str]] = None) -> StreamConfig:
    p = argparse.ArgumentParser(description="sediment streaming run")
    p.add_argument("--engine", choices=["mock", "vllm"], default=None)
    p.add_argument("--tasks", type=int, default=None, help="number of stream tasks")
    p.add_argument("--window", type=int, default=None, help="tasks per window")
    p.add_argument("--out", default=None, help="output root (run dir = out/run_id)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--trainer", choices=["stub", "torch"], default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="override any StreamConfig field (repeatable)")
    a = p.parse_args(argv)

    d = {field: getattr(a, flag) for flag, field in _FLAG_TO_FIELD.items()
         if getattr(a, flag) is not None}
    known = set(StreamConfig.__dataclass_fields__)
    for item in a.overrides:
        if "=" not in item:
            p.error(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        if k not in known:
            p.error(f"--set: unknown StreamConfig field {k!r}")
        d[k] = coerce(v)
    return StreamConfig.from_dict(d)


def build_engines(cfg: StreamConfig) -> list[Any]:
    if cfg.engine == "mock":
        from sediment.engine.mock import MockEngine

        return [MockEngine()]
    if cfg.engine == "vllm":
        import importlib

        mod = importlib.import_module("sediment.engine.vllm_client")
        cls = getattr(mod, "VllmClient", None) or getattr(mod, "VllmEngine")
        urls = cfg.extra.get("base_urls") or [
            f"http://127.0.0.1:{8000 + i}/v1" for i in range(cfg.num_engines)]
        return [cls(u, cfg.model, lora_prefix=f"{cfg.run_id}-") for u in urls]
    raise SystemExit(f"unknown engine: {cfg.engine}")


def _on_window(idx: int, records: list[Any]) -> None:
    ok = sum(1 for r in records if getattr(r, "success", None))
    n = max(1, len(records))
    adapters = sorted({str(getattr(r, "adapter", "?")) for r in records})
    print(f"[window {idx}] n={len(records)} success={ok / n:.3f} "
          f"adapter={','.join(adapters)}")


def _load_rl_tasks(cfg: StreamConfig) -> tuple[list[dict], list[dict]]:
    """EnvScaler stream + G3 probe holdout from the corpus tail.

    Pool = tail (num_tasks + gate_probe_tasks); stream gets the head of it,
    probes the rest. env_family is refined to the env prefix (env_188...) so
    buffer retrieval and the gate ledger see real families; each task carries
    its LOPD checkout for scheduler._make_env.
    """
    from sediment.envs.envscaler import list_tasks

    lopd = cfg.extra.get("lopd_dir", "/data/erv1n/resid/third_party/LOPD")
    pool = list_tasks("rl", cfg.data_dir, third_party_dir=lopd)[
        -(cfg.num_tasks + cfg.gate_probe_tasks):]
    for t in pool:
        t["lopd_dir"] = lopd
        t["env_family"] = t["task_id"].rsplit("_rl-task_", 1)[0]
    return pool[: cfg.num_tasks], pool[cfg.num_tasks:]


def main(argv: Optional[list[str]] = None) -> dict[str, Any]:
    cfg = parse_args(argv)
    if cfg.split not in ("toy", "rl"):
        raise SystemExit("run_stream supports split='toy' or split='rl' (EnvScaler)")
    run_dir = Path(cfg.out_dir) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # Isolate mutable run state under the run dir unless explicitly overridden.
    if cfg.buffer_path == StreamConfig.buffer_path:
        cfg.buffer_path = str(run_dir / "buffer.jsonl")
    if cfg.registry_dir == StreamConfig.registry_dir:
        cfg.registry_dir = str(run_dir / "registry")

    from sediment.buffer import Buffer
    from sediment.eval import write_report
    from sediment.gate import Gate
    from sediment.registry import Registry
    from sediment.scheduler import run_stream as stream_fn
    from sediment.trainer import train_candidate
    try:
        from sediment.envs import make_toy_tasks
    except ImportError:
        from sediment.envs.toy import make_toy_tasks

    run_probe = None
    probe_tasks: list[dict] = []
    if cfg.split == "toy":
        tasks = make_toy_tasks(cfg.num_tasks, cfg.seed)
    else:
        tasks, probe_tasks = _load_rl_tasks(cfg)
    engines = build_engines(cfg)
    buffer = Buffer(cfg.buffer_path)
    gate = Gate(cfg)
    registry = Registry(cfg.registry_dir)

    if probe_tasks:
        import dataclasses

        from sediment.rollout.agent_loop import run_episode
        from sediment.scheduler import _make_env

        probe_cfg = dataclasses.replace(cfg, temperature=0.0)

        def run_probe(adapter_name: str, ptasks: list[dict]) -> float:
            ok = 0
            for t in ptasks:
                traj = run_episode(engines[0], _make_env(t), t, probe_cfg,
                                   adapter=adapter_name)
                ok += int(bool(traj.success))
            return ok / max(1, len(ptasks))

    records = stream_fn(tasks, engines, buffer, gate, train_candidate, registry,
                        cfg, run_probe=run_probe, probe_tasks=probe_tasks,
                        on_window=_on_window)
    if asyncio.iscoroutine(records):
        records = asyncio.run(records)

    summary = write_report(run_dir, cfg, records, registry)
    print(f"wrote {run_dir / 'stream.jsonl'} and {run_dir / 'summary.json'}")
    return summary


if __name__ == "__main__":
    main()
