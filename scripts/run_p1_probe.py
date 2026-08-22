#!/usr/bin/env python3
"""P1 internalization probe: does belief-weighted distillation write the
in-context gain into the weights?

Per task (fresh env, weights reset to base each task): first attempt ->
evidence block from the task's own outcome -> hindsight pricing -> train a
candidate (stub by default) -> re-attempt WITHOUT the block in context.

Mock run: python scripts/run_p1_probe.py --engine mock --tasks 4
Writes {out}/{run_id}/p1.jsonl with one
{task_id, first_success, retry_success, obs_surprise} per line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sediment.config import StreamConfig

_FLAG_TO_FIELD = {
    "engine": "engine", "tasks": "num_tasks", "out": "out_dir",
    "seed": "seed", "trainer": "trainer", "run_id": "run_id",
}


def coerce(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


def parse_args(argv: Optional[list[str]] = None) -> StreamConfig:
    p = argparse.ArgumentParser(description="P1 per-task internalization probe")
    p.add_argument("--engine", choices=["mock", "vllm"], default=None)
    p.add_argument("--tasks", type=int, default=None, help="number of probe tasks")
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


def build_engine(cfg: StreamConfig) -> Any:
    if cfg.engine == "mock":
        from sediment.engine.mock import MockEngine

        return MockEngine()
    if cfg.engine == "vllm":
        import importlib

        mod = importlib.import_module("sediment.engine.vllm_client")
        cls = getattr(mod, "VllmClient", None) or getattr(mod, "VllmEngine")
        return cls(cfg.extra.get("base_url", "http://127.0.0.1:8000/v1"))
    raise SystemExit(f"unknown engine: {cfg.engine}")


def main(argv: Optional[list[str]] = None) -> list[dict[str, Any]]:
    cfg = parse_args(argv)
    run_dir = Path(cfg.out_dir) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    from sediment.experience import build_block
    from sediment.hindsight import score as hindsight_score
    from sediment.hindsight import to_train_sample
    from sediment.trainer import train_candidate
    from sediment.types import AdapterVersion
    try:
        from sediment.envs import ToyOrderEnv, make_toy_tasks
    except ImportError:
        from sediment.envs.toy import ToyOrderEnv, make_toy_tasks
    try:
        from sediment.rollout import run_episode
    except ImportError:
        from sediment.rollout.agent_loop import run_episode

    engine = build_engine(cfg)
    tasks = make_toy_tasks(cfg.num_tasks, cfg.seed)
    base = AdapterVersion(name="v0000", path=None, parent=None)
    out_path = run_dir / "p1.jsonl"
    records: list[dict[str, Any]] = []

    with out_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            first = run_episode(engine, ToyOrderEnv(), task, cfg,
                                adapter="base", experience=None)
            block = build_block([], first, cfg)
            hr = hindsight_score(engine, first, block, cfg)
            sample = to_train_sample(first, hr, cfg)
            cand = train_candidate([sample], base, cfg, str(run_dir / "candidates"))
            version = AdapterVersion(name=cand.candidate_id,
                                     path=cand.adapter_path, parent=base.name)
            engine.load_adapter(version)
            retry = run_episode(engine, ToyOrderEnv(), task, cfg,
                                adapter=version.name, experience=None)
            rec = {"task_id": task["task_id"], "first_success": first.success,
                   "retry_success": retry.success, "obs_surprise": hr.obs_surprise}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records.append(rec)
            print(f"[{rec['task_id']}] first={rec['first_success']} "
                  f"retry={rec['retry_success']} surprise={rec['obs_surprise']:.4f}")

    n = len(records)
    first_rate = sum(1 for r in records if r["first_success"]) / n if n else 0.0
    retry_rate = sum(1 for r in records if r["retry_success"]) / n if n else 0.0
    print(f"\nP1: n={n} first={first_rate:.3f} retry={retry_rate:.3f} "
          f"delta={retry_rate - first_rate:+.3f}")
    print(f"wrote {out_path}")
    return records


if __name__ == "__main__":
    main()
