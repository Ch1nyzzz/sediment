#!/usr/bin/env python3
"""Offline replay of G2/G3 for rejected candidates of a stream run.

Reconstructs the gate inputs from the run's persisted buffer + the candidate
dirs on disk, and prints per-candidate G2 behavior-change fractions and G3
probe rates vs the parent. Usage (on box):
    train_venv/bin/python scripts/gate_forensics.py --run p3long \
        --cands cand-554d35b2a1 cand-9905b9ba8f --url http://127.0.0.1:8105/v1
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sediment.buffer import Buffer
from sediment.config import StreamConfig
from sediment.engine.vllm_client import VllmClient
from sediment.envs.envscaler import EnvScalerAdapter, list_tasks
from sediment.rollout.agent_loop import run_episode
from sediment.types import AdapterVersion

LOPD = "/data/erv1n/resid/third_party/LOPD"
DATA = "/data/erv1n/resid/data"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="p3long")
    p.add_argument("--cands", nargs="+", required=True)
    p.add_argument("--url", default="http://127.0.0.1:8105/v1")
    p.add_argument("--tasks", type=int, default=160)
    p.add_argument("--probes", type=int, default=12)
    a = p.parse_args()

    cfg = StreamConfig(model="Qwen/Qwen3-4B-Instruct-2507", max_steps=30,
                       temperature=0.0, max_tokens=2048)
    engine = VllmClient(a.url, cfg.model)
    buf = Buffer.load(f"results/{a.run}/buffer.jsonl")
    replay = buf.replay_states(8, 0)
    print(f"buffer={len(buf)} trajs, replay states={len(replay)}")

    pool = list_tasks("rl", DATA, third_party_dir=LOPD)[-(a.tasks + a.probes):]
    probes = pool[a.tasks:]
    print(f"probe tasks: {[t['task_id'] for t in probes]}")

    def probe_rate(adapter: str) -> float:
        ok = 0
        for t in probes:
            traj = run_episode(engine, EnvScalerAdapter(LOPD), t, cfg, adapter=adapter)
            ok += int(bool(traj.success))
        return ok / len(probes)

    base_gen = [engine.generate(s, adapter="base", temperature=0.0,
                                max_tokens=cfg.max_tokens) for s in replay]
    parent_rate = probe_rate("base")
    print(f"parent(base): probe {parent_rate:.3f}")

    for cid in a.cands:
        path = f"results/candidates/{cid}"
        engine.load_adapter(AdapterVersion(cid, path, "v0000"))
        changed = sum(
            int(engine.generate(s, adapter=cid, temperature=0.0,
                                max_tokens=cfg.max_tokens) != g)
            for s, g in zip(replay, base_gen))
        rate = probe_rate(cid)
        print(f"{cid}: G2 changed {changed}/{len(replay)} ({changed/len(replay):.2f} "
              f"vs min 0.10) | G3 probe {rate:.3f} vs parent {parent_rate:.3f} "
              f"delta {rate-parent_rate:+.3f} (min -0.09)")
        engine.unload_adapter(cid)


if __name__ == "__main__":
    main()
