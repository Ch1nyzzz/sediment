#!/usr/bin/env python3
"""Poison-source attribution for one merged candidate of a stream run.

Re-scores the candidate's training samples under the parent adapter, splits
the hindsight weights by channel, trains three candidates from the parent
(both channels / observation-only / action-only) and probes each on the run's
G3 holdout — the channel whose single-channel candidate wrecks the probe is
the poison source. Usage (on box, trainer GPU via CUDA_VISIBLE_DEVICES):
    CUDA_VISIBLE_DEVICES=6 train_venv/bin/python scripts/channel_forensics.py \
        --run p3m_ng --parent v0004 --task-ids env_190_rl-task_7 env_189_rl-task_43 \
        --url http://127.0.0.1:8106/v1
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sediment import experience, hindsight
from sediment.buffer import Buffer
from sediment.config import StreamConfig
from sediment.engine.vllm_client import VllmClient
from sediment.envs.envscaler import EnvScalerAdapter, list_tasks
from sediment.rollout.agent_loop import run_episode
from sediment.trainer import train_candidate
from sediment.types import AdapterVersion, TrainSample

LOPD = "/data/erv1n/resid/third_party/LOPD"
DATA = "/data/erv1n/resid/data"


def mask_channel(sample: TrainSample, keep_role: str) -> TrainSample:
    ws = [list(w) if m.role == keep_role else [0.0] * len(w)
          for m, w in zip(sample.messages, sample.token_weights_by_msg)]
    return TrainSample(sample.task_id, list(sample.messages), ws)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--parent", required=True, help="registry version the candidate was trained from")
    p.add_argument("--task-ids", nargs="+", required=True)
    p.add_argument("--url", default="http://127.0.0.1:8106/v1")
    p.add_argument("--tasks", type=int, default=160)
    p.add_argument("--probes", type=int, default=12)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--cfg", action="append", default=[], help="extra StreamConfig key=value (json)")
    a = p.parse_args()

    run_dir = ROOT / "results" / a.run
    out_dir = run_dir / "forensics" / a.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = StreamConfig(model="Qwen/Qwen3-4B-Instruct-2507", max_steps=30, temperature=0.0,
                       max_tokens=2048, retrieval_k=4, lr=a.lr, lora_alpha=a.lora_alpha,
                       epochs=a.epochs, trainer="torch", out_dir=str(out_dir))
    for kv in a.cfg:
        k, v = kv.split("=", 1)
        setattr(cfg, k, json.loads(v))
    engine = VllmClient(a.url, cfg.model, lora_prefix=f"forensic-{a.run}-")
    parent = AdapterVersion(a.parent, str(run_dir / "registry" / a.parent), None)
    engine.load_adapter(parent)

    pool = list_tasks("rl", DATA, third_party_dir=LOPD)[-(a.tasks + a.probes):]
    for t in pool:
        t["lopd_dir"] = LOPD
        t["env_family"] = t["task_id"].rsplit("_rl-task_", 1)[0]
    by_id = {t["task_id"]: t for t in pool}
    probes = pool[a.tasks:]

    full = Buffer.load(run_dir / "buffer.jsonl")
    rows = full._trajs
    samples: list[TrainSample] = []
    report: dict = {"parent": a.parent, "samples": []}
    for tid in a.task_ids:
        idx = next(i for i, t in enumerate(rows) if t.task_id == tid and not t.is_retry)
        traj = rows[idx]
        traj.adapter = a.parent  # score under the parent, as the stream did
        past = Buffer(out_dir / "unused.jsonl"); past._trajs = rows[:idx]  # no future peers
        block = experience.build_block(past.retrieve(by_id[tid], cfg.retrieval_k), traj, cfg)
        hr = hindsight.score(engine, traj, block, cfg)
        s = hindsight.to_train_sample(traj, hr, cfg)
        samples.append(s)
        per_msg = []
        for m, w in zip(s.messages, s.token_weights_by_msg):
            if w and sum(w) > 0:
                per_msg.append({"role": m.role, "mass": round(sum(w), 3), "n": len(w),
                                "max": round(max(w), 3), "text": m.content[:140]})
        mass_act = sum(x["mass"] for x in per_msg if x["role"] == "assistant")
        mass_obs = sum(x["mass"] for x in per_msg if x["role"] == "tool")
        report["samples"].append({"task_id": tid, "obs_surprise": hr.obs_surprise,
                                  "act_gain": hr.act_gain, "mass_act": mass_act,
                                  "mass_obs": mass_obs, "messages": per_msg})
        print(f"[{tid}] obs_surprise={hr.obs_surprise:.3f} act_gain={hr.act_gain:.3f} "
              f"mass act={mass_act:.2f} obs={mass_obs:.2f}", flush=True)
        for x in sorted(per_msg, key=lambda x: -x["mass"])[:6]:
            print(f"    {x['role']:9s} mass={x['mass']:7.2f} max={x['max']:.2f} | {x['text']!r}", flush=True)

    def probe_rate(adapter: str) -> float:
        ok = 0
        for t in probes:
            traj = run_episode(engine, EnvScalerAdapter(LOPD), t, cfg, adapter=adapter)
            ok += int(bool(traj.success))
        return ok / len(probes)

    report["probe_parent"] = probe_rate(a.parent)
    print(f"probe parent {a.parent}: {report['probe_parent']:.3f}", flush=True)
    variants = {"both": samples,
                "obs_only": [mask_channel(s, "tool") for s in samples],
                "act_only": [mask_channel(s, "assistant") for s in samples]}
    report["variants"] = {}
    for name, ss in variants.items():
        cand = train_candidate(ss, parent, cfg, workdir=str(out_dir / name))
        ver = AdapterVersion(f"{a.parent}-{name}", cand.adapter_path, a.parent)
        engine.load_adapter(ver)
        rate = probe_rate(ver.name)
        engine.unload_adapter(ver.name)
        losses = cand.train_stats.get("losses", [])
        report["variants"][name] = {"probe": rate, "losses": losses}
        print(f"variant {name:9s}: probe {rate:.3f} (parent {report['probe_parent']:.3f}) "
              f"losses {[round(x, 3) for x in losses]}", flush=True)
    engine.unload_adapter(a.parent)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
