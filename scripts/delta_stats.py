#!/usr/bin/env python3
"""Action/observation δ quantiles from hindsight on a run's buffer (base model)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from sediment import experience, hindsight
from sediment.buffer import Buffer
from sediment.config import StreamConfig
from sediment.engine.vllm_client import VllmClient

p = argparse.ArgumentParser()
p.add_argument("--run", default="p3m800_frozen"); p.add_argument("--url", default="http://127.0.0.1:8106/v1")
p.add_argument("--start", type=int, default=400); p.add_argument("--n", type=int, default=24)
a = p.parse_args()
cfg = StreamConfig(model="Qwen/Qwen3-4B-Instruct-2507", retrieval_k=4)
eng = VllmClient(a.url, cfg.model)
buf = Buffer.load(ROOT / "results" / a.run / "buffer.jsonl")
rows = [t for t in buf._trajs if not t.is_retry]
act, obs, per_traj = [], [], []
for t in rows[a.start:a.start + a.n]:
    past = Buffer(ROOT / "results" / "unused.jsonl"); past._trajs = [x for x in rows if x is not t][:a.start]
    task = {"task_id": t.task_id, "env_family": t.env_family, "payload": next(m.content for m in t.messages if m.role == "user")}
    block = experience.build_block(past.retrieve(task, 4), t, cfg)
    t.adapter = "base"
    hr = hindsight.score(eng, t, block, cfg)
    da = [d for s in hr.spans if s.role == "assistant" for d in s.deltas]
    do = [d for s in hr.spans if s.role == "tool" for d in s.deltas]
    act += da; obs += do
    per_traj.append((t.task_id[-14:], len(da), sum(d > 0 for d in da), sum(d > 0.5 for d in da), round(sum(max(d, 0) for d in da), 1), round(max(da) if da else 0, 2)))
def q(x, ps):
    x = sorted(x); return [round(x[int(p * (len(x) - 1))], 3) for p in ps]
ps = (.5, .75, .9, .95, .99)
print(f"action tokens n={len(act)}  q50/75/90/95/99 = {q(act, ps)}  frac>0={sum(d>0 for d in act)/len(act):.3f} frac>0.5={sum(d>0.5 for d in act)/len(act):.3f} frac<-0.5={sum(d<-0.5 for d in act)/len(act):.3f}")
print(f"obs    tokens n={len(obs)}  q50/75/90/95/99 = {q(obs, ps)}  frac>0={sum(d>0 for d in obs)/len(obs):.3f}")
print("per traj: task, n_act_tok, n>0, n>0.5, relu mass, max")
for r in per_traj: print("  ", r)
