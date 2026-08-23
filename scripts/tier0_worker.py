#!/usr/bin/env python3
"""Tier-0 probe: does WITHIN-EPISODE residual learning help a single task?

Arms per task (temp LoRA per task, discarded afterwards; no cross-task state):
  std    one standard episode on base weights
  retry  a second standard episode (sampling-variance control)
  tier0  episode with mid-episode updates: after an ERROR step (settled
         step-level evidence; at most --max-updates), build a partial settled
         block (actions + ok/ERROR so far, no outcome — it does not exist
         yet), price the prefix observations by incremental hindsight, train
         the task's temp LoRA one step, and CONTINUE THE SAME EPISODE on it.

Copy-safety: actions are verbatim in the partial block -> belief channel only.
Runs one worker per GPU next to its vLLM server (same layout as p1_worker).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sediment.config import StreamConfig
from sediment.engine.vllm_client import VllmClient
from sediment.envs.envscaler import EnvScalerAdapter, list_tasks
from sediment.experience import EXPERIENCE_CLOSE, EXPERIENCE_OPEN, _truncate
from sediment.hindsight import score as hindsight_score
from sediment.hindsight import to_train_sample
from sediment.types import AdapterVersion, ExperienceBlock, Message, Trajectory

_spec = importlib.util.spec_from_file_location("p1w", ROOT / "scripts" / "p1_worker.py")
p1w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p1w)  # ResidentTrainer, STATUS_RE, zero_action_weights, ...


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tier-0 within-episode probe worker")
    p.add_argument("--base-url", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--pool", type=int, default=200)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-updates", type=int, default=2)
    p.add_argument("--data-dir", default="/data/erv1n/resid/data")
    p.add_argument("--lopd-dir", default="/data/erv1n/resid/third_party/LOPD")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--tag", default="tier0")
    return p.parse_args()


def build_partial_block(step_status: list[tuple[str, str]]) -> ExperienceBlock:
    """Settled facts of the UNFINISHED attempt: verbatim actions + ok/ERROR."""
    lines = [
        EXPERIENCE_OPEN,
        "Settled facts from your attempt so far (the task is NOT finished). "
        "Learn from the errors, then continue solving the task.",
        "",
    ]
    for n, (action, status) in enumerate(step_status, 1):
        lines.append(f"  {n}. {_truncate(action, 120)} -> {status}")
    lines.append(EXPERIENCE_CLOSE)
    return ExperienceBlock(text="\n".join(lines), source_task_ids=[],
                           includes_own_outcome=False)


class Tier0Trainer(p1w.ResidentTrainer):
    """ResidentTrainer whose LoRA can be continued across in-episode updates."""

    def train_step(self, sample, out_dir: str, reset: bool) -> list[float]:
        torch = self.torch
        cfg = self.cfg
        if reset:
            self._reset_lora()
        ids, ws = p1w._encode(self.tokenizer, sample, cfg.max_seq_len)
        losses: list[float] = []
        if len(ids) >= 2 and sum(ws[1:]) > 0.0:
            self.model.train()
            opt = torch.optim.AdamW(
                (p for p in self.model.parameters() if p.requires_grad), lr=cfg.lr)
            input_ids = torch.tensor([ids], device=self.device)
            w = torch.tensor([ws[1:]], dtype=torch.float32, device=self.device)
            logits = self.model(input_ids=input_ids).logits[:, :-1]
            loss = p1w.weighted_ce(logits, input_ids[:, 1:], w)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            del opt
            self.model.eval()
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir)
        torch.cuda.empty_cache()
        return losses


def std_episode(engine, task, cfg, lopd) -> dict[str, Any]:
    env = EnvScalerAdapter(lopd, max_steps=cfg.max_steps)
    from sediment.rollout.agent_loop import run_episode

    traj = run_episode(engine, env, task, cfg, adapter="base")
    return {"success": bool(traj.success), "reward": traj.reward, "steps": traj.steps}


def tier0_episode(engine, trainer, task, cfg, run_dir, tag, max_updates) -> dict[str, Any]:
    lopd = cfg.extra["lopd_dir"]
    env = EnvScalerAdapter(lopd, max_steps=cfg.max_steps)
    messages = env.reset(task)
    base_name = f"{tag}-{p1w.sanitize(task['task_id'])}"
    adir = str(run_dir / "candidates" / base_name)
    adapter = "base"
    step_status: list[tuple[str, str]] = []
    updates = 0
    upd_errors = 0
    done, reward, steps = False, 0.0, 0
    for _ in range(cfg.max_steps):
        action = engine.generate(messages, adapter=adapter,
                                 temperature=cfg.temperature, max_tokens=cfg.max_tokens)
        messages.append(Message("assistant", action))
        obs_msgs, done, reward = env.step(action)
        steps += 1
        if done:
            break
        messages.extend(obs_msgs)
        status = "ERROR" if p1w.STATUS_RE.search(obs_msgs[0].content[:300]) else "ok"
        step_status.append((action, status))
        if status == "ERROR" and updates < max_updates:
            try:
                block = build_partial_block(step_status)
                traj = Trajectory(task_id=str(task["task_id"]), env_family="envscaler",
                                  messages=list(messages), reward=0.0, success=False,
                                  adapter=adapter, steps=steps, meta={})
                hr = hindsight_score(engine, traj, block, cfg)
                sample = p1w.zero_action_weights(to_train_sample(traj, hr, cfg))
                trainer.train_step(sample, adir, reset=(updates == 0))
                updates += 1
                vname = f"{base_name}-u{updates}"
                engine.load_adapter(AdapterVersion(vname, adir, "v0000"))
                if updates > 1:
                    engine.unload_adapter(f"{base_name}-u{updates - 1}")
                adapter = vname
            except Exception:
                upd_errors += 1
    if updates:
        engine.unload_adapter(f"{base_name}-u{updates}")
    return {"success": bool(reward >= 0.999), "reward": float(reward), "steps": steps,
            "n_updates": updates, "upd_errors": upd_errors}


def main() -> None:
    args = parse_args()
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = StreamConfig(
        model=args.model, engine="vllm", trainer="torch", epochs=1,
        temperature=0.7, max_tokens=2048, max_steps=30,
        lora_r=32, lora_alpha=32, lr=1.5e-4, max_seq_len=12288,
        data_dir=args.data_dir, split="rl", out_dir=str(run_dir), run_id=args.tag,
        extra={"lopd_dir": args.lopd_dir, "base_url": args.base_url},
    )
    if args.shard == 0:
        (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    tasks = list_tasks("rl", args.data_dir, third_party_dir=args.lopd_dir)
    pool = tasks[-args.pool:] if args.pool else tasks
    mine = pool[args.shard :: args.num_shards]
    if args.limit:
        mine = mine[: args.limit]

    out_path = run_dir / f"t0_shard{args.shard}.jsonl"
    done_ids: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    done_ids.add(r["task_id"])

    print(f"[shard {args.shard}/{args.num_shards}] mine={len(mine)} done={len(done_ids)}",
          flush=True)
    engine = VllmClient(args.base_url, cfg.model)
    trainer = Tier0Trainer(cfg)
    print(f"[shard {args.shard}] trainer resident on {trainer.device}", flush=True)

    with out_path.open("a", encoding="utf-8") as f:
        for k, task in enumerate(mine):
            if task["task_id"] in done_ids:
                continue
            t0 = time.time()
            try:
                lopd = cfg.extra["lopd_dir"]
                rec: dict[str, Any] = {"task_id": task["task_id"], "tag": args.tag}
                rec["std"] = std_episode(engine, task, cfg, lopd)
                rec["retry"] = std_episode(engine, task, cfg, lopd)
                rec["tier0"] = tier0_episode(engine, trainer, task, cfg, run_dir,
                                             args.tag, args.max_updates)
            except Exception as e:
                rec = {"task_id": task["task_id"], "tag": args.tag,
                       "error": repr(e)[:300],
                       "trace": traceback.format_exc(limit=3)[-500:]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if "error" in rec:
                print(f"[{k + 1}/{len(mine)} {rec['task_id']}] ERROR {rec['error']}",
                      flush=True)
            else:
                print(f"[{k + 1}/{len(mine)} {rec['task_id']}] "
                      f"std={int(rec['std']['success'])} retry={int(rec['retry']['success'])} "
                      f"tier0={int(rec['tier0']['success'])} "
                      f"upd={rec['tier0']['n_updates']} dt={time.time() - t0:.0f}s",
                      flush=True)
    print(f"[shard {args.shard}] DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
