#!/usr/bin/env python3
"""P1 on real GPUs: per-task internalization arms on EnvScaler (one worker per GPU).

Arms per task, all sharing ONE first attempt (paired design):
  retry    resample, no experience, base weights   (sampling-variance control)
  icl      re-attempt with E_x in context          (ICL upper bound)
  ours     hindsight w_t-weighted CE -> fresh LoRA -> re-attempt WITHOUT E_x
  uniform  same token support, all weights 1.0     (unweighted-distillation baseline)

Runs inside train_venv (torch+peft) against a per-GPU vLLM server (serving venv).
Weights reset to a fresh LoRA every task (P1 protocol: per-task, no accumulation).
Resume-safe: task_ids already in this shard's jsonl are skipped.

  python scripts/p1_worker.py --base-url http://127.0.0.1:8104/v1 \
      --out results/p1_s0 --shard 0 --num-shards 4 --limit 2
"""
from __future__ import annotations

import argparse
import json
import math
import re
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
from sediment.experience import build_block
from sediment.hindsight import score as hindsight_score
from sediment.hindsight import to_train_sample
from sediment.rollout.agent_loop import run_episode
from sediment.trainer import LORA_TARGET_MODULES, _encode
from sediment.types import AdapterVersion, TrainSample, Trajectory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P1 real-GPU worker (one per GPU)")
    p.add_argument("--base-url", required=True, help="this GPU's vLLM /v1 endpoint")
    p.add_argument("--out", required=True, help="run directory (shared across shards)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--pool", type=int, default=200, help="task pool = corpus tail (holdout)")
    p.add_argument("--limit", type=int, default=0, help="max tasks this worker (0 = all)")
    p.add_argument("--data-dir", default="/data/erv1n/resid/data")
    p.add_argument("--lopd-dir", default="/data/erv1n/resid/third_party/LOPD")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--tag", default="s0", help="run tag (e.g. seed label)")
    return p.parse_args()


class ResidentTrainer:
    """Tokenizer + bf16 base loaded once; a fresh LoRA re-initialized per task."""

    def __init__(self, cfg: StreamConfig):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        base = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch.bfloat16)
        lora = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base, lora)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def _reset_lora(self) -> None:
        torch = self.torch
        with torch.no_grad():
            for mod in self.model.modules():
                if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
                    for a in mod.lora_A.values():
                        torch.nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
                    for b in mod.lora_B.values():
                        torch.nn.init.zeros_(b.weight)

    def train(self, sample: TrainSample, out_dir: str) -> list[float]:
        torch = self.torch
        cfg = self.cfg
        self._reset_lora()
        ids, ws = _encode(self.tokenizer, sample, cfg.max_seq_len)
        losses: list[float] = []
        if len(ids) >= 2 and sum(ws[1:]) > 0.0:
            self.model.train()
            opt = torch.optim.AdamW(
                (p for p in self.model.parameters() if p.requires_grad), lr=cfg.lr
            )
            input_ids = torch.tensor([ids], device=self.device)
            w = torch.tensor([ws[1:]], dtype=torch.float32, device=self.device)
            for _ in range(cfg.epochs):
                logits = self.model(input_ids=input_ids).logits[:, :-1].float()
                ce = torch.nn.functional.cross_entropy(
                    logits.transpose(1, 2), input_ids[:, 1:], reduction="none"
                )
                loss = (ce * w).sum() / w.sum().clamp_min(1e-8)
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


def uniform_sample(sample: TrainSample) -> TrainSample:
    """Same token support as the weighted sample, every supervised weight = 1."""
    return TrainSample(
        task_id=sample.task_id,
        messages=list(sample.messages),
        token_weights_by_msg=[[1.0] * len(ws) for ws in sample.token_weights_by_msg],
    )


def episode_result(traj: Trajectory) -> dict[str, Any]:
    return {
        "success": bool(traj.success),
        "reward": traj.reward,
        "steps": traj.steps,
        "n_msgs": len(traj.messages),
    }


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "-", name)[:60]


def run_task(
    task: dict[str, Any],
    engine: VllmClient,
    trainer: ResidentTrainer,
    cfg: StreamConfig,
    run_dir: Path,
    tag: str,
) -> dict[str, Any]:
    lopd = cfg.extra["lopd_dir"]
    timings: dict[str, float] = {}

    def roll(adapter: str = "base", experience=None) -> Trajectory:
        env = EnvScalerAdapter(lopd, max_steps=cfg.max_steps)
        return run_episode(engine, env, task, cfg, adapter=adapter, experience=experience)

    def timed(key: str, fn):
        t = time.time()
        out = fn()
        timings[key] = round(time.time() - t, 1)
        return out

    rec: dict[str, Any] = {"task_id": task["task_id"], "tag": tag}
    first = timed("first", lambda: roll())
    rec["first"] = episode_result(first)

    rec["retry"] = episode_result(timed("retry", lambda: roll()))

    block = build_block([], first, cfg)
    rec["block_chars"] = len(block.text)
    rec["icl"] = episode_result(timed("icl", lambda: roll(experience=block)))

    hr = timed("hindsight", lambda: hindsight_score(engine, first, block, cfg))
    rec["obs_surprise"] = round(hr.obs_surprise, 5)
    rec["act_gain"] = round(hr.act_gain, 5)
    sample = to_train_sample(first, hr, cfg)
    rec["n_supervised"] = sum(1 for ws in sample.token_weights_by_msg for w in ws if w > 0)

    for arm, s in (("ours", sample), ("uniform", uniform_sample(sample))):
        name = f"{tag}-{sanitize(task['task_id'])}-{arm}"
        adir = run_dir / "candidates" / name
        losses = timed(f"{arm}_train", lambda s=s, adir=adir: trainer.train(s, str(adir)))
        engine.load_adapter(AdapterVersion(name=name, path=str(adir), parent="v0000"))
        res = timed(arm, lambda name=name: roll(adapter=name))
        engine.unload_adapter(name)
        rec[arm] = episode_result(res)
        rec[arm]["losses"] = [round(x, 4) for x in losses]

    rec["timings"] = timings
    return rec


def main() -> None:
    args = parse_args()
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = StreamConfig(
        model=args.model,
        engine="vllm",
        trainer="torch",
        epochs=args.epochs,
        temperature=0.7,
        max_tokens=2048,
        max_steps=30,
        lora_r=32,
        lora_alpha=16,
        lr=5e-5,
        max_seq_len=12288,
        data_dir=args.data_dir,
        split="rl",
        out_dir=str(run_dir),
        run_id=args.tag,
        extra={"lopd_dir": args.lopd_dir, "base_url": args.base_url},
    )
    if args.shard == 0:
        (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    tasks = list_tasks("rl", args.data_dir, third_party_dir=args.lopd_dir)
    pool = tasks[-args.pool :] if args.pool else tasks
    mine = pool[args.shard :: args.num_shards]
    if args.limit:
        mine = mine[: args.limit]

    out_path = run_dir / f"p1_shard{args.shard}.jsonl"
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["task_id"])

    print(f"[shard {args.shard}/{args.num_shards}] pool={len(pool)} mine={len(mine)} "
          f"done={len(done)} url={args.base_url}", flush=True)
    engine = VllmClient(args.base_url, cfg.model)
    trainer = ResidentTrainer(cfg)
    print(f"[shard {args.shard}] trainer resident on {trainer.device}", flush=True)

    with out_path.open("a", encoding="utf-8") as f:
        for k, task in enumerate(mine):
            if task["task_id"] in done:
                continue
            t0 = time.time()
            try:
                rec = run_task(task, engine, trainer, cfg, run_dir, args.tag)
            except Exception as e:  # record and move on; relaunch skips it
                rec = {
                    "task_id": task["task_id"],
                    "tag": args.tag,
                    "error": repr(e)[:300],
                    "trace": traceback.format_exc(limit=3)[-600:],
                }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if "error" in rec:
                print(f"[{k + 1}/{len(mine)} {rec['task_id']}] ERROR {rec['error']}", flush=True)
            else:
                arms = " ".join(
                    f"{a}={int(rec[a]['success'])}" for a in ("first", "retry", "icl", "ours", "uniform")
                )
                print(
                    f"[{k + 1}/{len(mine)} {rec['task_id']}] {arms} "
                    f"surprise={rec['obs_surprise']:.4f} sup={rec['n_supervised']} "
                    f"dt={time.time() - t0:.0f}s",
                    flush=True,
                )
    print(f"[shard {args.shard}] DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
