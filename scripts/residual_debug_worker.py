#!/usr/bin/env python3
"""Scoring-only audit of latest-observation residuals; performs no training."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sediment.chat_template import load_tokenizer  # noqa: E402
from sediment.config import StreamConfig  # noqa: E402
from sediment.engine.vllm_client import VllmClient  # noqa: E402
from sediment.envs.envscaler import EnvScalerAdapter, list_tasks  # noqa: E402
from sediment.types import Message  # noqa: E402


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lf = _load("latest_factorial_worker", "latest_factorial_worker.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Residual token audit without training")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--pool", type=int, default=200)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--k-update", type=int, default=5)
    parser.add_argument("--max-selections", type=int, default=5)
    parser.add_argument("--data-dir", default="/data/erv1n/resid/data")
    parser.add_argument("--lopd-dir", default="/data/erv1n/resid/third_party/LOPD")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--tag", required=True)
    return parser.parse_args()


def score_observation(
    engine,
    tokenizer,
    natural_prefix: list[Message],
    observation: str,
    adapter: str = "base",
) -> dict[str, Any]:
    target = Message("user", observation)
    raw_ids = list(tokenizer.encode(observation, add_special_tokens=False))
    if not raw_ids:
        raise ValueError("latest observation has no content tokens")
    ctx_span = lf._message_span_ids(tokenizer, natural_prefix, target)
    solo_span = lf._message_span_ids(tokenizer, [], target)
    ctx_offset = lf._find_sublist(ctx_span, raw_ids)
    solo_offset = lf._find_sublist(solo_span, raw_ids)
    ctx_logps = lf._span_logps(engine, natural_prefix, target, adapter)
    solo_logps = lf._span_logps(engine, [], target, adapter)
    if len(ctx_logps) != len(ctx_span) or len(solo_logps) != len(solo_span):
        raise ValueError("engine/tokenizer chat-template mismatch")
    full = ctx_logps[ctx_offset : ctx_offset + len(raw_ids)]
    solo = solo_logps[solo_offset : solo_offset + len(raw_ids)]
    rows = [
        {
            "token_id": int(token_id),
            "token": tokenizer.decode([token_id]),
            "full_logp": round(float(full_logp), 6),
            "solo_logp": round(float(solo_logp), 6),
            "delta": round(float(full_logp - solo_logp), 6),
        }
        for token_id, full_logp, solo_logp in zip(raw_ids, full, solo, strict=True)
    ]
    return {
        "n_tok": len(rows),
        "pos_frac": sum(row["delta"] > 0 for row in rows) / len(rows),
        "mean_delta": sum(row["delta"] for row in rows) / len(rows),
        "observation": observation,
        "tokens": rows,
    }


def audit_episode(engine, tokenizer, task, cfg, k_update, max_selections):
    env = EnvScalerAdapter(cfg.extra["lopd_dir"], max_steps=cfg.max_steps)
    messages = env.reset(task)
    since = selected = steps = 0
    reward = 0.0
    diagnostics = []
    for _ in range(cfg.max_steps):
        action = engine.generate(
            messages, adapter="base", temperature=cfg.temperature,
            max_tokens=cfg.max_tokens)
        messages.append(Message("assistant", action))
        natural_prefix = list(messages)
        obs_msgs, done, reward = env.step(action)
        steps += 1
        if done:
            break
        messages.extend(obs_msgs)
        since += 1
        if since >= k_update and selected < max_selections:
            observation = "\n".join(message.content for message in obs_msgs)
            item = score_observation(engine, tokenizer, natural_prefix, observation)
            item.update({"step": steps, "source_role": obs_msgs[0].role})
            diagnostics.append(item)
            selected += 1
            since = 0
    return {
        "success": bool(reward >= 0.999),
        "reward": float(reward),
        "steps": steps,
        "n_selected": selected,
        "diagnostics": diagnostics,
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = StreamConfig(
        model=args.model, engine="vllm", trainer="stub", temperature=0.7,
        max_tokens=2048, max_steps=30, data_dir=args.data_dir, split="rl",
        out_dir=str(run_dir), run_id=args.tag,
        extra={"lopd_dir": args.lopd_dir, "base_url": args.base_url,
               "mode": "scoring_only", "training": False},
    )
    if args.shard == 0:
        (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    tasks = list_tasks("rl", args.data_dir, third_party_dir=args.lopd_dir)
    pool = tasks[-args.pool:] if args.pool else tasks
    mine = pool[args.shard :: args.num_shards]
    if args.limit:
        mine = mine[: args.limit]
    out_path = run_dir / f"debug_shard{args.shard}.jsonl"
    done_ids = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if "error" not in row:
                    done_ids.add(row["task_id"])
    engine = VllmClient(args.base_url, cfg.model)
    tokenizer = load_tokenizer(cfg.model)
    print(f"[shard {args.shard}] mine={len(mine)} done={len(done_ids)} no-training", flush=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for index, task in enumerate(mine):
            if task["task_id"] in done_ids:
                continue
            started = time.time()
            try:
                result = audit_episode(
                    engine, tokenizer, task, cfg, args.k_update,
                    args.max_selections)
                record = {"task_id": task["task_id"], "debug": result}
            except Exception as exc:
                record = {
                    "task_id": task["task_id"], "error": repr(exc)[:300],
                    "trace": traceback.format_exc(limit=3)[-500:],
                }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{index + 1}/{len(mine)} {task['task_id']}] "
                f"error={int('error' in record)} dt={time.time() - started:.0f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
