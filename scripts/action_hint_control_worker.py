#!/usr/bin/env python3
"""Scoring-only audit of action-hint copying and a matched-action control."""

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


p1w = _load("p1w", "p1_worker.py")
lf = _load("latest_factorial_worker", "latest_factorial_worker.py")

HINT_TMPL = (
    "Note (settled fact): the candidate action shown below produced the "
    "corresponding environment result.\n"
    "Candidate action: {action}\nEnvironment result: {result}"
)
MASKED_RESULT = "[RESULT MASKED]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matched action-hint score audit")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--pool", type=int, default=200)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-scored-steps", type=int, default=25)
    parser.add_argument("--data-dir", default="/data/erv1n/resid/data")
    parser.add_argument("--lopd-dir", default="/data/erv1n/resid/third_party/LOPD")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--tag", required=True)
    return parser.parse_args()


def hint(action: str, result: str) -> Message:
    return Message("user", HINT_TMPL.format(action=action, result=result))


def _content_logps(engine, tokenizer, prefix: list[Message], action: str) -> list[float]:
    target = Message("assistant", action)
    raw_ids = list(tokenizer.encode(action, add_special_tokens=False))
    span_ids = lf._message_span_ids(tokenizer, prefix, target)
    offset = lf._find_sublist(span_ids, raw_ids)
    full_logps = engine._prompt_logprobs(prefix + [target], "base")
    if len(full_logps) < len(span_ids):
        raise ValueError("scored prompt is shorter than final assistant span")
    span_logps = full_logps[-len(span_ids):]
    if len(span_logps) != len(span_ids):
        raise ValueError("engine/tokenizer chat-template mismatch")
    return span_logps[offset : offset + len(raw_ids)]


def score_action(engine, tokenizer, prefix: list[Message], action: str,
                 observation: str) -> dict[str, Any]:
    raw_ids = list(tokenizer.encode(action, add_special_tokens=False))
    if not raw_ids:
        raise ValueError("action has no content tokens")
    base = _content_logps(engine, tokenizer, prefix, action)
    masked = _content_logps(
        engine, tokenizer, prefix + [hint(action, MASKED_RESULT)], action)
    actual = _content_logps(
        engine, tokenizer, prefix + [hint(action, observation)], action)
    tokens = []
    for token_id, logp_base, logp_masked, logp_actual in zip(
        raw_ids, base, masked, actual, strict=True
    ):
        tokens.append({
            "token_id": int(token_id),
            "token": tokenizer.decode([token_id]),
            "base_logp": round(float(logp_base), 6),
            "masked_logp": round(float(logp_masked), 6),
            "actual_logp": round(float(logp_actual), 6),
            "hint_delta": round(float(logp_masked - logp_base), 6),
            "polluted_delta": round(float(logp_actual - logp_base), 6),
            "controlled_delta": round(float(logp_actual - logp_masked), 6),
        })

    def summarize(key: str) -> dict[str, float]:
        values = [float(token[key]) for token in tokens]
        return {
            "mean": sum(values) / len(values),
            "pos_frac": sum(value > 0 for value in values) / len(values),
        }

    return {
        "n_tok": len(tokens),
        "hint": summarize("hint_delta"),
        "polluted": summarize("polluted_delta"),
        "controlled": summarize("controlled_delta"),
        "tokens": tokens,
    }


def audit_episode(engine, tokenizer, task, cfg, max_scored_steps):
    env = EnvScalerAdapter(cfg.extra["lopd_dir"], max_steps=cfg.max_steps)
    messages = env.reset(task)
    diagnostics = []
    reward = 0.0
    steps = 0
    for _ in range(cfg.max_steps):
        prefix = list(messages)
        action = engine.generate(
            messages, adapter="base", temperature=cfg.temperature,
            max_tokens=cfg.max_tokens)
        messages.append(Message("assistant", action))
        obs_msgs, done, reward = env.step(action)
        steps += 1
        if done:
            break
        messages.extend(obs_msgs)
        observation = "\n".join(message.content for message in obs_msgs)
        if len(diagnostics) < max_scored_steps:
            item = score_action(engine, tokenizer, prefix, action, observation)
            item.update({
                "step": steps,
                "status": (
                    "ERROR" if p1w.STATUS_RE.search(observation[:300]) else "ok"
                ),
                "action": action,
                "observation": observation,
            })
            diagnostics.append(item)
    return {
        "success": bool(reward >= 0.999),
        "reward": float(reward),
        "steps": steps,
        "n_scored": len(diagnostics),
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
               "mode": "matched_action_hint_scoring_only", "training": False},
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
                    engine, tokenizer, task, cfg, args.max_scored_steps)
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
