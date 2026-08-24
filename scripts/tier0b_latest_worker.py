#!/usr/bin/env python3
"""Matched belief-residual arm for the aTTT Env comparison.

Every K settled, non-terminal steps, select exactly the same candidate as the
local aTTT Env reproduction: the latest environment observation, serialized as
one user message.  No hindsight block, reward, ERROR gate, or synthetic
evidence is used.

The only method-specific signal is trajectory-context gain.  For each content
token in the latest observation,

    delta = logp(token | natural trajectory prefix) - logp(token | observation)

and the training weight is relu(delta).  The natural prefix ends with the
assistant action that produced the observation.  Earlier trajectory messages
remain causal context but have zero loss.  Candidate cadence, paper dose, and
adapter lifetime match ``attt_worker.py --paper-recipe --candidate env``.
"""
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

from sediment.config import StreamConfig  # noqa: E402
from sediment.engine.vllm_client import VllmClient  # noqa: E402
from sediment.envs.envscaler import EnvScalerAdapter, list_tasks  # noqa: E402
from sediment.types import AdapterVersion, Message, TrainSample  # noqa: E402


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p1w = _load("p1w", "p1_worker.py")       # sanitize
t0w = _load("t0w", "tier0_worker.py")    # Tier0Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="K=5 latest-observation belief residual")
    p.add_argument("--base-url", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--pool", type=int, default=200)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--k-update", type=int, default=5)
    p.add_argument("--max-updates", type=int, default=5,
                   help="maximum scheduled candidate selections per episode")
    p.add_argument("--data-dir", default="/data/erv1n/resid/data")
    p.add_argument("--lopd-dir", default="/data/erv1n/resid/third_party/LOPD")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--tag", default="t0b_latest_k5_paper")
    return p.parse_args()


def _render_ids(tokenizer, messages: list[Message]) -> list[int]:
    return list(tokenizer.apply_chat_template(
        [m.to_dict() for m in messages], tokenize=True, return_dict=False))


def _find_sublist(xs: list[int], needle: list[int]) -> int:
    if not needle:
        raise ValueError("latest observation has no content tokens")
    for i in range(len(xs) - len(needle) + 1):
        if xs[i : i + len(needle)] == needle:
            return i
    raise ValueError(
        "observation content tokens are not a verbatim subspan of the rendered "
        "user message; refusing an unaligned residual update"
    )


def _message_span_ids(tokenizer, prefix: list[Message], final: Message) -> list[int]:
    before = _render_ids(tokenizer, prefix) if prefix else []
    full = _render_ids(tokenizer, prefix + [final])
    if full[: len(before)] != before:
        raise ValueError("chat template lacks the prefix property")
    return full[len(before) :]


def _span_logps(engine, prefix: list[Message], final: Message, adapter: str) -> list[float]:
    before = engine._prompt_logprobs(prefix, adapter) if prefix else []
    full = engine._prompt_logprobs(prefix + [final], adapter)
    if len(full) < len(before):
        raise ValueError("scored prompt shrank after appending latest observation")
    return full[len(before) :]


def latest_residual_sample(
    engine,
    tokenizer,
    task_id: str,
    natural_prefix: list[Message],
    observation: str,
    adapter: str,
) -> tuple[TrainSample, dict[str, Any]]:
    """Price one latest observation with natural-context gain.

    Both scoring branches serialize the candidate as the same user message.
    The context branch prepends the natural trajectory through the action that
    produced it; the standalone branch contains no task or trajectory.  The
    returned training sample keeps that natural prefix but places nonzero loss
    only on the candidate's raw content tokens.
    """
    target = Message("user", observation)
    raw_ids = list(tokenizer.encode(observation, add_special_tokens=False))

    ctx_span_ids = _message_span_ids(tokenizer, natural_prefix, target)
    solo_span_ids = _message_span_ids(tokenizer, [], target)
    ctx_offset = _find_sublist(ctx_span_ids, raw_ids)
    solo_offset = _find_sublist(solo_span_ids, raw_ids)

    ctx_logps = _span_logps(engine, natural_prefix, target, adapter)
    solo_logps = _span_logps(engine, [], target, adapter)
    if len(ctx_logps) != len(ctx_span_ids) or len(solo_logps) != len(solo_span_ids):
        raise ValueError(
            "engine/tokenizer chat-template mismatch while pricing latest observation"
        )

    n = len(raw_ids)
    deltas = [
        float(a - b)
        for a, b in zip(
            ctx_logps[ctx_offset : ctx_offset + n],
            solo_logps[solo_offset : solo_offset + n],
        )
    ]
    positive = [max(d, 0.0) for d in deltas]
    span_weights = [0.0] * len(ctx_span_ids)
    span_weights[ctx_offset : ctx_offset + n] = positive
    sample = TrainSample(
        task_id=task_id,
        messages=list(natural_prefix) + [target],
        token_weights_by_msg=[[] for _ in natural_prefix] + [span_weights],
    )
    diag = {
        "n_tok": n,
        "mean_delta": round(sum(deltas) / n, 6),
        "pos_frac": round(sum(d > 0.0 for d in deltas) / n, 6),
        "mean_pos_w": round(sum(positive) / n, 6),
        "max_pos_w": round(max(positive, default=0.0), 6),
    }
    return sample, diag


def residual_episode(engine, trainer, task, cfg, run_dir, tag, k_update, max_updates,
                     steps_per_update: int = 2) -> dict[str, Any]:
    env = EnvScalerAdapter(cfg.extra["lopd_dir"], max_steps=cfg.max_steps)
    messages = env.reset(task)
    base_name = f"{tag}-{p1w.sanitize(task['task_id'])}"
    adir = str(run_dir / "candidates" / base_name)
    adapter = "base"
    since = 0
    selected = 0
    updates = 0
    upd_errors = 0
    diag: list[dict[str, Any]] = []
    done, reward, steps = False, 0.0, 0

    for _ in range(cfg.max_steps):
        action = engine.generate(messages, adapter=adapter,
                                 temperature=cfg.temperature, max_tokens=cfg.max_tokens)
        messages.append(Message("assistant", action))
        natural_prefix = list(messages)
        obs_msgs, done, reward = env.step(action)
        steps += 1
        if done:
            break
        messages.extend(obs_msgs)
        observation = "\n".join(m.content for m in obs_msgs)
        since += 1

        if since >= k_update and selected < max_updates:
            selected += 1
            item: dict[str, Any] = {"step": steps, "source_role": obs_msgs[0].role}
            try:
                sample, priced = latest_residual_sample(
                    engine, trainer.tokenizer, str(task["task_id"]), natural_prefix,
                    observation, adapter)
                item.update(priced)
                losses: list[float] = []
                for si in range(steps_per_update):
                    losses += trainer.train_step(
                        sample, adir, reset=(updates == 0 and si == 0))
                item["trained"] = bool(losses)
                item["losses"] = [round(x, 6) for x in losses]
                if losses:
                    updates += 1
                    vname = f"{base_name}-u{updates}"
                    engine.load_adapter(AdapterVersion(vname, adir, "v0000"))
                    if updates > 1:
                        engine.unload_adapter(f"{base_name}-u{updates - 1}")
                    adapter = vname
            except Exception as exc:
                upd_errors += 1
                item.update({"trained": False, "error": repr(exc)[:240]})
            diag.append(item)
            since = 0

    if updates:
        engine.unload_adapter(f"{base_name}-u{updates}")
    return {
        "success": bool(reward >= 0.999),
        "reward": float(reward),
        "steps": steps,
        "n_selected": selected,
        "n_updates": updates,
        "upd_errors": upd_errors,
        "update_diag": diag,
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = StreamConfig(
        model=args.model, engine="vllm", trainer="torch", epochs=1,
        temperature=0.7, max_tokens=2048, max_steps=30,
        lora_r=8, lora_alpha=16, lr=5e-4, max_seq_len=12288,
        data_dir=args.data_dir, split="rl", out_dir=str(run_dir), run_id=args.tag,
        extra={
            "lopd_dir": args.lopd_dir,
            "base_url": args.base_url,
            "candidate": "latest_env",
            "cadence": args.k_update,
            "max_selections": args.max_updates,
            "pricing": "relu(logp_full_natural_prefix-logp_standalone)",
            "steps_per_update": 2,
        },
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
                row = json.loads(line)
                if "error" not in row:
                    done_ids.add(row["task_id"])

    print(f"[shard {args.shard}/{args.num_shards}] mine={len(mine)} "
          f"done={len(done_ids)} cadence={args.k_update} max={args.max_updates}", flush=True)
    engine = VllmClient(args.base_url, cfg.model)
    trainer = t0w.Tier0Trainer(cfg)
    print(f"[shard {args.shard}] trainer resident on {trainer.device}", flush=True)

    with out_path.open("a", encoding="utf-8") as handle:
        for k, task in enumerate(mine):
            if task["task_id"] in done_ids:
                continue
            started = time.time()
            try:
                rec: dict[str, Any] = {"task_id": task["task_id"], "tag": args.tag}
                rec["tier0"] = residual_episode(
                    engine, trainer, task, cfg, run_dir, args.tag,
                    args.k_update, args.max_updates, 2)
            except Exception as exc:
                rec = {
                    "task_id": task["task_id"], "tag": args.tag,
                    "error": repr(exc)[:300],
                    "trace": traceback.format_exc(limit=3)[-500:],
                }
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            if "error" in rec:
                print(f"[{k + 1}/{len(mine)} {rec['task_id']}] ERROR {rec['error']}",
                      flush=True)
            else:
                out = rec["tier0"]
                print(f"[{k + 1}/{len(mine)} {rec['task_id']}] "
                      f"tier0={int(out['success'])} sel={out['n_selected']} "
                      f"upd={out['n_updates']} uerr={out['upd_errors']} "
                      f"dt={time.time() - started:.0f}s", flush=True)
    print(f"[shard {args.shard}] DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
