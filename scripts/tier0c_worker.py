#!/usr/bin/env python3
"""Tier-0c probe: decision-point hint pricing + fixed-cadence (K-step) updates.

One arm per task (temp LoRA per task, discarded afterwards; no cross-task
state). std/retry baselines are NOT rerun: the tier0 run already sampled them
on the same 200-task pool with the same frozen base — join on task_id.
  tier0  episode with one LoRA update every K settled steps. Each step k of
         the window is priced at its own decision point: per-token
         delta_k = logp(a_k | P_k + hint(a_k -> f_k)) - logp(a_k | P_k),
         where the hint is a user message quoting the settled action and the
         feedback it produced (raw diff, no paired control — user-directed).
         The window trains ONE sample: the episode prefix through the last
         window action, with relu(delta_k) weights on each a_k's tokens and
         zero elsewhere; then the episode CONTINUES on the updated LoRA.

Channel note: this arm trains the ACTION channel only — f_k appears verbatim
in the hint, so pricing observation tokens here would be copy detection
(mirror image of the ERROR-triggered tier0, which trains belief only).
Diagnostics: per-step {status, mean delta, positive fraction} land in the
record, so "is delta>0 systematic on ERROR steps?" is answered by this run.
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

from sediment.config import StreamConfig
from sediment.engine.vllm_client import VllmClient
from sediment.envs.envscaler import EnvScalerAdapter, list_tasks
from sediment.experience import _truncate
from sediment.types import AdapterVersion, Message, TrainSample


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p1w = _load("p1w", "p1_worker.py")   # ResidentTrainer, STATUS_RE, weighted_ce, sanitize
t0w = _load("t0w", "tier0_worker.py")  # Tier0Trainer, std_episode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tier-0c decision-point hint worker")
    p.add_argument("--base-url", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--pool", type=int, default=200)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--k-update", type=int, default=5)
    p.add_argument("--max-updates", type=int, default=5)
    p.add_argument("--paper-recipe", action="store_true",
                   help="run OUR pricing objective at aTTT's write strength "
                        "(LoRA r=8 alpha=16 lr=5e-4, 2 grad steps/update) instead "
                        "of our config (r=32 alpha=32 lr=1.5e-4, 1 step)")
    p.add_argument("--data-dir", default="/data/erv1n/resid/data")
    p.add_argument("--lopd-dir", default="/data/erv1n/resid/third_party/LOPD")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--tag", default="tier0c")
    return p.parse_args()


HINT_TMPL = (
    "Note (settled fact): if you take the following action at this point, "
    "the environment will return the result shown.\n"
    "Action: {action}\nResult: {result}"
)


def hint_msg(action: str, feedback: str) -> Message:
    return Message("user", HINT_TMPL.format(
        action=_truncate(action, 2000), result=_truncate(feedback, 600)))


def span_logps(engine: VllmClient, prefix: list[Message], final: Message,
               adapter: str) -> list[float]:
    """Per-token logprobs of `final` rendered after `prefix` (2 prefill calls;
    prefix shares the episode's APC cache)."""
    base = engine._prompt_logprobs(prefix, adapter)
    full = engine._prompt_logprobs(prefix + [final], adapter)
    return full[len(base):]


def price_step(engine: VllmClient, p_msgs: list[Message], action: str,
               feedback: str, adapter: str) -> list[float]:
    amsg = Message("assistant", action)
    lo = span_logps(engine, p_msgs, amsg, adapter)
    lw = span_logps(engine, p_msgs + [hint_msg(action, feedback)], amsg, adapter)
    return [a - b for a, b in zip(lw, lo)]


def tier0c_episode(engine, trainer, task, cfg, run_dir, tag, k_update,
                   max_updates, steps_per_update: int = 1) -> dict[str, Any]:
    lopd = cfg.extra["lopd_dir"]
    env = EnvScalerAdapter(lopd, max_steps=cfg.max_steps)
    messages = env.reset(task)
    base_name = f"{tag}-{p1w.sanitize(task['task_id'])}"
    adir = str(run_dir / "candidates" / base_name)
    adapter = "base"
    window: list[tuple[int, str, str, str]] = []  # (prefix_len, action, feedback, status)
    step_diag: list[dict[str, Any]] = []
    updates = 0
    upd_errors = 0
    done, reward, steps = False, 0.0, 0
    for _ in range(cfg.max_steps):
        plen = len(messages)
        action = engine.generate(messages, adapter=adapter,
                                 temperature=cfg.temperature, max_tokens=cfg.max_tokens)
        messages.append(Message("assistant", action))
        obs_msgs, done, reward = env.step(action)
        steps += 1
        if done:
            break
        messages.extend(obs_msgs)
        feedback = "\n".join(m.content for m in obs_msgs)
        status = "ERROR" if p1w.STATUS_RE.search(obs_msgs[0].content[:300]) else "ok"
        window.append((plen, action, feedback, status))
        if len(window) >= k_update and updates < max_updates:
            try:
                # price each window step at its own decision point
                weights_by_idx: dict[int, list[float]] = {}
                for pl, act, fb, st in window:
                    deltas = price_step(engine, messages[:pl], act, fb, adapter)
                    if deltas:
                        step_diag.append({
                            "st": st,
                            "d": round(sum(deltas) / len(deltas), 4),
                            "pos": round(sum(1 for d in deltas if d > 0) / len(deltas), 3),
                        })
                    weights_by_idx[pl] = [max(d, 0.0) for d in deltas]
                # one sample: episode prefix through the last window action,
                # relu(delta) on each a_k's tokens, zero elsewhere
                last_pl = window[-1][0]
                sample_msgs = list(messages[: last_pl + 1])
                ws = [weights_by_idx.get(i, []) for i in range(len(sample_msgs))]
                sample = TrainSample(task_id=str(task["task_id"]),
                                     messages=sample_msgs, token_weights_by_msg=ws)
                losses = []
                for si in range(steps_per_update):
                    losses += trainer.train_step(
                        sample, adir, reset=(updates == 0 and si == 0))
                if losses:
                    updates += 1
                    vname = f"{base_name}-u{updates}"
                    engine.load_adapter(AdapterVersion(vname, adir, "v0000"))
                    if updates > 1:
                        engine.unload_adapter(f"{base_name}-u{updates - 1}")
                    adapter = vname
            except Exception:
                upd_errors += 1
            window = []
    if updates:
        engine.unload_adapter(f"{base_name}-u{updates}")
    return {"success": bool(reward >= 0.999), "reward": float(reward),
            "steps": steps, "n_updates": updates, "upd_errors": upd_errors,
            "step_deltas": step_diag}


def main() -> None:
    args = parse_args()
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    r, a, lr, spu = (8, 16, 5e-4, 2) if args.paper_recipe else (32, 32, 1.5e-4, 1)
    cfg = StreamConfig(
        model=args.model, engine="vllm", trainer="torch", epochs=1,
        temperature=0.7, max_tokens=2048, max_steps=30,
        lora_r=r, lora_alpha=a, lr=lr, max_seq_len=12288,
        data_dir=args.data_dir, split="rl", out_dir=str(run_dir), run_id=args.tag,
        extra={"lopd_dir": args.lopd_dir, "base_url": args.base_url,
               "recipe": "paper" if args.paper_recipe else "ours"},
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
    trainer = t0w.Tier0Trainer(cfg)
    print(f"[shard {args.shard}] trainer resident on {trainer.device}", flush=True)

    with out_path.open("a", encoding="utf-8") as f:
        for k, task in enumerate(mine):
            if task["task_id"] in done_ids:
                continue
            t0 = time.time()
            try:
                # std/retry baselines reuse the tier0 run's records (same 200-task
                # pool, frozen base weights): join on task_id at analysis time
                rec: dict[str, Any] = {"task_id": task["task_id"], "tag": args.tag}
                rec["tier0"] = tier0c_episode(engine, trainer, task, cfg, run_dir,
                                              args.tag, args.k_update, args.max_updates,
                                              spu)
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
                      f"t0c={int(rec['tier0']['success'])} "
                      f"upd={rec['tier0']['n_updates']} dt={time.time() - t0:.0f}s",
                      flush=True)
    print(f"[shard {args.shard}] DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
