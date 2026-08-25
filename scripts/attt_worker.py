#!/usr/bin/env python3
"""aTTT control arm: unconditional NTP + n-gram downweighting (arXiv 2607.03441).

Same harness/protocol as tier0c (one LoRA update every K settled steps, temp
LoRA per task, episode continues on the updated adapter); only the OBJECTIVE
differs, so the arms isolate aTTT's objective vs our residual pricing.

One arm per task. std/retry baselines are NOT rerun: the tier0 run already
sampled them on the same 200-task pool with the same frozen base -- join on
task_id.
  tier0  every K settled steps, ONE update on the MOST RECENT settled step
         only (paper: "we use only the most recent step rather than the full
         prefix"). Candidate text = Self (reasoning+action) or Env (latest
         observation); Summary (LLM progress note) is not implemented -- extra
         generator call, and not the winner at 4B. Loss is plain NTP over every
         candidate token with w_j = max(0.05, 1/(1+f_j)), f_j = largest count,
         in this episode's prior update history, of any 3-gram containing
         position j (paper Eq. 8-9). No reward, no advantage, no gating.

Upstream is closed-source (no repo as of 2026-08), so this follows the paper
text. Their LoRA is r=8 alpha=16 lr=5e-4 with 2 grad steps per update; we use
OUR config (r=32 alpha=32 lr=1.5e-4, 1 grad step) to hold the update budget
equal to the pricing arm.
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
from sediment.types import AdapterVersion, Message, TrainSample


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p1w = _load("p1w", "p1_worker.py")   # ResidentTrainer, sanitize
t0w = _load("t0w", "tier0_worker.py")  # Tier0Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="aTTT control-arm worker")
    p.add_argument("--base-url", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--pool", type=int, default=200)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--k-update", type=int, default=5)
    p.add_argument("--max-updates", type=int, default=5)
    p.add_argument("--candidate", choices=["self", "env", "both"], default="self",
                   help="update text: self=latest action, env=latest observation, "
                        "both=one sample each (2 grad steps/update)")
    p.add_argument("--paper-recipe", action="store_true",
                   help="use the paper's hyperparams (LoRA r=8 alpha=16 lr=5e-4, "
                        "2 grad steps/update) instead of our budget-aligned config")
    p.add_argument("--data-dir", default="/data/erv1n/resid/data")
    p.add_argument("--lopd-dir", default="/data/erv1n/resid/third_party/LOPD")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--tag", default="attt")
    return p.parse_args()


NGRAM = 3
W_MIN = 0.05


def ngram_weights(ids: list[int], hist: dict[tuple[int, ...], int]) -> list[float]:
    """Paper Eq. 8-9: f(j) = max over the n-grams containing j of their count in
    the prior update history; w(j) = max(W_MIN, 1/(1+f(j))). Novel tokens keep 1."""
    f = [0] * len(ids)
    for s in range(len(ids) - NGRAM + 1):
        c = hist.get(tuple(ids[s : s + NGRAM]), 0)
        for j in range(s, s + NGRAM):
            if c > f[j]:
                f[j] = c
    return [max(W_MIN, 1.0 / (1.0 + x)) for x in f]


def add_ngrams(hist: dict[tuple[int, ...], int], ids: list[int]) -> None:
    """Append this sample to the episode's update history H^tok. Counting per
    sample (not over the concatenation) only drops n-grams straddling two
    updates, which are artifacts of the concatenation anyway."""
    for s in range(len(ids) - NGRAM + 1):
        g = tuple(ids[s : s + NGRAM])
        hist[g] = hist.get(g, 0) + 1


def align_weights(tokenizer, text: str, ids: list[int], w: list[float]) -> list[float]:
    """Map content-token weights onto the trainer-side chat-template span.

    The sample holds one message, so _encode's span for it is exactly
    apply_chat_template([msg]); locating the content tokens inside it gives a
    1:1 map with framing tokens at 0. If they are not a verbatim sublist (BPE
    boundary effects at the join), fall back to the raw list -- _encode then
    broadcasts its mean over the span: still unconditional NTP, but the
    token-level reweighting flattens out.
    """
    full = list(tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=True, return_dict=False))
    n = len(ids)
    for i in range(len(full) - n + 1):
        if full[i] == ids[0] and full[i : i + n] == ids:
            return [0.0] * i + list(w) + [0.0] * (len(full) - i - n)
    return list(w)


def candidate_texts(kind: str, action: str, obs: str) -> list[tuple[str, str]]:
    if kind == "self":
        return [("self", action)]
    if kind == "env":
        return [("env", obs)]
    return [("self", action), ("env", obs)]


def attt_episode(engine, trainer, task, cfg, run_dir, tag, k_update, max_updates,
                 kind, steps_per_update: int = 1) -> dict[str, Any]:
    lopd = cfg.extra["lopd_dir"]
    env = EnvScalerAdapter(lopd, max_steps=cfg.max_steps)
    messages = env.reset(task)
    base_name = f"{tag}-{p1w.sanitize(task['task_id'])}"
    adir = str(run_dir / "candidates" / base_name)
    adapter = "base"
    hist: dict[tuple[int, ...], int] = {}  # per-task, reset between episodes
    diag: list[dict[str, Any]] = []
    since = 0
    updates = 0
    upd_errors = 0
    last = ("", "")  # most recent settled (action, observation)
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
        last = (action, "\n".join(m.content for m in obs_msgs))
        since += 1
        if since >= k_update and updates < max_updates:
            try:
                trained = False
                for cand, text in candidate_texts(kind, *last):
                    ids = trainer.tokenizer.encode(text, add_special_tokens=False)
                    if not ids:
                        continue
                    w = ngram_weights(ids, hist)
                    sample = TrainSample(
                        task_id=str(task["task_id"]),
                        messages=[Message("user", text)],
                        token_weights_by_msg=[align_weights(trainer.tokenizer, text, ids, w)])
                    for si in range(steps_per_update):
                        # fresh LoRA only on the episode's very first grad step
                        losses = trainer.train_step(
                            sample, adir, reset=(updates == 0 and not trained and si == 0))
                        trained = trained or bool(losses)
                    add_ngrams(hist, ids)
                    diag.append({"cand": cand, "n_tok": len(ids),
                                 "mean_w": round(sum(w) / len(w), 4)})
                if trained:
                    updates += 1
                    vname = f"{base_name}-u{updates}"
                    engine.load_adapter(AdapterVersion(vname, adir, "v0000"))
                    if updates > 1:
                        engine.unload_adapter(f"{base_name}-u{updates - 1}")
                    adapter = vname
            except Exception:
                upd_errors += 1
            since = 0
    if updates:
        engine.unload_adapter(f"{base_name}-u{updates}")
    return {"success": bool(reward >= 0.999), "reward": float(reward),
            "steps": steps, "n_updates": updates, "upd_errors": upd_errors,
            "upd_diag": diag}


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
               "candidate": args.candidate},
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

    print(f"[shard {args.shard}/{args.num_shards}] mine={len(mine)} done={len(done_ids)} "
          f"cand={args.candidate}", flush=True)
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
                rec: dict[str, Any] = {"task_id": task["task_id"], "tag": args.tag,
                                       "candidate": args.candidate}
                rec["tier0"] = attt_episode(engine, trainer, task, cfg, run_dir,
                                            args.tag, args.k_update, args.max_updates,
                                            args.candidate, spu)
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
                      f"t0a={int(rec['tier0']['success'])} "
                      f"upd={rec['tier0']['n_updates']} dt={time.time() - t0:.0f}s",
                      flush=True)
    print(f"[shard {args.shard}] DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
