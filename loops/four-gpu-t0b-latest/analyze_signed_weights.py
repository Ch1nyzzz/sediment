#!/usr/bin/env python3
"""Audit the exact capped/masked weights used by signed_action_worker."""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sediment.chat_template import load_tokenizer  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "signed_action_worker", ROOT / "scripts" / "signed_action_worker.py")
SIGNED = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SIGNED
SPEC.loader.exec_module(SIGNED)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--positive-cap", type=float, default=1.55)
    parser.add_argument("--negative-cap", type=float, default=4.51)
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.model)
    actions = []
    for path in sorted(glob.glob(str(Path(args.artifacts) / "debug_shard*.jsonl"))):
        for line in Path(path).read_text().splitlines():
            if line.strip():
                actions.extend(json.loads(line)["debug"]["diagnostics"])

    result = {}
    for status in ("ok", "ERROR"):
        subset = [action for action in actions if action["status"] == status]
        total_tokens = active_tokens = capped_tokens = 0
        total_mass = 0.0
        action_mass = []
        top = []
        for action in subset:
            raw_ids = [int(token["token_id"]) for token in action["tokens"]]
            mask = SIGNED.action_semantic_mask(tokenizer, action["action"], raw_ids)
            mass = 0.0
            for token, keep in zip(action["tokens"], mask, strict=True):
                delta = float(token["controlled_delta"])
                raw_weight = max(delta, 0.0) if status == "ok" else max(-delta, 0.0)
                cap = args.positive_cap if status == "ok" else args.negative_cap
                weight = min(raw_weight, cap) if keep else 0.0
                total_tokens += 1
                active_tokens += weight > 0
                capped_tokens += weight > 0 and raw_weight >= cap
                total_mass += weight
                mass += weight
                if weight > 0:
                    top.append({
                        "weight": weight,
                        "token": token["token"],
                        "action": action["action"][:120],
                        "observation": action["observation"][:120],
                    })
            action_mass.append(mass / len(action["tokens"]))
        result[status] = {
            "actions": len(subset),
            "tokens": total_tokens,
            "active_token_fraction": active_tokens / total_tokens,
            "mass_per_token": total_mass / total_tokens,
            "actions_with_mass_fraction": statistics.fmean(
                mass > 0 for mass in action_mass),
            "median_action_mass_per_token": statistics.median(action_mass),
            "capped_active_fraction": capped_tokens / max(active_tokens, 1),
            "top": sorted(top, key=lambda row: row["weight"], reverse=True)[:20],
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
