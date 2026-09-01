#!/usr/bin/env python3
"""Aggregate an InterCode rollout file over a frozen source-index manifest."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score(records: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> dict[str, Any]:
    by_source = {int(row["task_id"]): row for row in records}
    source_indices = [int(row["source_index"]) for row in manifest]
    missing = [index for index in source_indices if index not in by_source]
    if missing:
        raise ValueError(f"missing {len(missing)} source indices, first={missing[:10]}")
    selected = [by_source[index] for index in source_indices]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        groups[str(row["hardness"])].append(row)
    groups["all"] = selected

    def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        successes = sum(bool(row["success"]) for row in rows)
        return {
            "tasks": len(rows),
            "successes": successes,
            "success_rate": successes / len(rows) if rows else None,
            "sr_at_10": sum(
                row.get("success_turn") is not None
                and int(row["success_turn"]) <= 10
                for row in rows
            )
            / len(rows)
            if rows
            else None,
            "terminal_reasons": dict(
                sorted(Counter(str(row["terminal_reason"]) for row in rows).items())
            ),
            "transcript_tokens_max": max(
                (int(row["transcript_tokens"]) for row in rows), default=None
            ),
            "api_errors": sum(row.get("error") is not None for row in rows),
        }

    return {
        "manifest": str(manifest[0].get("split", "unknown")) if manifest else "empty",
        "by_hardness": {
            hardness: metrics(rows)
            for hardness, rows in sorted(groups.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(read_jsonl(args.records), read_jsonl(args.manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
