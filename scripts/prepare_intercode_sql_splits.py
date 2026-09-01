#!/usr/bin/env python3
"""Freeze an in-domain 500/534 InterCode-SQL split from Spider dev.

The released InterCode SQL artifact contains only Spider dev.  We therefore keep
all 20 databases represented and stratify by ``database x hardness``.  Selection
is by content hash (no random/rollout seed), with no question overlap.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


TRAIN_SIZE_DEFAULT = 500


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def normalized_question(value: str) -> str:
    return " ".join(value.split()).casefold()


def largest_remainder_quotas(
    capacities: dict[tuple[str, str], int], target: int
) -> dict[tuple[str, str], int]:
    total = sum(capacities.values())
    if target < 0 or target > total:
        raise ValueError(f"target {target} is outside [0, {total}]")
    raw = {key: target * value / total for key, value in capacities.items()}
    quotas = {key: int(raw[key]) for key in capacities}
    remaining = target - sum(quotas.values())
    for key in sorted(capacities, key=lambda item: (-(raw[item] % 1), item)):
        if not remaining:
            break
        if quotas[key] < capacities[key]:
            quotas[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("failed to allocate exact training size")
    return quotas


def manifest_row(
    row: dict[str, Any], *, source_index: int, split: str
) -> dict[str, Any]:
    identity = content_hash(
        {
            "db": row["db"],
            "question": normalized_question(row["query"]),
            "gold": row["gold"],
        }
    )[:20]
    return {
        "task_id": f"intercode_sql_{split}_{identity}",
        "benchmark": "intercode_sql",
        "split": split,
        "source_split": "spider_dev",
        "source_index": source_index,
        **row,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "hardness": dict(sorted(Counter(row["hardness"] for row in rows).items())),
        "database": dict(sorted(Counter(row["db"] for row in rows).items())),
    }


def prepare(source: Path, output_dir: Path, train_size: int) -> dict[str, Any]:
    rows = json.loads(source.read_text())
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("InterCode source must be a JSON list of objects")
    groups: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(str(row["db"]), str(row["hardness"]))].append((index, row))
    quotas = largest_remainder_quotas(
        {key: len(items) for key, items in groups.items()}, train_size
    )
    train_indices: set[int] = set()
    for key, items in groups.items():
        ranked = sorted(
            items,
            key=lambda item: (
                content_hash(
                    {
                        "db": item[1]["db"],
                        "question": normalized_question(item[1]["query"]),
                        "gold": item[1]["gold"],
                    }
                ),
                item[0],
            ),
        )
        train_indices.update(index for index, _ in ranked[: quotas[key]])

    train = [
        manifest_row(row, source_index=index, split="train")
        for index, row in enumerate(rows)
        if index in train_indices
    ]
    test = [
        manifest_row(row, source_index=index, split="test")
        for index, row in enumerate(rows)
        if index not in train_indices
    ]
    train_questions = {(row["db"], normalized_question(row["query"])) for row in train}
    test_questions = {(row["db"], normalized_question(row["query"])) for row in test}
    if len(train) != train_size or train_questions & test_questions:
        raise AssertionError("split size or question-disjointness invariant failed")

    train_path = output_dir / f"train{len(train)}.jsonl"
    test_path = output_dir / f"test{len(test)}.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(test_path, test)
    summary = {
        "benchmark": "InterCode-SQL",
        "source_path": str(source.resolve()),
        "source_rows": len(rows),
        "selection": (
            "content-hash ranking within database*hardness strata; "
            "no random or rollout seed"
        ),
        "database_boundary": "in-domain; all source databases may occur in both splits",
        "train": {
            "path": str(train_path.resolve()),
            "rows": len(train),
            "sha256": file_sha256(train_path),
            "distribution": distribution(train),
        },
        "test": {
            "path": str(test_path.resolve()),
            "rows": len(test),
            "sha256": file_sha256(test_path),
            "distribution": distribution(test),
        },
        "db_question_overlap": len(train_questions & test_questions),
        "stratum_quotas": {"|".join(key): quotas[key] for key in sorted(quotas)},
    }
    write_json(output_dir / "split_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=TRAIN_SIZE_DEFAULT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            prepare(args.source, args.output_dir, args.train_size),
            ensure_ascii=False,
            indent=2,
        )
    )
