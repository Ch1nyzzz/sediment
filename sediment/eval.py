"""Streaming-curve metrics (W-AUC, gain) and report writers.

Records are StreamRecord instances or their jsonl dict form; both are
accepted everywhere so metrics can be recomputed from written reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from sediment.config import StreamConfig


def _field(rec: Any, name: str, default: Any = None) -> Any:
    if isinstance(rec, dict):
        return rec.get(name, default)
    return getattr(rec, name, default)


def _success(rec: Any) -> float:
    """success=True -> 1.0; False or None (no signal) -> 0.0."""
    return 1.0 if _field(rec, "success") else 0.0


def window_means(records: Sequence[Any]) -> dict[int, float]:
    """Per-window mean success rate, keyed by window index (sorted)."""
    by_w: dict[int, list[float]] = {}
    for r in records:
        by_w.setdefault(int(_field(r, "window", 0)), []).append(_success(r))
    return {w: sum(v) / len(v) for w, v in sorted(by_w.items())}


def wauc(records: Sequence[Any]) -> float:
    """Windowed AUC of the streaming learning curve.

    Trapezoid of per-window success means over window index, normalized by
    the index span (a weighted mean of window means: interior windows weigh
    1, endpoints 1/2). Single window -> its mean; no records -> 0.0.
    """
    means = window_means(records)
    if not means:
        return 0.0
    xs = list(means)
    ys = list(means.values())
    if len(xs) == 1:
        return ys[0]
    area = sum((xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2.0 for i in range(len(xs) - 1))
    return area / (xs[-1] - xs[0])


def gain(records: Sequence[Any], baseline_records: Sequence[Any]) -> float:
    """Mean success delta of a stream vs a baseline stream (empty-safe)."""
    if not records or not baseline_records:
        return 0.0
    a = sum(_success(r) for r in records) / len(records)
    b = sum(_success(r) for r in baseline_records) / len(baseline_records)
    return a - b


def write_report(out_dir: str | Path, cfg: StreamConfig, records: Sequence[Any],
                 registry: Any) -> dict[str, Any]:
    """Write stream.jsonl + summary.json under out_dir and print a table.

    stream.jsonl holds one StreamRecord dict per line; summary.json holds
    the config, wauc, per-window success means, and the registry's version
    names. Returns the summary dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in records]
    with (out / "stream.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    means = window_means(records)
    history = registry.history() if registry is not None else []
    n = len(rows)
    summary = {
        "run_id": cfg.run_id,
        "config": cfg.to_dict(),
        "num_records": n,
        "success_rate": (sum(_success(r) for r in records) / n) if n else 0.0,
        "wauc": wauc(records),
        "window_means": {str(w): m for w, m in means.items()},
        "adapters": [getattr(v, "name", str(v)) for v in history],
    }
    metas = [_field(record, "meta", {}) or {} for record in records]
    reasons = [meta.get("termination_reason") for meta in metas]
    reasons = [str(reason) for reason in reasons if reason]
    if reasons:
        summary["termination_counts"] = {
            reason: reasons.count(reason) for reason in sorted(set(reasons))
        }
        summary["forced_settle_rate"] = (
            sum(bool(meta.get("forced_settle")) for meta in metas) / n if n else 0.0
        )

    checkpoint_names = sorted({
        str(name)
        for meta in metas
        for name in (meta.get("reward_checkpoints", {}) or {})
    }, key=int)
    if checkpoint_names:
        summary["checkpoint_metrics"] = {}
        for name in checkpoint_names:
            values = [
                float(meta["reward_checkpoints"][name])
                for meta in metas
                if name in (meta.get("reward_checkpoints", {}) or {})
            ]
            summary["checkpoint_metrics"][name] = {
                "num_records": len(values),
                "mean_reward": sum(values) / len(values),
                "success_rate": sum(value >= 0.999 for value in values) / len(values),
            }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_table(records, means, summary)
    return summary


def _print_table(records: Sequence[Any], means: dict[int, float],
                 summary: dict[str, Any]) -> None:
    by_w: dict[int, list[Any]] = {}
    for r in records:
        by_w.setdefault(int(_field(r, "window", 0)), []).append(r)
    print(f"\n=== stream {summary['run_id']}: {summary['num_records']} tasks, "
          f"success={summary['success_rate']:.3f}, wauc={summary['wauc']:.4f} ===")
    print(f"{'window':>6} {'n':>4} {'success':>8}  adapter")
    for w in sorted(by_w):
        adapters = sorted({str(_field(r, "adapter", "?")) for r in by_w[w]})
        print(f"{w:>6} {len(by_w[w]):>4} {means[w]:>8.3f}  {','.join(adapters)}")
    print()
