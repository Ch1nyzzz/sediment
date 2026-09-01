#!/usr/bin/env python3
"""Streaming experiment CLI (predict-then-update over a task stream).

Mock run (no GPU):
    python scripts/run_stream.py --engine mock --tasks 8 --window 4
Any StreamConfig field can be overridden with --set key=value (repeatable),
values parsed as json when possible:
    python scripts/run_stream.py --set gate_min_surprise=0.1 --set retry_on_fail=false
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sediment.config import StreamConfig  # noqa: E402

# argparse attribute -> StreamConfig field
_FLAG_TO_FIELD = {
    "engine": "engine", "tasks": "num_tasks", "window": "window_size",
    "out": "out_dir", "seed": "seed", "trainer": "trainer", "run_id": "run_id",
}


def coerce(text: str) -> Any:
    """Parse a --set value: json first (ints, floats, bools, lists), else str."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


def parse_args(argv: Optional[list[str]] = None) -> StreamConfig:
    p = argparse.ArgumentParser(description="sediment streaming run")
    p.add_argument("--engine", choices=["mock", "vllm"], default=None)
    p.add_argument("--tasks", type=int, default=None, help="number of stream tasks")
    p.add_argument("--window", type=int, default=None, help="tasks per window")
    p.add_argument("--out", default=None, help="output root (run dir = out/run_id)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--trainer", choices=["stub", "torch"], default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="override any StreamConfig field (repeatable)")
    a = p.parse_args(argv)

    d = {field: getattr(a, flag) for flag, field in _FLAG_TO_FIELD.items()
         if getattr(a, flag) is not None}
    known = set(StreamConfig.__dataclass_fields__)
    for item in a.overrides:
        if "=" not in item:
            p.error(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        if k not in known:
            p.error(f"--set: unknown StreamConfig field {k!r}")
        d[k] = coerce(v)
    return StreamConfig.from_dict(d)


def build_engines(cfg: StreamConfig) -> list[Any]:
    if cfg.engine == "mock":
        from sediment.engine.mock import MockEngine

        return [MockEngine()]
    if cfg.engine == "vllm":
        import importlib

        mod = importlib.import_module("sediment.engine.vllm_client")
        cls = getattr(mod, "VllmClient", None) or getattr(mod, "VllmEngine")
        urls = cfg.extra.get("base_urls") or [
            f"http://127.0.0.1:{8000 + i}/v1" for i in range(cfg.num_engines)]
        return [cls(u, cfg.model, max_context=cfg.max_model_len,
                    lora_prefix=f"{cfg.run_id}-") for u in urls]
    raise SystemExit(f"unknown engine: {cfg.engine}")


def _on_window(idx: int, records: list[Any]) -> None:
    ok = sum(1 for r in records if getattr(r, "success", None))
    n = max(1, len(records))
    adapters = sorted({str(getattr(r, "adapter", "?")) for r in records})
    print(f"[window {idx}] n={len(records)} success={ok / n:.3f} "
          f"adapter={','.join(adapters)}")


def _load_rl_tasks(cfg: StreamConfig) -> tuple[list[dict], list[dict]]:
    """EnvScaler stream + G3 probe holdout from the corpus tail.

    Pool = tail (num_tasks + gate_probe_tasks); stream gets the head of it,
    probes the rest (extra.corpus_slice=[a,b) restricts the corpus first, e.g.
    [0,1700] = env<175 meta-train head). env_family is refined to the env prefix (env_188...) so
    buffer retrieval and the gate ledger see real families; each task carries
    its LOPD checkout for scheduler._make_env.
    """
    from sediment.envs.envscaler import list_tasks

    lopd = cfg.extra.get("lopd_dir", "/data/erv1n/resid/third_party/LOPD")
    corpus = list_tasks("rl", cfg.data_dir, third_party_dir=lopd)
    for t in corpus:
        t["lopd_dir"] = lopd
        t["env_family"] = t["task_id"].rsplit("_rl-task_", 1)[0]
    if "corpus_slice" in cfg.extra:  # [start, end): meta-train head instead of the tail pool
        start, end = cfg.extra["corpus_slice"]
        corpus = corpus[start:end]
    keep = cfg.extra.get("keep_families")
    exclude = cfg.extra.get("exclude_families")
    if keep and exclude:
        raise ValueError("keep_families and exclude_families are mutually exclusive")
    if keep:
        # Six of the seventeen tail families produced 1 success in 256 tasks
        # under BOTH frozen and icl_refl (08-26): a third of the wall clock
        # carrying none of the signal. Stratifying to the discriminative
        # families is a change of pool, so absolute counts are not comparable
        # with the full-tail runs -- and peers now come only from these
        # families, which makes retrieval slightly richer too.
        corpus = [t for t in corpus if t["env_family"] in set(keep)]
    elif exclude:
        # Two-way protocol: online is an explicit family list and held-out is
        # its exact complement in rl_scenarios. Expressing the complement here
        # prevents a newly added family from silently falling outside both.
        excluded = set(exclude)
        corpus = [t for t in corpus if t["env_family"] not in excluded]
    kf = cfg.extra.get("keep_task_ids_file")
    if kf:
        # task-level stream composition (repairable-task split, 08-31): the id
        # list IS the split definition; ordering is handled below as usual
        import json as _json
        with open(kf) as fh:
            keep_ids = set(_json.load(fh))
        corpus = [t for t in corpus if t["task_id"] in keep_ids]
        print(f"[tasks] keep_task_ids_file: {len(corpus)} tasks kept", flush=True)
    if cfg.extra.get("interleave_families"):
        # round-robin across families (stable within a family): a family-sorted
        # stream front-loads the hardest families, so repairs (the only
        # training signal of the fork arms) arrive only when few tasks are
        # left to benefit -- and a single order is not a credible curve anyway.
        by_fam: dict[str, list] = {}
        for t in corpus:
            by_fam.setdefault(t["env_family"], []).append(t)
        fams = sorted(by_fam)
        corpus = [by_fam[f][i] for i in range(max(len(v) for v in by_fam.values()))
                  for f in fams if i < len(by_fam[f])]
    span = cfg.num_tasks + cfg.gate_probe_tasks
    pool = corpus[-span:]
    before = corpus[-span - cfg.probe_extra_before:-span] if cfg.probe_extra_before else []
    return pool[: cfg.num_tasks], pool[cfg.num_tasks:] + before


def _load_benchmark_tasks(cfg: StreamConfig) -> tuple[list[dict], list[dict]]:
    benchmark = str(cfg.extra.get("benchmark") or "")
    manifest = cfg.extra.get("task_manifest")
    if benchmark != "intercode_sql" or not manifest:
        raise ValueError(
            "benchmark split currently requires extra.benchmark='intercode_sql' "
            "and extra.task_manifest"
        )
    from sediment.envs.intercode_sql import read_manifest

    corpus = read_manifest(str(manifest))
    mysql_config = dict(cfg.extra.get("intercode_mysql") or {})
    for task in corpus:
        task["intercode_mysql"] = mysql_config
    if cfg.num_tasks > len(corpus):
        raise ValueError(
            f"requested {cfg.num_tasks} benchmark tasks from a {len(corpus)}-row manifest"
        )
    return corpus[: cfg.num_tasks], []


def main(argv: Optional[list[str]] = None) -> dict[str, Any]:
    cfg = parse_args(argv)
    if cfg.split not in ("toy", "rl", "benchmark"):
        raise SystemExit("run_stream supports split='toy', 'rl', or 'benchmark'")
    run_dir = Path(cfg.out_dir) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # Isolate mutable run state under the run dir unless explicitly overridden.
    if cfg.buffer_path == StreamConfig.buffer_path:
        cfg.buffer_path = str(run_dir / "buffer.jsonl")
    if cfg.registry_dir == StreamConfig.registry_dir:
        cfg.registry_dir = str(run_dir / "registry")

    from sediment.buffer import Buffer
    from sediment.eval import write_report
    from sediment.gate import Gate
    from sediment.registry import Registry
    from sediment.scheduler import run_stream as stream_fn
    from sediment.trainer import train_candidate
    try:
        from sediment.envs import make_toy_tasks
    except ImportError:
        from sediment.envs.toy import make_toy_tasks

    run_probe = None
    probe_tasks: list[dict] = []
    if cfg.split == "toy":
        tasks = make_toy_tasks(cfg.num_tasks, cfg.seed)
    elif cfg.split == "rl":
        tasks, probe_tasks = _load_rl_tasks(cfg)
    else:
        tasks, probe_tasks = _load_benchmark_tasks(cfg)
    engines = build_engines(cfg)
    from sediment.types import StreamRecord
    prior: list = []
    window_offset = 0
    if cfg.resume and Path(cfg.buffer_path).exists():
        buffer = Buffer.load(cfg.buffer_path)
        firsts = [t for t in buffer._trajs if not t.is_retry]
        done = (len(firsts) // cfg.window_size) * cfg.window_size  # whole windows only
        # trajectories of a partial last window are dropped from the buffer view so
        # the window is redone (records rebuilt from the persisted trajectories)
        keep_ids = {id(t) for t in firsts[:done]}
        buffer._trajs = [t for t in buffer._trajs if not (not t.is_retry and id(t) not in keep_ids)]
        prior = [StreamRecord(task_id=t.task_id, window=i // cfg.window_size, adapter=t.adapter,
                              success=t.success, reward=t.reward, meta={"env_family": t.env_family,
                              "resumed": True}) for i, t in enumerate(firsts[:done])]
        window_offset = done // cfg.window_size
        tasks = tasks[done:]
        print(f"[resume] {done} tasks / {window_offset} windows restored from buffer; "
              f"current adapter {Registry(cfg.registry_dir).current().name}; {len(tasks)} tasks left",
              flush=True)
    else:
        buffer = Buffer(cfg.buffer_path)
    retrieval_buffer = None
    if cfg.retrieval_buffer_paths:
        retrieval_buffer = Buffer(run_dir / ".immutable-retrieval-view.jsonl")
        for donor_path in cfg.retrieval_buffer_paths:
            donor = Buffer.load(donor_path)
            retrieval_buffer._trajs.extend(donor._trajs)
        print(f"[memory] loaded {len(retrieval_buffer)} immutable trajectories from "
              f"{len(cfg.retrieval_buffer_paths)} donor buffers", flush=True)
    gate = Gate(cfg)
    registry = Registry(cfg.registry_dir)
    if cfg.initial_adapter_path:
        if cfg.resume:
            raise SystemExit("initial_adapter_path and resume are mutually exclusive")
        if registry.current().name != "v0000":
            raise SystemExit("initial_adapter_path requires an empty run registry")
        adapter_path = Path(cfg.initial_adapter_path).resolve()
        if not adapter_path.is_dir() or not (adapter_path / "adapter_config.json").is_file():
            raise SystemExit(f"invalid initial_adapter_path: {adapter_path}")
        initial = registry.publish(
            str(adapter_path), parent="v0000", provenance=["initial_adapter"]
        )
        print(f"[initial adapter] {adapter_path} -> {initial.name}", flush=True)

    if probe_tasks:
        import dataclasses

        from sediment.rollout.agent_loop import run_episode
        from sediment.scheduler import _make_env

        probe_cfg = dataclasses.replace(cfg, temperature=0.0)

        def run_probe(adapter_name: str, ptasks: list[dict]) -> float:
            # G3 probes were the dominant per-window fixed cost (6-14 min
            # sequential); run them concurrently across all engines.
            from concurrent.futures import ThreadPoolExecutor

            def one(i_t):
                i, t = i_t
                eng = engines[i % len(engines)]
                return int(bool(run_episode(eng, _make_env(t), t, probe_cfg,
                                            adapter=adapter_name).success))
            with ThreadPoolExecutor(max_workers=max(1, len(ptasks))) as ex:
                ok = sum(ex.map(one, enumerate(ptasks)))
            return ok / max(1, len(ptasks))

    records = stream_fn(tasks, engines, buffer, gate, train_candidate, registry,
                        cfg, run_probe=run_probe, probe_tasks=probe_tasks,
                        on_window=_on_window, window_offset=window_offset,
                        retrieval_buffer=retrieval_buffer)
    if asyncio.iscoroutine(records):
        records = asyncio.run(records)
    records = prior + list(records)

    summary = write_report(run_dir, cfg, records, registry)
    print(f"wrote {run_dir / 'stream.jsonl'} and {run_dir / 'summary.json'}")
    return summary


if __name__ == "__main__":
    main()
