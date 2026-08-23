"""Streaming scheduler: predict-then-update window loop over a task stream.

Protocol per window of ``cfg.window_size`` tasks:

1. Capture ``registry.current()`` BEFORE the window body; run the FIRST
   attempt of every task concurrently with that adapter and build one
   StreamRecord per task immediately (predict-then-update: the learning
   curve is never contaminated by updates from the same window; with
   sequential windows this also satisfies ``cfg.staleness_windows == 1``).
2. Per task: buffer.retrieve -> experience.build_block -> hindsight.score ->
   gate.propose; a failed+proposed task gets ONE retry with the block in
   context (record marked ``retried``; its success/reward stay those of the
   first attempt). Every trajectory is added to the buffer.
3. TrainSamples of proposed tasks -> trainer_fn -> gate.validate -> on pass
   merge.merge publishes the new version and the window's contributing
   records get ``gated_in=True``.
4. Optional ``on_window(window_idx, window_records)`` callback.

Cross-module collaborators (envs, rollout, experience, hindsight, merge) are
imported lazily inside the run so importing this module needs only stdlib.
"""
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

from sediment.config import StreamConfig
from sediment.router import Router
from sediment.types import (
    AdapterVersion,
    StreamRecord,
    TrainSample,
    Trajectory,
    UpdateCandidate,
)

if TYPE_CHECKING:  # pragma: no cover - sibling modules land separately
    from sediment.buffer import Buffer
    from sediment.engine.base import Engine
    from sediment.gate import Gate
    from sediment.registry import Registry

TrainerFn = Callable[..., UpdateCandidate]  # (samples, parent, cfg, *, workdir)
RunProbe = Callable[[str, list[dict]], float]  # (adapter_name, tasks) -> success rate
OnWindow = Callable[[int, list[StreamRecord]], None]


def _make_env(task: dict[str, Any]) -> Any:
    """Build a fresh env instance for one attempt of `task`.

    Toy families ("toy*") map to ToyOrderEnv; the task itself is handed to the
    env by run_episode via ``env.reset(task)``.
    """
    family = str(task.get("env_family", ""))
    if family.startswith("toy"):
        from sediment.envs import ToyOrderEnv

        return ToyOrderEnv()
    if "lopd_dir" in task:  # EnvScaler task dicts carry their checkout path
        from sediment.envs.envscaler import EnvScalerAdapter

        return EnvScalerAdapter(task["lopd_dir"])
    raise ValueError(f"no env constructor for env_family {family!r}")


def run_stream(
    tasks: list[dict[str, Any]],
    engines: list["Engine"],
    buffer: "Buffer",
    gate: "Gate",
    trainer_fn: TrainerFn,
    registry: "Registry",
    cfg: StreamConfig,
    *,
    run_probe: Optional[RunProbe] = None,
    probe_tasks: Optional[list[dict[str, Any]]] = None,
    on_window: Optional[OnWindow] = None,
) -> list[StreamRecord]:
    """Run the full task stream; return one StreamRecord per task (first attempts).

    Deterministic under a fixed seed with MockEngine: attempts are gathered in
    task order and all bookkeeping (buffer adds, gate ledger, training) runs
    sequentially in task order. Records are plain-dict serializable (jsonl).
    """
    return asyncio.run(
        _stream(tasks, engines, buffer, gate, trainer_fn, registry, cfg,
                run_probe=run_probe, probe_tasks=probe_tasks, on_window=on_window)
    )


async def _stream(
    tasks: list[dict[str, Any]],
    engines: list["Engine"],
    buffer: "Buffer",
    gate: "Gate",
    trainer_fn: TrainerFn,
    registry: "Registry",
    cfg: StreamConfig,
    *,
    run_probe: Optional[RunProbe],
    probe_tasks: Optional[list[dict[str, Any]]] = None,
    on_window: Optional[OnWindow],
) -> list[StreamRecord]:
    from sediment import experience as experience_mod
    from sediment import hindsight as hindsight_mod
    from sediment import merge as merge_mod
    from sediment.rollout.agent_loop import run_episode

    probe_tasks = probe_tasks or []
    router = Router(engines)
    records: list[StreamRecord] = []
    windows = [tasks[i:i + cfg.window_size] for i in range(0, len(tasks), cfg.window_size)]

    async def attempt(task: dict[str, Any], adapter: str,
                      experience: Any = None) -> tuple[Trajectory, float]:
        """One episode in a worker thread; returns (trajectory, seconds)."""
        engine = router.for_task(task["task_id"])

        def _run() -> tuple[Trajectory, float]:
            t0 = time.monotonic()
            traj = run_episode(engine, _make_env(task), task, cfg,
                               adapter=adapter, experience=experience)
            return traj, time.monotonic() - t0

        return await asyncio.to_thread(_run)

    for w_idx, window in enumerate(windows):
        # Predict-then-update: serving adapter fixed before any window work.
        current = registry.current()
        serving = current.name

        # (1) first attempts, concurrent; results in task order.
        results = await asyncio.gather(*(attempt(t, serving) for t in window))
        w_records: list[StreamRecord] = []
        trajs: list[Trajectory] = []
        for task, (traj, dt) in zip(window, results):
            trajs.append(traj)
            w_records.append(StreamRecord(
                task_id=task["task_id"], window=w_idx, adapter=serving,
                success=traj.success, reward=traj.reward,
                timings={"attempt": dt},
                meta={"env_family": task.get("env_family", "")},
            ))
        records.extend(w_records)

        # (2) hindsight + gate proposal + optional retry, sequential.
        samples: list[TrainSample] = []
        contrib: list[StreamRecord] = []
        for task, rec, traj in zip(window, w_records, trajs):
            retrieved = buffer.retrieve(task, cfg.retrieval_k)
            block = experience_mod.build_block(retrieved, traj, cfg)
            engine = router.for_task(rec.task_id)
            t0 = time.monotonic()
            try:
                hr = await asyncio.to_thread(hindsight_mod.score, engine, traj, block, cfg)
            except Exception as e:  # oversized rendering etc.: no signal, stream goes on
                rec.timings["hindsight"] = time.monotonic() - t0
                rec.meta["hindsight_error"] = repr(e)[:160]
                buffer.add(traj)
                continue
            rec.timings["hindsight"] = time.monotonic() - t0
            proposed = bool(gate.propose(hr, traj))
            rec.meta["proposed"] = proposed
            buffer.add(traj)
            if proposed and not bool(traj.success) and cfg.retry_on_fail:
                retry_traj, retry_dt = await attempt(task, serving, experience=block)
                retry_traj.is_retry = True
                rec.retried = True
                rec.timings["retry"] = retry_dt
                buffer.add(retry_traj)
            if proposed:
                samples.append(hindsight_mod.to_train_sample(traj, hr, cfg))
                contrib.append(rec)

        # (3) train candidate from proposed tasks; gate-validate; merge/publish.
        if samples:
            t0 = time.monotonic()
            candidate = await asyncio.to_thread(
                trainer_fn, samples, current, cfg,
                workdir=cfg.out_dir + "/candidates")
            # G2/G3 A/B the candidate on the validating engine.
            cand_version = AdapterVersion(
                name=candidate.candidate_id, path=candidate.adapter_path,
                parent=current.name)
            engines[0].load_adapter(cand_version)
            decision = await asyncio.to_thread(
                gate.validate, candidate, current, engines[0],
                buffer.replay_states(cfg.gate_replay_states, cfg.seed),
                probe_tasks=probe_tasks, run_probe=run_probe)
            if decision.passed:
                version = merge_mod.merge(current, candidate, cfg, registry)
                for engine in engines:  # serve the new version from next window
                    engine.load_adapter(version)
                if current.path is not None and current.name != version.name:
                    for engine in engines:  # bound --max-loras in long streams
                        getattr(engine, "unload_adapter", lambda n: None)(current.name)
                for rec in contrib:
                    rec.gated_in = True
            for engine in engines:
                getattr(engine, "unload_adapter", lambda n: None)(candidate.candidate_id)
            rec_reason = decision.reason
            train_dt = time.monotonic() - t0
            for rec in contrib:
                rec.timings["train"] = train_dt
                rec.meta["gate_reason"] = rec_reason

        # (4) snapshot hook.
        if on_window is not None:
            on_window(w_idx, w_records)

    return records
