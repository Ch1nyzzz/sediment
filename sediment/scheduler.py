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


def _natural_task_text(task: dict[str, Any]) -> str:
    payload = task.get("payload", "")
    if isinstance(payload, dict):
        for key in ("task", "instruction", "query", "prompt"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return str(payload)


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


def strip_experience(messages: list) -> list:
    """Student view: drop the served <previous_attempts> block, call-time hints
    and working-state segments (harness.bare_view); no-op otherwise."""
    from sediment.harness import bare_view

    return bare_view(messages)


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
    window_offset: int = 0,
    retrieval_buffer: Optional["Buffer"] = None,
) -> list[StreamRecord]:
    """Run the full task stream; return one StreamRecord per task (first attempts).

    Deterministic under a fixed seed with MockEngine: attempts are gathered in
    task order and all bookkeeping (buffer adds, gate ledger, training) runs
    sequentially in task order. Records are plain-dict serializable (jsonl).
    """
    return asyncio.run(
        _stream(tasks, engines, buffer, gate, trainer_fn, registry, cfg,
                run_probe=run_probe, probe_tasks=probe_tasks, on_window=on_window,
                window_offset=window_offset, retrieval_buffer=retrieval_buffer)
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
    window_offset: int = 0,
    retrieval_buffer: Optional["Buffer"] = None,
) -> list[StreamRecord]:
    from sediment import experience as experience_mod
    from sediment import hindsight as hindsight_mod
    from sediment import merge as merge_mod
    from sediment.rollout.agent_loop import run_episode
    if cfg.reflect:
        from sediment.reflect import reflect as reflect_fn
    probe_tasks = probe_tasks or []
    memory = retrieval_buffer or buffer
    router = Router(engines)
    records: list[StreamRecord] = []
    windows = [tasks[i:i + cfg.window_size] for i in range(0, len(tasks), cfg.window_size)]

    harness_on = bool(cfg.serve_experience or cfg.calltime_hints or cfg.working_state)

    def _retrieve(task: dict[str, Any], k: int) -> list[Trajectory]:
        return memory.retrieve(task, k, scope=cfg.retrieval_scope,
                               score_mode=cfg.retrieval_score, offset=cfg.retrieval_offset,
                               diversity=cfg.retrieval_diversity)

    async def attempt(task: dict[str, Any], adapter: str,
                      experience: Any = None) -> tuple[Trajectory, float]:
        """One episode in a worker thread; returns (trajectory, seconds).
        Harness state (call-time hinter, working state) is fresh per attempt."""
        engine = router.for_task(task["task_id"])
        hinter = ws = None
        if cfg.calltime_hints:
            from sediment.calltime import CallTimeHinter
            hinter = CallTimeHinter(
                _retrieve(task, cfg.calltime_pool_k), triggers=cfg.calltime_triggers,
                max_hints=cfg.calltime_max_hints, max_chars=cfg.calltime_max_chars,
                result_chars=cfg.max_result_chars, intercept=cfg.calltime_intercept,
                style=cfg.calltime_style)
        if cfg.working_state:
            from sediment.working_state import WorkingState
            ws = WorkingState(_natural_task_text(task), max_goals=cfg.working_state_max_goals)

        harness = {k: v for k, v in (("hinter", hinter), ("working_state", ws)) if v is not None}

        def _run() -> tuple[Trajectory, float]:
            t0 = time.monotonic()
            traj = run_episode(engine, _make_env(task), task, cfg, adapter=adapter,
                               experience=experience, **harness)
            return traj, time.monotonic() - t0

        return await asyncio.to_thread(_run)

    recheck: Optional[AdapterVersion] = None  # pre-merge version to re-probe
    cur0 = registry.current()
    if cur0.path is not None:  # resume: serve the persisted current version
        for engine in engines:
            engine.load_adapter(cur0)
    if window_offset > 0 and cfg.rollback_thr > 0 and cur0.parent:
        # resumed mid-stream: the last merge's re-check flag was not persisted,
        # so always re-probe the pre-merge version at the first validation
        try:
            recheck = registry.get(cur0.parent)
            print(f"[resume] will re-probe {cur0.parent} against {cur0.name} at the first "
                  "validation", flush=True)
        except Exception as e:  # pragma: no cover - missing dir
            print(f"[resume] cannot load {cur0.parent} for re-probe: {e!r}", flush=True)
    for w_idx, window in enumerate(windows, start=window_offset):
        # Predict-then-update: serving adapter fixed before any window work.
        current = registry.current()
        serving = current.name

        # (1) first attempts, concurrent; results in task order. ICL arm:
        # retrieval happens before this window's trajectories enter the buffer.
        served = [experience_mod.build_block(_retrieve(t, cfg.retrieval_k), None, cfg,
                                             query_text=_natural_task_text(t))
                  if cfg.serve_experience else None for t in window]
        results = await asyncio.gather(*(attempt(t, serving, experience=e)
                                         for t, e in zip(window, served)))
        w_records: list[StreamRecord] = []
        trajs: list[Trajectory] = []
        for task, (traj, dt), sb in zip(window, results, served):
            if harness_on:
                # memory/harness-served rollout (teacher distribution); everything
                # downstream -- hindsight with/without, training target, buffer --
                # sees the bare, no-harness rendering (student view). The served
                # view is kept verbatim in meta for forensics; block provenance
                # stays in traj.meta["experience_source_task_ids"].
                if cfg.calltime_hints or cfg.working_state:
                    traj.meta["served_messages"] = [m.to_dict() for m in traj.messages]
                traj.messages = strip_experience(traj.messages)
            trajs.append(traj)
            ct = traj.meta.get("calltime", {})
            events = ct.get("events", [])
            w_records.append(StreamRecord(
                task_id=task["task_id"], window=w_idx, adapter=serving,
                success=traj.success, reward=traj.reward,
                timings={"attempt": dt},
                meta={"env_family": task.get("env_family", ""),
                      "served_experience": sorted(traj.meta.get("experience_source_task_ids", []))
                      if cfg.serve_experience else [],
                      "served_peer_count": len(sb.source_task_ids) if sb is not None else 0,
                      "served_block_chars": len(sb.text) if sb is not None else 0,
                      "steps": traj.steps,
                      **({"hints": len(events),
                          "hint_triggers": [e["trigger"] for e in events],
                          "hint_changed": sum(1 for e in events if e.get("changed")),
                          "hint_chars": sum(e["chars"] for e in events)}
                         if cfg.calltime_hints else {}),
                      **({"working_state": traj.meta.get("working_state", {})}
                         if cfg.working_state else {})},
            ))
        records.extend(w_records)

        # (2) reflection + hindsight for the whole window concurrently (window-
        # atomic: every task retrieves against the buffer as it stood BEFORE the
        # window), then gate proposals in task order, then retries concurrently,
        # then buffer adds in task order.
        async def hindsight_one(task, rec, traj, served_block=None):
            engine = router.for_task(rec.task_id)
            if cfg.reflect:  # stored on the trajectory so peers retrieve it later
                t0 = time.monotonic()
                traj.meta["reflection"] = await asyncio.to_thread(
                    reflect_fn, engine, traj, cfg, adapter=serving)
                rec.timings["reflect"] = time.monotonic() - t0
            retrieved = memory.retrieve(task, cfg.retrieval_k, scope=cfg.retrieval_scope,
                                        score_mode=cfg.retrieval_score,
                                        offset=cfg.retrieval_offset,
                                        diversity=cfg.retrieval_diversity)
            # serve_experience: the trajectory was GENERATED under served_block,
            # so that exact block -- not a rebuilt one carrying the own-outcome
            # summary -- is the teacher whose distribution we are matching.
            block = served_block or experience_mod.build_block(
                retrieved, traj, cfg, query_text=_natural_task_text(task)
            )
            if not cfg.score_hindsight:  # corpus collection: nothing consumes the deltas
                return None, block
            t0 = time.monotonic()
            try:
                hr = await asyncio.to_thread(hindsight_mod.score, engine, traj, block, cfg)
            except Exception as e:  # oversized rendering etc.: no signal, stream goes on
                rec.meta["hindsight_error"] = repr(e)[:160]
                hr = None
            rec.timings["hindsight"] = time.monotonic() - t0
            return hr, block

        scored = await asyncio.gather(*(hindsight_one(t, r, tj, sb)
                                        for t, r, tj, sb in zip(window, w_records, trajs, served)))
        samples: list[TrainSample] = []
        contrib: list[StreamRecord] = []
        retry_jobs = []
        deferred = []  # sft/advantage: proposal waits on the hint-carrying retry
        for task, rec, traj, (hr, block) in zip(window, w_records, trajs, scored):
            if hr is None:
                continue
            sample = hindsight_mod.to_train_sample(traj, hr, cfg)
            if cfg.weight_mode == "sft":
                # every action token is a KL position; the only question is
                # whether this task's hint is worth internalising at all
                rank = float(traj.reward or 0.0)
                if cfg.propose_rule == "advantage":
                    # decided after the retry: propose iff the hint turned a
                    # failure into a success, i.e. reward(p+hint) > reward(p)
                    if traj.success:
                        rec.meta["proposed"] = False  # no headroom to demonstrate
                        continue
                    retry_jobs.append((rec, attempt(task, serving, experience=block)))
                    deferred.append((rec, sample, rank))
                    continue
                proposed = True
            elif cfg.weight_mode == "binary":
                n_gate = sum(1 for ws in sample.token_weights_by_msg for w in ws if w != 0.0)
                proposed = n_gate >= cfg.min_gate_tokens
                rec.meta["n_gate"] = n_gate
                rank = float(n_gate)
            else:
                proposed = bool(gate.propose(hr, traj))
                rank = hr.obs_surprise
            rec.meta["proposed"] = proposed
            if proposed and not bool(traj.success) and cfg.retry_on_fail:
                retry_jobs.append((rec, attempt(task, serving, experience=block)))
            if proposed:
                samples.append((rank, sample))
                contrib.append(rec)
        retries = await asyncio.gather(*(job for _, job in retry_jobs))
        retry_by_id = {}
        for (rec, _), (retry_traj, retry_dt) in zip(retry_jobs, retries):
            retry_traj.is_retry = True
            rec.retried = True
            rec.timings["retry"] = retry_dt
            retry_by_id[rec.task_id] = retry_traj
        for rec, sample, rank in deferred:
            rt = retry_by_id.get(rec.task_id)
            helped = bool(rt is not None and rt.success)
            rec.meta["proposed"] = helped
            if helped:
                samples.append((rank, sample))
                contrib.append(rec)
        for traj in trajs:
            buffer.add(traj)
            if traj.task_id in retry_by_id:
                buffer.add(retry_by_id[traj.task_id])

        # (3) train candidate from proposed tasks; gate-validate; merge/publish.
        if samples:
            t0 = time.monotonic()
            # Dose control: total steps per merge = n_samples * epochs. Keep the
            # highest-surprise K (P1.7: update magnitude past the knee wrecks
            # retention; stream candidates must not multiply the dose by n).
            samples.sort(key=lambda p: p[0], reverse=True)
            kept = [s for _, s in samples[: cfg.max_candidate_samples]]
            if len(samples) > len(kept):
                print(f"[dose w{w_idx}] {len(samples)} proposals -> top {len(kept)} "
                      f"by surprise", flush=True)
            candidate = await asyncio.to_thread(
                trainer_fn, kept, current, cfg,
                workdir=cfg.out_dir + "/candidates")
            # G2/G3 A/B the candidate on the validating engine.
            cand_version = AdapterVersion(
                name=candidate.candidate_id, path=candidate.adapter_path,
                parent=current.name)
            for engine in engines:  # probes run across all engines
                engine.load_adapter(cand_version)
            decision = await asyncio.to_thread(
                gate.validate, candidate, current, engines[0],
                buffer.replay_states(cfg.gate_replay_states, cfg.seed),
                probe_tasks=probe_tasks, run_probe=run_probe)
            rolled = False
            if recheck is not None and run_probe is not None and probe_tasks:
                for engine in engines:
                    engine.load_adapter(recheck)
                prev_rate = await asyncio.to_thread(run_probe, recheck.name, probe_tasks)
                now_best = max(decision.parent_rate or 0.0, decision.probe_rate or 0.0)
                if prev_rate > now_best + cfg.rollback_margin:
                    if recheck.path is None:  # rolling back to the base model
                        registry._set_current(recheck.name)
                        version = recheck
                    else:  # republish the pre-merge weights as a new version
                        version = registry.publish(recheck.path, current.name,
                                                   [f"rollback:{recheck.name}"])
                    for engine in engines:
                        engine.load_adapter(version)
                    rolled = True
                    print(f"[rollback w{w_idx}] {recheck.name} probe {prev_rate:.3f} > "
                          f"max(current {decision.parent_rate:.3f}, cand "
                          f"{decision.probe_rate:.3f}) -> republished as {version.name}; "
                          f"candidate discarded", flush=True)
                else:
                    print(f"[rollback w{w_idx}] keep: {recheck.name} probe {prev_rate:.3f} "
                          f"<= {now_best:.3f}", flush=True)
                for engine in engines:
                    getattr(engine, "unload_adapter", lambda n: None)(recheck.name)
                recheck = None
            if decision.passed and not rolled:
                version = merge_mod.merge(current, candidate, cfg, registry)
                if (cfg.rollback_thr > 0 and decision.probe_delta is not None
                        and decision.probe_delta <= -cfg.rollback_thr):
                    recheck = current  # re-probe the pre-merge version next window
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
            print(f"[gate w{w_idx}] {'PASS' if decision.passed else 'REJECT'}: "
                  f"{rec_reason}", flush=True)
            train_dt = time.monotonic() - t0
            for rec in contrib:
                rec.timings["train"] = train_dt
                rec.meta["gate_reason"] = rec_reason

        # (4) snapshot hook.
        if on_window is not None:
            on_window(w_idx, w_records)

    return records
