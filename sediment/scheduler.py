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
   ``cfg.kl_states`` in ("diverge", "retry"): every failed first attempt is
   redone with the block and the redo's success is the
   ``propose_rule="advantage"`` filter. "diverge" trains the first attempt's
   own states up to the turn where the redo deviates; "retry" trains the
   redo itself, stripped to the student view and scored against the block.
   ``cfg.min_merge_samples`` holds proposed samples across windows until
   enough are pending for one candidate.
3. TrainSamples of proposed tasks -> trainer_fn -> gate.validate -> on pass
   merge.merge publishes the new version and the window's contributing
   records get ``gated_in=True``.
4. Optional ``on_window(window_idx, window_records)`` callback.

Cross-module collaborators (envs, rollout, experience, hindsight, merge) are
imported lazily inside the run so importing this module needs only stdlib.
"""
from __future__ import annotations

import json
import re

import asyncio
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, Optional

from sediment.config import StreamConfig
from sediment.router import Router
from sediment.types import (
    AdapterVersion,
    Message,
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
    if family == "intercode_sql":
        from sediment.envs.intercode_sql import InterCodeSQLEnv

        return InterCodeSQLEnv(**dict(task.get("intercode_mysql") or {}))
    if family == "alfworld":
        from sediment.envs.alfworld import ALFWorldEnv

        return ALFWorldEnv(**dict(task.get("alfworld") or {}))
    raise ValueError(f"no env constructor for env_family {family!r}")


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _parsed_tool_calls(content: str) -> tuple[str, ...]:
    """Canonical tool-call payloads from one assistant message.

    Invalid JSON remains comparable as whitespace-normalized raw text. Pure
    narration yields no call and is ignored by tool-diff localization.
    """
    found = _TOOL_CALL_RE.findall(content)
    if not found:
        return ()
    out = []
    for raw in found:
        try:
            out.append(json.dumps(json.loads(raw), sort_keys=True,
                                  separators=(",", ":"), ensure_ascii=False))
        except (TypeError, ValueError, json.JSONDecodeError):
            out.append(" ".join(raw.split()))
    return tuple(out)


def _divergence_pair(
    first: Trajectory, redo: Trajectory, locator: str = "tool_diff"
) -> Optional[tuple[int, int]]:
    """(first-message, redo-message) of the first assistant decision that differs.

    ``text`` compares the whole assistant content. ``tool_diff`` compares only
    canonical parsed tool calls at the same assistant-turn ordinal, ignoring
    reasoning/narration changes. Returning the redo index makes the result the
    exact rejected/target step pair used by fork CE and preference losses.
    """
    if locator not in ("text", "tool_diff", "sql"):
        raise ValueError(f"unknown fork_locator: {locator!r}")
    firsts = [(i, m.content) for i, m in enumerate(first.messages)
              if m.role == "assistant"]
    redos = [(i, m.content) for i, m in enumerate(redo.messages)
             if m.role == "assistant"]
    if locator == "tool_diff":
        first_calls = [(i, call) for i, content in firsts
                       for call in _parsed_tool_calls(content)]
        redo_calls = [(i, call) for i, content in redos
                      for call in _parsed_tool_calls(content)]
        for (i, left), (j, right) in zip(first_calls, redo_calls):
            if left != right:
                return i, j
        # An extra redo call has no failed action at the same call ordinal, so
        # it cannot define the paired fork required by these experiments.
        return None
    if locator == "sql":
        # InterCode actions are fenced SQL rather than tool-call JSON.  Fork CE
        # pairs the student's *first attempted answer* with the privileged
        # teacher's final successful correction.  SHOW/DESC are information
        # gathering, not erroneous answer SQL.  The teacher may need several
        # failed intermediate attempts, so comparing equal turn ordinals can
        # accidentally supervise another wrong query instead of the correction.
        from sediment.envs.intercode_sql import parse_sql_action

        def answer_sql(items):
            out = []
            for i, content in items:
                sql, valid = parse_sql_action(content)
                if not valid:
                    continue
                head = sql.lstrip().split(None, 1)[0].rstrip(";").upper()
                if head in {"SHOW", "DESC", "DESCRIBE"}:
                    continue
                out.append((i, sql))
            return out

        first_answers = answer_sql(firsts)
        redo_answers = answer_sql(redos)
        if not first_answers or not redo_answers:
            return None
        i, left = first_answers[0]
        j, right = redo_answers[-1]
        if " ".join(left.split()) != " ".join(right.split()):
            return i, j
        return None
    for (i, left), (j, right) in zip(firsts, redos):
        if left != right:
            return i, j
    return None


def _divergence_index(
    first: Trajectory, redo: Trajectory, locator: str = "tool_diff"
) -> Optional[int]:
    """Compatibility wrapper returning only the redo-side fork index."""
    pair = _divergence_pair(first, redo, locator)
    return pair[1] if pair is not None else None


_ARG_VALUE_RE = re.compile(r'"[^"]*"\s*:\s*"([^"]+)"')
_ARG_NUM_RE = re.compile(r'"[^"]*"\s*:\s*([0-9][0-9.\-]*)')


def _stepwise_feedback_samples(
    traj: Trajectory,
    behavior_logprobs: list[list[float]],
    gamma: float,
) -> tuple[list[TrainSample], dict[str, Any]]:
    """Compile one hindsight-distillation sample per executed assistant turn.

    The sample ends at the decision being scored. Its student view is therefore
    exactly ``H_{t-1} + (reasoning, action)_t``. The teacher receives only the
    observation returned *after* that decision through a separate context; the
    decision itself is never copied into that context. Earlier decisions and
    observations remain ordinary causal history. The final environment output
    lives in ``traj.meta['final_obs']`` in the real rollout loop, so it is paired
    with the last assistant turn explicitly.
    """
    if len(behavior_logprobs) != len(traj.messages):
        raise ValueError(
            f"behavior logprob/message mismatch for {traj.task_id}: "
            f"{len(behavior_logprobs)} != {len(traj.messages)}"
        )
    assistant = [i for i, m in enumerate(traj.messages) if m.role == "assistant"]
    out: list[TrainSample] = []
    feedback_chars: list[int] = []
    redacted = 0
    for turn, i in enumerate(assistant):
        stop = assistant[turn + 1] if turn + 1 < len(assistant) else len(traj.messages)
        observations = [
            m.content for m in traj.messages[i + 1:stop]
            if m.role in ("tool", "user") and m.content.strip()
        ]
        is_last = turn == len(assistant) - 1
        final_obs = str(traj.meta.get("final_obs", "") or "").strip()
        if is_last and final_obs and final_obs not in observations:
            observations.append(final_obs)
        # An intercepted/unexecuted draft has no environment response. It is
        # history, not an action-feedback training pair.
        if not observations:
            continue
        action = traj.messages[i].content.strip()
        cleaned = []
        for observation in observations:
            if action and action in observation:
                observation = observation.replace(action, "[current assistant decision omitted]")
                redacted += 1
            cleaned.append(observation)
        lines = [
            "<hindsight_feedback>",
            "The environment returned the following observation after the assistant "
            "decision currently being re-evaluated. Use it to reassess both the "
            "reasoning and the action; do not copy an action from this block.",
            "",
            *cleaned,
        ]
        if is_last:
            outcome = "SUCCEEDED" if bool(traj.success) else "FAILED"
            reward = "n/a" if traj.reward is None else f"{float(traj.reward):.6g}"
            lines.extend(["", f"Verified trajectory outcome: {outcome} (reward={reward})."])
        lines.append("</hindsight_feedback>")
        feedback = "\n".join(lines)
        prefix = list(traj.messages[:i + 1])
        weights: list[list[float]] = [[] for _ in prefix]
        weights[-1] = [1.0]  # whole visible reasoning + action assistant turn
        old: list[list[float]] = [[] for _ in prefix]
        old[-1] = list(behavior_logprobs[i])
        out.append(TrainSample(
            task_id=f"{traj.task_id}@turn{turn:02d}",
            messages=prefix,
            token_weights_by_msg=weights,
            teacher_contexts={"feedback": feedback},
            behavior_logprobs_by_msg=old,
            loss_scale=float(gamma) ** turn,
        ))
        feedback_chars.append(len(feedback))
    return out, {
        "assistant_turns": len(assistant),
        "supervised_turns": len(out),
        "feedback_chars": feedback_chars,
        "action_echo_redactions": redacted,
    }


def _apply_turn_decay(sample: TrainSample, gamma: float) -> TrainSample:
    """Scale the j-th weighted assistant turn by gamma**j (in place)."""
    if gamma >= 1.0:
        return sample
    j = 0
    for i, (m, ws) in enumerate(zip(sample.messages, sample.token_weights_by_msg)):
        if m.role != "assistant" or not ws or not any(w != 0.0 for w in ws):
            continue
        f = gamma ** j
        sample.token_weights_by_msg[i] = [w * f for w in ws]
        j += 1
    return sample


def _fork_grounded(first: Trajectory, redo: Trajectory, k: int) -> bool:
    """Are the redo's tool-call argument values at the fork copyable from the
    student's own context (everything before message k)? Values that only the
    block revealed (a later observation of the failed attempt) are not."""
    if k >= len(redo.messages):
        return True
    action = redo.messages[k].content
    values = [v for v in _ARG_VALUE_RE.findall(action) + _ARG_NUM_RE.findall(action)
              if len(v) >= 3]
    if not values:
        return True
    context = "\n".join(m.content for m in first.messages[:k])
    return all(v in context for v in values)


def _fork_action_grounded(first: Trajectory, action: str, prefix_end: int) -> bool:
    """Grounding check for a target action paired with the student's prefix."""
    values = [v for v in _ARG_VALUE_RE.findall(action) + _ARG_NUM_RE.findall(action)
              if len(v) >= 3]
    if not values:
        return True
    context = "\n".join(m.content for m in first.messages[:prefix_end])
    return all(v in context for v in values)


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
    rollout_workers = cfg.extra.get("rollout_workers")
    if rollout_workers is not None and (
        isinstance(rollout_workers, bool)
        or not isinstance(rollout_workers, int)
        or rollout_workers < 1
    ):
        raise ValueError("extra.rollout_workers must be a positive integer")

    async def _run() -> list[StreamRecord]:
        if rollout_workers is not None:
            # asyncio.to_thread otherwise uses Python's bounded default pool
            # (at most 32 workers), so a larger inference-only held-out window
            # silently fails to reach its requested rollout concurrency.
            asyncio.get_running_loop().set_default_executor(
                ThreadPoolExecutor(
                    max_workers=rollout_workers,
                    thread_name_prefix="sediment-rollout",
                )
            )
        return await _stream(
            tasks, engines, buffer, gate, trainer_fn, registry, cfg,
            run_probe=run_probe, probe_tasks=probe_tasks, on_window=on_window,
            window_offset=window_offset, retrieval_buffer=retrieval_buffer,
        )

    return asyncio.run(_run())


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
    if cfg.kl_states not in ("first", "diverge", "fork", "retry"):
        raise ValueError(f"unknown kl_states: {cfg.kl_states!r}")
    redo_mode = cfg.kl_states in ("diverge", "fork", "retry")
    if cfg.own_view == "full" and not redo_mode:
        # 08-25 icl_gate forensics: with the scored trajectory verbatim in the
        # block, the teacher partly measures copying and reverse KL can pull
        # the student toward its own failed actions. Allowed anyway (08-31,
        # user decision): in multi-turn agentic failures the error signal is
        # spread across turns and a compact verdict may not carry it -- the
        # single-response refs (SDPO/CLaaS) never faced this. Watch the copy
        # indicators (diverge_delta-style) when running this combination.
        print("[warn] own_view='full' with kl_states='first': scored trajectory "
              "is verbatim in the teacher block (copy risk, 08-25 forensics)",
              flush=True)
    if cfg.kl_teacher not in ("topk", "local"):
        raise ValueError(f"unknown kl_teacher: {cfg.kl_teacher!r}")
    teacher_ctx_names = {c.strip() for c in cfg.teacher_contexts.split(",") if c.strip()}
    if cfg.kl_teacher == "local" and not teacher_ctx_names <= {"donor", "failure", "feedback"}:
        raise ValueError(f"unknown teacher_contexts: {cfg.teacher_contexts!r}")
    if not (0.0 < cfg.turn_decay <= 1.0):
        raise ValueError(f"turn_decay must be in (0, 1], got {cfg.turn_decay}")
    if cfg.fork_objective == "trajectory_dpo" and (
            cfg.kl_states != "retry" or cfg.kl_target):
        raise ValueError("trajectory_dpo requires kl_states='retry' and kl_target=false")
    if cfg.kl_states in ("fork", "diverge") and not cfg.kl_target and cfg.weight_mode != "sft":
        # 09-01: the same 55 r1 fork samples score +52 with whole-turn credit
        # and -9 with call-interior-only credit -- the decision forms in the
        # narration, so fork-CE must never run under the gated weight modes.
        raise ValueError("fork/diverge CE requires weight_mode='sft' (whole turn); "
                         "call-only credit measured -9 vs +52 on identical samples")
    if cfg.stepwise_feedback_distill:
        required = {
            "kl_states='first'": cfg.kl_states == "first",
            "kl_teacher='local'": cfg.kl_teacher == "local",
            "kl_target=true": cfg.kl_target,
            "weight_mode='sft'": cfg.weight_mode == "sft",
            "propose_rule='all'": cfg.propose_rule == "all",
            "retry_on_fail=false": not cfg.retry_on_fail,
            "calltime_hints=false": not cfg.calltime_hints,
            "working_state=false": not cfg.working_state,
        }
        bad = [name for name, ok in required.items() if not ok]
        if bad:
            raise ValueError("stepwise_feedback_distill requires " + ", ".join(bad))
    sw_pool = None
    if cfg.stepwise_experience:
        import os

        from sediment import stepwise as stepwise_mod
        from sediment.stepwise import PrinciplePool
        if cfg.calltime_hints or cfg.serve_experience or cfg.stepwise_feedback_distill:
            raise ValueError("stepwise_experience does not combine with "
                             "calltime_hints/serve_experience/stepwise_feedback_distill")
        if cfg.kl_states not in ("first", "retry", "fork", "diverge"):
            raise ValueError("stepwise_experience requires kl_states in "
                             "('first', 'retry', 'fork', 'diverge')")
        if cfg.kl_target and cfg.kl_teacher != "topk":
            raise ValueError("stepwise_experience teacher views are per-turn; "
                             "only the served kl_teacher='topk' path supports them")
        pp = cfg.stepwise_pool_path or os.path.join(
            os.path.dirname(cfg.buffer_path) or ".", "pool.jsonl")
        sw_pool = PrinciplePool(pp, max_items=cfg.stepwise_max_pool)
        print(f"[stepwise] pool {pp}: {len(sw_pool)} items", flush=True)
    pending: list[tuple[float, TrainSample, StreamRecord]] = []  # min_merge_samples
    async_trainer = None
    if cfg.async_train:
        if cfg.gate_validate:
            raise ValueError("async_train requires gate_validate=false")
        from sediment.async_train import AsyncTrainer
        async_trainer = AsyncTrainer(trainer_fn, registry, engines, cfg,
                                     workdir=cfg.out_dir + "/candidates")
        async_trainer.start()

    def _retrieve(task: dict[str, Any], k: int) -> list[Trajectory]:
        return memory.retrieve(task, k, scope=cfg.retrieval_scope,
                               score_mode=cfg.retrieval_score, offset=cfg.retrieval_offset,
                               diversity=cfg.retrieval_diversity)

    async def attempt(task: dict[str, Any], adapter: str,
                      experience: Any = None,
                      stepwise: bool = False,
                      feedback_from: Optional[Trajectory] = None) -> tuple[Trajectory, float]:
        """One episode in a worker thread; returns (trajectory, seconds).
        Harness state (call-time hinter, working state) is fresh per attempt."""
        engine = router.for_task(task["task_id"])
        hinter = ws = sw = None
        if stepwise:
            from sediment.stepwise import StepwiseHinter
            sw = StepwiseHinter(sw_pool, engine, adapter, cfg, _natural_task_text(task))
        elif feedback_from is not None:
            from sediment.stepwise import AttemptFeedbackHinter
            sw = AttemptFeedbackHinter(feedback_from)
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

        harness = {k: v for k, v in (("hinter", hinter), ("working_state", ws),
                                     ("stepwise", sw))
                   if v is not None}

        def _run() -> tuple[Trajectory, float]:
            t0 = time.monotonic()
            traj = run_episode(engine, _make_env(task), task, cfg, adapter=adapter,
                               experience=experience, **harness)
            return traj, time.monotonic() - t0

        return await asyncio.to_thread(_run)

    def _critic_post(path: str, payload: dict) -> dict:
        import json as _json
        import urllib.request
        req = urllib.request.Request(cfg.critic_url.rstrip("/") + path, data=_json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return _json.loads(r.read().decode())

    async def guided_redo(task: dict[str, Any], first: Trajectory, adapter: str) -> tuple[Trajectory, float]:
        """cfg.critic_mode="redo_rank": sample candidate actions at the branch
        turn of the FAILED first attempt, rank them with the critic service,
        replay the prefix and execute the top candidate, then let the served
        policy finish. Returns (trajectory, seconds) like attempt()."""
        from sediment.branch import continue_from_prefix, sample_siblings

        engine = router.for_task(task["task_id"])

        def _run() -> tuple[Trajectory, float]:
            t0 = time.monotonic()
            msgs = first.messages
            asst = [i for i, m in enumerate(msgs) if m.role == "assistant"]
            meta: dict[str, Any] = {"critic_mode": cfg.critic_mode}
            if not asst:
                return Trajectory(task_id=first.task_id, env_family=first.env_family, messages=list(msgs),
                                  reward=first.reward, success=False, adapter=adapter, is_retry=True,
                                  meta={**meta, "critic_skip": "no_assistant_turn"}), 0.0
            j = asst[0]
            if cfg.critic_turns == "localize":
                resp = _critic_post("/score_turns", {"messages": [m.to_dict() for m in msgs], "key": task["task_id"]})
                cand = [(t["A"], t["msg"]) for t in resp.get("turns", [])
                        if t["A"] is not None and t["msg"] in asst[: cfg.critic_max_turn]]
                if cand:
                    j = min(cand)[1]
                    meta["critic_turn_scores"] = cand
            prefix = list(msgs[:j])
            sib, smeta = sample_siblings(engine, prefix, msgs[j].content, engine._tokenizer(), cfg.critic_k,
                                         cfg.critic_sample_temperature, adapter=adapter)
            meta.update(critic_branch_msg=j, critic_n_candidates=len(sib), critic_n_raw=smeta.get("n_raw"))
            if not sib:
                return Trajectory(task_id=first.task_id, env_family=first.env_family, messages=list(msgs),
                                  reward=first.reward, success=False, adapter=adapter, is_retry=True,
                                  meta={**meta, "critic_skip": "no_distinct_candidate"}), time.monotonic() - t0
            if cfg.critic_mode == "redo_random":  # control: same candidates, no critic
                import random as _random
                A = [None] * len(sib)
                pick = _random.Random(hash(task["task_id"]) & 0xFFFF).randrange(len(sib))
            else:
                resp = _critic_post("/score", {"prefix": [m.to_dict() for m in prefix],
                                               "actions": [x["action"] for x in sib], "key": task["task_id"]})
                A = resp.get("A", [None] * len(sib))
                valid = [i for i, a in enumerate(A) if a is not None]
                pick = max(valid, key=lambda i: A[i]) if valid else 0
            meta.update(critic_A=A, critic_pick=pick, critic_pick_source=sib[pick]["source"])
            prefix_actions = [msgs[i].content for i in range(j) if msgs[i].role == "assistant"]
            traj = continue_from_prefix(engine, _make_env(task), task, cfg, adapter, prefix_actions, sib[pick]["action"])
            traj.meta.update(meta)
            return traj, time.monotonic() - t0

        return await asyncio.to_thread(_run)

    async def critic_dpo_pairs(task: dict[str, Any], first: Trajectory,
                               adapter: str) -> list[tuple[float, TrainSample]]:
        """cfg.critic_mode="dpo_steps": preference pairs without any redo.

        The critic scores every executed action of the first attempt (whether
        the task succeeded or not); at the cfg.critic_dpo_states lowest-valued
        turns we DRAFT cfg.critic_k alternatives -- they are never executed, so
        the environment is only a trajectory source, never a judge -- rank them
        with the original action, and emit (best, worst) pairs. Rank = the
        critic gap, which is what the dose sort keeps."""
        from sediment.branch import sample_siblings

        engine = router.for_task(task["task_id"])

        def _run() -> list[tuple[float, TrainSample]]:
            msgs = first.messages
            asst = [i for i, m in enumerate(msgs) if m.role == "assistant"]
            if not asst:
                return []
            diag = first.meta.setdefault("dpo_diag", {})
            resp = _critic_post("/score_turns",
                                {"messages": [m.to_dict() for m in msgs], "key": task["task_id"]})
            turns = sorted((t["A"], t["msg"]) for t in resp.get("turns", [])
                           if t["A"] is not None and t["msg"] in asst[: cfg.critic_max_turn])
            if cfg.critic_dpo_state_pick == "random":  # control: the critic ranks, but does not locate
                _rng = random.Random(f"{task['task_id']}:states")
                _rng.shuffle(turns)
            # depth bias check: "lowest A" must not just mean "deepest turn"
            diag["turn_A"] = [(round(a, 4), m) for a, m in sorted(turns, key=lambda x: x[1])]
            diag["n_turns"] = len(asst)
            out: list[tuple[float, TrainSample]] = []
            for _a0, j in turns[: cfg.critic_dpo_states]:
                prefix = list(msgs[:j])
                sib, _meta = sample_siblings(engine, prefix, msgs[j].content, engine._tokenizer(),
                                             cfg.critic_k, cfg.critic_sample_temperature,
                                             adapter=adapter)
                cands = [x["action"] for x in sib]
                if cfg.critic_dpo_include_original:
                    cands.append(msgs[j].content)
                if len(cands) < 2:
                    continue
                key = f"{task['task_id']}@{j}"
                A = _critic_post("/score", {"prefix": [m.to_dict() for m in prefix],
                                            "actions": cands, "key": key}).get("A", [])
                diag.setdefault("cand_A", {})[str(j)] = [
                    None if a is None else round(a, 4) for a in A]
                diag.setdefault("orig_rank", {})[str(j)] = (
                    sum(1 for a in A[:-1] if a is not None and A[-1] is not None and a > A[-1])
                    if cfg.critic_dpo_include_original else None)
                ranked = [(a, c) for a, c in zip(A, cands) if a is not None]
                if cfg.critic_dpo_rank == "random":  # control: same drafts, arbitrary direction
                    random.Random(key).shuffle(ranked)
                else:
                    ranked.sort(key=lambda x: x[0], reverse=True)
                for p in range(min(cfg.critic_dpo_pairs, len(ranked) // 2)):
                    (a_hi, hi), (a_lo, lo) = ranked[p], ranked[-1 - p]
                    gap = a_hi - a_lo
                    if cfg.critic_dpo_rank == "critic" and gap < cfg.critic_dpo_min_margin:
                        continue
                    out.append((abs(float(gap)), TrainSample(
                        task_id=f"{key}#{p}",
                        messages=prefix + [Message("assistant", hi)],
                        token_weights_by_msg=[[] for _ in prefix] + [[1.0]],
                        rejected=lo)))
            return out

        return await asyncio.to_thread(_run)

    if cfg.critic_mode or cfg.episode_token_budget > 0:
        # Candidate sampling and strict episode budgets both tokenize inside
        # worker threads. Load Transformers' lazy AutoTokenizer once on the
        # main thread; simultaneous first imports otherwise race in fresh
        # processes (observed by the InterCode three-arm smoke).
        for e in engines:  # load it once here, on the main thread (transformers' lazy
            if hasattr(e, "_tokenizer"):  # imports race when first triggered from threads)
                e._tokenizer()

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
        # (async_train: this picks up the freshest async-published version at
        # every window boundary; the pin keeps it loaded for in-flight episodes.)
        current = registry.current()
        serving = current.name
        if async_trainer is not None:
            async_trainer.pin(serving)
        w_start_step = async_trainer.step if async_trainer is not None else 0

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
                      "natural_finish": traj.meta.get("natural_finish"),
                      "forced_settle": traj.meta.get("forced_settle"),
                      "termination_reason": traj.meta.get("termination_reason"),
                      "termination_step": traj.meta.get("termination_step"),
                      "reward_checkpoints": traj.meta.get("reward_checkpoints", {}),
                      "tool_protocol": traj.meta.get("tool_protocol", {}),
                      **({"hints": len(events),
                          "hint_triggers": [e["trigger"] for e in events],
                          "hint_changed": sum(1 for e in events if e.get("changed")),
                          "hint_chars": sum(e["chars"] for e in events)}
                         if cfg.calltime_hints else {}),
                      **({"working_state": traj.meta.get("working_state", {})}
                         if cfg.working_state else {})},
            ))
        records.extend(w_records)

        # (1.5) stepwise: distill this window's episodes into the principle
        # pool BEFORE any redo or teacher scoring, so a failed task's guided
        # redo / teacher can select its own freshly extracted lessons (the
        # paper's pool refresh, at window granularity).
        if sw_pool is not None and cfg.stepwise_extract:
            t0 = time.monotonic()

            def _safe_extract(eng, tr):
                try:
                    return stepwise_mod.extract_principles(eng, tr, cfg, serving)
                except Exception as e:  # extraction never kills the stream
                    print(f"[stepwise] extract failed {tr.task_id}: {e!r}"[:200], flush=True)
                    return []

            ex = await asyncio.gather(*(
                asyncio.to_thread(_safe_extract, router.for_task(t["task_id"]), tr)
                for t, tr in zip(window, trajs)))
            added = sw_pool.add([it for lst in ex for it in lst])
            print(f"[stepwise w{w_idx}] pool +{added} -> {len(sw_pool)} "
                  f"({time.monotonic() - t0:.0f}s)", flush=True)

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
            if cfg.stepwise_feedback_distill:
                if cfg.stepwise_feedback_failures_only and traj.success:
                    rec.meta["stepwise"] = {
                        "n_samples": 0,
                        "skipped": "successful_first_attempt",
                    }
                    return None, None, {}, []
                t0 = time.monotonic()
                # Score with the exact pinned adapter that generated this
                # window. This is collection-time behavior logp even if newer
                # async versions publish before the replay sample is drawn.
                behavior = await asyncio.to_thread(
                    engine.score, traj.messages, adapter=serving
                )
                step_samples, diag = _stepwise_feedback_samples(
                    traj, behavior, cfg.turn_decay
                )
                rec.timings["behavior_score"] = time.monotonic() - t0
                rec.meta["stepwise"] = diag
                return None, None, {}, step_samples
            if sw_pool is not None and cfg.kl_states == "first":
                # on-policy regime (2606.04703 Eq.2): the bare first attempt is
                # the sample; the teacher is the same policy under per-turn
                # step-wise views, scored here post-hoc. No block is built.
                t0 = time.monotonic()
                try:
                    hr = await asyncio.to_thread(stepwise_mod.teacher_score,
                                                 engine, traj, cfg, serving, sw_pool)
                except Exception as e:
                    rec.meta["hindsight_error"] = repr(e)[:160]
                    hr = None
                rec.timings["hindsight"] = time.monotonic() - t0
                if hr is not None:
                    rec.meta["stepwise_act_gain"] = round(hr.act_gain, 4)
                    rec.meta["stepwise_turns"] = len(hr.spans)
                return hr, experience_mod.build_block([], None, cfg), {}, []
            retrieved = memory.retrieve(task, cfg.retrieval_k, scope=cfg.retrieval_scope,
                                        score_mode=cfg.retrieval_score,
                                        offset=cfg.retrieval_offset,
                                        diversity=cfg.retrieval_diversity)
            # serve_experience: the trajectory was GENERATED under served_block,
            # so that exact block -- not a rebuilt one carrying the own-outcome
            # summary -- is the teacher whose distribution we are matching.
            own = traj if cfg.hindsight_include_own_outcome else None
            block = served_block or experience_mod.build_block(
                retrieved, own, cfg, query_text=_natural_task_text(task)
            )
            # local teacher (cfg.kl_teacher="local"): SEPARATE donor / own-failure
            # blocks for the trainer -- they never share a prompt; the trainer
            # combines their residuals in logit space. The redo prompt stays
            # `block` (retrieval_k peers + own), so teacher_retrieval_k can
            # supply a donor to the trainer while the redo sees own failure only.
            ctx: dict[str, str] = {}
            if cfg.kl_teacher == "local":
                qtext = _natural_task_text(task)
                if "donor" in teacher_ctx_names:
                    donors = retrieved if cfg.retrieval_k > 0 else (
                        memory.retrieve(task, cfg.teacher_retrieval_k, scope=cfg.retrieval_scope,
                                        score_mode=cfg.retrieval_score,
                                        offset=cfg.retrieval_offset,
                                        diversity=cfg.retrieval_diversity)
                        if cfg.teacher_retrieval_k > 0 else [])
                    if donors:
                        ctx["donor"] = experience_mod.build_block(
                            donors, None, cfg, query_text=qtext).text
                        rec.meta["teacher_donor_ids"] = [d.task_id for d in donors]
                if "failure" in teacher_ctx_names and own is not None and not traj.success:
                    ctx["failure"] = experience_mod.build_block([], own, cfg, query_text=qtext).text
                rec.meta["teacher_contexts"] = {k: len(v) for k, v in ctx.items()}
            # corpus collection: nothing consumes the deltas; kl_states=retry:
            # the redo is scored instead
            if not cfg.score_hindsight or cfg.kl_states == "retry":
                return None, block, ctx, []
            t0 = time.monotonic()
            try:
                hr = await asyncio.to_thread(hindsight_mod.score, engine, traj, block, cfg)
            except Exception as e:  # oversized rendering etc.: no signal, stream goes on
                rec.meta["hindsight_error"] = repr(e)[:160]
                hr = None
            rec.timings["hindsight"] = time.monotonic() - t0
            return hr, block, ctx, []

        dpo_mode = cfg.critic_mode == "dpo_steps"
        scored = ([(None, None, {}, [])] * len(window) if dpo_mode else
                  await asyncio.gather(*(hindsight_one(t, r, tj, sb)
                                         for t, r, tj, sb in zip(window, w_records, trajs, served))))
        samples: list[tuple[float, TrainSample]] = []
        contrib: list[StreamRecord] = []
        retry_jobs = []
        dpo_jobs = []  # dpo_steps: (rec, coroutine) -- no redo, no retry
        deferred = []  # sft/advantage: proposal waits on the hint-carrying retry
        redo = []  # diverge/fork/retry: (rec, block, first, hr, ctx) whose redo decides the sample
        for task, rec, traj, (hr, block, ctx, step_samples) in zip(window, w_records, trajs, scored):
            if dpo_mode:
                dpo_jobs.append((rec, traj, critic_dpo_pairs(task, traj, serving)))
                continue
            if cfg.stepwise_feedback_distill:
                rec.meta["proposed"] = bool(step_samples)
                rank = float(traj.reward or 0.0)
                for sample in step_samples:
                    samples.append((rank, sample))
                    contrib.append(rec)
                continue
            if redo_mode:
                if traj.success or not cfg.retry_on_fail:
                    rec.meta["proposed"] = False  # nothing failed, nothing to redo
                    continue
                n_multi = (cfg.redo_block_samples + cfg.redo_stepwise_samples
                           + cfg.redo_feedback_samples)
                if cfg.critic_mode in ("redo_rank", "redo_random"):
                    retry_jobs.append((rec, guided_redo(task, traj, serving)))
                elif n_multi > 1 or (cfg.redo_stepwise_samples and cfg.redo_block_samples):
                    # Best-of-N mixed-context rejection sampling. Each request
                    # draws independently from the serving engine; no request
                    # seed is supplied or recorded.
                    jobs = []
                    for i in range(n_multi):
                        if i < cfg.redo_block_samples:
                            jobs.append(attempt(task, serving, experience=block))
                        elif i < cfg.redo_block_samples + cfg.redo_stepwise_samples:
                            jobs.append(attempt(task, serving, stepwise=True))
                        else:
                            jobs.append(attempt(task, serving, feedback_from=traj))

                    async def multi_redo(jobs=jobs, n_block=cfg.redo_block_samples,
                                         n_step=cfg.redo_stepwise_samples):
                        outs = await asyncio.gather(*jobs)
                        succ = [i for i, (t, _) in enumerate(outs) if t.success]
                        pick = succ[0] if succ else 0
                        t, _ = outs[pick]
                        t.meta["redo_group"] = {
                            "n": len(outs), "n_success": len(succ), "chosen": pick,
                            "chosen_context": (
                                "block" if pick < n_block else
                                "stepwise" if pick < n_block + n_step else
                                "previous_attempt_feedback"
                            ),
                            "success_idx": succ}
                        return t, sum(d for _, d in outs)

                    retry_jobs.append((rec, multi_redo()))
                elif cfg.stepwise_experience:
                    # off-policy teacher rollout (2606.04703 Eq.1): step-wise
                    # selected principles instead of a static block; no
                    # transcript of the failure ever enters the redo prompt
                    retry_jobs.append((rec, attempt(task, serving, stepwise=True)))
                elif cfg.redo_feedback_samples:
                    retry_jobs.append((rec, attempt(task, serving, feedback_from=traj)))
                else:
                    retry_jobs.append((rec, attempt(task, serving, experience=block)))
                redo.append((rec, block, traj, hr, ctx))
                continue
            if hr is None:
                continue
            sample = hindsight_mod.to_train_sample(traj, hr, cfg)
            # soft fork prior applies to first-attempt states too (08-31): the
            # same statistic that motivated it for redos (first changed turn =
            # turn 1 in 54-85% of repairs) says first-attempt mistakes sit in
            # the early turns as well. 1.0 = off, unchanged behavior.
            sample = _apply_turn_decay(sample, cfg.turn_decay)
            if ctx:
                sample.teacher_contexts = dict(ctx)
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
        dpo_out = await asyncio.gather(*(j for _, _, j in dpo_jobs))
        for (rec, traj, _), pairs in zip(dpo_jobs, dpo_out):
            rec.meta["dpo_pairs"] = len(pairs)
            rec.meta["dpo_margin"] = [round(r, 4) for r, _ in pairs]
            rec.meta["proposed"] = bool(pairs)
            rec.meta.update(traj.meta.pop("dpo_diag", {}))
            for pair in pairs:
                samples.append(pair)
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

        async def score_redo(rec, block, first, hr, ctx):
            """(rank, sample) decided by the redo, or None when the task is not
            a train sample. diverge: the first attempt's own sample, masked
            after the turn where the redo deviates; retry: the redo in the
            student view, scored against the block it was generated with."""
            rt = retry_by_id[rec.task_id]
            rt.messages = strip_experience(rt.messages)  # block = teacher context only
            rt.meta["served_block_chars"] = len(block.text)
            rec.meta["redo_success"] = bool(rt.success)
            rec.meta["redo_reward"] = rt.reward
            for k_ in ("critic_branch_msg", "critic_n_candidates", "critic_pick", "critic_pick_source", "critic_A", "critic_skip"):
                if k_ in rt.meta:
                    rec.meta[k_] = rt.meta[k_]
            if cfg.stepwise_experience:
                rec.meta["stepwise_hints"] = len(rt.meta.get("stepwise_events", []))
            if cfg.redo_feedback_samples:
                rec.meta["attempt_feedback_hints"] = len(
                    rt.meta.get("stepwise_events", [])
                )
            if "redo_group" in rt.meta:
                rec.meta["redo_group"] = rt.meta["redo_group"]
            if cfg.propose_rule == "advantage" and not rt.success:
                rec.meta["proposed"] = False  # the evidence did not repair it
                return None
            if cfg.kl_states == "retry" and cfg.fork_objective == "trajectory_dpo":
                # Whole-trajectory preference, not a fork objective: chosen is
                # the complete verified redo after stripping privileged hints;
                # rejected is the complete failed first attempt. Only assistant
                # tokens are scored by the trainer; tool observations remain
                # conditioning context along their respective trajectories.
                weights = [[1.0] if m.role == "assistant" else []
                           for m in rt.messages]
                sample = TrainSample(
                    task_id=rec.task_id,
                    messages=list(rt.messages),
                    token_weights_by_msg=weights,
                    rejected_messages=list(first.messages),
                )
                rec.meta["proposed"] = True
                rec.meta["preference_scope"] = "trajectory"
                return float(rt.reward or 0.0), sample
            if cfg.stepwise_experience and cfg.kl_states == "retry":
                # rejection-sampled guided redo (propose_rule="advantage" above
                # is the filter). KL: forward-KL targets from the exact per-turn
                # views the redo was generated under; CE (kl_target=false): the
                # hard-target limit, plain SFT on the stripped student view.
                if cfg.kl_target:
                    t0 = time.monotonic()
                    try:
                        hr2 = await asyncio.to_thread(
                            stepwise_mod.teacher_score, router.for_task(rec.task_id),
                            rt, cfg, serving)
                    except Exception as e:
                        rec.meta["hindsight_error"] = repr(e)[:160]
                        rec.meta["proposed"] = False
                        return None
                    rec.timings["hindsight"] = time.monotonic() - t0
                    if hr2 is None:
                        rec.meta["proposed"] = False
                        return None
                    rec.meta["stepwise_act_gain"] = round(hr2.act_gain, 4)
                    sample = hindsight_mod.to_train_sample(rt, hr2, cfg)
                else:
                    sample = hindsight_mod.sft_sample(rt, cfg)
                rec.meta["proposed"] = True
                return float(rt.reward or 0.0), _apply_turn_decay(sample, cfg.turn_decay)
            if cfg.kl_states in ("diverge", "fork"):
                pair = _divergence_pair(first, rt, cfg.fork_locator)
                first_k, redo_k = pair if pair is not None else (None, None)
                rec.meta["diverge_msg"] = redo_k
                rec.meta["diverge_first_msg"] = first_k
                rec.meta["fork_locator"] = cfg.fork_locator
                if pair is None:  # the redo never produced a paired decision change
                    rec.meta["proposed"] = False
                    return None
                span = next((sp for sp in hr.spans if sp.msg_idx == first_k), None) if hr else None
                if span is not None and span.deltas:
                    # log q - log p on the failed action itself: negative =
                    # the teacher lowered it, positive = it copied/endorsed it
                    rec.meta["diverge_delta"] = sum(span.deltas) / len(span.deltas)
                if (first_k >= len(first.messages) or redo_k >= len(rt.messages)
                        or first.messages[first_k].role != "assistant"
                        or rt.messages[redo_k].role != "assistant"):
                    rec.meta["proposed"] = False
                    return None
                target = rt.messages[redo_k]
                rejected = first.messages[first_k]
                grounded = _fork_action_grounded(first, target.content, first_k)
                rec.meta["fork_grounded"] = grounded
                if cfg.fork_require_grounded and not grounded:
                    # the repaired call's values came from the block: the bare
                    # student would be trained to emit what it cannot see
                    rec.meta["proposed"] = False
                    return None
                if cfg.kl_states == "fork" and not cfg.kl_target:
                    # Pair both actions with the failed student's exact bare
                    # prefix.  The privileged redo supplies only the target
                    # assistant step; its hint-bearing history is never part of
                    # the train input.
                    msgs = list(first.messages[:first_k]) + [target]
                    weights: list[list[float]] = [[] for _ in msgs]
                    weights[-1] = [1.0]
                    rec.meta["proposed"] = True
                    sample = TrainSample(
                        task_id=rec.task_id,
                        messages=msgs,
                        token_weights_by_msg=weights,
                    )
                    if cfg.fork_objective in ("ce_suffix", "margin_token", "dpo"):
                        sample.rejected = rejected.content
                    elif cfg.fork_objective != "ce":
                        raise ValueError(f"unknown fork_objective: {cfg.fork_objective!r}")
                    return float(rt.reward or 0.0), sample
                if hr is None:  # unscored: no teacher for the KL objective
                    rec.meta["proposed"] = False
                    return None
                sample = hindsight_mod.to_train_sample(first, hr, cfg)
                keep = ((lambda i: i == first_k) if cfg.kl_states == "fork"
                        else (lambda i: i <= first_k))
                sample.token_weights_by_msg = [
                    ws if keep(i) else [] for i, ws in enumerate(sample.token_weights_by_msg)]
                if ctx:
                    sample.teacher_contexts = dict(ctx)
                rec.meta["proposed"] = True
                return float(rt.reward or 0.0), _apply_turn_decay(sample, cfg.turn_decay)
            t0 = time.monotonic()
            try:
                hr = await asyncio.to_thread(
                    hindsight_mod.score, router.for_task(rec.task_id), rt, block, cfg)
            except Exception as e:
                rec.meta["hindsight_error"] = repr(e)[:160]
                rec.meta["proposed"] = False
                return None
            rec.timings["hindsight"] = time.monotonic() - t0
            rec.meta["proposed"] = True
            sample = hindsight_mod.to_train_sample(rt, hr, cfg)
            if ctx:
                sample.teacher_contexts = dict(ctx)
            return float(rt.reward or 0.0), _apply_turn_decay(sample, cfg.turn_decay)

        outs = await asyncio.gather(*(score_redo(*job) for job in redo))
        for (rec, *_), out in zip(redo, outs):
            if out is not None:
                samples.append(out)
                contrib.append(rec)
        for traj in trajs:
            buffer.add(traj)
            if traj.task_id in retry_by_id:
                buffer.add(retry_by_id[traj.task_id])

        # (3) train candidate from proposed tasks; gate-validate; merge/publish.
        if async_trainer is not None:
            # local-teacher samples without a context (e.g. successful first
            # attempts in kl_states="first") would be skipped by the trainer on
            # every draw -- keep them out of the replay buffer entirely
            usable = [(rank, sample, rec) for (rank, sample), rec in zip(samples, contrib)
                      if not (cfg.kl_target and cfg.kl_teacher == "local"
                              and not sample.teacher_contexts)]
            size = async_trainer.add(
                [(rank, sample, bool(rec.success or rec.meta.get("redo_success")))
                 for rank, sample, rec in usable])
            for rec in contrib:
                rec.meta["async_buffered"] = True
            print(f"[async w{w_idx}] +{len(usable)}/{len(samples)} samples -> buffer "
                  f"size {size} (step {async_trainer.step})", flush=True)
        else:
            pending.extend((rank, sample, rec) for (rank, sample), rec in zip(samples, contrib))
        if pending and len(pending) < cfg.min_merge_samples:
            print(f"[merge w{w_idx}] holding {len(pending)}/{cfg.min_merge_samples} samples",
                  flush=True)
        elif pending:
            t0 = time.monotonic()
            # Dose control: total steps per merge = n_samples * epochs. Keep the
            # highest-surprise K (P1.7: update magnitude past the knee wrecks
            # retention; stream candidates must not multiply the dose by n).
            pending.sort(key=lambda p: p[0], reverse=True)
            batch, rest = pending[: cfg.max_candidate_samples], pending[cfg.max_candidate_samples:]
            kept = [s for _, s, _ in batch]
            contrib = [r for _, _, r in batch]
            # with a minimum, the cap only bounds one merge; the rest waits
            pending = rest if cfg.min_merge_samples > 0 else []
            if rest:
                print(f"[dose w{w_idx}] {len(batch) + len(rest)} proposals -> top {len(kept)} "
                      f"by surprise" + (f", {len(rest)} carried" if pending else ""), flush=True)
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

        # (3b) pacing: the stream may not outrun the trainer by more than one
        # window (cfg.async_min_steps_per_window steps). Skips waiting when the
        # trainer cannot make progress (thin buffer, parked thread) -- no
        # deadlock on an all-success window.
        if async_trainer is not None and cfg.async_min_steps_per_window > 0:
            target = w_start_step + cfg.async_min_steps_per_window
            waited = 0.0
            while async_trainer.step < target and async_trainer.waitable():
                await asyncio.sleep(10.0)
                waited += 10.0
            if waited:
                print(f"[pace w{w_idx}] waited {waited:.0f}s for trainer "
                      f"steps {w_start_step}->{async_trainer.step}", flush=True)

        # (4) snapshot hook.
        if async_trainer is not None:
            async_trainer.unpin(serving)
        if on_window is not None:
            on_window(w_idx, w_records)

    if async_trainer is not None:
        # Match CLaaS split semantics: once rollout collection ends, finish
        # age-bounded replay until fewer than B_min eligible samples remain.
        async_trainer.stop(drain=True)
    return records
