"""Hindsight pass: per-token logprob shifts induced by the experience block.

The SAME trajectory is scored twice (with the block injected into the first
user message, and without); delta_t = logp_with - logp_without on the aligned
suffix messages. obs_surprise = mean(max(delta, 0)) over tool-role tokens is
the belief signal; act_gain = mean(delta) over assistant-role tokens is the
action signal. Two prefill-only score calls, no sampling, no teacher model.

With cfg.kl_target the with-block call also returns each position's top-k
distribution: that IS the teacher of context distillation, so the deltas (which
still select the tokens) and the KL target come out of the same request.
"""
from __future__ import annotations

import re

from sediment.config import StreamConfig
from sediment.experience import inject
from sediment.spans import align_suffix, spans_from_lengths
from sediment.types import ExperienceBlock, HindsightResult, SpanScore, TrainSample, Trajectory


def score(
    engine, traj: Trajectory, block: ExperienceBlock, cfg: StreamConfig
) -> HindsightResult:
    """Score traj with/without the block and diff the aligned suffix.

    Uses `engine.score` per-message logprob lists (aligned with sediment.spans
    tokenization) under the trajectory's own adapter. Suffix messages after
    the injection point are paired via align_suffix; only assistant/tool
    messages produce SpanScores. Empty spans yield 0.0 aggregates.
    """
    without = traj.messages
    withm, injected_index = inject(without, block.text)
    dists_w = None
    if cfg.kl_target:
        # raises on token-alignment failure -- never silently degrade to CE,
        # that would turn this arm back into the one it is being compared to
        ids_w, dists_w = engine.score_topk(withm, adapter=traj.adapter, k=cfg.kl_topk)
        logp_w = [[d.get(t, min(d.values()) if d else 0.0) for t, d in zip(ids, ds)]
                  for ids, ds in zip(ids_w, dists_w)]
    else:
        logp_w = engine.score(withm, adapter=traj.adapter)
    logp_o = engine.score(without, adapter=traj.adapter)
    spans_w = spans_from_lengths(withm, [len(x) for x in logp_w])
    spans_o = spans_from_lengths(without, [len(x) for x in logp_o])

    spans: list[SpanScore] = []
    teacher: list[list[dict[int, float]]] = []
    for sw, so in align_suffix(spans_w, spans_o, injected_index):
        if so.role not in ("assistant", "tool"):
            continue
        deltas = [float(a - b) for a, b in zip(logp_w[sw.index], logp_o[so.index])]
        spans.append(SpanScore(msg_idx=so.index, role=so.role, deltas=deltas))
        if dists_w is not None:
            teacher.append(dists_w[sw.index])

    tool = [d for s in spans if s.role == "tool" for d in s.deltas]
    act = [d for s in spans if s.role == "assistant" for d in s.deltas]
    obs_surprise = sum(max(d, 0.0) for d in tool) / len(tool) if tool else 0.0
    act_gain = sum(act) / len(act) if act else 0.0
    return HindsightResult(
        task_id=traj.task_id, spans=spans, obs_surprise=obs_surprise, act_gain=act_gain,
        teacher=teacher if dists_w is not None else None,
    )


STATUS_RE = re.compile(r"error|invalid|fail|exceed|denied|cannot|unable", re.I)


def _error_action_indices(messages) -> set[int]:
    """Indices of assistant messages whose next env message reads as an error."""
    bad: set[int] = set()
    for i, m in enumerate(messages):
        if m.role == "assistant" and i + 1 < len(messages) and messages[i + 1].role in ("tool", "user"):
            if STATUS_RE.search(messages[i + 1].content[:300]):
                bad.add(i)
    return bad


def sft_sample(traj: Trajectory, cfg: StreamConfig) -> TrainSample:
    """Every token of the trajectory's own turns, weight 1.0 -- no residual.

    Used by weight_mode="sft": the trajectory was rolled out WITH the evidence
    block and is stored with it stripped, so this is teacher-student
    distillation over the whole action sequence, including when to stop.
    A one-element weight list is broadcast over the message's span by
    trainer._encode (its documented fallback when the lengths disagree).
    """
    off = {"both": set(), "act": {"tool"}, "obs": {"assistant"}}[cfg.train_channels]
    return TrainSample(
        task_id=traj.task_id, messages=list(traj.messages),
        token_weights_by_msg=[[1.0] if (m.role in ("assistant", "tool")
                                        and m.role not in off) else []
                              for m in traj.messages],
    )


def to_train_sample(traj: Trajectory, hr: HindsightResult, cfg: StreamConfig) -> TrainSample:
    """Per-token training weights from hindsight deltas.

    Scored tool/assistant messages get max(relu(delta), cfg.w_floor) per
    token; every other message (system/prompt/unscored) carries an empty
    list, meaning weight 0.0 for all its tokens. token_weights_by_msg is
    index-parallel to traj.messages.
    """
    by_msg = {s.msg_idx: s for s in hr.spans}
    bad = _error_action_indices(traj.messages) if cfg.gate_error_actions else set()
    off = {"both": set(), "act": {"tool"}, "obs": {"assistant"}}[cfg.train_channels]
    weights: list[list[float]] = []
    for i in range(len(traj.messages)):
        s = by_msg.get(i)
        if s is None:
            weights.append([])
        elif s.role in off:  # channel ablation
            weights.append([0.0] * len(s.deltas))
        elif cfg.weight_mode == "sft":
            # no residual selection at all: the whole turn is the target. Used
            # with serve_experience, where the trajectory IS the teacher's, so
            # there is nothing to select -- only a distribution to match.
            weights.append([1.0] * len(s.deltas))
        elif cfg.signed and s.role == "assistant":  # two gates: +/- (binary: ±1)
            if cfg.weight_mode == "binary":
                weights.append([1.0 if d > cfg.pos_thr else -1.0 if d < -cfg.neg_thr else 0.0
                                for d in s.deltas])
            else:
                weights.append([min(d, cfg.pos_cap) if d > cfg.pos_thr
                                else -min(-d, cfg.neg_cap) if d < -cfg.neg_thr else 0.0
                                for d in s.deltas])
        elif i in bad:  # P1.6 status gate: never reinforce a known-bad action
            weights.append([0.0] * len(s.deltas))
        elif cfg.weight_mode == "binary":
            weights.append([1.0 if d > cfg.gate_thr else 0.0 for d in s.deltas])
        else:
            weights.append([max(max(d, 0.0), cfg.w_floor) for d in s.deltas])
    teacher_by_msg = None
    if hr.teacher is not None:
        by_idx = {s.msg_idx: t for s, t in zip(hr.spans, hr.teacher)}
        teacher_by_msg = [by_idx.get(i, []) for i in range(len(traj.messages))]
    return TrainSample(
        task_id=traj.task_id, messages=list(traj.messages), token_weights_by_msg=weights,
        teacher_by_msg=teacher_by_msg,
    )
