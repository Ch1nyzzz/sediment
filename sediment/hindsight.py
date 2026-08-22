"""Hindsight pass: per-token logprob shifts induced by the experience block.

The SAME trajectory is scored twice (with the block injected into the first
user message, and without); delta_t = logp_with - logp_without on the aligned
suffix messages. obs_surprise = mean(max(delta, 0)) over tool-role tokens is
the belief signal; act_gain = mean(delta) over assistant-role tokens is the
action signal. Two prefill-only score calls, no sampling, no teacher model.
"""
from __future__ import annotations

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
    logp_w = engine.score(withm, adapter=traj.adapter)
    logp_o = engine.score(without, adapter=traj.adapter)
    spans_w = spans_from_lengths(withm, [len(x) for x in logp_w])
    spans_o = spans_from_lengths(without, [len(x) for x in logp_o])

    spans: list[SpanScore] = []
    for sw, so in align_suffix(spans_w, spans_o, injected_index):
        if so.role not in ("assistant", "tool"):
            continue
        deltas = [float(a - b) for a, b in zip(logp_w[sw.index], logp_o[so.index])]
        spans.append(SpanScore(msg_idx=so.index, role=so.role, deltas=deltas))

    tool = [d for s in spans if s.role == "tool" for d in s.deltas]
    act = [d for s in spans if s.role == "assistant" for d in s.deltas]
    obs_surprise = sum(max(d, 0.0) for d in tool) / len(tool) if tool else 0.0
    act_gain = sum(act) / len(act) if act else 0.0
    return HindsightResult(
        task_id=traj.task_id, spans=spans, obs_surprise=obs_surprise, act_gain=act_gain
    )


def to_train_sample(traj: Trajectory, hr: HindsightResult, cfg: StreamConfig) -> TrainSample:
    """Per-token training weights from hindsight deltas.

    Scored tool/assistant messages get max(relu(delta), cfg.w_floor) per
    token; every other message (system/prompt/unscored) carries an empty
    list, meaning weight 0.0 for all its tokens. token_weights_by_msg is
    index-parallel to traj.messages.
    """
    by_msg = {s.msg_idx: s for s in hr.spans}
    weights: list[list[float]] = []
    for i in range(len(traj.messages)):
        s = by_msg.get(i)
        weights.append([max(max(d, 0.0), cfg.w_floor) for d in s.deltas] if s else [])
    return TrainSample(
        task_id=traj.task_id, messages=list(traj.messages), token_weights_by_msg=weights
    )
