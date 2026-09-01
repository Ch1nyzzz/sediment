"""Candidate trainer: hindsight-weighted token CE -> LoRA adapter dir.

cfg.kl_target swaps the objective on the gated tokens from CE to KL between
the student and the memory-conditioned teacher (reverse KL by default),
the teacher being the base model reading the evidence
block (context distillation): the gate still picks the tokens, but instead of
"reproduce this token" the target becomes "reproduce the distribution shift the
memory caused", which is zero wherever the memory did not actually help.

`train_candidate` is the single entry point. cfg.trainer selects "stub"
(metadata only, no heavy deps -- used by all tests) or "torch" (lazy
transformers/peft LoRA fine-tune).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import contextmanager
from typing import Any

from .chat_template import load_tokenizer
from .config import StreamConfig
from .semantic import action_semantic_mask, framing_mask, tool_call_region_mask
from .types import AdapterVersion, TrainSample, UpdateCandidate

_RESIDENT: dict[str, Any] = {}  # process-wide resident base model
LORA_TARGET_MODULES = ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]


def candidate_id_for(task_ids: list[str], parent_name: str) -> str:
    """Short deterministic id from the trained task ids + parent version."""
    h = hashlib.sha256(json.dumps([task_ids, parent_name]).encode()).hexdigest()
    return f"cand-{h[:10]}"


def _weight_stats(sample: TrainSample) -> dict[str, Any]:
    """Per-sample stats over the flattened token weights."""
    flat = [w for ws in sample.token_weights_by_msg for w in ws]
    if not flat:
        return {"task_id": sample.task_id, "mean": 0.0, "max": 0.0, "nonzero": 0}
    # per-channel weight mass (forensics: which channel a poison merge came from)
    mass = {"assistant": 0.0, "tool": 0.0}
    for m, ws in zip(sample.messages, sample.token_weights_by_msg):
        if m.role in mass:
            mass[m.role] += float(sum(ws))
    return {
        "task_id": sample.task_id,
        "mean": float(sum(flat) / len(flat)),
        "max": float(max(flat)),
        "nonzero": int(sum(1 for w in flat if w > 0.0)),
        "mass_act": mass["assistant"],
        "mass_obs": mass["tool"],
        # gate counts (binary mode: exactly the tokens that passed each gate)
        "n_pos": int(sum(1 for w in flat if w > 0.0)),
        "n_neg": int(sum(1 for w in flat if w < 0.0)),
    }


def _effective_micro_batch(n_samples: int, cfg: StreamConfig) -> int:
    if cfg.steps_per_merge <= 0:
        return cfg.micro_batch
    return max(1, -(-n_samples * cfg.epochs // cfg.steps_per_merge))


def train_candidate(
    samples: list[TrainSample],
    parent: AdapterVersion,
    cfg: StreamConfig,
    workdir: str,
) -> UpdateCandidate:
    """Train (or stub out) one update candidate from weighted samples.

    Writes the adapter into ``{workdir}/{candidate_id}/``: stub mode emits
    only ``adapter_meta.json`` (task ids, parent, per-sample weight stats);
    torch mode saves a peft LoRA checkpoint via ``save_pretrained``.
    """
    task_ids = [s.task_id for s in samples]
    cid = candidate_id_for(task_ids, parent.name)
    out_dir = os.path.join(workdir, cid)
    os.makedirs(out_dir, exist_ok=True)
    stats = [_weight_stats(s) for s in samples]
    for st in stats:  # per-sample gate log line (grep "[gates]")
        print(f"[gates] {st['task_id']} n_pos={st.get('n_pos', 0)} n_neg={st.get('n_neg', 0)} "
              f"mass_act={st.get('mass_act', 0.0):.2f}", flush=True)
    if cfg.trainer == "stub":
        train_stats: dict[str, Any] = {"mode": "stub", "n_samples": len(samples), "samples": stats}
    elif cfg.trainer == "torch":
        losses = _train_torch(samples, parent, cfg, out_dir)
        effective_micro_batch = _effective_micro_batch(len(samples), cfg)
        train_stats = {
            "mode": "torch",
            "n_samples": len(samples),
            "samples": stats,
            "losses": losses,
            "effective_micro_batch": effective_micro_batch,
            "optimizer_steps": math.ceil(len(losses) / effective_micro_batch),
        }
    else:
        raise ValueError(f"unknown trainer: {cfg.trainer!r}")
    meta = {"candidate_id": cid, "task_ids": task_ids, "parent": parent.name,
            "samples": stats, "train_stats": train_stats}
    with open(os.path.join(out_dir, "adapter_meta.json"), "w") as f:  # both modes
        json.dump(meta, f, indent=2)
    return UpdateCandidate(
        candidate_id=cid,
        task_ids=task_ids,
        adapter_path=out_dir,
        parent=parent.name,
        train_stats=train_stats,
    )


def _encode(tokenizer, sample: TrainSample, cfg: StreamConfig):
    """(token ids, weights, per-token teacher dists) for one sample.

    The teacher list is all-None unless cfg.kl_target.

    Alignment assumption: ``token_weights_by_msg`` was computed against the
    engine-side tokenization of the same messages (sediment.spans). Here the
    trainer re-derives per-message token spans by applying the chat template
    incrementally over message prefixes (the same construction as
    ``spans.message_spans``, relying on the template's token prefix
    property). When a message's trainer-side span length equals its weight
    list length the weights map 1:1; otherwise the message's mean weight is
    broadcast uniformly over every token of its span (framing tokens
    included). Prompt/system messages carry all-zero weights either way. A
    teacher DISTRIBUTION cannot be broadcast, so kl positions in a misaligned
    message get None and are dropped from the KL mask.
    """
    kl = cfg.kl_target
    sft = cfg.weight_mode == "sft"
    max_seq_len = cfg.max_seq_len
    msgs = [m.to_dict() for m in sample.messages]
    prev: list[int] = []
    ids: list[int] = []
    weights: list[float] = []
    teacher: list = []
    behavior: list[float] = []
    assistant: list[bool] = []
    for i in range(1, len(msgs) + 1):
        # return_dict=False: transformers 5.x defaults to a BatchEncoding here
        # the tokenizer carries the right template (sediment.chat_template)
        toks = list(tokenizer.apply_chat_template(msgs[:i], tokenize=True, return_dict=False))
        if toks[: len(prev)] != prev:
            raise AssertionError(
                f"chat template broke the token prefix property at message {i - 1}; "
                "per-message weight alignment would be invalid"
            )
        span = len(toks) - len(prev)
        ws = sample.token_weights_by_msg[i - 1]
        tch = (sample.teacher_by_msg or [[]] * len(msgs))[i - 1] if kl else []
        mtch = list(tch) if len(tch) == span else [None] * span
        old = ((sample.behavior_logprobs_by_msg or [[]] * len(msgs))[i - 1]
               if sample.behavior_logprobs_by_msg is not None else [])
        # Importance weights require exact token alignment. Unlike scalar
        # message weights, behavior log-probs must never be broadcast.
        mold = [float(x) for x in old] if len(old) == span else [float("nan")] * span
        if len(ws) == span:
            mws = [float(w) for w in ws]
        else:
            mean = float(sum(ws) / len(ws)) if ws else 0.0
            mws = [mean] * span
        span_ids = toks[len(prev):]
        if sft and any(mws):
            # whole message, not just the tool-call interior: the drift the gated
            # modes pay for lives outside the region they credit. Framing tokens
            # still get nothing (their prompt-logprob artefact is 10-21 nats).
            fr = framing_mask(tokenizer, span_ids)
            mws = [w if f else 0.0 for w, f in zip(mws, fr)]
            # (grounded_weight token damping removed 08-31: the 0.2-vs-1.0 CE
            # ablation showed no benefit and the fork-CE main line trains a
            # single decision turn, not whole trajectories)
        elif sample.messages[i - 1].role == "assistant" and any(w != 0 for w in mws):
            # no credit of either sign on chat/template framing tokens; negative
            # credit additionally never on JSON keys or the tool name (semantic)
            content = sample.messages[i - 1].content
            fr = framing_mask(tokenizer, span_ids)
            mws = [w if f else 0.0 for w, f in zip(mws, fr)]
            keep = action_semantic_mask(tokenizer, content, span_ids)  # values / tool name
            if kl:
                # KL has no sign, so BOTH gates feed one mask, under the strict
                # (positive-credit) region rule: KL self-masks unpredictable
                # tokens but not narration, and narration collapse is a measured
                # failure mode of this arm.
                region = tool_call_region_mask(tokenizer, content, span_ids)
                mws = [1.0 if (w != 0 and f and k and r) else 0.0
                       for w, f, k, r in zip(mws, fr, keep, region)]
            else:
                if any(w < 0 for w in mws):
                    mws = [w if (w >= 0 or k) else 0.0 for w, k in zip(mws, keep)]
                if any(w > 0 for w in mws):
                    # positive credit = endorsed DECISIONS only: semantic tokens inside
                    # <tool_call> (never terminal phrases, narration, keys, framing)
                    region = tool_call_region_mask(tokenizer, content, span_ids)
                    mws = [w if (w <= 0 or (k and r)) else 0.0
                           for w, k, r in zip(mws, keep, region)]
        if kl:
            # KL carries its own sign, so every gated token -- either gate, any
            # channel -- becomes one unsigned mask entry that KEEPS its
            # magnitude. With the served top-k teacher a position without a
            # distribution is not a KL position; the local teacher scores
            # every weighted position itself.
            local = cfg.kl_teacher == "local"
            mws = [abs(w) if (w != 0.0 and (t or local)) else 0.0 for w, t in zip(mws, mtch)]
        weights.extend(mws)
        teacher.extend(mtch)
        behavior.extend(mold)
        ids.extend(span_ids)
        assistant.extend([sample.messages[i - 1].role == "assistant"] * span)
        prev = toks
    _encode.last_assistant_mask = assistant[:max_seq_len]  # consumed by _train_torch
    _encode.last_behavior_logprobs = behavior[:max_seq_len]
    return ids[:max_seq_len], weights[:max_seq_len], teacher[:max_seq_len]


def weighted_ce(logits, targets, weights, chunk: int = 2048, ul_lambda: float = 0.0,
                norm_floor: float = 0.0):
    """Weighted token CE with the float32 cast done per chunk (memory-bound
    otherwise: full-length fp32 logits on a 12k-token sequence are ~7GB).

    Negative weights (signed credit) contribute an unlikelihood term
    -log(1 - p_t) scaled by ul_lambda; each sign is normalised by its own mass."""
    import torch

    parts = []
    for s in range(0, targets.shape[1], chunk):
        lg = logits[:, s : s + chunk].float()
        parts.append(
            torch.nn.functional.cross_entropy(
                lg.transpose(1, 2), targets[:, s : s + chunk], reduction="none"
            )
        )
    ce = torch.cat(parts, dim=1)
    pos = weights.clamp_min(0.0)
    loss = (ce * pos).sum() / pos.sum().clamp_min(max(norm_floor, 1e-8))
    neg = (-weights).clamp_min(0.0)
    if ul_lambda > 0.0 and bool((neg > 0).any()):
        p = torch.exp(-ce).clamp(max=1.0 - 1e-6)  # ce = -log p_t
        ul = (-torch.log1p(-p) * neg).sum() / neg.sum().clamp_min(max(norm_floor, 1e-8))
        loss = loss + ul_lambda * ul
    return loss


def sparse_weighted_ce(model, input_ids, weights, norm_floor: float = 0.0):
    """Weighted CE materialising logits ONLY at the weighted positions
    (`logits_to_keep`), for non-negative weights and no anchor: identical to
    weighted_ce on the same weights, without the [seq, vocab] tensor that a
    full-sequence pass needs (the 08-29 RFT arm OOMed on it beside a resident
    service). Row j of the logits predicts token j+1; weights index j."""
    import torch

    nz = (weights[0] != 0.0).nonzero(as_tuple=True)[0]
    if nz.numel() == 0:
        return None
    logits = model(input_ids=input_ids, logits_to_keep=nz).logits[0].float()
    targets = input_ids[0, 1:].index_select(0, nz)
    ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    ww = weights[0].index_select(0, nz)
    return (ce * ww).sum() / ww.sum().clamp_min(max(norm_floor, 1e-8))


def _dpo_spans(tokenizer, prefix, action: str, max_prefix_tokens: int = 9000):
    """(input ids, logits rows, target ids) for one continuation of `prefix`.

    Framing tokens are excluded, so the score covers exactly the action's own
    tokens -- the same span construction the tree critic ranks candidates with
    (scripts/train_tree_critic.py: Critic.encode). None when the template
    breaks the prefix property, the prefix is too long, or nothing is left."""
    pre = list(tokenizer.apply_chat_template(
        [m.to_dict() for m in prefix], tokenize=True, add_generation_prompt=True, return_dict=False))
    full = list(tokenizer.apply_chat_template(
        [m.to_dict() for m in prefix] + [{"role": "assistant", "content": action}],
        tokenize=True, return_dict=False))
    if full[: len(pre)] != pre or len(pre) > max_prefix_tokens:
        return None
    act = full[len(pre):]
    fr = framing_mask(tokenizer, act)
    pos = [len(pre) + i - 1 for i, f in enumerate(fr) if f]
    tgt = [t for t, f in zip(act, fr) if f]
    return (full, pos, tgt) if pos else None


def _trajectory_dpo_spans(tokenizer, messages, max_tokens: int = 12288):
    """(full ids, policy-logit rows, targets) for a complete trajectory.

    Every assistant message is scored under the history actually observed on
    that trajectory. User/tool messages are retained as causal context but are
    not policy outputs. Returning one flattened span makes the DPO log-ratio a
    preference over the complete correction sequence rather than over a fork.
    """
    serial = [m.to_dict() if hasattr(m, "to_dict") else dict(m) for m in messages]
    full = list(tokenizer.apply_chat_template(
        serial, tokenize=True, return_dict=False))
    if len(full) > max_tokens:
        return None
    pos: list[int] = []
    tgt: list[int] = []
    for i, m in enumerate(messages):
        role = m.role if hasattr(m, "role") else m["role"]
        if role != "assistant":
            continue
        content = m.content if hasattr(m, "content") else m["content"]
        span = _dpo_spans(tokenizer, messages[:i], content,
                          max_prefix_tokens=max_tokens)
        if span is None:
            continue
        partial, rows, targets = span
        if full[:len(partial)] != partial:
            return None
        pos.extend(rows)
        tgt.extend(targets)
    return (full, pos, tgt) if pos else None


def _mean_logp(model, spans, device, grad: bool):
    """Mean log pi(action token) over the scored positions; logits are
    materialised only there (`logits_to_keep`), as in sparse_weighted_ce."""
    import torch

    ids, pos, tgt = spans
    input_ids = torch.tensor([ids], device=device)
    p = torch.tensor(pos, device=device)
    t = torch.tensor(tgt, device=device)
    with torch.enable_grad() if grad else torch.no_grad():
        logits = model(input_ids=input_ids, logits_to_keep=p).logits[0].float()
        return logits.log_softmax(-1).gather(-1, t.unsqueeze(-1)).squeeze(-1).mean()


def _margin_token_loss(model, chosen, rejected, device, margin: float):
    """Raw-logit rank loss at the first token where two fork steps differ.

    The two continuations must still have an identical causal prefix at that
    row.  This is what makes ``z_target - z_wrong`` a well-defined comparison
    from one student state, rather than comparing logits from two already
    diverged histories.
    """
    import torch

    c_full, c_pos, c_tgt = chosen
    r_full, r_pos, r_tgt = rejected
    diff = next((i for i, (c, r) in enumerate(zip(c_tgt, r_tgt)) if c != r), None)
    if diff is None:
        return None
    cp, rp = c_pos[diff], r_pos[diff]
    if c_full[:cp + 1] != r_full[:rp + 1]:
        return None
    input_ids = torch.tensor([c_full], device=device)
    row = torch.tensor([cp], device=device)
    logits = model(input_ids=input_ids, logits_to_keep=row).logits[0, 0].float()
    delta = logits[c_tgt[diff]] - logits[r_tgt[diff]] - float(margin)
    return torch.nn.functional.softplus(-delta)


def _suffix_start(chosen_tokens, rejected_tokens):
    """First chosen-token index not shared with the rejected continuation."""
    diff = next(
        (
            i
            for i, (chosen, rejected) in enumerate(
                zip(chosen_tokens, rejected_tokens)
            )
            if chosen != rejected
        ),
        None,
    )
    if diff is not None:
        return diff
    return len(rejected_tokens) if len(chosen_tokens) > len(rejected_tokens) else None


def _suffix_ce_loss(model, chosen, rejected, device):
    """Teacher-forced CE after the chosen/rejected continuation's first fork.

    ``chosen`` is the privileged correction and ``rejected`` is the student's
    erroneous SQL at the identical causal prefix. Shared prefix tokens receive
    no gradient; every corrected token from the first token difference onward
    does. This is InterCode fork CE, not whole-turn CE.
    """
    import torch

    c_full, c_pos, c_tgt = chosen
    _r_full, _r_pos, r_tgt = rejected
    diff = _suffix_start(c_tgt, r_tgt)
    if diff is None:
        return None
    pos = c_pos[diff:]
    tgt = c_tgt[diff:]
    if not pos:
        return None
    input_ids = torch.tensor([c_full], device=device)
    rows = torch.tensor(pos, device=device)
    targets = torch.tensor(tgt, device=device)
    logits = model(input_ids=input_ids, logits_to_keep=rows).logits[0].float()
    return torch.nn.functional.cross_entropy(logits, targets)


def _kl_digit_share(tokenizer, ids: list[int], task_id: str, idx, kl, ce=None) -> str:
    """How much of the gradient mass sits on digit-bearing tokens -- KL vs the
    CE that would have been spent on the very same tokens.

    The decisive diagnostic for this arm. Plain CE put 65% of its gradient on
    random hex ids the student cannot infer (sft_gradient_audit, 08-26); KL
    should put far less, because a teacher reading a leak-free block is just as
    clueless there. A KL share that stays close to the CE share means the
    evidence block leaks instance values into the teacher -- i.e. the compiler
    is load-bearing, not optional.
    """
    pos = idx.tolist()
    digit = [j + 1 < len(ids) and any(c.isdigit() for c in tokenizer.decode([ids[j + 1]]))
             for j in pos]

    def share(v):
        tot = sum(v)
        return sum(x for x, d in zip(v, digit) if d) / max(tot, 1e-9), tot

    kl_share, kl_tot = share([float(x) for x in kl.tolist()])
    n = len(pos)
    line = (f"[kl] {task_id} n_tok={n} kl_mean={kl_tot / max(n, 1):.3f} "
            f"digit_frac={sum(digit) / max(n, 1):.3f} digit_share={kl_share:.3f}")
    if ce is not None:
        ce_share, ce_tot = share([float(x) for x in ce.tolist()])
        line += f" | ce_mean={ce_tot / max(n, 1):.3f} ce_digit_share={ce_share:.3f}"
    return line


def _is_summary(ratio, clip: float) -> str:
    """Compact token-level importance diagnostics for one replay sample."""
    if ratio is None or ratio.numel() == 0 or clip <= 0.0:
        return ""
    r = ratio.float()
    p50 = float(r.quantile(0.50))
    p95 = float(r.quantile(0.95))
    clipped = float((r >= float(clip) * (1.0 - 1e-6)).float().mean())
    ess = float(r.sum().square() / r.square().sum().clamp_min(1e-12))
    return (f" | is_p50={p50:.3f} is_p95={p95:.3f} is_max={float(r.max()):.3f}"
            f" is_clipped={clipped:.3f} is_ess={ess:.1f}/{r.numel()}")


def topk_kl(logits, t_ids, t_logq, mask, targets=None, norm_floor: float = 0.0,
            reverse: bool = False):
    """KL between teacher and student on the teacher's top-k support.

    reverse=False: KL(teacher || student), mode-covering, correct when the
    sequence was sampled from the teacher. reverse=True: KL(student || teacher),
    mode-seeking, the GKD-style on-policy objective -- correct here because at
    inference the model runs prompt-only, so the states are the student's and
    the expectation must be under the student. Forward KL on teacher-sampled
    trajectories measured 0.008 nats (kl_screen 08-26): at its own states the
    student already agrees, so there is nothing there to learn.

    Everything outside the support is grouped into ONE tail bucket; grouping
    can only lose information, so this is a lower bound on the full-vocabulary
    KL -- the objective never over-claims, and the student cannot escape by
    dumping mass on tokens the teacher never scored.

    Only gated positions are materialised in fp32 (a handful per trajectory vs
    the full 12k x 152k logit tensor), so this is far cheaper than weighted_ce.
    Returns (loss, (positions, per_token_kl, per_token_ce)): the CE of the same
    tokens is what plain CE would have spent its gradient on, so the two
    digit-shares are measured on one identical token set.
    """
    import torch

    idx = mask[0].nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return logits.sum() * 0.0, None
    lg = logits[0].index_select(0, idx).float()  # [n, V]
    ti = t_ids[0].index_select(0, idx)
    lq = t_logq[0].index_select(0, idx)
    w = mask[0].index_select(0, idx)
    tg = targets[0].index_select(0, idx) if targets is not None else None
    kl, ce = _topk_kl_per_token(lg, ti, lq, targets=tg, reverse=reverse)
    loss = (kl * w).sum() / w.sum().clamp_min(max(norm_floor, 1e-8))
    return loss, (idx, kl.detach(), ce)


def _topk_kl_per_token(logits, t_ids, t_logq, targets=None, reverse: bool = False,
                       jsd_alpha: float = 0.0):
    """Per-position grouped-tail KL for already-selected logits.

    Keeping this calculation separate lets the 8B trainer request only a
    bounded set of output positions from the causal-LM head.  The formula is
    identical to :func:`topk_kl`; only the order in which positions are
    materialised changes.

    jsd_alpha > 0 switches to the generalized JSD (GKD/SDPO):
        a*KL(teacher||m) + (1-a)*KL(student||m),  m = a*teacher + (1-a)*student
    on the same top-k+tail support. It pulls the student up on teacher-
    preferred tokens the student assigns ~0 mass (RKL's gradient there is
    ~q_student ~ 0) while staying bounded on tokens outside the teacher's
    support (log(q/m) <= -log(1-a)), so the top-k truncation cannot blow it up.
    """
    import torch

    logZ = torch.logsumexp(logits, dim=-1, keepdim=True)
    lp = logits.gather(-1, t_ids) - logZ
    ce = None
    if targets is not None:
        ce = (logZ - logits.gather(-1, targets.unsqueeze(-1))).squeeze(-1).detach()
    q = t_logq.exp()                  # teacher on its own top-k
    p = lp.exp()                      # student there
    q_tail = (1.0 - q.sum(-1)).clamp_min(1e-9)
    p_tail = (1.0 - p.sum(-1)).clamp_min(1e-9)
    if jsd_alpha > 0.0:
        a = float(jsd_alpha)
        la, l1a = math.log(a), math.log(1.0 - a)
        log_m = torch.logaddexp(la + t_logq, l1a + lp)
        log_m_tail = torch.logaddexp(la + torch.log(q_tail), l1a + torch.log(p_tail))
        kl_t = (q * (t_logq - log_m)).sum(-1) + q_tail * (torch.log(q_tail) - log_m_tail)
        kl_s = (p * (lp - log_m)).sum(-1) + p_tail * (torch.log(p_tail) - log_m_tail)
        return a * kl_t + (1.0 - a) * kl_s, ce
    if reverse:
        kl = ((p * (lp - t_logq)).sum(-1)
              + p_tail * (torch.log(p_tail) - torch.log(q_tail)))
    else:
        kl = ((q * (t_logq - lp)).sum(-1)
              + q_tail * (torch.log(q_tail) - torch.log(p_tail)))
    return kl, ce


def _backward_chunked_kl(
    model, input_ids, t_ids, t_logq, mask, targets, cfg: StreamConfig,
    micro_batch: int, chunk: int = 1024, anchorable=None,
):
    """Backpropagate the exact KL objective without full-sequence logits.

    Qwen3 accepts a tensor-valued ``logits_to_keep`` argument.  Each chunk
    therefore runs the same full prefix states but projects only the requested
    positions through the 152k-way LM head.  Chunk numerators share the same
    global denominator, so summing their backward passes is algebraically the
    same gradient as one monolithic loss while avoiding the 4--5 GiB logits
    allocation that prevents 8B training beside the resident service.
    """
    import torch

    selected = mask[0].nonzero(as_tuple=True)[0]
    selected_mass = float(mask[0].index_select(0, selected).sum())
    selected_den = max(selected_mass, cfg.w_norm_floor, 1e-8)
    main_value = 0.0
    dbg_kl = []
    dbg_ce = []

    for start in range(0, selected.numel(), chunk):
        pos = selected[start:start + chunk]
        logits = model(input_ids=input_ids, logits_to_keep=pos).logits[0].float()
        ti = t_ids[0].index_select(0, pos)
        lq = t_logq[0].index_select(0, pos)
        tg = targets[0].index_select(0, pos)
        weight = mask[0].index_select(0, pos)
        kl, ce = _topk_kl_per_token(
            logits, ti, lq, targets=tg, reverse=cfg.kl_reverse,
            jsd_alpha=cfg.kl_jsd_alpha,
        )
        numerator = (kl * weight).sum()
        (numerator / selected_den / micro_batch).backward()
        main_value += float(numerator.detach()) / selected_den
        dbg_kl.append(kl.detach().cpu())
        dbg_ce.append(ce.detach().cpu())
        del logits, kl, ce, numerator

    anchor_value = _backward_anchor(model, input_ids, mask, cfg, micro_batch, anchorable=anchorable)

    debug = None
    if selected.numel():
        debug = (
            selected.detach().cpu(),
            torch.cat(dbg_kl) if dbg_kl else torch.empty(0),
            torch.cat(dbg_ce) if dbg_ce else torch.empty(0),
        )
    return main_value + anchor_value, debug


def _backward_anchor(model, input_ids, mask, cfg: StreamConfig, micro_batch: int,
                     chunk: int = 2048, anchorable=None) -> float:
    """Forward KL(base top-k || student) on the positions the objective does not
    teach (mask == 0; with cfg.anchor_assistant_only also restricted to
    `anchorable` = assistant-token positions), backpropagated chunk by chunk.
    Returns the scaled value."""
    import torch

    anchor_value = 0.0
    if cfg.anchor_kl_coef <= 0.0:
        return anchor_value
    free_mask = mask[0] == 0.0
    if cfg.anchor_assistant_only and anchorable is not None:
        free_mask = free_mask & anchorable
    free = free_mask.nonzero(as_tuple=True)[0]
    free_den = max(float(free.numel()), 1e-8)
    for start in range(0, free.numel(), chunk):
        pos = free[start:start + chunk]
        with torch.no_grad(), model.disable_adapter():
            base_logits = model(input_ids=input_ids, logits_to_keep=pos).logits[0].float()
            values, anchor_ids = base_logits.topk(cfg.kl_topk, dim=-1)
            anchor_logq = values - torch.logsumexp(base_logits, dim=-1, keepdim=True)
        del base_logits, values

        logits = model(input_ids=input_ids, logits_to_keep=pos).logits[0].float()
        anchor_kl, _ = _topk_kl_per_token(
            logits, anchor_ids, anchor_logq, reverse=False
        )
        numerator = anchor_kl.sum()
        scaled = cfg.anchor_kl_coef * numerator / free_den
        (scaled / micro_batch).backward()
        anchor_value += cfg.anchor_kl_coef * float(numerator.detach()) / free_den
        del logits, anchor_ids, anchor_logq, anchor_kl, numerator, scaled
    return anchor_value


def _context_ids(tokenizer, sample: TrainSample, block_text: str,
                 student_ids: list[int]) -> tuple[list[int], int]:
    """Token ids of the block-injected view of `sample.messages` and the offset
    that maps a student position after the first user message to the same
    token in that view. The block only lengthens the first user message, so
    the suffix tokenisation must be identical -- verified, never assumed."""
    from .experience import inject

    withm, inj = inject(list(sample.messages), block_text)
    msgs_s = [m.to_dict() for m in sample.messages]
    msgs_t = [m.to_dict() for m in withm]
    s_pre = list(tokenizer.apply_chat_template(msgs_s[: inj + 1], tokenize=True, return_dict=False))
    t_pre = list(tokenizer.apply_chat_template(msgs_t[: inj + 1], tokenize=True, return_dict=False))
    t_all = list(tokenizer.apply_chat_template(msgs_t, tokenize=True, return_dict=False))
    off = len(t_pre) - len(s_pre)
    n = min(len(t_all) - len(t_pre), len(student_ids) - len(s_pre))
    if n <= 0 or t_all[len(t_pre):len(t_pre) + n] != student_ids[len(s_pre):len(s_pre) + n]:
        raise RuntimeError("teacher context changed the suffix tokenisation; "
                           "cannot align the KL positions")
    return t_all, off


@contextmanager
def _swapped_trainable(model, state):
    """Temporarily load ``state`` into the matching (LoRA) parameters,
    restoring the originals on exit.

    Callers must keep any grad-requiring forward OUTSIDE the swap: linear
    layers save their weight tensors for backward, so swapping between a
    forward and its backward would silently corrupt gradients.
    """
    import torch

    saved = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in state:
                saved[n] = p.detach().clone()
                p.data.copy_(state[n])
    try:
        yield
    finally:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in saved:
                    p.data.copy_(saved[n])


def _pack_bins(entries, max_len: int, max_rows: int = 4096) -> list[list[int]]:
    """Greedy first-fit binning. Cost = max(student length, longest teacher
    view), so one bin's total bounds both the packed student sequence and
    every packed teacher sequence; max_rows additionally caps the KL rows per
    bin (the [rows, vocab] fp32 tensors are the memory ceiling)."""
    sums, rows, bins = [], [], []
    for i, e in enumerate(entries):
        cost = min(max_len, max(len(e["ids"]),
                                max((len(c) for c, _ in e["ctx"].values()), default=0)))
        r = e["n_sel"]
        for k in range(len(bins)):
            if sums[k] + cost <= max_len and rows[k] + r <= max_rows:
                bins[k].append(i)
                sums[k] += cost
                rows[k] += r
                break
        else:
            bins.append([i])
            sums.append(cost)
            rows.append(r)
    return bins


def _local_divergence_per_token(logp, target, cfg: StreamConfig):
    """Per-row local-teacher divergence used by packed and unpacked replay.

    For JSD, ``kl_student_topk`` is deliberately interpreted as TEACHER
    top-k: the privileged teacher's modes must remain in the support even when
    the student currently assigns them almost no probability.  This is the
    CLaaS/SDPO-style installation regime.  The non-JSD KL path preserves the
    historical student-support top-k behavior.
    """
    import torch

    if cfg.kl_jsd_alpha > 0.0:
        a = float(cfg.kl_jsd_alpha)
        if not 0.0 < a < 1.0:
            raise ValueError(f"kl_jsd_alpha must be in (0, 1), got {a}")
        if cfg.kl_student_topk > 0:
            k = min(cfg.kl_student_topk, target.shape[-1])
            support = target.detach().topk(k, dim=-1).indices
            kl, _ = _topk_kl_per_token(
                logp, support, target.gather(-1, support), jsd_alpha=a
            )
            return kl
        logm = torch.logaddexp(math.log(a) + target,
                               math.log(1.0 - a) + logp)
        return (a * (target.exp() * (target - logm)).sum(-1)
                + (1.0 - a) * (logp.exp() * (logp - logm)).sum(-1))
    if cfg.kl_student_topk > 0:
        k = min(cfg.kl_student_topk, logp.shape[-1])
        support = logp.detach().topk(k, dim=-1).indices
        lp_k = logp.gather(-1, support)
        lq_k = target.gather(-1, support)
        p_tail = (1.0 - lp_k.exp().sum(-1)).clamp_min(1e-9)
        q_tail = (1.0 - lq_k.exp().sum(-1)).clamp_min(1e-9)
        if cfg.kl_reverse:
            return ((lp_k.exp() * (lp_k - lq_k)).sum(-1)
                    + p_tail * (p_tail.log() - q_tail.log()))
        return ((lq_k.exp() * (lq_k - lp_k)).sum(-1)
                + q_tail * (q_tail.log() - p_tail.log()))
    if cfg.kl_reverse:
        return (logp.exp() * (logp - target)).sum(-1)
    return (target.exp() * (target - logp)).sum(-1)


def _backward_local_kl_packed(model, entries, cfg: StreamConfig, micro_batch: int,
                              ema_state=None, row_chunk: int = 512):
    """Packed variant of _backward_local_kl: ONE trunk forward per bin for the
    student, one per teacher-context pack, one for the EMA bare view --
    instead of one forward per sample per 256-position chunk. Samples are
    concatenated with per-segment position_ids; flash-attention-2 derives the
    segment boundaries from the position resets, so tokens never attend
    across samples. Per-sample loss normalization, student top-k tail
    buckets, EMA teacher, turn decay (baked into the weights) and replay
    IS-weights are identical to the unpacked path. poe_conflict and the
    anchor are not supported here.

    entries: [{task_id, sample, ids, w [1,T-1] on device, ctx{name:(cids,off)},
    n_sel}]. Returns [(task_id, loss_value, dbg, ctx_names, ids)].
    """
    import torch

    if cfg.poe_conflict != "none":
        raise ValueError("pack_samples does not support poe_conflict")
    if getattr(model.config, "_attn_implementation", "") != "flash_attention_2":
        raise RuntimeError("pack_samples requires attn_implementation=flash_attention_2")
    device = next(model.parameters()).device
    coef = {"donor": float(cfg.poe_alpha), "failure": float(cfg.poe_beta)}
    results = []
    for bin_idx in _pack_bins(entries, cfg.pack_max_len):
        segs = [entries[i] for i in bin_idx]
        sel_loc = [(e["w"][0] != 0).nonzero(as_tuple=True)[0] for e in segs]
        keep = [k for k, sl in enumerate(sel_loc) if sl.numel()]
        if not keep:
            continue
        segs = [segs[k] for k in keep]
        sel_loc = [sel_loc[k] for k in keep]
        s_lens = [len(e["ids"]) for e in segs]
        s_starts, acc = [], 0
        for L in s_lens:
            s_starts.append(acc)
            acc += L
        pack = torch.tensor([[t for e in segs for t in e["ids"]]], device=device)
        pos_ids = torch.cat([torch.arange(L, device=device) for L in s_lens]).unsqueeze(0)
        counts = [int(sl.numel()) for sl in sel_loc]
        bounds = [0]
        for c in counts:
            bounds.append(bounds[-1] + c)
        sel_glob = torch.cat([st + sl for st, sl in zip(s_starts, sel_loc)])
        tgt = pack[0].index_select(0, sel_glob + 1)

        def _ctx_target(base_rows):
            """log q* rows (same order as sel_glob) from packed teacher views."""
            target = base_rows.clone()
            for name in sorted({n for e in segs for n in e["ctx"]}):
                parts = [(k, segs[k]["ctx"][name]) for k in range(len(segs))
                         if name in segs[k]["ctx"]]
                t_lens = [len(c) for _, (c, _) in parts]
                t_ids = torch.tensor([[t for _, (c, _) in parts for t in c]], device=device)
                t_pos = torch.cat([torch.arange(L, device=device) for L in t_lens]).unsqueeze(0)
                t_starts, a = [], 0
                for L in t_lens:
                    t_starts.append(a)
                    a += L
                t_sel = torch.cat([t_starts[j] + parts[j][1][1] + sel_loc[parts[j][0]]
                                   for j in range(len(parts))])
                lq = model(input_ids=t_ids, position_ids=t_pos,
                           logits_to_keep=t_sel).logits[0].float().log_softmax(-1)
                o = 0
                for j in range(len(parts)):
                    k = parts[j][0]
                    sl = slice(bounds[k], bounds[k + 1])
                    target[sl] = target[sl] + coef.get(name, 1.0) * (lq[o:o + counts[k]]
                                                                     - base_rows[sl])
                    o += counts[k]
            return target.log_softmax(-1)

        target = None
        with torch.no_grad():
            if ema_state is not None:
                with _swapped_trainable(model, ema_state):
                    base_rows = model(input_ids=pack, position_ids=pos_ids,
                                      logits_to_keep=sel_glob).logits[0].float().log_softmax(-1)
                    target = _ctx_target(base_rows)
        logits = model(input_ids=pack, position_ids=pos_ids,
                       logits_to_keep=sel_glob).logits[0]
        if target is None:
            with torch.no_grad():
                target = _ctx_target(logits.detach().float().log_softmax(-1))
        # fp32 KL math in row chunks over the single logits tensor
        kl_parts, ce_parts = [], []
        n_rows = logits.shape[0]
        for r in range(0, n_rows, row_chunk):
            lp = logits[r:r + row_chunk].float().log_softmax(-1)
            tq = target[r:r + row_chunk]
            tg_r = tgt[r:r + row_chunk]
            ce_parts.append((-lp.gather(-1, tg_r.unsqueeze(-1)).squeeze(-1)).detach())
            kl_parts.append(_local_divergence_per_token(lp, tq, cfg))
        kl = torch.cat(kl_parts)
        ce = torch.cat(ce_parts)
        bin_losses = []
        for k in range(len(segs)):
            sl = slice(bounds[k], bounds[k + 1])
            w_sel = segs[k]["w"][0].index_select(0, sel_loc[k])
            ratio = torch.ones_like(kl[sl])
            if cfg.replay_is_clip > 0.0:
                old = segs[k]["old_logp"][0].index_select(0, sel_loc[k])
                if not bool(torch.isfinite(old).all()):
                    raise ValueError(
                        f"{segs[k]['task_id']} lacks collection-time token logprobs; "
                        "cannot apply replay IS"
                    )
                log_ratio = (-ce[sl] - old).clamp(max=math.log(float(cfg.replay_is_clip)))
                ratio = log_ratio.exp()
            den = max(float(w_sel.sum()), cfg.w_norm_floor, 1e-8)
            num = (kl[sl] * w_sel * ratio).sum() * float(segs[k]["sample"].loss_scale)
            bin_losses.append(num / den)
            results.append((segs[k]["task_id"], float(num.detach()) / den,
                            (sel_loc[k].cpu(), kl[sl].detach().cpu(), ce[sl].cpu(),
                             ratio.detach().cpu()),
                            ",".join(sorted(segs[k]["ctx"])), segs[k]["ids"]))
        (torch.stack(bin_losses).sum() / micro_batch).backward()
        del logits, target, kl, ce
    return results


def _local_teacher_target(model, ctx, pos, tg, base, coef, cfg):
    """log q* from the context-teacher residuals, relative to ``base``.

    ``base`` is the bare-view forward under the TEACHER's weights: the
    student's own logp when the teacher is zero-lag, the EMA copy's when
    cfg.kl_teacher_ema > 0 (then, with a single context of coef 1, the target
    reduces to exactly the EMA teacher conditioned on that context). Must be
    called under torch.no_grad(). Returns (log-softmaxed target, n_conflict).
    """
    import torch

    n_conflict = 0
    deltas = {}
    for name, (cids, off) in ctx.items():
        lq = model(input_ids=cids, logits_to_keep=pos + off).logits[0].float().log_softmax(-1)
        deltas[name] = lq - base
        del lq
    target = base.clone()
    for name, d in deltas.items():
        c = torch.full((pos.numel(),), coef.get(name, 1.0), device=base.device)
        if cfg.poe_conflict == "failure_wins" and name == "donor" and "failure" in deltas:
            d_m = d.gather(-1, tg.unsqueeze(-1)).squeeze(-1)
            d_f = deltas["failure"].gather(-1, tg.unsqueeze(-1)).squeeze(-1)
            conflict = (d_m * d_f < 0) & (d_m.abs() > 0.25) & (d_f.abs() > 0.25)
            n_conflict += int(conflict.sum())
            c = torch.where(conflict, torch.zeros_like(c), c)
        target = target + c.unsqueeze(-1) * d
    return target.log_softmax(-1), n_conflict


def _backward_local_kl(
    model, input_ids, ctx: dict[str, tuple], mask, targets, cfg: StreamConfig,
    micro_batch: int, chunk: int = 256, anchorable=None, ema_state=None,
    loss_scale: float = 1.0, behavior_logp=None,
):
    """KL between the student and the logit-space combination of the local
    context teachers, at the masked positions: exact full-vocabulary by
    default, student top-k + grouped tail when cfg.kl_student_topk > 0; the
    teacher forwards run under the EMA copy when ``ema_state`` is given.
    JSD is supported both exactly and on the privileged teacher's top-k.

    ctx: name -> (ids [1, T_c], offset). For each chunk of selected student
    positions `pos`, every teacher view is forwarded (no grad, same weights)
    at `pos + offset`; the residual of view c is
        delta_c = log q_c - log p        (p = the student, detached)
    and the target is
        log q* = log_softmax(log p + sum_c coef_c * delta_c),
    coef = poe_alpha (donor) / poe_beta (failure). One context with coef 1
    is plain full-vocab KL to that teacher. poe_conflict="failure_wins" zeroes
    the donor coefficient at positions where, on the observed token, the donor
    residual and the failure residual disagree in sign (both > 0.25 nats).
    Returns (value, (positions, per_token_kl, per_token_ce, n_conflict)).
    """
    import torch

    selected = mask[0].nonzero(as_tuple=True)[0]
    selected_mass = float(mask[0].index_select(0, selected).sum())
    den = max(selected_mass, cfg.w_norm_floor, 1e-8)
    coef = {"donor": float(cfg.poe_alpha), "failure": float(cfg.poe_beta)}
    main_value = 0.0
    n_conflict = 0
    dbg_kl, dbg_ce, dbg_ratio = [], [], []
    for start in range(0, selected.numel(), chunk):
        pos = selected[start:start + chunk]
        tg = targets[0].index_select(0, pos)
        weight = mask[0].index_select(0, pos)
        if ema_state is not None:
            # Teacher passes run under the EMA weights and BEFORE the student's
            # grad forward: the swap must never sit between the student's
            # forward and its backward (see _swapped_trainable).
            with torch.no_grad(), _swapped_trainable(model, ema_state):
                base = model(input_ids=input_ids,
                             logits_to_keep=pos).logits[0].float().log_softmax(-1)
                target, nc = _local_teacher_target(model, ctx, pos, tg, base, coef, cfg)
            n_conflict += nc
            logits = model(input_ids=input_ids, logits_to_keep=pos).logits[0].float()
            logp = logits.log_softmax(-1)
        else:
            logits = model(input_ids=input_ids, logits_to_keep=pos).logits[0].float()
            logp = logits.log_softmax(-1)
            with torch.no_grad():
                target, nc = _local_teacher_target(
                    model, ctx, pos, tg, logp.detach(), coef, cfg)
            n_conflict += nc
        kl = _local_divergence_per_token(logp, target, cfg)
        ce = (-logp.gather(-1, tg.unsqueeze(-1)).squeeze(-1)).detach()
        ratio = torch.ones_like(kl)
        if cfg.replay_is_clip > 0.0:
            if behavior_logp is None:
                raise ValueError("collection-time token logprobs are required for replay IS")
            old = behavior_logp[0].index_select(0, pos)
            if not bool(torch.isfinite(old).all()):
                raise ValueError("supervised token lacks a collection-time behavior logprob")
            log_ratio = (-ce - old).clamp(max=math.log(float(cfg.replay_is_clip)))
            ratio = log_ratio.exp()
        numerator = (kl * weight * ratio).sum() * loss_scale
        (numerator / den / micro_batch).backward()
        main_value += float(numerator.detach()) / den
        dbg_kl.append(kl.detach().cpu())
        dbg_ce.append(ce.cpu())
        dbg_ratio.append(ratio.detach().cpu())
        del logits, logp, target, kl, ce, numerator
    anchor_value = _backward_anchor(model, input_ids, mask, cfg, micro_batch, anchorable=anchorable)
    debug = None
    if selected.numel():
        debug = (selected.detach().cpu(), torch.cat(dbg_kl), torch.cat(dbg_ce),
                 n_conflict, torch.cat(dbg_ratio))
    return main_value + anchor_value, debug


def base_topk(model, input_ids, k: int, chunk: int = 512):
    """(ids, log q) of the BASE model's top-k at every position, LoRA disabled.

    The anchor target. Chunked so the fp32 cast never materialises a full
    [seq, vocab] tensor, and taken under no_grad -- the anchor pulls the
    student toward the base, never the other way round.
    """
    import torch

    with torch.no_grad(), model.disable_adapter():
        logits = model(input_ids=input_ids).logits[:, :-1]
        ids, logq = [], []
        for s in range(0, logits.shape[1], chunk):
            lg = logits[:, s:s + chunk].float()
            v, i = lg.topk(k, dim=-1)
            logq.append(v - torch.logsumexp(lg, dim=-1, keepdim=True))
            ids.append(i)
    return torch.cat(ids, dim=1), torch.cat(logq, dim=1)


def _teacher_tensors(teacher, weights, k: int, device):
    """Dense [1, T, k] (token_ids, logprobs) from the per-position top-k dicts.

    Unselected positions keep id 0 / logprob -inf; they are masked out anyway.
    """
    import torch

    ids = [[0] * k for _ in weights]
    logq = [[-1e9] * k for _ in weights]
    for j, (w, d) in enumerate(zip(weights, teacher)):
        if not w or not d:
            continue
        for c, (tid, lp) in enumerate(sorted(d.items(), key=lambda kv: -kv[1])[:k]):
            ids[j][c], logq[j][c] = int(tid), float(lp)
    return (torch.tensor([ids], dtype=torch.long, device=device),
            torch.tensor([logq], dtype=torch.float32, device=device))


def _train_torch(
    samples: list[TrainSample],
    parent: AdapterVersion,
    cfg: StreamConfig,
    out_dir: str,
) -> list[float]:
    """LoRA fine-tune with per-token weighted CE; returns per-sample-pass losses."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(cfg.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Resident base: loaded once per process and reused across windows (a 4B
    # reload cost 30-60 s per window). The LoRA is (re)attached per candidate
    # and detached (peft unload) afterwards so the base stays pristine.
    resident_key = f"{cfg.model}|pack={bool(cfg.pack_samples)}"
    if _RESIDENT.get("name") != resident_key:
        _RESIDENT.clear()
        kwargs = {"attn_implementation": "flash_attention_2"} if cfg.pack_samples else {}
        base = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=torch.bfloat16, **kwargs)
        base.config.use_cache = False
        base.to(device)
        _RESIDENT.update(name=resident_key, model=base)
    base = _RESIDENT["model"]
    if parent.path is not None:
        model = PeftModel.from_pretrained(base, parent.path, is_trainable=True)
    else:
        lora = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.lr)

    # SDPO-style EMA self-teacher (kl_teacher_ema > 0): the local-teacher
    # forwards run under a slow EMA copy of the LoRA params instead of the
    # student's instantaneous weights, persisted across merges so the lag is
    # meaningful at 1-4 optimizer steps per merge (CLaaS Table 4: 0.01).
    from pathlib import Path

    state_root = Path(cfg.registry_dir).parent
    # Persistent optimizer state (cfg.persist_opt_state): Adam moments resume
    # instead of cold-starting every merge. Loading casts state onto the params'
    # device; an architecture change fails the load loudly, by design.
    opt_path = state_root / "opt_state.pt"
    if cfg.persist_opt_state and opt_path.exists():
        opt.load_state_dict(torch.load(opt_path, map_location="cpu"))

    ema_state = None
    ema_path = None
    if cfg.kl_target and cfg.kl_teacher == "local" and cfg.kl_teacher_ema > 0.0:
        ema_path = state_root / "teacher_ema.pt"
        trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if ema_path.exists():
            saved = torch.load(ema_path, map_location="cpu")
            ema_state = {n: saved[n].to(device=p.device, dtype=p.dtype)
                         for n, p in trainable.items()}
        else:
            ema_state = {n: p.detach().clone() for n, p in trainable.items()}

    def _optim_step():
        opt.step()
        opt.zero_grad()
        if ema_state is not None:
            beta = float(cfg.kl_teacher_ema)
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in ema_state:
                        ema_state[n].mul_(1.0 - beta).add_(p.detach(), alpha=beta)

    losses: list[float] = []
    step = 0
    # Fixed dose: accumulate all sample passes into K optimizer steps.
    micro_batch = _effective_micro_batch(len(samples), cfg)
    # Preference-pair references are frozen before the first optimizer step.
    # Historical critic DPO uses the parent.  Preference DPO may instead request the
    # stable adapter-disabled base so replay age does not silently move the
    # reference between buffer draws.  Margin-token ranking has no reference.
    ref: dict[int, tuple] = {}
    for i, sample in enumerate(samples):
        if ((sample.rejected is None and sample.rejected_messages is None)
                or cfg.fork_objective in ("ce_suffix", "margin_token")):
            continue
        if sample.rejected_messages is not None:
            sc = _trajectory_dpo_spans(tokenizer, sample.messages, cfg.max_seq_len)
            sr = _trajectory_dpo_spans(tokenizer, sample.rejected_messages,
                                       cfg.max_seq_len)
        else:
            sc = _dpo_spans(tokenizer, sample.messages[:-1], sample.messages[-1].content)
            sr = _dpo_spans(tokenizer, sample.messages[:-1], sample.rejected)
        if sc is None or sr is None:
            print(f"[dpo] {sample.task_id} unscorable pair; skipped", flush=True)
            continue
        if cfg.dpo_reference == "parent":
            rc = float(_mean_logp(model, sc, device, False))
            rr = float(_mean_logp(model, sr, device, False))
        elif cfg.dpo_reference == "base":
            if not hasattr(model, "disable_adapter"):
                raise RuntimeError("dpo_reference='base' requires a PEFT model")
            with model.disable_adapter():
                rc = float(_mean_logp(model, sc, device, False))
                rr = float(_mean_logp(model, sr, device, False))
        else:
            raise ValueError(f"unknown dpo_reference: {cfg.dpo_reference!r}")
        ref[i] = (sc, sr, rc, rr)
    packed_done: set[int] = set()
    if cfg.pack_samples and cfg.kl_target and cfg.kl_teacher == "local":
        assert cfg.anchor_kl_coef == 0.0, "pack_samples does not support the anchor"
        entries = []
        for i, sample in enumerate(samples):
            if sample.rejected is not None or sample.rejected_messages is not None:
                continue
            ids, ws, tch = _encode(tokenizer, sample, cfg)
            old_logp = list(_encode.last_behavior_logprobs)
            packed_done.add(i)
            if len(ids) < 2 or not any(w_ != 0.0 for w_ in ws):
                continue
            ctx = {}
            for name, text in (sample.teacher_contexts or {}).items():
                if text:
                    cids, off = _context_ids(tokenizer, sample, text, ids)
                    ctx[name] = (cids, off)
            if not ctx:
                print(f"[poe] {sample.task_id} no teacher context; skipped", flush=True)
                continue
            w_t = torch.tensor([ws[1:]], dtype=torch.float32, device=device)
            old_t = torch.tensor([old_logp[1:]], dtype=torch.float32, device=device)
            entries.append({"task_id": sample.task_id, "sample": sample, "ids": ids,
                            "w": w_t, "old_logp": old_t, "ctx": ctx,
                            "n_sel": int((w_t[0] != 0).sum())})
        for _ in range(cfg.epochs):
            for task_id, loss_value, dbg, names, ids_ in _backward_local_kl_packed(
                    model, entries, cfg, micro_batch, ema_state=ema_state):
                losses.append(loss_value)
                if dbg is not None:
                    print(_kl_digit_share(tokenizer, ids_, task_id, *dbg[:3])
                          + f" | ctx={names} packed"
                          + _is_summary(dbg[3], cfg.replay_is_clip), flush=True)
                step += 1
                if step % micro_batch == 0:
                    _optim_step()
    for _ in range(cfg.epochs):
        for i, sample in enumerate(samples):
            if i in packed_done:
                continue
            if sample.rejected is not None:
                if cfg.fork_objective == "ce_suffix":
                    sc = _dpo_spans(tokenizer, sample.messages[:-1],
                                    sample.messages[-1].content)
                    sr = _dpo_spans(tokenizer, sample.messages[:-1], sample.rejected)
                    loss = (None if sc is None or sr is None else
                            _suffix_ce_loss(model, sc, sr, device))
                    if loss is None:
                        print(f"[ce-suffix] {sample.task_id} has no corrected token "
                              "suffix; skipped", flush=True)
                        continue
                    (loss / micro_batch).backward()
                    losses.append(float(loss.detach()))
                    step += 1
                    if step % micro_batch == 0:
                        _optim_step()
                    continue
                if cfg.fork_objective == "margin_token":
                    sc = _dpo_spans(tokenizer, sample.messages[:-1],
                                    sample.messages[-1].content)
                    sr = _dpo_spans(tokenizer, sample.messages[:-1], sample.rejected)
                    loss = (None if sc is None or sr is None else
                            _margin_token_loss(model, sc, sr, device,
                                               cfg.margin_token_m))
                    if loss is None:
                        print(f"[margin-token] {sample.task_id} has no common-prefix "
                              "token divergence; skipped", flush=True)
                        continue
                    (loss / micro_batch).backward()
                    losses.append(float(loss.detach()))
                    step += 1
                    if step % micro_batch == 0:
                        _optim_step()
                    continue
                if i not in ref:
                    continue
                sc, sr, rc, rr = ref[i]
                # d(a) = mean log pi_theta(a) - mean log pi_parent(a);
                # L = -log sigmoid(beta * (d(chosen) - d(rejected)))
                margin = cfg.dpo_beta * ((_mean_logp(model, sc, device, True) - rc)
                                         - (_mean_logp(model, sr, device, True) - rr))
                loss = torch.nn.functional.softplus(-margin)
                (loss / micro_batch).backward()
                losses.append(float(loss.detach()))
                step += 1
                if step % micro_batch == 0:
                    _optim_step()
                continue
            ids, ws, tch = _encode(tokenizer, sample, cfg)
            old_logp = list(_encode.last_behavior_logprobs)
            if len(ids) < 2 or not any(w != 0.0 for w in ws):
                continue
            input_ids = torch.tensor([ids], device=device)
            w = torch.tensor([ws[1:]], dtype=torch.float32, device=device)
            # position j (logits row) predicts token j+1: anchorable follows the target token
            anchorable = torch.tensor(_encode.last_assistant_mask[1:], dtype=torch.bool, device=device)
            if cfg.kl_target and cfg.kl_teacher == "local":
                ctx = {}
                for name, text in (sample.teacher_contexts or {}).items():
                    if text:
                        cids, off = _context_ids(tokenizer, sample, text, ids)
                        ctx[name] = (torch.tensor([cids], device=device), off)
                if not ctx:
                    print(f"[poe] {sample.task_id} no teacher context; skipped", flush=True)
                    continue
                behavior_t = (torch.tensor([old_logp[1:]], dtype=torch.float32, device=device)
                              if cfg.replay_is_clip > 0.0 else None)
                loss_value, dbg = _backward_local_kl(
                    model, input_ids, ctx, w, input_ids[:, 1:], cfg, micro_batch,
                    anchorable=anchorable, ema_state=ema_state,
                    loss_scale=float(sample.loss_scale), behavior_logp=behavior_t,
                )
                if dbg is not None:
                    print(_kl_digit_share(tokenizer, ids, sample.task_id, *dbg[:3])
                          + f" | ctx={','.join(sorted(ctx))} conflict_dropped={dbg[3]}"
                          + _is_summary(dbg[4], cfg.replay_is_clip), flush=True)
            elif cfg.kl_target:
                # weights/teacher at position j predict token j, i.e. logits[j-1]
                t_ids, t_logq = _teacher_tensors(tch[1:], ws[1:], cfg.kl_topk, device)
                loss_value, dbg = _backward_chunked_kl(
                    model, input_ids, t_ids, t_logq, w, input_ids[:, 1:], cfg,
                    micro_batch, anchorable=anchorable,
                )
                if dbg is not None:
                    print(_kl_digit_share(tokenizer, ids, sample.task_id, *dbg), flush=True)
            elif cfg.anchor_kl_coef <= 0.0 and not bool((w < 0).any()):
                loss = sparse_weighted_ce(model, input_ids, w, norm_floor=cfg.w_norm_floor)
                if loss is None:
                    continue
                (loss / micro_batch).backward()
                loss_value = float(loss.detach())
            else:
                logits = model(input_ids=input_ids).logits[:, :-1]
                loss = weighted_ce(logits, input_ids[:, 1:], w, ul_lambda=cfg.ul_lambda,
                                   norm_floor=cfg.w_norm_floor)
                if cfg.anchor_kl_coef > 0.0:
                    # hold the distribution the objective does NOT teach: the gated
                    # modes credit ~20 of ~4000 positions and pay for their loss with
                    # 0.3 nats of anti-repetition prior per merge (policy_shift.py)
                    a_ids, a_logq = base_topk(model, input_ids, cfg.kl_topk)
                    free = (w == 0.0).float()
                    if float(free.sum()) > 0:
                        anchor, _ = topk_kl(logits, a_ids, a_logq, free)
                        loss = loss + cfg.anchor_kl_coef * anchor
                (loss / micro_batch).backward()
                loss_value = float(loss.detach())
            losses.append(loss_value)
            step += 1
            if step % micro_batch == 0:
                _optim_step()
    if step % micro_batch != 0:
        _optim_step()
    if ema_state is not None:
        torch.save({n: t.detach().cpu() for n, t in ema_state.items()}, ema_path)
    if cfg.persist_opt_state:
        torch.save(opt.state_dict(), opt_path)
    model.save_pretrained(out_dir)
    _RESIDENT["model"] = model.unload()  # strip LoRA layers, keep the base resident
    return losses
