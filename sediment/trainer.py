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
from typing import Any

from .chat_template import load_tokenizer
from .config import StreamConfig
from .semantic import (action_semantic_mask, framing_mask, grounded_value_mask,
                       tool_call_region_mask)
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
    context = ""  # everything rendered so far: what a value could be copied from
    prev: list[int] = []
    ids: list[int] = []
    weights: list[float] = []
    teacher: list = []
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
            content = sample.messages[i - 1].content
            fr = framing_mask(tokenizer, span_ids)
            mws = [w if f else 0.0 for w, f in zip(mws, fr)]
            if cfg.grounded_weight != 1.0 and sample.messages[i - 1].role == "assistant":
                g = grounded_value_mask(tokenizer, content, span_ids, context)
                mws = [w * cfg.grounded_weight if hit else w for w, hit in zip(mws, g)]
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
            # channel -- becomes one unsigned mask entry; a position with no
            # teacher distribution is not a KL position at all.
            mws = [1.0 if (w != 0.0 and t) else 0.0 for w, t in zip(mws, mtch)]
        weights.extend(mws)
        teacher.extend(mtch)
        ids.extend(span_ids)
        context += sample.messages[i - 1].content
        prev = toks
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
    logZ = torch.logsumexp(lg, dim=-1, keepdim=True)
    lp = lg.gather(-1, ti) - logZ
    ce = None
    if targets is not None:
        tg = targets[0].index_select(0, idx).unsqueeze(-1)
        ce = (logZ - lg.gather(-1, tg)).squeeze(-1).detach()
    q = lq.exp()                      # teacher on its own top-k
    p = lp.exp()                      # student there
    q_tail = (1.0 - q.sum(-1)).clamp_min(1e-9)
    p_tail = (1.0 - p.sum(-1)).clamp_min(1e-9)
    if reverse:
        kl = (p * (lp - lq)).sum(-1) + p_tail * (torch.log(p_tail) - torch.log(q_tail))
    else:
        kl = (q * (lq - lp)).sum(-1) + q_tail * (torch.log(q_tail) - torch.log(p_tail))
    loss = (kl * w).sum() / w.sum().clamp_min(max(norm_floor, 1e-8))
    return loss, (idx, kl.detach(), ce)


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
    if _RESIDENT.get("name") != cfg.model:
        _RESIDENT.clear()
        base = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=torch.bfloat16)
        base.config.use_cache = False
        base.to(device)
        _RESIDENT.update(name=cfg.model, model=base)
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

    losses: list[float] = []
    step = 0
    # Fixed dose: accumulate all sample passes into K optimizer steps.
    micro_batch = _effective_micro_batch(len(samples), cfg)
    for _ in range(cfg.epochs):
        for sample in samples:
            ids, ws, tch = _encode(tokenizer, sample, cfg)
            if len(ids) < 2 or not any(w != 0.0 for w in ws):
                continue
            input_ids = torch.tensor([ids], device=device)
            w = torch.tensor([ws[1:]], dtype=torch.float32, device=device)
            logits = model(input_ids=input_ids).logits[:, :-1]
            if cfg.kl_target:
                # weights/teacher at position j predict token j, i.e. logits[j-1]
                t_ids, t_logq = _teacher_tensors(tch[1:], ws[1:], cfg.kl_topk, device)
                loss, dbg = topk_kl(logits, t_ids, t_logq, w, targets=input_ids[:, 1:],
                                    norm_floor=cfg.w_norm_floor, reverse=cfg.kl_reverse)
                if dbg is not None:
                    print(_kl_digit_share(tokenizer, ids, sample.task_id, *dbg), flush=True)
            else:
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
            losses.append(float(loss.detach()))
            step += 1
            if step % micro_batch == 0:
                opt.step()
                opt.zero_grad()
    if step % micro_batch != 0:
        opt.step()
        opt.zero_grad()
    model.save_pretrained(out_dir)
    _RESIDENT["model"] = model.unload()  # strip LoRA layers, keep the base resident
    return losses
