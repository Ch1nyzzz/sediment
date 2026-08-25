"""Candidate trainer: hindsight-weighted token CE -> LoRA adapter dir.

`train_candidate` is the single entry point. cfg.trainer selects "stub"
(metadata only, no heavy deps -- used by all tests) or "torch" (lazy
transformers/peft LoRA fine-tune).
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from .chat_template import load_tokenizer
from .config import StreamConfig
from .semantic import action_semantic_mask
from .types import AdapterVersion, TrainSample, UpdateCandidate

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
    }


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
    if cfg.trainer == "stub":
        train_stats: dict[str, Any] = {"mode": "stub", "n_samples": len(samples), "samples": stats}
    elif cfg.trainer == "torch":
        losses = _train_torch(samples, parent, cfg, out_dir)
        train_stats = {"mode": "torch", "n_samples": len(samples), "samples": stats, "losses": losses}
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


def _encode(tokenizer, sample: TrainSample, max_seq_len: int) -> tuple[list[int], list[float]]:
    """Token ids + parallel CE weights for one sample.

    Alignment assumption: ``token_weights_by_msg`` was computed against the
    engine-side tokenization of the same messages (sediment.spans). Here the
    trainer re-derives per-message token spans by applying the chat template
    incrementally over message prefixes (the same construction as
    ``spans.message_spans``, relying on the template's token prefix
    property). When a message's trainer-side span length equals its weight
    list length the weights map 1:1; otherwise the message's mean weight is
    broadcast uniformly over every token of its span (framing tokens
    included). Prompt/system messages carry all-zero weights either way.
    """
    msgs = [m.to_dict() for m in sample.messages]
    prev: list[int] = []
    ids: list[int] = []
    weights: list[float] = []
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
        if len(ws) == span:
            mws = [float(w) for w in ws]
        else:
            mean = float(sum(ws) / len(ws)) if ws else 0.0
            mws = [mean] * span
        span_ids = toks[len(prev):]
        if sample.messages[i - 1].role == "assistant" and any(w < 0 for w in mws):
            # negative credit only on value/content tokens, never on framing,
            # JSON keys or the tool name (sediment.semantic)
            keep = action_semantic_mask(tokenizer, sample.messages[i - 1].content, span_ids)
            mws = [w if (w >= 0 or k) else 0.0 for w, k in zip(mws, keep)]
        weights.extend(mws)
        ids.extend(span_ids)
        prev = toks
    return ids[:max_seq_len], weights[:max_seq_len]


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


def _train_torch(
    samples: list[TrainSample],
    parent: AdapterVersion,
    cfg: StreamConfig,
    out_dir: str,
) -> list[float]:
    """LoRA fine-tune with per-token weighted CE; returns per-step losses."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(cfg.model)
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch.bfloat16)
    if parent.path is not None:
        model = PeftModel.from_pretrained(model, parent.path, is_trainable=True)
    else:
        lora = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.lr)

    losses: list[float] = []
    step = 0
    for _ in range(cfg.epochs):
        for sample in samples:
            ids, ws = _encode(tokenizer, sample, cfg.max_seq_len)
            if len(ids) < 2 or not any(w != 0.0 for w in ws):
                continue
            input_ids = torch.tensor([ids], device=device)
            w = torch.tensor([ws[1:]], dtype=torch.float32, device=device)
            logits = model(input_ids=input_ids).logits[:, :-1]
            loss = weighted_ce(logits, input_ids[:, 1:], w, ul_lambda=cfg.ul_lambda,
                               norm_floor=cfg.w_norm_floor)
            (loss / cfg.micro_batch).backward()
            losses.append(float(loss.detach()))
            step += 1
            if step % cfg.micro_batch == 0:
                opt.step()
                opt.zero_grad()
    if step % cfg.micro_batch != 0:
        opt.step()
        opt.zero_grad()
    model.save_pretrained(out_dir)
    return losses
