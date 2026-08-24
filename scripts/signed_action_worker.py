#!/usr/bin/env python3
"""Matched-action signed credit: OK CE plus ERROR-only unlikelihood.

Every five settled steps, score each executed action under two hindsight
prompts that contain the identical action and differ only in the returned
observation. Positive controlled residual is trained only for ok actions;
negative controlled residual is trained only for explicit ERROR actions.

Unlike the historical weighted CE, both branches divide by a fixed count of
action content tokens. Residual magnitude therefore controls update dose.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sediment.config import StreamConfig  # noqa: E402
from sediment.engine.vllm_client import VllmClient  # noqa: E402
from sediment.envs.envscaler import EnvScalerAdapter, list_tasks  # noqa: E402
from sediment.types import AdapterVersion, Message, TrainSample  # noqa: E402


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p1w = _load("p1w", "p1_worker.py")
t0w = _load("t0w", "tier0_worker.py")
lf = _load("latest_factorial_worker", "latest_factorial_worker.py")
ahc = _load("action_hint_control_worker", "action_hint_control_worker.py")

SHARED_TEMPLATE_WORDS = {"name", "arguments", "tool_call", "json"}
WORD_RE = re.compile(r"[A-Za-z0-9]")
JSON_KEY_RE = re.compile(r'["\u0027]([^"\u0027]+)["\u0027]\s*:')
TOOL_NAME_RE = re.compile(
    r'["\u0027]name["\u0027]\s*:\s*["\u0027]([^"\u0027]+)["\u0027]', re.I)


@dataclass
class SignedSample:
    task_id: str
    messages: list[Message]
    positive_weights_by_msg: list[list[float]]
    negative_weights_by_msg: list[list[float]]
    positive_denominator: int
    negative_denominator: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matched-action signed worker")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--pool", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--task-id", action="append", default=[],
        help="exact task id for focused smoke; repeatable")
    parser.add_argument("--k-update", type=int, default=5)
    parser.add_argument("--max-updates", type=int, default=5)
    parser.add_argument("--lambda-ul", type=float, default=0.02)
    parser.add_argument("--beta-kl", type=float, default=0.01)
    parser.add_argument("--post-kl-limit", type=float, default=0.0)
    parser.add_argument("--positive-cap", type=float, default=1.55)
    parser.add_argument("--negative-cap", type=float, default=4.51)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--max-backtracks", type=int, default=3)
    parser.add_argument("--direction-min-dose", type=float, default=1e-5)
    parser.add_argument("--direction-tolerance", type=float, default=1e-5)
    parser.add_argument("--steps-per-update", type=int, default=2)
    parser.add_argument("--data-dir", default="/data/erv1n/resid/data")
    parser.add_argument("--lopd-dir", default="/data/erv1n/resid/third_party/LOPD")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--tag", required=True)
    return parser.parse_args()


def is_semantic_token(piece: str) -> bool:
    stripped = piece.strip()
    if not stripped or stripped in {"<tool_call>", "</tool_call>", "```"}:
        return False
    normalized = stripped.strip("\"'`{}[]():,.").lower()
    if normalized in SHARED_TEMPLATE_WORDS:
        return False
    return bool(WORD_RE.search(normalized))


def action_semantic_mask(tokenizer, action: str, raw_ids: list[int]) -> list[bool]:
    """Exclude chat/JSON framing, tool names, and argument keys.

    A failed tool call often has the correct function but a bad entity value.
    Penalizing the function-name tokens would generalize the error in the
    wrong direction, so only value/content tokens remain eligible.
    """
    excluded_words = set(SHARED_TEMPLATE_WORDS)
    fields = JSON_KEY_RE.findall(action)
    tool_match = TOOL_NAME_RE.search(action)
    if tool_match:
        fields.append(tool_match.group(1))
    for field in fields:
        excluded_words.update(
            part.lower() for part in re.split(r"[^A-Za-z0-9]+", field) if part)
        for variant in (field, f" {field}", f'"{field}"'):
            for token_id in tokenizer.encode(variant, add_special_tokens=False):
                piece = tokenizer.decode([token_id])
                normalized = re.sub(
                    r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", piece.strip()).lower()
                if normalized:
                    excluded_words.add(normalized)
    mask = []
    for token_id in raw_ids:
        piece = tokenizer.decode([token_id])
        normalized = re.sub(
            r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", piece.strip()).lower()
        mask.append(is_semantic_token(piece) and normalized not in excluded_words)
    return mask


def direction_violation(positive_dose: float, positive_change: float,
                        negative_dose: float, negative_change: float,
                        minimum_dose: float, tolerance: float) -> str | None:
    if positive_dose >= minimum_dose and positive_change < -tolerance:
        return "positive_direction_violation"
    if negative_dose >= minimum_dose and negative_change > tolerance:
        return "negative_direction_violation"
    return None


def _content_logps(engine, tokenizer, prefix: list[Message], action: str,
                   adapter: str) -> tuple[list[int], list[float]]:
    target = Message("assistant", action)
    raw_ids = list(tokenizer.encode(action, add_special_tokens=False))
    if not raw_ids:
        raise ValueError("action has no content tokens")
    span_ids = lf._message_span_ids(tokenizer, prefix, target)
    offset = lf._find_sublist(span_ids, raw_ids)
    full_logps = engine._prompt_logprobs(prefix + [target], adapter)
    if len(full_logps) < len(span_ids):
        raise ValueError("scored prompt is shorter than final assistant span")
    span_logps = full_logps[-len(span_ids):]
    if len(span_logps) != len(span_ids):
        raise ValueError("engine/tokenizer chat-template mismatch")
    return raw_ids, span_logps[offset: offset + len(raw_ids)]


def controlled_deltas(engine, tokenizer, prefix: list[Message], action: str,
                      observation: str, adapter: str) -> tuple[list[int], list[float]]:
    raw_ids, masked = _content_logps(
        engine, tokenizer, prefix + [ahc.hint(action, ahc.MASKED_RESULT)],
        action, adapter)
    actual_ids, actual = _content_logps(
        engine, tokenizer, prefix + [ahc.hint(action, observation)],
        action, adapter)
    if actual_ids != raw_ids:
        raise ValueError("action tokenization changed across control branches")
    return raw_ids, [float(a - b) for a, b in zip(actual, masked, strict=True)]


def _span_weights(tokenizer, prefix: list[Message], action: str,
                  content_weights: list[float]) -> list[float]:
    target = Message("assistant", action)
    raw_ids = list(tokenizer.encode(action, add_special_tokens=False))
    if len(raw_ids) != len(content_weights):
        raise ValueError("content weight length mismatch")
    span_ids = lf._message_span_ids(tokenizer, prefix, target)
    offset = lf._find_sublist(span_ids, raw_ids)
    weights = [0.0] * len(span_ids)
    weights[offset: offset + len(raw_ids)] = content_weights
    return weights


def build_signed_sample(engine, tokenizer, task_id: str, messages: list[Message],
                        window: list[tuple[int, str, str, str]], adapter: str,
                        positive_cap: float, negative_cap: float,
                        max_seq_len: int) -> tuple[SignedSample, list[dict[str, Any]]]:
    last_prefix_len = window[-1][0]
    sample_messages = list(messages[:last_prefix_len + 1])
    full_ids = tokenizer.apply_chat_template(
        [message.to_dict() for message in sample_messages], tokenize=True,
        return_dict=False)
    if len(full_ids) > max_seq_len:
        raise ValueError(
            f"signed sample length {len(full_ids)} exceeds max_seq_len {max_seq_len}")

    positive = [[] for _ in sample_messages]
    negative = [[] for _ in sample_messages]
    positive_denominator = 0
    negative_denominator = 0
    diagnostics: list[dict[str, Any]] = []

    for prefix_len, action, observation, status in window:
        raw_ids, deltas = controlled_deltas(
            engine, tokenizer, messages[:prefix_len], action, observation, adapter)
        semantic = action_semantic_mask(tokenizer, action, raw_ids)
        pos_content = [
            min(max(delta, 0.0), positive_cap) if keep and status == "ok" else 0.0
            for delta, keep in zip(deltas, semantic, strict=True)
        ]
        neg_content = [
            min(max(-delta, 0.0), negative_cap)
            if keep and status == "ERROR" else 0.0
            for delta, keep in zip(deltas, semantic, strict=True)
        ]
        positive[prefix_len] = _span_weights(
            tokenizer, messages[:prefix_len], action, pos_content)
        negative[prefix_len] = _span_weights(
            tokenizer, messages[:prefix_len], action, neg_content)
        if status == "ok":
            positive_denominator += len(raw_ids)
        else:
            negative_denominator += len(raw_ids)
        diagnostics.append({
            "status": status,
            "n_tok": len(raw_ids),
            "semantic_tok": sum(semantic),
            "positive_tok": sum(weight > 0 for weight in pos_content),
            "negative_tok": sum(weight > 0 for weight in neg_content),
            "positive_mass_per_token": round(sum(pos_content) / len(raw_ids), 8),
            "negative_mass_per_token": round(sum(neg_content) / len(raw_ids), 8),
            "mean_delta": round(sum(deltas) / len(deltas), 8),
        })

    return SignedSample(
        task_id=task_id,
        messages=sample_messages,
        positive_weights_by_msg=positive,
        negative_weights_by_msg=negative,
        positive_denominator=positive_denominator,
        negative_denominator=negative_denominator,
    ), diagnostics


class SignedTrainer(t0w.Tier0Trainer):
    """Transactional two-branch trainer with exact active-position KL."""

    def _snapshot(self) -> dict[str, Any]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }

    def _restore(self, state: dict[str, Any]) -> None:
        with self.torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if name in state:
                    parameter.copy_(state[name].to(
                        device=parameter.device, dtype=parameter.dtype))

    def _save(self, out_dir: str) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir)

    def rollback_pending(self, out_dir: str) -> None:
        if getattr(self, "_pending_state", None) is not None:
            self._restore(self._pending_state)
            self._save(out_dir)
            self._pending_state = None

    def commit_pending(self) -> None:
        self._pending_state = None

    @staticmethod
    def _exact_kl(torch, reference_logits, current_logits):
        reference_logp = torch.nn.functional.log_softmax(reference_logits.float(), dim=-1)
        current_logp = torch.nn.functional.log_softmax(current_logits.float(), dim=-1)
        return (reference_logp.exp() * (reference_logp - current_logp)).sum(-1).mean()

    def train_update(self, sample: SignedSample, out_dir: str, reset: bool,
                     lambda_ul: float, beta_kl: float, grad_clip: float,
                     steps_per_update: int, post_kl_limit: float,
                     max_backtracks: int, direction_min_dose: float,
                     direction_tolerance: float) -> dict[str, Any]:
        torch = self.torch
        if reset:
            self._reset_lora()
        positive_sample = TrainSample(
            sample.task_id, sample.messages, sample.positive_weights_by_msg)
        negative_sample = TrainSample(
            sample.task_id, sample.messages, sample.negative_weights_by_msg)
        ids, positive_weights = p1w._encode(
            self.tokenizer, positive_sample, self.cfg.max_seq_len)
        negative_ids, negative_weights = p1w._encode(
            self.tokenizer, negative_sample, self.cfg.max_seq_len)
        if ids != negative_ids:
            raise AssertionError("positive and negative encodings differ")
        if len(ids) < 2:
            return {"trained": False, "reason": "short_sample"}

        positive_weights = positive_weights[1:]
        negative_weights = negative_weights[1:]
        active_positions = [
            index for index, (pos, neg) in enumerate(zip(
                positive_weights, negative_weights, strict=True))
            if pos > 0 or neg > 0
        ]
        if not active_positions:
            return {"trained": False, "reason": "no_nonzero_signed_weights"}

        input_ids = torch.tensor([ids], device=self.device)
        active = torch.tensor(active_positions, device=self.device, dtype=torch.long)
        targets = input_ids[:, 1:].index_select(1, active).squeeze(0)
        pos_w = torch.tensor(
            [positive_weights[index] for index in active_positions],
            device=self.device, dtype=torch.float32)
        neg_w = torch.tensor(
            [negative_weights[index] for index in active_positions],
            device=self.device, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            reference_full = self.model(input_ids=input_ids).logits[:, :-1]
            reference_logits = reference_full.index_select(1, active).squeeze(0).float()
            del reference_full
        state = self._snapshot()
        self._pending_state = state
        attempts: list[dict[str, Any]] = []
        try:
            reference_target_logp = torch.nn.functional.log_softmax(
                reference_logits, dim=-1).gather(
                    1, targets.unsqueeze(1)).squeeze(1)
            for backtrack in range(max_backtracks + 1):
                self._restore(state)
                for parameter in self.model.parameters():
                    parameter.grad = None
                learning_rate = self.cfg.lr * (0.25 ** backtrack)
                attempt: dict[str, Any] = {
                    "backtrack": backtrack,
                    "lr": learning_rate,
                    "steps": [],
                }
                try:
                    self.model.train()
                    for _ in range(steps_per_update):
                        optimizer = torch.optim.AdamW(
                            (parameter for parameter in self.model.parameters()
                             if parameter.requires_grad), lr=learning_rate)
                        current_full = self.model(input_ids=input_ids).logits[:, :-1]
                        current_logits = current_full.index_select(1, active).squeeze(0)
                        ce = torch.nn.functional.cross_entropy(
                            current_logits.float(), targets, reduction="none")
                        probabilities = torch.nn.functional.softmax(
                            current_logits.float(), dim=-1).gather(
                                1, targets.unsqueeze(1)).squeeze(1)
                        unlikelihood = -torch.log1p(
                            -probabilities.clamp(max=1.0 - 1e-6))
                        zero = current_logits.sum() * 0.0
                        positive_loss = (
                            (ce * pos_w).sum() / sample.positive_denominator
                            if sample.positive_denominator else zero)
                        negative_loss = (
                            (unlikelihood * neg_w).sum() / sample.negative_denominator
                            if sample.negative_denominator else zero)
                        kl = self._exact_kl(torch, reference_logits, current_logits)
                        total = positive_loss + lambda_ul * negative_loss + beta_kl * kl
                        if not bool(torch.isfinite(total)):
                            raise FloatingPointError("nonfinite signed loss")
                        total.backward()
                        parameters = [
                            parameter for parameter in self.model.parameters()
                            if parameter.requires_grad
                        ]
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            parameters, grad_clip, error_if_nonfinite=True)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        if not all(
                            bool(torch.isfinite(parameter).all())
                            for parameter in parameters
                        ):
                            raise FloatingPointError("nonfinite trainable parameter")
                        attempt["steps"].append({
                            "positive_loss": round(float(positive_loss.detach()), 10),
                            "unlikelihood_loss": round(float(negative_loss.detach()), 10),
                            "kl_loss": round(float(kl.detach()), 10),
                            "total_loss": round(float(total.detach()), 10),
                            "preclip_grad_norm": round(float(grad_norm.detach()), 10),
                        })
                        del optimizer, current_full, current_logits

                    self.model.eval()
                    with torch.no_grad():
                        post_full = self.model(input_ids=input_ids).logits[:, :-1]
                        post_logits = post_full.index_select(1, active).squeeze(0)
                        post_kl = float(self._exact_kl(
                            torch, reference_logits, post_logits).detach())
                        post_target_logp = torch.nn.functional.log_softmax(
                            post_logits.float(), dim=-1).gather(
                                1, targets.unsqueeze(1)).squeeze(1)
                        neg_mass = neg_w.sum().clamp_min(1e-8)
                        pos_mass = pos_w.sum().clamp_min(1e-8)
                        neg_logp_change = float((
                            (post_target_logp - reference_target_logp) * neg_w
                        ).sum() / neg_mass) if bool(neg_w.sum() > 0) else 0.0
                        pos_logp_change = float((
                            (post_target_logp - reference_target_logp) * pos_w
                        ).sum() / pos_mass) if bool(pos_w.sum() > 0) else 0.0
                        del post_full, post_logits
                    attempt.update({
                        "post_kl": round(post_kl, 10),
                        "positive_weighted_logp_change": round(pos_logp_change, 8),
                        "negative_weighted_logp_change": round(neg_logp_change, 8),
                    })
                    if not math.isfinite(post_kl):
                        raise FloatingPointError("nonfinite post-update KL")
                    positive_dose = (
                        float(pos_w.sum()) / sample.positive_denominator
                        if sample.positive_denominator else 0.0)
                    negative_dose = (
                        lambda_ul * float(neg_w.sum()) / sample.negative_denominator
                        if sample.negative_denominator else 0.0)
                    attempt.update({
                        "positive_dose": round(positive_dose, 10),
                        "negative_dose": round(negative_dose, 10),
                    })
                    if post_kl_limit > 0 and post_kl > post_kl_limit:
                        attempt["accepted"] = False
                        attempt["reason"] = "post_kl_limit"
                        attempts.append(attempt)
                        continue
                    violation = direction_violation(
                        positive_dose, pos_logp_change,
                        negative_dose, neg_logp_change,
                        direction_min_dose, direction_tolerance)
                    if violation:
                        attempt["accepted"] = False
                        attempt["reason"] = violation
                        attempts.append(attempt)
                        continue
                    attempt["accepted"] = True
                    attempts.append(attempt)
                    self._save(out_dir)
                    torch.cuda.empty_cache()
                    return {
                        "trained": True,
                        "active_positions": len(active_positions),
                        "positive_denominator": sample.positive_denominator,
                        "negative_denominator": sample.negative_denominator,
                        "positive_weight_sum": round(float(pos_w.sum()), 8),
                        "negative_weight_sum": round(float(neg_w.sum()), 8),
                        "post_kl": round(post_kl, 10),
                        "positive_weighted_logp_change": round(pos_logp_change, 8),
                        "negative_weighted_logp_change": round(neg_logp_change, 8),
                        "accepted_lr": learning_rate,
                        "attempts": attempts,
                    }
                except (FloatingPointError, RuntimeError) as exc:
                    attempt["accepted"] = False
                    attempt["reason"] = repr(exc)[:180]
                    attempts.append(attempt)
            raise RuntimeError(
                f"all {max_backtracks + 1} signed attempts rejected: {attempts}")
        except Exception:
            self._restore(state)
            self._pending_state = None
            self.model.eval()
            torch.cuda.empty_cache()
            raise


def signed_episode(engine, trainer, task, cfg, run_dir, tag, args):
    env = EnvScalerAdapter(cfg.extra["lopd_dir"], max_steps=cfg.max_steps)
    messages = env.reset(task)
    base_name = f"{tag}-{p1w.sanitize(task['task_id'])}"
    adapter_dir = str(run_dir / "candidates" / base_name)
    adapter = "base"
    window: list[tuple[int, str, str, str]] = []
    diagnostics: list[dict[str, Any]] = []
    selected = updates = update_errors = rollbacks = steps = 0
    reward = 0.0

    for _ in range(cfg.max_steps):
        prefix_len = len(messages)
        action = engine.generate(
            messages, adapter=adapter, temperature=cfg.temperature,
            max_tokens=cfg.max_tokens)
        messages.append(Message("assistant", action))
        observation_messages, done, reward = env.step(action)
        steps += 1
        if done:
            break
        messages.extend(observation_messages)
        observation = "\n".join(message.content for message in observation_messages)
        status = "ERROR" if p1w.STATUS_RE.search(observation[:300]) else "ok"
        window.append((prefix_len, action, observation, status))

        if len(window) >= args.k_update and selected < args.max_updates:
            selected += 1
            item: dict[str, Any] = {"step": steps}
            try:
                sample, action_diags = build_signed_sample(
                    engine, trainer.tokenizer, str(task["task_id"]), messages,
                    window, adapter, args.positive_cap, args.negative_cap,
                    cfg.max_seq_len)
                item["actions"] = action_diags
                train_diag = trainer.train_update(
                    sample, adapter_dir, reset=(updates == 0),
                    lambda_ul=args.lambda_ul, beta_kl=args.beta_kl,
                    grad_clip=args.grad_clip,
                    steps_per_update=args.steps_per_update,
                    post_kl_limit=args.post_kl_limit,
                    max_backtracks=args.max_backtracks,
                    direction_min_dose=args.direction_min_dose,
                    direction_tolerance=args.direction_tolerance)
                item["train"] = train_diag
                if train_diag.get("trained"):
                    version = f"{base_name}-u{updates + 1}"
                    try:
                        engine.load_adapter(AdapterVersion(
                            version, adapter_dir, "v0000"))
                    except Exception:
                        trainer.rollback_pending(adapter_dir)
                        try:
                            engine.unload_adapter(version)
                        except Exception:
                            pass
                        rollbacks += 1
                        raise
                    trainer.commit_pending()
                    if updates:
                        engine.unload_adapter(f"{base_name}-u{updates}")
                    adapter = version
                    updates += 1
            except Exception as exc:
                update_errors += 1
                item.update({
                    "error": repr(exc)[:300],
                    "trace": traceback.format_exc(limit=3)[-500:],
                })
            diagnostics.append(item)
            window = []

    if updates:
        engine.unload_adapter(f"{base_name}-u{updates}")
    return {
        "success": bool(reward >= 0.999),
        "reward": float(reward),
        "steps": steps,
        "n_selected": selected,
        "n_updates": updates,
        "upd_errors": update_errors,
        "rollbacks": rollbacks,
        "update_diag": diagnostics,
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = StreamConfig(
        model=args.model, engine="vllm", trainer="torch", epochs=1,
        temperature=0.7, max_tokens=2048, max_steps=30,
        lora_r=8, lora_alpha=16, lr=5e-4, max_seq_len=12288,
        data_dir=args.data_dir, split="rl", out_dir=str(run_dir), run_id=args.tag,
        extra={
            "lopd_dir": args.lopd_dir,
            "base_url": args.base_url,
            "method": "matched_action_signed_fixed_denominator",
            "cadence": args.k_update,
            "max_updates": args.max_updates,
            "lambda_ul": args.lambda_ul,
            "beta_kl": args.beta_kl,
            "post_kl_limit": args.post_kl_limit,
            "positive_cap": args.positive_cap,
            "negative_cap": args.negative_cap,
            "grad_clip": args.grad_clip,
            "max_backtracks": args.max_backtracks,
            "direction_min_dose": args.direction_min_dose,
            "direction_tolerance": args.direction_tolerance,
            "steps_per_update": args.steps_per_update,
            "template_mask": True,
        },
    )
    if args.shard == 0:
        (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    tasks = list_tasks("rl", args.data_dir, third_party_dir=args.lopd_dir)
    pool = tasks[-args.pool:] if args.pool else tasks
    if args.task_id:
        requested = set(args.task_id)
        pool = [task for task in pool if task["task_id"] in requested]
        found = {task["task_id"] for task in pool}
        missing = requested - found
        if missing:
            raise ValueError(f"requested task ids outside pool: {sorted(missing)}")
    mine = pool[args.shard::args.num_shards]
    if args.limit:
        mine = mine[:args.limit]
    output_path = run_dir / f"t0_shard{args.shard}.jsonl"
    done_ids = set()
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if "error" not in row:
                    done_ids.add(row["task_id"])

    engine = VllmClient(args.base_url, cfg.model)
    trainer = SignedTrainer(cfg)
    print(
        f"[shard {args.shard}/{args.num_shards}] mine={len(mine)} "
        f"done={len(done_ids)} trainer={trainer.device}", flush=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for index, task in enumerate(mine):
            if task["task_id"] in done_ids:
                continue
            started = time.time()
            try:
                result = signed_episode(
                    engine, trainer, task, cfg, run_dir, args.tag, args)
                record = {"task_id": task["task_id"], "tier0": result}
            except Exception as exc:
                record = {
                    "task_id": task["task_id"],
                    "error": repr(exc)[:300],
                    "trace": traceback.format_exc(limit=3)[-500:],
                }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{index + 1}/{len(mine)} {task['task_id']}] "
                f"error={int('error' in record)} "
                f"success={int(record.get('tier0', {}).get('success', False))} "
                f"updates={record.get('tier0', {}).get('n_updates', 0)} "
                f"dt={time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
