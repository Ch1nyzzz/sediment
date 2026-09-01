"""Branching at a state: sample alternative actions from the served policy with
first-decision-token exclusion, and continue an episode from a replayed prefix.

Shared by the offline tree sampler (scripts/tree_sample.py) and the online
critic-guided redo (scheduler, cfg.critic_mode="redo_rank").
"""
from __future__ import annotations

import time
from typing import Any

from sediment.harness import is_error_obs
from sediment.semantic import is_semantic_token
from sediment.types import Message, Trajectory


def norm_action(a: str) -> str:
    return "".join(a.split())  # whitespace-insensitive: " get_x" == "get_x"


def ban_variants(tok, piece: str) -> list[int]:
    """Token ids of a decision piece and its space/strip variants ('get', ' get')."""
    ids = set()
    for v in {piece, piece.strip(), " " + piece.strip()}:
        enc = tok.encode(v, add_special_tokens=False)
        if len(enc) == 1:
            ids.add(int(enc[0]))
    return sorted(ids)


def first_decision_split(tok, action: str):
    """(framing prefix text, banned token id) at the first semantic token of the action."""
    ids = tok.encode(action, add_special_tokens=False)
    for pos, tid in enumerate(ids):
        if is_semantic_token(tok.decode([tid])):
            return tok.decode(ids[:pos]), int(tid)
    return None, None


def chat(engine, messages, prefix: str, adapter: str = "base", **kw):
    """Chat completion; a non-empty `prefix` continues the assistant turn from
    that text (vLLM continue_final_message) instead of starting it fresh."""
    payload = {"model": engine._model_name(adapter), "messages": [m.to_dict() for m in messages], **kw}
    if prefix:
        payload["messages"] = payload["messages"] + [{"role": "assistant", "content": prefix}]
        payload["continue_final_message"] = True
        payload["add_generation_prompt"] = False
    return engine._post("/v1/chat/completions", payload)


def sample_siblings(engine, messages, orig_action: str, tok, n_each: int, temperature: float,
                    adapter: str = "base"):
    """(siblings, meta). 'banned': cumulative exclusion of the original action's
    first decision token and of the alternatives already taken (two-stage: one
    token under logit_bias with the framing prefix, then free continuation);
    'free': plain samples at a higher temperature. Deduplicated (whitespace-
    insensitive) against the original and each other."""
    out, seen = [], {norm_action(orig_action)}
    meta = {"banned_token": None, "banned_piece": None, "n_raw": 0}
    prefix, banned = first_decision_split(tok, orig_action)
    if banned is not None:
        meta.update(banned_token=banned, banned_piece=tok.decode([banned]))
        try:
            bias = {str(t): -100 for t in ban_variants(tok, tok.decode([banned])) or [banned]}
            for _ in range(n_each):
                d1 = chat(engine, messages, prefix, adapter, temperature=temperature, max_tokens=1, n=1, logit_bias=bias)
                t = d1["choices"][0]["message"]["content"] or ""
                if not t:
                    break
                d2 = chat(engine, messages, prefix + t, adapter, temperature=temperature, max_tokens=2048)
                text = prefix + t + (d2["choices"][0]["message"]["content"] or "")
                meta["n_raw"] += 1
                for tid in ban_variants(tok, t) or tok.encode(t, add_special_tokens=False)[:1]:
                    bias[str(int(tid))] = -100
                key = norm_action(text)
                if key and key not in seen:
                    seen.add(key)
                    out.append({"source": "banned", "action": text})
        except RuntimeError as e:
            if "maximum context length" not in str(e):
                raise
    try:
        d3 = chat(engine, messages, "", adapter, temperature=min(1.2, temperature + 0.2), max_tokens=2048, n=n_each + 1)
        for ch in d3.get("choices", []):
            text = engine._strip_think(ch["message"]["content"] or "")
            meta["n_raw"] += 1
            key = norm_action(text)
            if key and key not in seen:
                seen.add(key)
                out.append({"source": "free", "action": text})
    except RuntimeError as e:
        if "maximum context length" not in str(e):
            raise
    meta["n_distinct"] = len(out)
    return out, meta


def continue_from_prefix(engine, env, task: dict[str, Any], cfg, adapter: str,
                         prefix_actions: list[str], action: str) -> Trajectory:
    """Replay `prefix_actions` in a fresh env, execute `action`, then let the
    served policy finish (cfg.max_steps, cfg.temperature). Student view only."""
    messages = env.reset(task)
    steps = 0
    done, reward = False, 0.0
    for act in prefix_actions:
        messages.append(Message("assistant", act))
        obs_msgs, done, reward = env.step(act)
        steps += 1
        if done:
            break
        messages.extend(obs_msgs)
    branch_msg = len(messages)
    final_obs = ""
    if not done:
        messages.append(Message("assistant", action))
        obs_msgs, done, reward = env.step(action)
        steps += 1
        if done:
            final_obs = "\n".join(m.content for m in obs_msgs)
        else:
            messages.extend(obs_msgs)
            while steps < cfg.max_steps:
                nxt = engine.generate(messages, adapter=adapter, temperature=cfg.temperature, max_tokens=cfg.max_tokens)
                messages.append(Message("assistant", nxt))
                obs_msgs, done, reward = env.step(nxt)
                steps += 1
                if done:
                    final_obs = "\n".join(m.content for m in obs_msgs)
                    break
                messages.extend(obs_msgs)
    if not done:
        try:
            reward = float(env.score_current_state())
        except Exception:
            pass
    return Trajectory(task_id=task["task_id"], env_family=task.get("env_family", ""), messages=messages,
                      reward=float(reward), success=bool(done and reward >= 0.999), adapter=adapter, steps=steps,
                      is_retry=True, meta={"final_obs": final_obs[:2000], "branch_msg": branch_msg,
                                           "done": done, "termination_reason": "done" if done else "max_steps"})
