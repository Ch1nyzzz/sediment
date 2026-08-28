"""Verified working state: a harness-maintained goal ledger (Recuris §2.2).

The harness -- not the model -- tracks what the task asked for and what the
environment has actually confirmed. Goals come from the task text (bullets
or sentences); a goal flips to ``done`` only when a SUCCESSFUL modification
call lexically supports it (verb or object of the tool name, or an argument
value, appears in the goal). Failed calls stay on the ledger. Nothing here
calls the model, and the initial state is not rendered (it would restate the
task), so the first generation stays byte-identical to the frozen arm.

This is the lexical stand-in for Recuris' per-goal checker predicates; it
cannot mark a goal done on the model's say-so, which is the property that
carried their working-memory gain.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from sediment.harness import STATE_CLOSE, STATE_OPEN, is_write_tool, tokens, tool_verb

_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*\S)", re.M)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_GENERIC_PARTS = frozenset({"by", "id", "ids", "all", "the", "a", "an", "and", "of",
                            "for", "to", "in", "on", "with", "new", "info"})


def extract_goals(task_text: str, max_goals: int = 12, max_chars: int = 140) -> list[str]:
    items = [g.strip() for g in _BULLET_RE.findall(task_text)]
    items = [g for g in items if not g.endswith(":")]  # section headers
    if not items:
        items = [s.strip() for s in _SENTENCE_RE.split(task_text)]
    items = [g for g in items if len(g) >= 12]
    return [g if len(g) <= max_chars else g[:max_chars] + "…" for g in items[:max_goals]]


class WorkingState:
    def __init__(self, task_text: str, *, max_goals: int = 12, max_chars: int = 2600):
        self.goals = extract_goals(task_text, max_goals)
        self._goal_tokens = [tokens(g) for g in self.goals]
        self.done: dict[int, str] = {}  # goal index -> "step k tool"
        self.ledger: list[dict[str, Any]] = []
        self.max_chars = max_chars

    # -- evidence ----------------------------------------------------------
    def record(self, step: int, call: Optional[dict[str, Any]], obs_text: str, ok: bool) -> None:
        name = str((call or {}).get("name") or "")
        args = (call or {}).get("arguments") or {}
        entry = {"step": step, "name": name, "ok": ok, "write": is_write_tool(name)}
        if not ok:
            entry["error"] = obs_text.replace("\n", " ")[:80]
        self.ledger.append(entry)
        if ok and entry["write"]:
            i = self._supported(name, args)
            if i is not None:
                self.done[i] = f"step {step} {name}"

    def _supported(self, name: str, args: dict[str, Any]) -> Optional[int]:
        """Best-matching pending goal for one successful write, or None.

        One receipt supports at most one goal (the lexical checker's only
        guard against 'cancel' completing every goal that mentions it).
        Argument values that appear verbatim in a goal are the strongest
        evidence (3 per value, up to two), then the exact verb token (2), the
        verb stem cancel~cancelling (1), then the tool's object words (1).
        Ties go to the earliest pending goal.
        """
        parts = [p for p in name.lower().split("_") if p and p not in _GENERIC_PARTS]
        verb = tool_verb(name)
        stem = verb.rstrip("e")
        values = {str(v).lower() for v in args.values() if isinstance(v, (str, int, float))
                  and not isinstance(v, bool) and len(str(v)) >= 3}
        best, best_score = None, 0
        for i, gtoks in enumerate(self._goal_tokens):
            if i in self.done:
                continue
            goal = self.goals[i].lower()
            n_values = min(2, sum(1 for v in values if v in goal))
            verb_score = 2 if verb in gtoks else (
                1 if len(stem) >= 4 and any(t.startswith(stem) for t in gtoks) else 0)
            if not n_values and not verb_score:
                continue
            score = 3 * n_values + verb_score + any(p in gtoks for p in parts if p != verb)
            if score > best_score:
                best, best_score = i, score
        return best

    # -- rendering ---------------------------------------------------------
    def render(self) -> str:
        lines = [STATE_OPEN,
                 "Task goals tracked by the harness. A goal is done only when a "
                 "successful modification call supports it; the model cannot mark it done."]
        for i, g in enumerate(self.goals):
            tag = f"[done: {self.done[i]}]" if i in self.done else "[pending]"
            lines.append(f"{i + 1}. {tag} {g}")
        ok_r = sum(1 for e in self.ledger if e["ok"] and not e["write"])
        ok_w = sum(1 for e in self.ledger if e["ok"] and e["write"])
        fails = [e for e in self.ledger if not e["ok"]]
        receipt = f"Receipts: {ok_r} reads ok, {ok_w} writes ok, {len(fails)} failed calls"
        if fails:
            receipt += " (" + "; ".join(
                f"step {e['step']} {e['name'] or 'malformed'} -> {e['error']}" for e in fails[-3:]) + ")"
        lines.append(receipt + ".")
        pending = [str(i + 1) for i in range(len(self.goals)) if i not in self.done]
        lines.append(
            f"Goals without a supporting receipt yet: {', '.join(pending)}. Reply 'Task Completed' "
            "only when each is covered by a successful modification receipt or verified impossible."
            if pending else "Every tracked goal has a supporting receipt; verify, then finish.")
        lines.append(STATE_CLOSE)
        text = "\n".join(lines)
        if len(text) > self.max_chars:  # never drop a goal: shorten goal lines instead
            budget = max(40, (self.max_chars - 400) // max(1, len(self.goals)))
            lines[2:2 + len(self.goals)] = [
                ln if len(ln) <= budget else ln[:budget] + "…" for ln in lines[2:2 + len(self.goals)]]
            text = "\n".join(lines)
        return text

    def summary(self) -> dict[str, Any]:
        return {"goals": len(self.goals), "done": len(self.done),
                "calls": len(self.ledger), "failed": sum(1 for e in self.ledger if not e["ok"])}
