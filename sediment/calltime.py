"""Call-time single-snippet memory (Recuris §2.2.2, cross-family version).

Instead of one long retrieved block before the first action, memory enters
at execution events, one short snippet at a time:

  write   the actor drafts a modification call whose tool has not been
          hinted yet -> the draft is NOT executed; a synthetic user turn
          carries the matching donor step(s) and the actor re-drafts
  error   the env returned a failure -> the donor step with the most similar
          failure and its recovery step are appended to that observation
  tool    like ``write`` but for every first use of any tool (off by default)

Donors are the top-k retrieved trajectories (the same candidate pool the
start-block arms rank); the snippet is chosen by the CURRENT execution state
(tool name parts, argument keys, observation text), not by the task text, so
it works across families whose tool names differ. No model call is made to
build a hint and the intercepted draft does not count as an env step.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from sediment.experience import _reflection_units
from sediment.harness import (
    HINT_CLOSE,
    HINT_INTERCEPT_OPEN,
    HINT_OPEN,
    first_call,
    is_error_obs,
    is_write_tool,
    tokens,
)
from sediment.types import Trajectory


@dataclass
class DonorStep:
    donor: int
    step: int
    name: str
    args: dict[str, Any]
    result: str
    is_error: bool
    is_write: bool
    toks: set[str] = field(default_factory=set)


def _short(text: str, n: int) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= n else text[:n] + "…"


def _render_call(s: DonorStep, n: int) -> str:
    args = json.dumps(s.args, ensure_ascii=False)
    return f"{s.name or '(no tool call)'}({_short(args, n)}) -> {_short(s.result, n)}"


class CallTimeHinter:
    def __init__(self, donors: list[Trajectory], *, triggers: str = "error,write",
                 max_hints: int = 4, max_chars: int = 900, result_chars: int = 200,
                 intercept: bool = True, style: str = "hold"):
        if style not in ("hold", "confirm"):
            raise ValueError(f"unknown intercept style: {style!r}")
        self.style = style
        self.triggers = {t.strip() for t in triggers.split(",") if t.strip()}
        self.max_hints = max_hints
        self.max_chars = max_chars
        self.result_chars = result_chars
        self.intercept = intercept
        self.events: list[dict[str, Any]] = []
        self._hinted: set[str] = set()
        self._pending: Optional[dict[str, Any]] = None
        self.donor_ids = [d.task_id for d in donors]
        self._success = [bool(d.success) if d.success is not None
                         else bool(d.reward is not None and d.reward >= 0.999) for d in donors]
        self._reward = [float(d.reward or 0.0) for d in donors]
        self._lessons = [_reflection_units(str(d.meta.get("reflection", "") or "")) for d in donors]
        self._steps: list[list[DonorStep]] = [self._extract(i, d) for i, d in enumerate(donors)]

    @staticmethod
    def _extract(idx: int, traj: Trajectory) -> list[DonorStep]:
        out: list[DonorStep] = []
        msgs = traj.messages
        step = 0
        for i, m in enumerate(msgs):
            if m.role != "assistant":
                continue
            step += 1
            nxt = msgs[i + 1] if i + 1 < len(msgs) else None
            result = nxt.content if nxt is not None and nxt.role in ("tool", "user") else "(episode end)"
            call = first_call(m.content) or {}
            name = str(call.get("name") or "")
            args = call.get("arguments") or {}
            s = DonorStep(idx, step, name, args, result, is_error_obs(result), is_write_tool(name))
            s.toks = tokens(" ".join(name.split("_")) + " " + " ".join(map(str, args.keys()))
                            + " " + result[:300])
            out.append(s)
        return out

    # -- budget (per trigger type, so write intercepts cannot starve errors) --
    def _spent(self, trigger: str) -> bool:
        return sum(1 for e in self.events if e["trigger"] == trigger) >= self.max_hints

    @property
    def exhausted(self) -> bool:
        return all(self._spent(t) for t in self.triggers)

    # -- events ------------------------------------------------------------
    def on_draft(self, action_text: str, last_obs: str) -> Optional[str]:
        """Hint for a drafted action, or None. Non-None => caller must NOT
        execute the draft (intercept) when ``intercept`` is on."""
        if self._pending is not None:  # this draft is the re-draft after a hint
            call = first_call(action_text) or {}
            self._pending["changed"] = (call.get("name"), call.get("arguments")) != (
                self._pending["draft_name"], self._pending["draft_args"])
            self._pending["redraft_name"] = call.get("name", "")
            self._pending = None
            return None
        if not self.intercept:
            return None
        call = first_call(action_text)
        if not call or not call.get("name"):
            return None
        name = call["name"]
        trig = "write" if is_write_tool(name) else "tool"
        if trig not in self.triggers or name in self._hinted or self._spent(trig):
            return None
        query = tokens(" ".join(name.split("_")) + " " + " ".join(call["arguments"].keys())
                       + " " + last_obs[:300])
        pick = self._best(query, want_error=False, want_write=trig == "write")
        if pick is None:
            return None
        self._hinted.add(name)
        text = self._render(pick, query, intercept_name=name)
        self._pending = {"trigger": trig, "tool": name, "draft_name": name,
                         "draft_args": call["arguments"], "changed": None}
        self._log(self._pending, pick, text)
        return text

    def on_observation(self, call: Optional[dict[str, Any]], obs_text: str) -> Optional[str]:
        """Hint to append to an observation (error trigger), or None."""
        if "error" not in self.triggers or self._spent("error") or not is_error_obs(obs_text):
            return None
        name = str((call or {}).get("name") or "")
        key = f"error:{name}"
        if key in self._hinted:
            return None
        query = tokens(" ".join(name.split("_")) + " " + obs_text[:300])
        pick = self._best(query, want_error=True, want_write=False)
        if pick is None:
            return None
        self._hinted.add(key)
        text = self._render(pick, query, intercept_name=None)
        self._log({"trigger": "error", "tool": name}, pick, text)
        return text

    def _log(self, event: dict[str, Any], pick: DonorStep, text: str) -> None:
        event.update({"donor": self.donor_ids[pick.donor], "donor_step": pick.step,
                      "donor_tool": pick.name, "donor_success": self._success[pick.donor],
                      "chars": len(text)})
        self.events.append(event)

    # -- selection ---------------------------------------------------------
    def _best(self, query: set[str], *, want_error: bool, want_write: bool) -> Optional[DonorStep]:
        best, best_key = None, None
        for steps in self._steps:
            for s in steps:
                if not s.name:
                    continue
                union = query | s.toks
                sim = len(query & s.toks) / len(union) if union else 0.0
                score = sim + (0.5 if self._success[s.donor] else 0.0)
                if want_error:
                    score += 0.5 if s.is_error else -0.5
                    score += 0.2 if s.step < len(steps) else 0.0  # has a recovery step
                else:
                    score += 0.3 if s.is_write == want_write else 0.0
                    score += 0.2 if not s.is_error else -0.2
                key = (score, -s.donor, -s.step)
                if best_key is None or key > best_key:
                    best, best_key = s, key
        return best

    def _lesson(self, s: DonorStep, query: set[str]) -> str:
        units = self._lessons[s.donor]
        if not units:
            return ""
        ref = query | s.toks
        unit = max(units, key=lambda u: len(tokens(u) & ref))
        return _short(unit, 220)

    def _render(self, s: DonorStep, query: set[str], *, intercept_name: Optional[str]) -> str:
        """Hint text within ``max_chars``: drop the lesson, then the preceding
        step, then the recovery step, then shorten call/result text."""
        steps = self._steps[s.donor]
        outcome = f"outcome {'SUCCESS' if self._success[s.donor] else 'FAILED'}, r={self._reward[s.donor]:.2f}"
        lesson = self._lesson(s, query)
        if intercept_name is not None and self.style == "confirm":
            # 08-27 forensics: "was NOT executed" reads as a rejection to a 4B
            # actor -- 72% of re-drafts abandoned the write for a query and the
            # write was often never re-issued. This wording asks for the write back.
            head = (f"{HINT_INTERCEPT_OPEN}\nYour call `{intercept_name}` is queued, not yet executed. "
                    f"A related task did this ({outcome}):")
            tail = ("Tools and ids here may differ; do not copy identifiers. Re-issue "
                    f"`{intercept_name}` now, with the same arguments or corrected ones.")
        elif intercept_name is not None:
            head = (f"{HINT_INTERCEPT_OPEN}\nYour drafted call `{intercept_name}` was NOT executed. "
                    f"Before this modification, compare with a related task ({outcome}):")
            tail = ("Tools and ids in the current environment may differ; do not copy identifiers. "
                    "Now issue your next action (the same call if it is still correct).")
        else:
            head = f"{HINT_OPEN}\nA related task hit a similar failure ({outcome}):"
            tail = "Tools and ids here may differ; do not copy identifiers."

        def build(n: int, prev: bool, recovery: bool, with_lesson: bool) -> str:
            body = [head]
            if prev and s.step >= 2:
                body.append(f"  step {s.step - 1}: {_render_call(steps[s.step - 2], n)}")
            body.append(f"  step {s.step}: {_render_call(s, n)}")
            if recovery and s.is_error and s.step < len(steps):
                body.append(f"  step {s.step + 1} (recovery): {_render_call(steps[s.step], n)}")
            if with_lesson and lesson:
                body.append(f"Lesson from that task: {lesson}")
            body.extend([tail, HINT_CLOSE])
            return "\n".join(body)

        text = ""
        for n in (self.result_chars, self.result_chars // 2, 60):
            for prev, recovery, with_lesson in ((True, True, True), (True, True, False),
                                                (False, True, False), (False, False, False)):
                text = build(n, prev, recovery, with_lesson)
                if len(text) <= self.max_chars:
                    return text
        return text
