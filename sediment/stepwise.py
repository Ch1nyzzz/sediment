"""Step-wise principle-pool experience (ExpInternalization, 2606.04703).

The pool stores PRINCIPLE-level items -- short, reusable strategy / failure
rules the serving model itself distills from finished episodes (the paper's
"Qwen self-generated" setting) -- never trajectory transcripts: instance-level
experience (concrete ids, urls, values) was the paper's collapse case, and our
own probes (08-31) showed a raw own-failure transcript teacher carries ~0
signal while dilution from transcripts hurts a success demo.

Injection is step-wise: before EVERY generation the selector picks the
``cfg.stepwise_k`` most state-relevant items (lexical prefilter -> one short
LLM call) and attaches them to the newest user/tool message; earlier stepwise
segments are dropped first, so the model reads exactly one current selection
(working_state's "one current ledger" rule). The tag is
``<experience_hint stepwise>`` -- harness.bare_view / strip_experience already
remove it, so stored student views remain prompt-only.

Three consumers (the 2606.04703 regime grid):
  on-policy  (kl_states="first"):  bare first attempts; ``teacher_score``
             builds, per assistant turn, the view the teacher WOULD have seen
             (selection on the bare prefix) and returns per-token top-k
             teacher distributions for reverse-KL at the student's states.
  off-policy (kl_states="retry"):  the redo is GENERATED with a StepwiseHinter
             (rolling injection); redo success = rejection sampling; the
             stored student view is trained with forward KL against the
             per-turn views actually seen (``teacher_score`` over the recorded
             events) -- or with plain CE (kl_target=false), the hard-target
             limit of the same objective.
"""
from __future__ import annotations

import json
import os
import re
import threading
from typing import Any, Optional

from sediment.config import StreamConfig
from sediment.harness import tokens
from sediment.spans import spans_from_lengths
from sediment.types import HindsightResult, Message, SpanScore, Trajectory

STEP_OPEN = "<experience_hint stepwise>"
STEP_CLOSE = "</experience_hint>"
_STEP_RE = re.compile(r"\s*<experience_hint stepwise>.*?</experience_hint>\s*", re.DOTALL)
FEEDBACK_OPEN = "<experience_hint previous_attempt>"
_FEEDBACK_RE = re.compile(
    r"\s*<experience_hint previous_attempt>.*?</experience_hint>\s*", re.DOTALL
)
# Instance-value leak filter: 3+ digit runs (ids, years, amounts), long hex,
# or quoted literals containing digits. Principles must survive without them.
_LEAK_RE = re.compile(r"\d{3,}|[0-9a-fA-F]{8,}|['\"][^'\"]{0,60}\d[^'\"]{0,60}['\"]")
_ITEM_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.{10,})$")


def strip_stepwise(text: str) -> str:
    return _STEP_RE.sub("\n", text).strip("\n") if STEP_OPEN in text else text


def render_hint(items: list[str], max_chars: int) -> str:
    head = ("Guidance distilled from past tasks (general principles; tools and "
            "ids in the current environment may differ):")
    body: list[str] = []
    for it in items:
        cand = body + [f"- {it}"]
        text = "\n".join([STEP_OPEN, head, *cand, STEP_CLOSE])
        if len(text) > max_chars and body:
            break
        body = cand
    return "\n".join([STEP_OPEN, head, *body, STEP_CLOSE])


class PrinciplePool:
    """Append-only jsonl pool of principle items, deduplicated lexically.

    Reads during a window race only with the between-window ``add`` in the
    scheduler (extraction is gathered, then added once, before any scoring or
    redo of the same window), but a lock keeps stray concurrent adds safe.
    """

    def __init__(self, path: str, max_items: int = 4000):
        self.path = path
        self.max_items = max_items
        self.items: list[dict[str, Any]] = []
        self._toks: list[set[str]] = []
        self._lock = threading.Lock()
        if path and os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        it = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.items.append(it)
                    self._toks.append(tokens(it["text"]))

    def __len__(self) -> int:
        return len(self.items)

    def add(self, items: list[dict[str, Any]]) -> int:
        added = 0
        with self._lock:
            new_lines = []
            for it in items:
                if len(self.items) >= self.max_items:
                    break
                tk = tokens(it["text"])
                if len(tk) < 3:
                    continue
                dup = any(len(tk & t) / max(len(tk | t), 1) > 0.66
                          for t in self._toks[-800:])
                if dup:
                    continue
                self.items.append(it)
                self._toks.append(tk)
                new_lines.append(json.dumps(it, ensure_ascii=False))
                added += 1
            if new_lines and self.path:
                with open(self.path, "a") as f:
                    f.write("\n".join(new_lines) + "\n")
        return added

    def candidates(self, query_text: str, n: int) -> list[dict[str, Any]]:
        q = tokens(query_text)
        scored = sorted(
            ((len(q & t), -i, it) for i, (t, it) in enumerate(zip(self._toks, self.items))),
            reverse=True)
        return [it for s, _, it in scored[:n] if s > 0]


# -- extraction (self-generated principles) --------------------------------

_EXTRACT_PROMPT = """You review an agent's attempt at a task and distill reusable lessons.

Task: {task}

Attempt ({outcome}):
{digest}

Write up to 3 general lessons a future agent should apply on SIMILAR tasks in
similar environments. Each lesson must be:
- a strategy, ordering rule, checking habit, or failure pattern to avoid
- at most 30 words, one line, starting with "- "
- fully generic: NO concrete ids, names, dates, numbers, or quoted values

Lessons:"""


def _digest(traj: Trajectory, max_lines: int = 14) -> str:
    from sediment.harness import first_call, is_error_obs

    lines: list[str] = []
    msgs = traj.messages
    for i, m in enumerate(msgs):
        if m.role != "assistant":
            continue
        call = first_call(m.content) or {}
        if call:
            name = str(call.get("name") or "(unnamed tool call)")
            args = json.dumps(call.get("arguments") or {}, ensure_ascii=False)[:160]
        else:
            # InterCode emits fenced SQL rather than tool-call JSON. Preserve
            # the executed action in the principle extractor's evidence; the
            # generic "(no tool call)({})" digest otherwise removes the exact
            # mistake the privileged teacher is meant to repair.
            try:
                from sediment.envs.intercode_sql import parse_sql_action

                sql, valid = parse_sql_action(m.content)
            except Exception:
                sql, valid = "", False
            name = "SQL" if valid else "assistant"
            args = " ".join((sql if valid else m.content).split())[:240]
        nxt = msgs[i + 1] if i + 1 < len(msgs) else None
        res = nxt.content if nxt is not None and nxt.role in ("tool", "user") else "(end)"
        tag = "ERROR" if is_error_obs(res) else "ok"
        lines.append(f"{len(lines) + 1}. {name}({args}) -> [{tag}] {res[:120]}")
    if len(lines) > max_lines:  # keep the head and the failing tail
        lines = lines[:6] + [f"... ({len(lines) - 10} steps omitted) ..."] + lines[-4:]
    return "\n".join(lines) if lines else "(no actions)"


def extract_principles(engine, traj: Trajectory, cfg: StreamConfig,
                       adapter: str) -> list[dict[str, Any]]:
    """<=3 leak-filtered principle items from one finished episode."""
    outcome = (f"SUCCEEDED, reward {traj.reward:.2f}" if traj.success
               else f"FAILED, reward {float(traj.reward or 0.0):.2f}")
    task_text = ""
    for m in traj.messages:
        if m.role == "user":
            task_text = m.content[:500]
            break
    prompt = _EXTRACT_PROMPT.format(task=task_text, outcome=outcome, digest=_digest(traj))
    out = engine.generate([Message("user", prompt)], adapter=adapter,
                          temperature=0.0, max_tokens=200)
    items: list[dict[str, Any]] = []
    for line in out.splitlines():
        m = _ITEM_RE.match(line)
        if not m:
            continue
        text = m.group(1).strip()
        if len(text) > 240 or _LEAK_RE.search(text):
            continue
        items.append({"text": text, "src": traj.task_id, "family": traj.env_family,
                      "kind": "success" if traj.success else "failure"})
        if len(items) >= 3:
            break
    return items


# -- step-wise selection ----------------------------------------------------

_SELECT_PROMPT = """You pick which experience notes are relevant to an agent's CURRENT state.

Task: {task}
Last action: {action}
Last result: {obs}

Candidate notes:
{cands}

Reply with ONLY the numbers of the {k} most relevant notes, comma-separated."""


def _state_of(messages: list[Message]) -> tuple[str, str]:
    action = obs = "(start of task)"
    for m in reversed(messages):
        if obs == "(start of task)" and m.role in ("tool", "user"):
            obs = strip_stepwise(m.content)[:300]
        if m.role == "assistant":
            action = m.content[:300]
            break
    return action, obs


def select_items(engine, pool: PrinciplePool, cfg: StreamConfig, adapter: str,
                 messages: list[Message], task_text: str) -> list[str]:
    """cfg.stepwise_k pool items for the state at the end of ``messages``."""
    action, obs = _state_of(messages)
    cands = pool.candidates(f"{task_text} {action} {obs}", cfg.stepwise_candidates)
    if len(cands) <= cfg.stepwise_k:
        return [c["text"] for c in cands]
    listing = "\n".join(f"{i + 1}. {c['text']}" for i, c in enumerate(cands))
    out = engine.generate(
        [Message("user", _SELECT_PROMPT.format(
            task=task_text[:400], action=action, obs=obs, cands=listing,
            k=cfg.stepwise_k))],
        adapter=adapter, temperature=0.0,
        max_tokens=cfg.stepwise_selector_max_tokens)
    picks: list[int] = []
    for tok in re.findall(r"\d+", out):
        i = int(tok) - 1
        if 0 <= i < len(cands) and i not in picks:
            picks.append(i)
        if len(picks) >= cfg.stepwise_k:
            break
    if not picks:  # unparseable selector output: fall back to the prefilter
        picks = list(range(cfg.stepwise_k))
    return [cands[i]["text"] for i in picks]


class StepwiseHinter:
    """Rolling per-turn injection for a guided rollout (off-policy redo).

    ``pre_generate`` mutates the message list in place: drops any earlier
    stepwise segment, selects for the current state, and appends the hint to
    the newest user/tool message. Message COUNT never changes, so recorded
    attach indices stay valid in the stored (stripped) student view and each
    turn's exact context is reconstructible from ``events``.
    """

    def __init__(self, pool: PrinciplePool, engine, adapter: str,
                 cfg: StreamConfig, task_text: str):
        self.pool, self.engine, self.adapter = pool, engine, adapter
        self.cfg, self.task_text = cfg, task_text
        self.events: list[dict[str, Any]] = []
        self._gen_i = 0

    def pre_generate(self, messages: list[Message]) -> None:
        gen_i = self._gen_i
        self._gen_i += 1
        for i, m in enumerate(messages):
            if m.role in ("user", "tool") and STEP_OPEN in m.content:
                messages[i] = Message(m.role, strip_stepwise(m.content))
        try:
            items = select_items(self.engine, self.pool, self.cfg, self.adapter,
                                 messages, self.task_text)
        except Exception:  # selector failure never kills the episode
            items = []
        if not items:
            return
        attach = next((i for i in range(len(messages) - 1, -1, -1)
                       if messages[i].role in ("user", "tool")), None)
        if attach is None:
            return
        hint = render_hint(items, self.cfg.stepwise_max_chars)
        messages[attach] = Message(messages[attach].role,
                                   messages[attach].content + "\n\n" + hint)
        self.events.append({"gen_i": gen_i, "attach": attach, "text": hint,
                            "n_items": len(items)})


class AttemptFeedbackHinter:
    """State-aligned privileged feedback from one failed first attempt.

    Redo turn ``t`` sees first-attempt turn ``t`` only while every earlier redo
    action and observation exactly matches the first-attempt prefix. At the
    first difference, no later old feedback is injected: the trajectories are
    now in different conversational/observation states. ``bare_view`` strips
    all privileged segments before CE/DPO training.
    """

    def __init__(self, first: Trajectory):
        self.turns: list[tuple[str, str]] = []
        for i, message in enumerate(first.messages):
            if message.role != "assistant":
                continue
            feedback = next(
                (m.content for m in first.messages[i + 1:]
                 if m.role in ("tool", "user")),
                "(no environment feedback recorded)",
            )
            self.turns.append((message.content, feedback))
        self.events: list[dict[str, Any]] = []
        self._gen_i = 0
        self._aligned = True

    @staticmethod
    def _render(turn: int, action: str, feedback: str) -> str:
        return "\n".join([
            FEEDBACK_OPEN,
            f"At turn {turn} of the previous attempt:",
            "Previous action:",
            action,
            "Environment feedback:",
            feedback,
            "Re-select the best action for the current redo state using this evidence.",
            STEP_CLOSE,
        ])

    def pre_generate(self, messages: list[Message]) -> None:
        gen_i = self._gen_i
        self._gen_i += 1
        for i, message in enumerate(messages):
            if (message.role in ("user", "tool")
                    and FEEDBACK_OPEN in message.content):
                messages[i] = Message(
                    message.role,
                    _FEEDBACK_RE.sub("\n", message.content).strip("\n"),
                )
        # The previous redo turn has now executed. Keep consuming old feedback
        # only if its action AND resulting observation are byte-identical to the
        # first-attempt turn. This conservative equality is the state-validity
        # gate; semantically equivalent SQL with different rendered history is
        # still treated as a divergence.
        if gen_i > 0 and self._aligned:
            redo_actions = [
                (i, m.content) for i, m in enumerate(messages)
                if m.role == "assistant"
            ]
            if len(redo_actions) < gen_i:
                self._aligned = False
            else:
                idx, action = redo_actions[gen_i - 1]
                observation = next(
                    (m.content for m in messages[idx + 1:]
                     if m.role in ("tool", "user")),
                    None,
                )
                old_action, old_observation = self.turns[gen_i - 1]
                self._aligned = (
                    action == old_action and observation == old_observation
                )
        if not self._aligned or gen_i >= len(self.turns):
            return
        attach = next(
            (i for i in range(len(messages) - 1, -1, -1)
             if messages[i].role in ("user", "tool")),
            None,
        )
        if attach is None:
            return
        action, feedback = self.turns[gen_i]
        hint = self._render(gen_i + 1, action, feedback)
        messages[attach] = Message(
            messages[attach].role,
            messages[attach].content + "\n\n" + hint,
        )
        self.events.append({
            "gen_i": gen_i,
            "attach": attach,
            "text": hint,
            "source": "previous_attempt_action_feedback",
        })


# -- per-turn teacher scoring ----------------------------------------------

def teacher_score(engine, traj: Trajectory, cfg: StreamConfig, adapter: str,
                  pool: Optional[PrinciplePool] = None) -> Optional[HindsightResult]:
    """Per-assistant-turn top-k teacher distributions under step-wise views.

    traj.messages must be the BARE student view. When the trajectory was
    generated with a StepwiseHinter, ``traj.meta['stepwise_events']`` replays
    the exact per-turn contexts; otherwise (on-policy first attempts) the
    selection is made here from ``pool`` on the bare prefix -- extraction runs
    before scoring in the same window, so the pool already holds this
    episode's own lessons. Views only ever CHANGE message contents (never the
    count), so message indices map 1:1 and the scored turn's tokens are
    byte-identical to the student view's. Returns None when no turn got a
    teacher (empty pool), so callers can skip the task cleanly.
    """
    msgs = traj.messages
    events = traj.meta.get("stepwise_events")
    by_gen = ({int(e["gen_i"]): e for e in events} if events is not None else {})
    logp_o = engine.score(msgs, adapter=adapter)
    spans_o = spans_from_lengths(msgs, [len(x) for x in logp_o])
    a_spans = [s for s in spans_o if s.role == "assistant"]
    task_text = next((m.content[:500] for m in msgs if m.role == "user"), "")

    spans: list[SpanScore] = []
    teacher: list[list[dict[int, float]]] = []
    for ordinal, sp in enumerate(a_spans):
        if events is not None:
            e = by_gen.get(ordinal)
            if e is None:
                continue
            hint, attach = str(e["text"]), int(e["attach"])
        else:
            items = select_items(engine, pool, cfg, adapter, msgs[: sp.index], task_text)
            if not items:
                continue
            hint = render_hint(items, cfg.stepwise_max_chars)
            attach = next((i for i in range(sp.index - 1, -1, -1)
                           if msgs[i].role in ("user", "tool")), None)
            if attach is None:
                continue
        view = list(msgs)
        view[attach] = Message(view[attach].role, view[attach].content + "\n\n" + hint)
        ids_w, dists_w = engine.score_topk(view, adapter=adapter, k=cfg.kl_topk)
        if len(dists_w[sp.index]) != len(logp_o[sp.index]):
            raise RuntimeError(
                f"stepwise view changed the scored turn's tokenization: "
                f"{len(dists_w[sp.index])} vs {len(logp_o[sp.index])} "
                f"(task {traj.task_id}, msg {sp.index})")
        deltas = [float(d.get(t, min(d.values()) if d else 0.0) - b)
                  for t, d, b in zip(ids_w[sp.index], dists_w[sp.index], logp_o[sp.index])]
        spans.append(SpanScore(msg_idx=sp.index, role="assistant", deltas=deltas))
        teacher.append(dists_w[sp.index])
    if not spans:
        return None
    act = [d for s in spans for d in s.deltas]
    return HindsightResult(task_id=traj.task_id, spans=spans, obs_surprise=0.0,
                           act_gain=sum(act) / len(act), teacher=teacher)
