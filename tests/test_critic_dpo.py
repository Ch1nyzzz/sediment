"""critic_mode="dpo_steps": preference pairs drafted at the critic's lowest-valued
executed turns -- no redo, no environment verdict -- and the trainer-side spans
those pairs are scored on."""
from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest

import sediment
import sediment.harness  # noqa: F401  (binds sediment.envs.base before the stub replaces sediment.envs)
from sediment.config import StreamConfig
from sediment.scheduler import run_stream
from sediment.trainer import _dpo_spans, _trajectory_dpo_spans
from sediment.types import AdapterVersion, Message, TrainSample, UpdateCandidate

from test_scheduler import (  # noqa: E402
    EASY,
    ScriptedMockEngine,
    StubBuffer,
    StubGate,
    StubRegistry,
    stub_modules,  # noqa: F401  (fixture)
)


class CriticEngine(ScriptedMockEngine):
    def _tokenizer(self):  # sample_siblings is stubbed; only the attribute matters
        return None


class FakeCritic:
    """Stands in for scripts/critic_server.py. Turn values: later turns are worth
    less. Candidate values: by a marker in the action text."""

    def __init__(self) -> None:
        self.turn_calls: list[list[dict]] = []
        self.score_calls: list[dict] = []

    def __call__(self, req, timeout=0):  # urllib.request.urlopen replacement
        payload = json.loads(req.data.decode())
        if req.full_url.endswith("/score_turns"):
            self.turn_calls.append(payload["messages"])
            turns = [{"msg": i, "A": -float(i)}
                     for i, m in enumerate(payload["messages"]) if m["role"] == "assistant"]
            body = {"turns": turns}
        else:
            self.score_calls.append(payload)
            body = {"A": [float(a.count("+")) - float(a.count("-")) for a in payload["actions"]]}
        return _Resp(body)


class _Resp:
    def __init__(self, body: dict) -> None:
        self._b = json.dumps(body).encode()

    def read(self) -> bytes:
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_siblings(engine, prefix, orig_action, tok, n_each, temperature, adapter="base"):
    return ([{"source": "banned", "action": "++best"}, {"source": "free", "action": "--worst"}],
            {"n_raw": 2, "n_distinct": 2})


@pytest.fixture()
def critic(monkeypatch):
    c = FakeCritic()
    monkeypatch.setattr("urllib.request.urlopen", c)
    branch = types.ModuleType("sediment.branch")
    branch.sample_siblings = fake_siblings
    monkeypatch.setitem(sys.modules, "sediment.branch", branch)
    monkeypatch.setattr(sediment, "branch", branch, raising=False)
    return c


def make_tasks(n: int) -> list[dict[str, Any]]:
    return [{"task_id": f"t{i}", "env_family": EASY, "payload": f"order-{i}"} for i in range(n)]


def run(tmp_path, critic_dpo_states=1, critic_dpo_pairs=1, critic_dpo_min_margin=0.0,
        include_original=True, n_tasks=4):
    cfg = StreamConfig(window_size=4, out_dir=str(tmp_path), seed=0, temperature=0.0,
                       retrieval_k=0, retry_on_fail=False, score_hindsight=False,
                       critic_url="http://127.0.0.1:9/", critic_mode="dpo_steps",
                       critic_max_turn=30, critic_dpo_states=critic_dpo_states,
                       critic_dpo_pairs=critic_dpo_pairs,
                       critic_dpo_min_margin=critic_dpo_min_margin,
                       critic_dpo_include_original=include_original,
                       min_merge_samples=99)  # hold: assert on the samples, not on a merge
    seen: list[list[TrainSample]] = []

    def trainer(samples, parent, cfg_, workdir):
        seen.append(list(samples))
        return UpdateCandidate(candidate_id="c", task_ids=[s.task_id for s in samples],
                               adapter_path=str(tmp_path / "c"), parent=parent.name)

    records = run_stream(make_tasks(n_tasks), [CriticEngine()], StubBuffer(), StubGate(cfg),
                         trainer, StubRegistry(), cfg)
    return records, seen


def test_pairs_come_from_every_task_without_a_redo(stub_modules, critic, tmp_path):
    records, _ = run(tmp_path)

    # every task -- the toy EASY family always succeeds -- still yields a pair
    assert [r.success for r in records] == [True] * 4
    assert [r.meta["dpo_pairs"] for r in records] == [1] * 4
    assert all(r.meta["proposed"] for r in records)
    assert not any(r.retried for r in records)  # no redo, ever
    # the critic saw the executed trajectory and then ranked drafts at one state
    assert len(critic.turn_calls) == 4 and len(critic.score_calls) == 4
    # the original executed action is ranked alongside the drafts
    assert critic.score_calls[0]["actions"] == ["++best", "--worst", "CORRECT"]


def test_sample_carries_the_chosen_and_rejected_continuations(stub_modules, critic, tmp_path):
    samples: list[TrainSample] = []

    def trainer(s, parent, cfg_, workdir):
        samples.extend(s)
        return UpdateCandidate(candidate_id="c", task_ids=[x.task_id for x in s],
                               adapter_path=str(tmp_path / "c"), parent=parent.name)

    cfg = StreamConfig(window_size=4, out_dir=str(tmp_path), seed=0, temperature=0.0,
                       retrieval_k=0, retry_on_fail=False, score_hindsight=False,
                       critic_url="http://127.0.0.1:9/", critic_mode="dpo_steps",
                       critic_max_turn=30, min_merge_samples=1, gate_validate=False)
    run_stream(make_tasks(4), [CriticEngine()], StubBuffer(), StubGate(cfg), trainer,
               StubRegistry(), cfg)
    assert len(samples) == 4  # one pair per task, merged in the same window
    s = samples[0]
    assert s.messages[-1].content == "++best"  # highest critic score
    assert s.rejected == "--worst"  # lowest
    assert s.messages[-2].role == "tool" or s.messages[-2].role == "user"  # prefix, not the action
    assert s.token_weights_by_msg[-1] == [1.0]
    assert len(s.token_weights_by_msg) == len(s.messages)


def test_min_margin_drops_pairs(stub_modules, critic, tmp_path):
    records, _ = run(tmp_path, critic_dpo_min_margin=99.0)
    assert [r.meta["dpo_pairs"] for r in records] == [0] * 4
    assert not any(r.meta["proposed"] for r in records)


class FakeTok:
    """Byte-level stand-in: one token per character, plus a template wrapper."""

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=False,
                            return_dict=False):
        text = "".join(f"<{m['role']}>{m['content']}" for m in msgs)
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(c) for c in text]


def test_dpo_spans_scores_only_the_action_tokens():
    tok = FakeTok()
    prefix = [Message("user", "hi")]
    spans = _dpo_spans(tok, prefix, "ab")
    assert spans is not None
    ids, pos, tgt = spans
    assert "".join(chr(i) for i in ids).endswith("<assistant>ab")
    assert tok.decode(tgt) == "ab"
    # row j predicts token j+1, so the first scored row is the one before "a"
    assert pos == [len(ids) - 3, len(ids) - 2]
    assert _dpo_spans(tok, prefix, "ab", max_prefix_tokens=1) is None
    assert _dpo_spans(tok, prefix, "   ") is None  # only framing/blank tokens


def test_trajectory_dpo_spans_scores_every_assistant_but_not_observations():
    tok = FakeTok()
    messages = [
        Message("user", "question"),
        Message("assistant", "SHOW X"),
        Message("tool", "schema"),
        Message("assistant", "SELECT Y"),
        Message("tool", "ok"),
    ]
    spans = _trajectory_dpo_spans(tok, messages)
    assert spans is not None
    _ids, _pos, tgt = spans
    # Whitespace-only byte tokens are framing-masked, as in single-action DPO.
    assert tok.decode(tgt) == "SHOWXSELECTY"
    assert "schema" not in tok.decode(tgt) and "ok" not in tok.decode(tgt)


def test_rank_control_keeps_the_drafts_and_drops_the_direction(stub_modules, critic, tmp_path):
    """critic_dpo_rank="random": same candidates, arbitrary preference -- the null
    arm. It must still produce pairs (the margin filter is a critic-only gate)."""
    samples: list[TrainSample] = []

    def trainer(s, parent, cfg_, workdir):
        samples.extend(s)
        return UpdateCandidate(candidate_id="c", task_ids=[x.task_id for x in s],
                               adapter_path=str(tmp_path / "c"), parent=parent.name)

    cfg = StreamConfig(window_size=4, out_dir=str(tmp_path), seed=0, temperature=0.0,
                       retrieval_k=0, retry_on_fail=False, score_hindsight=False,
                       critic_url="http://127.0.0.1:9/", critic_mode="dpo_steps",
                       critic_max_turn=30, critic_dpo_rank="random",
                       min_merge_samples=1, gate_validate=False)
    run_stream(make_tasks(4), [CriticEngine()], StubBuffer(), StubGate(cfg), trainer,
               StubRegistry(), cfg)
    assert len(samples) == 4
    assert {s.messages[-1].content for s in samples} | {s.rejected for s in samples} \
        <= {"++best", "--worst", "CORRECT"}
    assert any(s.messages[-1].content != "++best" for s in samples)  # not the critic's order
