"""Shared datatypes for all sediment modules.

This file is a contract: module writers implement against it and must not
edit it. Everything is plain dataclasses + dicts so records round-trip
through jsonl without custom encoders.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Message":
        return Message(role=d["role"], content=d["content"])


@dataclass
class Trajectory:
    task_id: str
    env_family: str
    messages: list[Message]
    reward: Optional[float] = None  # None when the env gives no scalar
    success: Optional[bool] = None
    adapter: str = "base"  # adapter version name used to generate
    steps: int = 0
    is_retry: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Trajectory":
        d = dict(d)
        d["messages"] = [Message.from_dict(m) for m in d["messages"]]
        return Trajectory(**d)


@dataclass
class ExperienceBlock:
    """Evidence block injected for hindsight scoring / retry attempts."""

    text: str
    source_task_ids: list[str] = field(default_factory=list)
    includes_own_outcome: bool = False


@dataclass
class SpanScore:
    """Per-token hindsight deltas for one suffix message of a trajectory.

    deltas[i] = logp(token_i | ctx with block) - logp(token_i | ctx without).
    """

    msg_idx: int
    role: str  # "assistant" | "tool"
    deltas: list[float]


@dataclass
class HindsightResult:
    task_id: str
    spans: list[SpanScore]
    obs_surprise: float  # mean over max(delta, 0) on tool-role tokens
    act_gain: float  # mean delta on assistant-role tokens
    # Context-distillation teacher (cfg.kl_target), parallel to `spans`: per
    # token, the top-k {token_id: logprob} of the model READING the block.
    teacher: Optional[list[list[dict[int, float]]]] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("teacher", None)  # megabytes per trajectory; never persisted
        return d


@dataclass
class TrainSample:
    """One trajectory prepared for the trainer.

    token_weights aligns 1:1 with the tokenization of `messages` produced by
    the trainer's tokenizer via sediment.spans; weights are already floored
    and normalized by hindsight.to_train_sample. Non-supervised positions
    (prompt/system) carry weight 0.0.
    """

    task_id: str
    messages: list[Message]
    token_weights_by_msg: list[list[float]]  # parallel to messages
    # Context-distillation teacher (cfg.kl_target): per message, per token, the
    # top-k {token_id: logprob} of the model reading the evidence block. None
    # (or an empty per-message list) means plain CE on that message.
    teacher_by_msg: Optional[list[list[dict[int, float]]]] = None
    # Local-teacher contexts (cfg.kl_teacher="local"): block texts keyed by
    # "donor" / "failure"; the trainer injects each into the first user message
    # and forwards it itself at the KL positions.
    teacher_contexts: Optional[dict[str, str]] = None
    # Exact behavior-policy log-prob for each rendered message token, recorded
    # under the adapter that generated the trajectory BEFORE replay updates.
    # Empty message entries are unsupervised. cfg.replay_is_clip uses these
    # token-aligned values for truncated importance sampling.
    behavior_logprobs_by_msg: Optional[list[list[float]]] = None
    # Legacy sequence-level approximation retained for old serialized/in-memory
    # callers. New step-wise samples never write it.
    is_birth_logp: Optional[float] = None
    # Multiplicative scale applied after the sample's token-mass normalization.
    # Step-wise turn decay must live here: multiplying every token weight in a
    # one-turn sample would otherwise cancel between numerator and denominator.
    loss_scale: float = 1.0
    # Preference pair (cfg.critic_mode="dpo_steps"): the rejected assistant
    # action at the SAME prefix -- `messages` holds prefix + the chosen action
    # as its last message. When set, the trainer optimises the Bradley-Terry
    # log-ratio of the two continuations instead of weighted CE, and
    # token_weights_by_msg is only carried for the [gates] forensics line.
    rejected: Optional[str] = None
    # Whole-trajectory preference pair. ``messages`` is the complete successful
    # privileged redo in the stripped student view; this is the complete failed
    # first attempt. The trainer scores every assistant action token on each
    # trajectory and never treats environment observations as policy outputs.
    rejected_messages: Optional[list[Message]] = None


@dataclass
class UpdateCandidate:
    candidate_id: str
    task_ids: list[str]
    adapter_path: str  # dir with adapter weights, or stub metadata json
    parent: str  # adapter version name it was trained from
    train_stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateDecision:
    candidate_id: str
    passed: bool
    magnitude: float  # aggregated surprise that proposed this candidate
    behavioral_changed: Optional[float] = None  # frac replayed states w/ changed action
    probe_delta: Optional[float] = None  # probe success delta vs parent
    probe_rate: Optional[float] = None  # candidate probe success
    parent_rate: Optional[float] = None  # parent probe success
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdapterVersion:
    name: str  # "v0000" = base (no adapter); "v0001", ...
    path: Optional[str]  # None for base
    parent: Optional[str]
    provenance: list[str] = field(default_factory=list)  # consolidated task ids


@dataclass
class StreamRecord:
    """One row of the streaming learning curve (first attempts only)."""

    task_id: str
    window: int
    adapter: str
    success: Optional[bool]
    reward: Optional[float]
    gated_in: bool = False
    retried: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
