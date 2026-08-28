"""StreamConfig: one flat dataclass, CLI-overridable, jsonl-serializable."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class StreamConfig:
    # model / engine
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    engine: str = "mock"  # "mock" | "vllm"
    num_engines: int = 3  # serving workers (GPU 0..n-1)
    temperature: float = 0.7
    max_tokens: int = 2048
    max_model_len: int = 12288
    # Optional common-random-number control for paired ICL arms. The seed is a
    # stable hash of the prompt after removing any retrieved-memory block, so
    # two donor interventions start from the same sampling noise.
    generation_seed_mode: str = "none"  # "none" | "bare_prompt_hash"
    # Nonzero salts create independent, reproducible request-seed replicates
    # while preserving common seeds across all arms within one replicate.
    generation_seed_salt: int = 0
    gpu_memory_utilization: float = 0.85
    # Evaluation-only warm start. The adapter is copied into the run-local
    # registry as v0001 before any task is served. Training runs leave this
    # empty and build their adapter lineage online from v0000.
    initial_adapter_path: str = ""

    # stream protocol
    window_size: int = 16
    staleness_windows: int = 1  # window k served by adapter published at k-1
    max_steps: int = 30  # env steps per episode
    retry_on_fail: bool = True
    max_retries: int = 1

    # buffer / retrieval
    buffer_path: str = "results/buffer.jsonl"
    # Optional immutable donor banks used only for retrieval. Generated eval
    # trajectories still go to buffer_path and never contaminate this memory.
    retrieval_buffer_paths: list[str] = field(default_factory=list)
    retrieval_k: int = 4
    # Rank-wise ICL ablations use one donor at a time: offset=0 is top-1,
    # offset=1 is rank-2, etc. This measures whether the top-n candidate set
    # contains complementary interventions without concatenating them.
    retrieval_offset: int = 0
    retrieval_scope: str = "all"  # "all" | "same_family" | "cross_family"
    # "task" keeps distinct source task ids; "family" first takes at most one
    # donor per source environment family, then backfills if fewer than k
    # families exist. The latter prevents a nominal cross-family top-n set from
    # collapsing onto several near-duplicate donors from one source group.
    retrieval_diversity: str = "task"  # "task" | "family"
    # "task": historical str(payload)-to-task Jaccard; "task_text": compare
    # the natural-language payload["task"] to the donor task;
    # "task_text_success": successful source trajectories first, then natural
    # task similarity; "legacy": old +2 same-family route; "transfer": overlap
    # of canonical workflow features extracted from the query and donor.
    retrieval_score: str = "task"
    serve_experience: bool = False  # ICL arm: first attempt carries the retrieved block

    # experience block
    max_result_chars: int = 200
    max_block_chars: int = 16000  # soft cap; task + full stored reflection are mandatory
    # Cross-domain memories can omit source task/API details and expose only the
    # distilled lesson. This is an explicit experimental arm, not a truncation.
    # "matched_reflection" additionally gates canonical rules by intersection
    # with the current query's workflow features; no match means no injection.
    # "compiled_reflection" extractively merges and deduplicates lessons from
    # top-n peers into one short block before a single actor rollout.
    experience_view: str = "full"  # full | reflection_only | matched_reflection | compiled_reflection | legacy
    # harness arms (08-27, after Recuris 2608.24876): memory at execution
    # events instead of one standing block, and a harness-verified goal
    # ledger. Neither makes an extra model call; intercepted drafts are not
    # env steps; the frozen arm's first-generation seed is untouched.
    calltime_hints: bool = False
    calltime_pool_k: int = 4  # donor candidates (same retrieval as the block)
    calltime_triggers: str = "error,write"  # subset of error,write,tool
    calltime_max_hints: int = 4  # per episode and trigger type
    calltime_max_chars: int = 900  # per hint
    calltime_intercept: bool = True  # write/tool: hold the draft, re-draft
    calltime_style: str = "hold"  # "hold" (NOT executed, compare) | "confirm" (queued, re-issue)
    working_state: bool = False
    working_state_max_goals: int = 12
    memory_compile_max_rules: int = 8
    memory_compile_max_chars: int = 2400
    memory_compile_rule_max_chars: int = 260
    reflect: bool = False  # actor writes a reflection per first attempt; rendered with trajectories
    reflection_mode: str = "local"  # "local" | "transfer"
    reflect_max_tokens: int = 400
    max_reflection_chars: int = 1200
    gate_error_actions: bool = False  # P1.6: zero action weights on ERROR-status steps
    train_channels: str = "both"  # "both" | "act" | "obs": which token channels get weight
    # "relu": w = relu(δ) (magnitude-weighted); "binary": w = 1[δ > gate_thr]
    # (residual only selects tokens, plain CE on the selected set);
    # "sft": no residual at all -- every action token of a TEACHER trajectory
    # (rolled out with the block, trained with it stripped). The gated modes
    # credit ~20 tool-call-interior tokens out of ~4000 and leave the rest of
    # the distribution unconstrained, which measurably costs 0.3 nats of
    # anti-repetition prior per merge (scripts/policy_shift.py, 08-26).
    weight_mode: str = "relu"
    # which tasks contribute an update in weight_mode="sft":
    #   "all"       every task -- the hint's distribution shift is always matched
    #   "advantage" only where the hint actually helped: the bare attempt failed
    #               and the retry carrying the hint succeeded, i.e.
    #               reward(prompt+hint) > reward(prompt). A wrong hint otherwise
    #               gets its wrong posterior frozen into the weights.
    propose_rule: str = "all"
    grounded_weight: float = 1.0  # weight of copyable argument-value tokens (0.2 = damped)
    gate_thr: float = 0.0
    # binary mode: a task is proposed iff its gated action tokens (n_pos+n_neg)
    # reach min_gate_tokens -- ONE filter, the same one that picks the tokens
    # (no obs-surprise G1 / top-K by surprise). Dose = fixed optimizer steps
    # per merge via gradient accumulation over all proposed samples.
    min_gate_tokens: int = 8
    steps_per_merge: int = 0  # 0 = legacy (micro_batch as configured)
    # signed action credit on EVERY action token (no step-status gate):
    # δ > pos_thr -> +min(δ, pos_cap) CE; δ < -neg_thr -> -min(-δ, neg_cap)
    # unlikelihood -log(1-p) scaled by ul_lambda; |δ| inside the dead band -> 0.
    signed: bool = False
    # context distillation: the gate still SELECTS tokens (same thresholds), but
    # the objective on them becomes KL(teacher || student) with the teacher =
    # base model reading the evidence block. CE penalises unpredictability, so
    # it dumps 65% of its gradient on random hex ids the student cannot infer
    # (sft_gradient_audit, 08-26); KL penalises disagreement, and on those ids
    # the teacher is equally clueless -> zero gradient, no mask needed.
    kl_target: bool = False
    # reverse KL (student || teacher), sampled on-policy from the student: at
    # inference the model runs prompt-only, so the expectation must be under the
    # student's own distribution. Forward KL on teacher-sampled trajectories has
    # essentially no signal (0.008 nats, kl_screen 08-26).
    kl_reverse: bool = True
    kl_topk: int = 20  # teacher support per position (+ one tail bucket);
    # vLLM refuses more than its --max-logprobs, 20 by default
    pos_thr: float = 0.5
    neg_thr: float = 0.5
    pos_cap: float = 1.55
    neg_cap: float = 4.51
    ul_lambda: float = 0.1

    # gate
    # corpus collection (no gate, no retry): skip the with/without delta scoring
    # entirely -- it is the dominant per-window cost (60 s mean vs 23 s attempt).
    score_hindsight: bool = True
    # The historical hindsight block also summarized the current trajectory's
    # terminal outcome.  Disable this for pure memory distillation: the teacher
    # then differs from the student only by the pre-existing retrieved memory,
    # with no target-outcome leakage.
    hindsight_include_own_outcome: bool = True
    gate_validate: bool = True  # False: skip G2/G3 measurement entirely (always pass)
    gate_min_surprise: float = 0.05  # G1 magnitude threshold on obs_surprise
    gate_recurrence: int = 2  # ledger: distinct tasks with similar surprise before write
    gate_replay_states: int = 8  # G2: replayed decision points per candidate
    gate_min_behavior_change: float = 0.1  # G2: min frac of changed actions
    gate_probe_tasks: int = 4  # G3: held-out probe tasks per validation (corpus tail)
    probe_extra_before: int = 0  # extra G3 probes taken from just BEFORE the stream
    # window, so the stream itself stays identical across probe-set sizes
    gate_min_probe_delta: float = 0.0  # G3: probe success delta must be >= this
    # rollback (user 08-26): a merge whose probe delta <= -rollback_thr flags a
    # re-check; at the next validation the pre-merge version is probed too and,
    # if it beats both the current version and the new candidate by more than
    # rollback_margin, it is re-published (weights copied) and the new candidate
    # is discarded. 0 = off.
    rollback_thr: float = 0.0
    rollback_margin: float = 0.0

    # trainer
    trainer: str = "stub"  # "stub" | "torch"
    max_candidate_samples: int = 64  # top-K by surprise per window candidate
    # (dose control: total steps per merge = K * epochs; P1.7 knee ~2-4 steps)
    lora_r: int = 8  # 08-26: 32 gave the adapter room to memorise instance ids
    lora_alpha: int = 8  # alpha/r = 1, same effective scale as the r=32/alpha=32 arms
    lr: float = 5e-5
    epochs: int = 1
    micro_batch: int = 1
    max_seq_len: int = 12288
    w_floor: float = 0.0  # weight floor for supervised tokens
    # dose ∝ evidence: per-sample loss = Σ w·ce / max(Σ w, w_norm_floor). With 0 the
    # loss is a weighted mean, so a sample carrying ~1 unit of weight mass still
    # drives a full-strength step (forensics 08-25: act-only merges trained on
    # mass 1.7 / 0.6 wrecked probes). Signed: applied to each sign separately.
    w_norm_floor: float = 0.0
    # NCA anchoring: KL(base || student) on the positions the objective does
    # NOT supervise, so the update cannot pay for its loss with drift elsewhere.
    # Present since day one and never switched on; the 08-26 margin measurement
    # says that omission is what the six collapses have in common.
    anchor_kl_coef: float = 0.0

    # merge / registry
    registry_dir: str = "results/registry"
    merge_alpha: float = 0.5  # EMA weight of candidate on merge

    # eval
    snapshot_every_windows: int = 4
    snapshot_tasks: int = 0  # 0 = skip snapshots

    # data
    data_dir: str = "data"
    split: str = "toy"
    num_tasks: int = 32
    seed: int = 0
    out_dir: str = "results"
    run_id: str = "dev"
    resume: bool = False  # continue an existing run dir (buffer + registry) from its next window

    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "StreamConfig":
        known = {f for f in StreamConfig.__dataclass_fields__}
        clean = {k: v for k, v in d.items() if k in known}
        return StreamConfig(**clean)
