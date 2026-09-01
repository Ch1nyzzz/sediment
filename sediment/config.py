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
    gpu_memory_utilization: float = 0.85
    # Evaluation-only warm start. The adapter is copied into the run-local
    # registry as v0001 before any task is served. Training runs leave this
    # empty and build their adapter lineage online from v0000.
    initial_adapter_path: str = ""

    # stream protocol
    window_size: int = 16
    staleness_windows: int = 1  # window k served by adapter published at k-1
    max_steps: int = 30  # env steps per episode
    # Optional strict episode budget over the rendered multi-turn transcript.
    # 0 disables it. When enabled, success, max_steps, or this token budget
    # ends the episode as soon as the first condition is reached.
    episode_token_budget: int = 0
    # Optional non-mutating state-score checkpoints for budget curves. A
    # naturally finished episode carries its terminal reward forward to later
    # checkpoints; a truncated episode is always force-scored at max_steps.
    reward_checkpoint_steps: list[int] = field(default_factory=list)
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
    # What the teacher block shows of the CURRENT task's own first attempt:
    # "outcome" = SUCCEEDED/FAILED tag, reward, final feedback only (the body
    # cannot be copied from context); "full" = the complete failed trajectory,
    # every action and observation -- the self-feedback teacher. Only sound
    # with kl_states="retry": the block then holds the first attempt verbatim,
    # and a teacher scored on those very tokens measures copying, not belief
    # change (icl_gate forensics, 08-25).
    own_view: str = "outcome"  # "outcome" | "full"
    # Whose states carry the KL positions. "first": the bare first attempt,
    # the student's own on-policy sample, every action token. "diverge": the
    # same student states, but only up to the first turn where an independently
    # sampled, block-conditioned redo of a FAILED first attempt differs from
    # it. This is a localization heuristic, not a common-random-number causal
    # certificate: sampling noise can also move the divergence point. The
    # later first-attempt turns sit verbatim in the block with nothing
    # certifying the teacher is not just copying them, so they get no target.
    # "retry": the redo's own states (teacher-sampled,
    # off-policy for the student), stored and trained in the student view --
    # the ablation. Both redo modes propose nothing after a successful first
    # attempt and use the redo's success as the propose_rule="advantage" filter.
    # "fork": like "diverge" but ONLY the turn where the redo deviates carries
    # weight -- the earlier turns sit verbatim in the block (copy bias).
    kl_states: str = "first"  # "first" | "diverge" | "fork" | "retry"
    # harness arms (08-27, after Recuris 2608.24876): memory at execution
    # events instead of one standing block, and a harness-verified goal
    # ledger. Neither makes an extra model call; intercepted drafts are not
    # env steps; none of these harness edits controls generation randomness.
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
    gate_thr: float = 0.0
    # binary mode: a task is proposed iff its gated action tokens (n_pos+n_neg)
    # reach min_gate_tokens -- ONE filter, the same one that picks the tokens
    # (no obs-surprise G1 / top-K by surprise). Dose = fixed optimizer steps
    # per merge via gradient accumulation over all proposed samples.
    min_gate_tokens: int = 8
    steps_per_merge: int = 0  # 0 = legacy (micro_batch as configured)
    # Fixed evidence per merge: proposed samples accumulate across windows and
    # a candidate is trained only once at least this many are pending (the
    # excess over max_candidate_samples waits for the next merge instead of
    # being dropped). 0 = one candidate per window from whatever it proposed,
    # so a window with a single repaired task drives a full-strength step.
    # Pending samples are not persisted across resume.
    min_merge_samples: int = 0
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
    # Where the teacher distribution comes from. "topk": the served engine's
    # prompt_logprobs top-k on the TEACHER context (a lower bound whose slack
    # sits exactly on the student's off-support modes -- the failed action).
    # "local": the trainer forwards the block-injected view itself at the KL
    # positions (same weights, no grad) and matches the FULL vocabulary; with
    # several contexts the targets combine in logit space
    #   log q* = log p + poe_alpha * (log q_donor - log p) + poe_beta * (log q_failure - log p)
    # (product of experts), i.e. the two residuals ADD in the update instead
    # of competing for attention in one prompt (donor+own in the redo prompt
    # hurt: 14 vs 30 repairs, 08-29).
    kl_teacher: str = "topk"  # "topk" | "local"
    # Generalized JSD (GKD / SDPO): alpha*KL(p||m) + (1-alpha)*KL(q||m),
    # m = alpha*p + (1-alpha)*q, teacher p detached. 0 = off (plain KL, the
    # direction set by kl_reverse). alpha=0.5 is the symmetric SDPO default.
    # Measured gradient geometry (08-31, do NOT expect mode installation):
    # on a repair token the student assigns probability q~0, the pull-up
    # gradient is FKL ~ -p (finite, grows with student error) but BOTH RKL
    # and JSD vanish ~q (every JSD term touches the student only through
    # q_j; at q=4.5e-5: FKL -1.0, JSD -2e-4, RKL -9e-4). JSD's actual value
    # is boundedness (loss <= log 2, robust to top-k truncation and teacher
    # tail noise) for all-position matching at moderate q -- a safer RKL,
    # not a weaker CE. For installing fork repairs use CE/FKL.
    # topk-teacher path only.
    kl_jsd_alpha: float = 0.0
    # Local-teacher support truncation: 0 = exact full vocabulary. Ordinary KL
    # keeps student top-k plus a grouped tail (historical behavior); JSD keeps
    # TEACHER top-k plus a grouped tail so a currently-low-probability privileged
    # mode cannot disappear from the installation objective. The field name is
    # retained for config compatibility (CLaaS/SDPO use k=100).
    kl_student_topk: int = 0
    # EMA self-teacher (SDPO/CLaaS Table 4: 0.01). 0 = the teacher shares the
    # student's weights instantly (zero lag). beta>0: the teacher forwards run
    # under a separate EMA copy of the trainable params, updated
    # teacher <- (1-beta)*teacher + beta*student after every optimizer step and
    # persisted across merges at {run_dir}/teacher_ema.pt. CLaaS C.4: the slow
    # teacher is what lets SDPO tolerate replay age 50 monotonically.
    kl_teacher_ema: float = 0.0
    teacher_contexts: str = "failure"  # comma list of "donor" | "failure" | "feedback"
    # Multi-turn standard: compile every executed assistant decision into one
    # state-aligned sample. The student sees H_{t-1}; the local teacher sees the
    # same history plus the observation returned after that decision; the
    # original reasoning+action remains a causal continuation. Both successful
    # and failed trajectories enter replay.
    stepwise_feedback_distill: bool = False
    # OPD failure-only protocol: successful first attempts remain evaluation
    # records but do not yield privileged-feedback distillation samples.
    stepwise_feedback_failures_only: bool = False
    # Step-wise principle-pool experience (2606.04703, self-generated setting;
    # sediment/stepwise.py). The pool holds <=30-word leak-filtered principle
    # items the serving model distills from every finished episode; before
    # every generation of a guided redo (kl_states="retry") the selector
    # injects the stepwise_k most state-relevant items into the newest
    # observation, and for kl_states="first" the same per-turn views define
    # the teacher distributions scored post-hoc on the bare first attempt.
    stepwise_experience: bool = False
    stepwise_pool_path: str = ""  # default: pool.jsonl next to buffer_path
    stepwise_k: int = 3  # items injected per turn
    stepwise_candidates: int = 24  # lexical prefilter fed to the LLM selector
    stepwise_extract: bool = True  # distill principles from every episode
    stepwise_max_pool: int = 4000
    stepwise_max_chars: int = 900  # rendered hint budget per turn
    stepwise_selector_max_tokens: int = 32
    # Multi-redo (best-of-N rejection sampling, 08-31 gate): a failed first
    # attempt is redone redo_block_samples times with the own-failure block,
    # redo_stepwise_samples times with principle-pool injection, and/or
    # redo_feedback_samples times with the first action/environment feedback
    # from that same failed first attempt. Redo turn t receives first-attempt
    # turn t only while every earlier redo action+observation exactly matches
    # the first-attempt prefix. At the first divergence, later old feedback is
    # disabled and the redo follows only its own state.
    # (mixed contexts save disjoint tasks: 4+4 union 5.0% vs 3.1%/3.8% pure).
    # Every redo is sampled independently by the serving engine. No request
    # seed or prompt-derived common-random-number control is used.
    redo_block_samples: int = 1
    redo_stepwise_samples: int = 0
    redo_feedback_samples: int = 0
    # retrieval used ONLY to build the trainer's donor context (the redo prompt
    # keeps retrieval_k; 0 there = own failure only)
    teacher_retrieval_k: int = 0
    poe_alpha: float = 1.0  # donor residual weight (local mode)
    poe_beta: float = 1.0  # own-failure residual weight (local mode)
    # "failure_wins": at positions where the donor raises the token the failure
    # teacher lowers (or vice versa), drop the donor residual there
    poe_conflict: str = "none"  # "none" | "failure_wins"
    # fork mode: drop the sample when the redo's tool-call argument values at
    # the fork are not copyable from the student's own context (they came from
    # the block -> the bare student would be trained to hallucinate them)
    fork_require_grounded: bool = False
    # Every new fork experiment uses parsed tool-call difference: narration or
    # reasoning-only changes do not define a fork. "text" remains solely for
    # reproducing historical audits made before this convention was locked.
    fork_locator: str = "tool_diff"  # "tool_diff" | "sql" | legacy "text"
    # Objective applied after a verified best-of-N redo at the tool-diff fork.
    # ce: target step only; margin_token: first token divergence with a raw-logit
    # margin; dpo: complete chosen/rejected step with a reference-policy ratio;
    # trajectory_dpo: complete successful redo vs complete failed first attempt.
    fork_objective: str = "ce"  # ce | ce_suffix | margin_token | dpo | trajectory_dpo
    # Soft fork prior for all-position distillation (kl_states="retry"/"diverge"):
    # the j-th supervised assistant turn (j=0 first) is scaled by turn_decay**j.
    # Forks sit in the first turns (first changed turn = turn 1 in 54-85% of
    # repairs, 08-29/30), so later turns -- where the teacher-sampled redo is
    # shaped most by the block -- get less weight. 1.0 = no decay.
    turn_decay: float = 1.0
    # Critic-guided redo (offline-trained tree critic served by scripts/critic_server.py).
    # "redo_rank": a failed first attempt is redone by sampling critic_k alternative
    # actions at the branch turn (critic_turns: "first" = the first action, "localize" =
    # the executed action the critic values least among the first critic_max_turn
    # turns), executing the critic's top-ranked candidate from the replayed prefix
    # and letting the served policy finish; the fork/CE path then treats it like
    # any other redo (the environment still verifies).
    # "dpo_steps": no redo and no environment verification at all -- the critic
    # scores every executed action of the first attempt (success or failure),
    # the critic_dpo_states lowest-valued turns are branched, critic_k
    # alternatives are DRAFTED there (never executed), the critic ranks them
    # together with the original action, and each (best, worst) pair becomes one
    # DPO sample. This is the arm that does not need a successful redo: sample
    # count is n_tasks * critic_dpo_states * critic_dpo_pairs, not the ~5% of
    # tasks whose own-failure redo happens to succeed.
    critic_url: str = ""
    critic_mode: str = ""  # "" | "redo_rank" | "redo_random" (same candidates, random pick) | "dpo_steps"
    critic_k: int = 3  # per request: banned-token alternatives + free samples (<= 2k+1 candidates)
    critic_turns: str = "first"  # "first" | "localize"
    critic_max_turn: int = 3
    critic_sample_temperature: float = 1.0
    critic_dpo_states: int = 2  # executed turns branched per task
    critic_dpo_pairs: int = 1  # (best, worst), (2nd best, 2nd worst), ... per state
    # The critic has two jobs here; each has its own null control.
    # "random": branch at random turns instead of the lowest-valued ones.
    critic_dpo_state_pick: str = "lowest"  # "lowest" | "random"
    # "random": keep the same drafts but pair them arbitrarily -- the arm that
    # says whether the preference direction carries any information at all.
    critic_dpo_rank: str = "critic"  # "critic" | "random"
    critic_dpo_min_margin: float = 0.0  # skip pairs whose critic gap is below this
    critic_dpo_include_original: bool = True  # rank the executed action with the drafts
    dpo_beta: float = 2.0  # Bradley-Terry temperature on the mean log-ratio
    dpo_reference: str = "parent"  # "parent" | stable adapter-disabled "base"
    margin_token_m: float = 1.0
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
    # Persist AdamW state across merges at {run_dir}/opt_state.pt (CLaaS trains
    # with one resident optimizer whose moments accumulate across all updates;
    # a fresh AdamW per merge never leaves moment cold-start at 1-4 steps).
    # Param-index keyed, so it requires an unchanged LoRA architecture.
    persist_opt_state: bool = False
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
    # anchor only the untaught ASSISTANT positions (prompt / observation tokens
    # are never generated; anchoring them costs a full forward per 1k positions)
    anchor_assistant_only: bool = False

    # --- async continual training (CLaaS-style, 08-30) ---
    # Background trainer thread: steps as soon as the replay buffer holds
    # replay_min samples, publishes + hot-loads the adapter after EVERY step.
    # Rollout keeps the window structure (all 16 first attempts start together
    # under the window-start version = predict-then-update intact); each new
    # window picks up the freshest published version. Requires
    # gate_validate=false; pair with persist_opt_state=true and a small lr.
    async_train: bool = False
    replay_cap: int = 512  # B_max: buffer capacity (FIFO beyond this)
    replay_min: int = 16  # B_min: don't step below this many buffered samples
    replay_batch: int = 16  # M: samples uniformly drawn per optimizer step
    replay_max_age: int = 100  # A_max: evict after this many optimizer steps
    # Per-entry successful-training cap. 0 keeps the historical age-bounded
    # behavior. A positive value suppresses age eviction and the end-of-stream
    # drain uses tail batches to bring every accepted entry to exactly N uses.
    replay_max_uses: int = 0
    # sampling weight multiplier for verified redo-success samples (the redo
    # passed the environment): >1 tilts replay toward anchored supervision
    replay_success_boost: float = 1.0
    # Token-level truncated importance correction for replayed samples. 0 =
    # off. When >0, every supervised token must carry its exact log-prob under
    # the collection adapter; replay multiplies that token's KL by
    # min(exp(logp_current - logp_behavior), clip). No sequence product and no
    # first-replay approximation. Local-teacher KL path only.
    replay_is_clip: float = 0.0
    async_keep_versions: int = 3  # published versions kept loaded on engines
    async_step_interval: float = 0.0  # optional pause between steps (seconds)
    # Pacing (slime-style bounded staleness, the missing half of CLaaS's
    # B_min valve): before starting window w+1, the stream waits until the
    # trainer has taken at least this many steps since window w began. 0 = no
    # pacing (08-31 plan-D lesson: 3 fast servers finished a 400-task stream
    # in 22 min while the trainer landed 6 steps -- near-zero total dose).
    async_min_steps_per_window: int = 0
    # Sequence packing for the local-teacher trainer (FA2 varlen via reset
    # position_ids): student views and teacher views are greedily binned into
    # sequences of at most pack_max_len tokens, one forward per bin instead of
    # per sample. Requires flash-attn; per-sample loss normalization unchanged.
    pack_samples: bool = False
    pack_max_len: int = 16384

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
