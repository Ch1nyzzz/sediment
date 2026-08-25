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

    # stream protocol
    window_size: int = 16
    staleness_windows: int = 1  # window k served by adapter published at k-1
    max_steps: int = 30  # env steps per episode
    retry_on_fail: bool = True
    max_retries: int = 1

    # buffer / retrieval
    buffer_path: str = "results/buffer.jsonl"
    retrieval_k: int = 4
    serve_experience: bool = False  # ICL arm: first attempt carries the retrieved block

    # experience block
    max_result_chars: int = 200
    max_block_chars: int = 4000  # total block budget; peers are dropped past it
    reflect: bool = False  # actor writes a reflection per first attempt; rendered with trajectories
    reflect_max_tokens: int = 400
    max_reflection_chars: int = 1200
    gate_error_actions: bool = False  # P1.6: zero action weights on ERROR-status steps
    train_channels: str = "both"  # "both" | "act" | "obs": which token channels get weight
    # signed action credit on EVERY action token (no step-status gate):
    # δ > pos_thr -> +min(δ, pos_cap) CE; δ < -neg_thr -> -min(-δ, neg_cap)
    # unlikelihood -log(1-p) scaled by ul_lambda; |δ| inside the dead band -> 0.
    signed: bool = False
    pos_thr: float = 0.5
    neg_thr: float = 0.5
    pos_cap: float = 1.55
    neg_cap: float = 4.51
    ul_lambda: float = 0.1

    # gate
    gate_validate: bool = True  # False: skip G2/G3 measurement entirely (always pass)
    gate_min_surprise: float = 0.05  # G1 magnitude threshold on obs_surprise
    gate_recurrence: int = 2  # ledger: distinct tasks with similar surprise before write
    gate_replay_states: int = 8  # G2: replayed decision points per candidate
    gate_min_behavior_change: float = 0.1  # G2: min frac of changed actions
    gate_probe_tasks: int = 4  # G3: held-out probe tasks per validation
    gate_min_probe_delta: float = 0.0  # G3: probe success delta must be >= this

    # trainer
    trainer: str = "stub"  # "stub" | "torch"
    max_candidate_samples: int = 64  # top-K by surprise per window candidate
    # (dose control: total steps per merge = K * epochs; P1.7 knee ~2-4 steps)
    lora_r: int = 32
    lora_alpha: int = 16
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
    anchor_kl_coef: float = 0.0  # NCA anchoring (0 = off)

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

    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "StreamConfig":
        known = {f for f in StreamConfig.__dataclass_fields__}
        clean = {k: v for k, v in d.items() if k in known}
        return StreamConfig(**clean)
