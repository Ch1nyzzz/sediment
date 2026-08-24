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

    # experience block
    max_result_chars: int = 200
    max_block_chars: int = 4000  # total block budget; peers are dropped past it

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
