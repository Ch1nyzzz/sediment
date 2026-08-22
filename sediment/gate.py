"""Gate: decides when hindsight signal becomes a weight update.

G1 (`propose`) fires on a large observation surprise that recurs within an
env family. G2/G3 (`validate`) admit a trained candidate only if it changed
behavior on replayed decision points AND did not regress on probe tasks.
Pure logic: engine and probe runner are injected, so everything is
mock-testable with no heavy deps.
"""
from __future__ import annotations

from typing import Callable, Optional

from sediment.config import StreamConfig
from sediment.types import (
    AdapterVersion,
    GateDecision,
    HindsightResult,
    Message,
    Trajectory,
    UpdateCandidate,
)

# run_probe(adapter_name, probe_tasks) -> success rate in [0, 1]
ProbeFn = Callable[[str, list[dict]], float]


class Gate:
    """Residual gate over the update stream.

    `ledger` maps env_family -> list of above-threshold obs_surprise values
    seen this stream (count = len); it is plain data, exposed for
    inspection and jsonl serialization.
    """

    def __init__(self, cfg: StreamConfig) -> None:
        self.cfg = cfg
        self.ledger: dict[str, list[float]] = {}

    def propose(self, hr: HindsightResult, traj: Trajectory) -> bool:
        """G1: record an above-threshold surprise; fire once the trajectory's
        env_family has accumulated cfg.gate_recurrence of them (this one
        included). Below-threshold surprises leave the ledger untouched."""
        if hr.obs_surprise < self.cfg.gate_min_surprise:
            return False
        seen = self.ledger.setdefault(traj.env_family, [])
        seen.append(hr.obs_surprise)
        return len(seen) >= self.cfg.gate_recurrence

    def validate(
        self,
        candidate: UpdateCandidate,
        parent: AdapterVersion,
        engine,
        replay_states: list[list[Message]],
        probe_tasks: list[dict],
        run_probe: Optional[ProbeFn],
    ) -> GateDecision:
        """G2 behavioral A/B + G3 probe; the candidate passes only if both
        pass. A stage without the means to run (no replay states / no probe
        fn or tasks) is skipped and counts as a pass, noted in `reason`.
        Generation is greedy (temperature 0) so the A/B diff is meaningful.
        """
        cfg = self.cfg
        notes: list[str] = []

        behavioral_changed: Optional[float] = None
        g2_pass = True
        if replay_states:
            changed = 0
            for state in replay_states:
                a = engine.generate(state, adapter=parent.name,
                                    temperature=0.0, max_tokens=cfg.max_tokens)
                b = engine.generate(state, adapter=candidate.candidate_id,
                                    temperature=0.0, max_tokens=cfg.max_tokens)
                changed += int(a != b)
            behavioral_changed = changed / len(replay_states)
            g2_pass = behavioral_changed >= cfg.gate_min_behavior_change
            notes.append(
                f"G2 {'pass' if g2_pass else 'fail'}: {changed}/{len(replay_states)}"
                f" replayed actions changed ({behavioral_changed:.3f}"
                f" vs min {cfg.gate_min_behavior_change})"
            )
        else:
            notes.append("G2 skipped (no replay states), counts as pass")

        probe_delta: Optional[float] = None
        g3_pass = True
        if run_probe is not None and probe_tasks:
            cand_rate = run_probe(candidate.candidate_id, probe_tasks)
            parent_rate = run_probe(parent.name, probe_tasks)
            probe_delta = cand_rate - parent_rate
            g3_pass = probe_delta >= cfg.gate_min_probe_delta
            notes.append(
                f"G3 {'pass' if g3_pass else 'fail'}: probe {cand_rate:.3f}"
                f" vs parent {parent_rate:.3f}, delta {probe_delta:+.3f}"
                f" vs min {cfg.gate_min_probe_delta}"
            )
        else:
            why = "no probe fn" if run_probe is None else "no probe tasks"
            notes.append(f"G3 skipped ({why}), counts as pass")

        return GateDecision(
            candidate_id=candidate.candidate_id,
            passed=g2_pass and g3_pass,
            magnitude=self._magnitude(),
            behavioral_changed=behavioral_changed,
            probe_delta=probe_delta,
            reason="; ".join(notes),
        )

    def _magnitude(self) -> float:
        """Mean above-threshold obs_surprise accumulated in the ledger."""
        vals = [v for vs in self.ledger.values() for v in vs]
        return sum(vals) / len(vals) if vals else 0.0
