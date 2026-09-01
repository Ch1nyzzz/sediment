"""CLaaS-style asynchronous continual training (cfg.async_train).

A resident background thread owns a training replay buffer. Whenever the
buffer holds at least cfg.replay_min samples it draws cfg.replay_batch of them
(weighted: verified redo-success samples get cfg.replay_success_boost), runs
one train_candidate step FROM the currently published version, publishes the
result to the registry and hot-loads it into every engine -- so the policy
improves continuously while rollouts happen, and every sample is reused across
many optimizer steps instead of being trained once and discarded (CLaaS
Alg. 1; samples are evicted after cfg.replay_max_age steps).

Pair with cfg.persist_opt_state=true (Adam moments carry across steps) and
cfg.kl_teacher_ema>0 (the slow teacher is what makes replayed self-distillation
stable -- CLaaS C.4). The scheduler pins the version serving the current
window so it is never unloaded from under in-flight episodes.
"""
from __future__ import annotations

import random
import threading
import time
from typing import Any

from .config import StreamConfig
from .types import TrainSample


class AsyncTrainer:
    def __init__(self, trainer_fn, registry, engines, cfg: StreamConfig, workdir: str):
        if not (0 < cfg.replay_batch <= cfg.replay_min <= cfg.replay_cap):
            raise ValueError("async replay requires "
                             "0 < replay_batch <= replay_min <= replay_cap")
        if cfg.replay_max_age < 0:
            raise ValueError("replay_max_age must be non-negative")
        if cfg.replay_max_uses < 0:
            raise ValueError("replay_max_uses must be non-negative")
        if cfg.replay_success_boost <= 0:
            raise ValueError("replay_success_boost must be positive")
        self._trainer_fn = trainer_fn
        self._registry = registry
        self._engines = engines
        self._cfg = cfg
        self._workdir = workdir
        self._lock = threading.Lock()
        self._buf: list[dict[str, Any]] = []  # {rank, sample, success, birth, uses}
        self._pinned: set[str] = set()
        self._published: list[str] = []
        self._stop = threading.Event()
        self._draining = False
        self._rng = random.Random(cfg.seed)
        self.step = 0
        self.trained_passes = 0
        self._failures = 0
        self.parked = False
        self._thread = threading.Thread(target=self._loop, name="async-trainer", daemon=True)

    # -- scheduler-facing API (main thread) --------------------------------
    def start(self) -> None:
        self._thread.start()

    def add(self, items: list[tuple[float, TrainSample, bool]]) -> int:
        """Add (rank, sample, verified_success) entries; returns buffer size."""
        with self._lock:
            for rank, sample, success in items:
                self._buf.append({"rank": float(rank), "sample": sample,
                                  "success": bool(success), "birth": self.step,
                                  "uses": 0})
            drop = len(self._buf) - self._cfg.replay_cap
            if drop > 0:  # FIFO beyond capacity
                del self._buf[:drop]
            return len(self._buf)

    def pin(self, name: str) -> None:
        """Protect a version from unloading while a window still serves it."""
        with self._lock:
            self._pinned.add(name)

    def unpin(self, name: str) -> None:
        with self._lock:
            self._pinned.discard(name)

    def waitable(self) -> bool:
        """Can the trainer still advance? (pacing must not wait otherwise)"""
        with self._lock:
            self._evict_expired_locked()
            enough = len(self._buf) >= self._cfg.replay_min
        started_and_dead = self._thread.ident is not None and not self._thread.is_alive()
        return (enough and not self.parked and not started_and_dead
                and not self._stop.is_set())

    def drain(self, timeout: float = 600.0) -> bool:
        """Let the resident trainer consume every currently eligible batch.

        CLaaS finishes a scenario split only after age eviction takes the
        buffer below B_min.  Without this end-of-stream drain, a final batch
        that reaches B_min can be collected but never trained.  Returns False
        on timeout or a parked/dead trainer so shutdown remains bounded.

        With ``replay_max_uses > 0``, drain has stronger semantics: every
        accepted sample is trained exactly that many times.  Tail batches may
        therefore be smaller than replay_min/replay_batch.  Age never evicts a
        capped sample before it reaches its use quota.
        """
        deadline = time.monotonic() + timeout
        with self._lock:
            self._draining = self._cfg.replay_max_uses > 0
        while True:
            with self._lock:
                self._evict_expired_locked()
                enough = bool(self._buf) if self._draining else (
                    len(self._buf) >= self._cfg.replay_min
                )
            if not enough:
                with self._lock:
                    self._draining = False
                return True
            if self.parked or (self._thread.ident is not None
                               and not self._thread.is_alive()):
                with self._lock:
                    self._draining = False
                return False
            if time.monotonic() >= deadline:
                print(f"[async] drain timed out after {timeout:.0f}s at step "
                      f"{self.step}, buffer {len(self._buf)}", flush=True)
                with self._lock:
                    self._draining = False
                return False
            time.sleep(0.1)

    def stop(self, *, drain: bool = False) -> None:
        if drain:
            ok = self.drain()
            print(f"[async] drain {'complete' if ok else 'incomplete'} at step "
                  f"{self.step}", flush=True)
        self._stop.set()
        self._thread.join(timeout=600)
        print(f"[async] stopped after {self.step} steps, "
              f"{self.trained_passes} sample passes, buffer {len(self._buf)}", flush=True)

    # -- trainer thread ----------------------------------------------------
    def _evict_expired_locked(self) -> None:
        max_uses = self._cfg.replay_max_uses
        if max_uses > 0:
            # A use quota is an exact replay contract.  Do not let wall-clock
            # optimizer age silently discard an under-trained late sample.
            self._buf = [e for e in self._buf if e["uses"] < max_uses]
        else:
            self._buf = [e for e in self._buf
                         if self.step - e["birth"] <= self._cfg.replay_max_age]

    def _draw(self) -> list[dict[str, Any]]:
        """Age-evict, then weighted sample without replacement
        (Efraimidis-Spirakis keys: u^(1/w))."""
        with self._lock:
            self._evict_expired_locked()
            min_size = 1 if self._draining and self._cfg.replay_max_uses > 0 \
                else self._cfg.replay_min
            if len(self._buf) < min_size:
                return []
            boost = float(self._cfg.replay_success_boost)
            keyed = sorted(
                self._buf,
                key=lambda e: self._rng.random() ** (1.0 / (boost if e["success"] else 1.0)),
                reverse=True)
            batch = keyed[: min(self._cfg.replay_batch, len(keyed))]
            return batch

    def _step_once(self, batch: list[dict[str, Any]]) -> None:
        current = self._registry.current()
        candidate = self._trainer_fn([e["sample"] for e in batch], current,
                                     self._cfg, workdir=self._workdir)
        version = self._registry.publish(candidate, current.name,
                                         provenance=[f"async-step-{self.step}"])
        # Two-phase publish: registry.publish points `current` at the new
        # version BEFORE any engine holds the weights -- a window starting in
        # that gap serves a 404 and kills the stream (nr_dec10, 08-31). Park
        # the pointer on the old version until every engine has loaded.
        self._registry._set_current(current.name)
        for engine in self._engines:
            for retry in range(3):  # a busy vLLM can time out the load; a
                try:                # published-but-unloaded version 404s later
                    engine.load_adapter(version)
                    break
                except Exception as e:
                    if retry == 2:
                        raise
                    print(f"[async s{self.step}] load {version.name} retrying: {e!r}",
                          flush=True)
                    time.sleep(5.0)
        self._registry._set_current(version.name)  # engines ready: go live
        with self._lock:
            self._published.append(version.name)
            keep = self._cfg.async_keep_versions
            excess = [n for n in self._published[:-keep] if n not in self._pinned]
            self._published = [n for n in self._published
                               if n in self._pinned or n in self._published[-keep:]]
        for name in excess:
            for engine in self._engines:
                getattr(engine, "unload_adapter", lambda n: None)(name)
        # Consume replay quota only after the optimizer result is published and
        # live.  A failed trainer/load attempt must not make an under-trained
        # sample look complete during the final exact-quota drain.
        with self._lock:
            for e in batch:
                e["uses"] += 1
            self.trained_passes += len(batch)
        ages = [self.step - e["birth"] for e in batch]
        uses = [e["uses"] for e in batch]
        print(f"[async s{self.step}] {current.name}->{version.name} "
              f"batch={len(batch)} age={min(ages)}-{max(ages)} "
              f"uses_max={max(uses)} buf={len(self._buf)}", flush=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            batch = self._draw()
            if not batch:
                time.sleep(2.0)
                continue
            try:
                self._step_once(batch)
                self._failures = 0
                self.step += 1
            except Exception as e:  # the stream must outlive a bad step
                self._failures += 1
                print(f"[async s{self.step}] step failed ({self._failures}): {e!r}",
                      flush=True)
                if self._failures >= 3:
                    self.parked = True
                    print("[async] 3 consecutive failures; trainer thread parking "
                          "(stream continues frozen)", flush=True)
                    self._stop.wait()
                    return
                time.sleep(5.0)
            if self._cfg.async_step_interval > 0:
                time.sleep(self._cfg.async_step_interval)
