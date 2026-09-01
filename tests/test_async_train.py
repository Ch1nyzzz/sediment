"""AsyncTrainer (cfg.async_train): stepping from the buffer, age eviction,
FIFO cap, success-boosted sampling, pinned versions never unloaded."""
from __future__ import annotations

import time

from sediment.async_train import AsyncTrainer
from sediment.config import StreamConfig
from sediment.types import AdapterVersion, TrainSample, UpdateCandidate


def _sample(tid):
    return TrainSample(task_id=tid, messages=[], token_weights_by_msg=[])


class _Registry:
    def __init__(self):
        self.n = 0
        self.cur = AdapterVersion("v0000", None, None)
        self._by_name = {"v0000": self.cur}

    def current(self):
        return self.cur

    def publish(self, candidate, parent, provenance):
        self.n += 1
        self.cur = AdapterVersion(f"v{self.n:04d}", f"/fake/{self.n}", parent)
        self._by_name[self.cur.name] = self.cur
        return self.cur

    def _set_current(self, name):
        self.cur = self._by_name[name]


class _Engine:
    def __init__(self, registry=None):
        self.loaded, self.unloaded = [], []
        self._registry = registry

    def load_adapter(self, version):
        if self._registry is not None:
            # two-phase publish: the pointer must still be parked on the OLD
            # version while engines load (404-race regression check)
            assert self._registry.current().name != version.name
        self.loaded.append(version.name)

    def unload_adapter(self, name):
        self.unloaded.append(name)


def _trainer_fn(samples, parent, cfg, workdir):
    return UpdateCandidate(candidate_id="c", task_ids=[s.task_id for s in samples],
                           adapter_path=None, parent=parent.name, train_stats={})


def _make(**kw):
    cfg = StreamConfig(async_train=True, replay_min=2, replay_batch=2,
                       replay_cap=4, replay_max_age=3, async_keep_versions=2, **kw)
    reg = _Registry()
    eng = _Engine(registry=reg)
    return AsyncTrainer(_trainer_fn, reg, [eng], cfg, workdir="/tmp/x"), reg, eng


def _spin(t, cond, timeout=5.0):
    t.start()
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        time.sleep(0.05)
    t.stop()


def test_steps_and_reuse_and_eviction():
    t, reg, eng = _make()
    t.add([(1.0, _sample("a"), False), (1.0, _sample("b"), False)])
    _spin(t, lambda: t.step >= 4)
    # replay_max_age=3: samples born at step 0 evict at step 4 -> exactly 4 steps
    assert t.step == 4 and reg.n == 4
    assert t.trained_passes == 8  # 2 samples x 4 steps: reuse across steps
    # replay_max_age=3: both samples born at step 0 must be evicted by now
    assert len(t._buf) == 0
    assert eng.loaded[: reg.n] == [f"v{i:04d}" for i in range(1, reg.n + 1)]


def test_fifo_cap():
    t, _, _ = _make()
    t.add([(1.0, _sample(str(i)), False) for i in range(10)])
    assert len(t._buf) == 4  # replay_cap
    assert [e["sample"].task_id for e in t._buf] == ["6", "7", "8", "9"]


def test_success_boost_bias():
    t, _, _ = _make(replay_success_boost=1000.0)
    t.add([(1.0, _sample("fail"), False), (1.0, _sample("ok"), True),
           (1.0, _sample("ok2"), True), (1.0, _sample("fail2"), False)])
    picks = [e["sample"].task_id for _ in range(50) for e in t._draw()]
    frac_ok = sum(p.startswith("ok") for p in picks) / len(picks)
    assert frac_ok > 0.9


def test_pin_blocks_unload():
    t, reg, eng = _make()
    t.add([(1.0, _sample("a"), False), (1.0, _sample("b"), False)])
    t.pin("v0001")
    _spin(t, lambda: t.step >= 4)
    assert "v0001" not in eng.unloaded  # pinned survives keep_versions=2
    assert any(n in eng.unloaded for n in ("v0002", "v0003"))
    t.unpin("v0001")


def test_waitable_reflects_buffer_park_and_liveness():
    t, _, _ = _make()
    assert not t.waitable()  # empty buffer: pacing must not wait
    t.add([(1.0, _sample("a"), False), (1.0, _sample("b"), False)])
    assert t.waitable()  # enough samples, thread startable
    t.parked = True
    assert not t.waitable()
    t.parked = False
    t._buf.clear()
    assert not t.waitable()


def test_drain_finishes_age_bounded_replay():
    t, reg, _ = _make()
    t.start()
    t.add([(1.0, _sample("a"), False), (1.0, _sample("b"), False)])
    t.stop(drain=True)
    # Birth step 0 with Amax=3 is eligible at steps 0,1,2,3, then evicted.
    assert t.step == 4
    assert reg.n == 4
    assert len(t._buf) == 0


def test_replay_max_uses_is_a_hard_per_sample_cap():
    t, reg, _ = _make(replay_max_uses=2)
    t.add([(1.0, _sample("a"), False), (1.0, _sample("b"), False)])
    _spin(t, lambda: t.step >= 2)
    assert t.step == 2
    assert reg.n == 2
    assert t.trained_passes == 4
    assert len(t._buf) == 0


def test_replay_quota_is_consumed_only_after_a_successful_step():
    calls = 0

    def fail_once(samples, parent, cfg, workdir):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic optimizer failure")
        return _trainer_fn(samples, parent, cfg, workdir)

    cfg = StreamConfig(async_train=True, replay_min=2, replay_batch=2,
                       replay_cap=4, replay_max_age=0, replay_max_uses=1,
                       async_keep_versions=2)
    reg = _Registry()
    t = AsyncTrainer(fail_once, reg, [_Engine(registry=reg)], cfg, workdir="/tmp/x")
    t.add([(1.0, _sample("a"), False), (1.0, _sample("b"), False)])
    batch = t._draw()
    assert [e["uses"] for e in batch] == [0, 0]
    try:
        t._step_once(batch)
    except RuntimeError:
        pass
    else:
        raise AssertionError("synthetic optimizer failure was swallowed")
    assert [e["uses"] for e in batch] == [0, 0]
    t._step_once(batch)
    assert [e["uses"] for e in batch] == [1, 1]
    assert t.trained_passes == 2


def test_capped_drain_finishes_every_sample_quota_with_tail_batches():
    batch_sizes = []

    def trainer(samples, parent, cfg, workdir):
        batch_sizes.append(len(samples))
        return _trainer_fn(samples, parent, cfg, workdir)

    cfg = StreamConfig(async_train=True, replay_min=2, replay_batch=2,
                       replay_cap=4, replay_max_age=0, replay_max_uses=5,
                       async_keep_versions=2)
    reg = _Registry()
    t = AsyncTrainer(trainer, reg, [_Engine(registry=reg)], cfg, workdir="/tmp/x")
    t.start()
    t.add([(1.0, _sample("a"), False), (1.0, _sample("b"), False),
           (1.0, _sample("late-tail"), False)])
    t.stop(drain=True)
    assert t.trained_passes == 15  # three samples, exactly five uses each
    assert len(t._buf) == 0
    assert 1 in batch_sizes  # below replay_min tail was still completed


def test_replay_geometry_is_validated():
    cfg = StreamConfig(async_train=True, replay_min=8, replay_batch=9,
                       replay_cap=512)
    try:
        AsyncTrainer(_trainer_fn, _Registry(), [_Engine()], cfg, workdir="/tmp/x")
    except ValueError as e:
        assert "replay_batch <= replay_min" in str(e)
    else:
        raise AssertionError("invalid replay geometry was accepted")

    cfg = StreamConfig(async_train=True, replay_min=8, replay_batch=8,
                       replay_cap=512, replay_max_uses=-1)
    try:
        AsyncTrainer(_trainer_fn, _Registry(), [_Engine()], cfg, workdir="/tmp/x")
    except ValueError as e:
        assert "replay_max_uses must be non-negative" in str(e)
    else:
        raise AssertionError("negative replay_max_uses was accepted")


def test_pack_bins_respects_len_and_row_caps():
    from sediment.trainer import _pack_bins
    def e(s, t, r):
        return {"ids": [0] * s, "ctx": {"failure": ([0] * t, 0)}, "n_sel": r}
    # cost = max(student, teacher); rows capped separately
    bins = _pack_bins([e(100, 200, 10), e(150, 100, 10), e(900, 950, 10)], max_len=1000)
    assert bins == [[0, 1], [2]]  # 200+150 fits; 950 opens a new bin
    bins = _pack_bins([e(10, 10, 3000), e(10, 10, 3000)], max_len=1000, max_rows=4096)
    assert bins == [[0], [1]]  # row cap splits despite tiny lengths
    assert _pack_bins([], 1000) == []
