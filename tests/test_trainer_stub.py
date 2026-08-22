import json
import os

from sediment.config import StreamConfig
from sediment.trainer import candidate_id_for, train_candidate
from sediment.types import AdapterVersion, Message, TrainSample


def make_sample(task_id="t1", weights=None):
    messages = [
        Message("system", "you are an agent"),
        Message("user", "do the task"),
        Message("assistant", "calling tool"),
        Message("tool", "Error: cannot cancel"),
    ]
    if weights is None:
        weights = [[0.0], [0.0, 0.0], [0.5, 1.5], [0.0, 2.0, 0.0]]
    return TrainSample(task_id=task_id, messages=messages, token_weights_by_msg=weights)


def test_stub_writes_adapter_meta(tmp_path):
    cfg = StreamConfig(trainer="stub")
    parent = AdapterVersion("v0000", None, None)
    samples = [make_sample("t1"), make_sample("t2", weights=[[0.0], [0.0], [1.0, 3.0], [0.0]])]
    cand = train_candidate(samples, parent, cfg, str(tmp_path))

    assert cand.candidate_id.startswith("cand-")
    assert cand.task_ids == ["t1", "t2"]
    assert cand.parent == "v0000"
    assert cand.adapter_path == os.path.join(str(tmp_path), cand.candidate_id)

    with open(os.path.join(cand.adapter_path, "adapter_meta.json")) as f:
        meta = json.load(f)
    assert meta["candidate_id"] == cand.candidate_id
    assert meta["task_ids"] == ["t1", "t2"]
    assert meta["parent"] == "v0000"
    # t1 flat weights: [0, 0, 0, 0.5, 1.5, 0, 2.0, 0] -> mean 0.5, max 2.0, 3 nonzero
    assert meta["samples"][0] == {"task_id": "t1", "mean": 0.5, "max": 2.0, "nonzero": 3}
    # t2 flat weights: [0, 0, 1.0, 3.0, 0] -> mean 0.8, max 3.0, 2 nonzero
    assert meta["samples"][1] == {"task_id": "t2", "mean": 0.8, "max": 3.0, "nonzero": 2}
    assert cand.train_stats["n_samples"] == 2


def test_candidate_id_deterministic(tmp_path):
    cfg = StreamConfig(trainer="stub")
    parent = AdapterVersion("v0000", None, None)
    c1 = train_candidate([make_sample()], parent, cfg, str(tmp_path / "a"))
    c2 = train_candidate([make_sample()], parent, cfg, str(tmp_path / "b"))
    assert c1.candidate_id == c2.candidate_id == candidate_id_for(["t1"], "v0000")
    # different parent or task set -> different id
    other = AdapterVersion("v0001", "/x", "v0000")
    c3 = train_candidate([make_sample()], other, cfg, str(tmp_path / "c"))
    c4 = train_candidate([make_sample("t9")], parent, cfg, str(tmp_path / "d"))
    assert len({c1.candidate_id, c3.candidate_id, c4.candidate_id}) == 3


def test_empty_weights_sample_stats(tmp_path):
    cfg = StreamConfig(trainer="stub")
    sample = TrainSample(task_id="t0", messages=[Message("user", "x")], token_weights_by_msg=[[]])
    cand = train_candidate([sample], AdapterVersion("v0000", None, None), cfg, str(tmp_path))
    with open(os.path.join(cand.adapter_path, "adapter_meta.json")) as f:
        meta = json.load(f)
    assert meta["samples"][0] == {"task_id": "t0", "mean": 0.0, "max": 0.0, "nonzero": 0}


def test_unknown_trainer_raises(tmp_path):
    import pytest

    cfg = StreamConfig(trainer="nope")
    with pytest.raises(ValueError):
        train_candidate([make_sample()], AdapterVersion("v0000", None, None), cfg, str(tmp_path))
