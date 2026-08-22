import json
import os

from sediment.config import StreamConfig
from sediment.merge import merge
from sediment.registry import Registry
from sediment.types import UpdateCandidate


def make_candidate(tmp_path, cid="cand-abc", task_ids=("t1",), parent="v0000"):
    d = tmp_path / cid
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_meta.json").write_text(json.dumps({"candidate_id": cid}))
    return UpdateCandidate(
        candidate_id=cid, task_ids=list(task_ids), adapter_path=str(d), parent=parent
    )


def test_base_and_initial_current(tmp_path):
    reg = Registry(str(tmp_path / "reg"))
    base = reg.base()
    assert (base.name, base.path, base.parent, base.provenance) == ("v0000", None, None, [])
    assert reg.current().name == "v0000"
    assert [v.name for v in reg.history()] == ["v0000"]


def test_publish_increments_and_updates_current(tmp_path):
    reg = Registry(str(tmp_path / "reg"))
    cand = make_candidate(tmp_path)

    v1 = reg.publish(cand, parent="v0000", provenance=["t1"])
    assert (v1.name, v1.parent, v1.provenance) == ("v0001", "v0000", ["t1"])
    assert os.path.isfile(os.path.join(v1.path, "adapter_meta.json"))  # contents copied
    with open(os.path.join(v1.path, "meta.json")) as f:
        meta = json.load(f)
    assert meta == {
        "name": "v0001",
        "parent": "v0000",
        "provenance": ["t1"],
        "created_from": cand.adapter_path,
    }
    assert reg.current().name == "v0001"

    # publish by path too; version increments and current moves
    v2 = reg.publish(cand.adapter_path, parent="v0001", provenance=["t1", "t2"])
    assert v2.name == "v0002"
    assert reg.current().name == "v0002"
    assert reg.get("v0001").provenance == ["t1"]
    assert [v.name for v in reg.history()] == ["v0000", "v0001", "v0002"]

    # a fresh Registry over the same dir sees the same state
    reg2 = Registry(reg.dir)
    assert reg2.current().name == "v0002"
    assert reg2.publish(cand, parent="v0002", provenance=[]).name == "v0003"


def test_current_pointer_fallback_without_symlink(tmp_path, monkeypatch):
    def no_symlink(*args, **kwargs):
        raise OSError("symlinks unavailable")

    monkeypatch.setattr(os, "symlink", no_symlink)
    reg = Registry(str(tmp_path / "reg"))
    v1 = reg.publish(make_candidate(tmp_path), parent="v0000", provenance=["t1"])
    assert not os.path.islink(os.path.join(reg.dir, "current"))
    assert os.path.isfile(os.path.join(reg.dir, "current.json"))
    assert reg.current().name == v1.name == "v0001"


def test_merge_stub_provenance_union(tmp_path):
    cfg = StreamConfig(trainer="stub")
    reg = Registry(str(tmp_path / "reg"))
    session = reg.publish(
        make_candidate(tmp_path, "cand-a", ["t1", "t2"]), parent="v0000", provenance=["t1", "t2"]
    )
    cand = make_candidate(tmp_path, "cand-b", ["t2", "t3"], parent=session.name)

    merged = merge(session, cand, cfg, reg)
    assert merged.name == "v0002"
    assert merged.parent == "v0001"
    assert merged.provenance == ["t1", "t2", "t3"]
    assert reg.current().name == "v0002"
    with open(os.path.join(merged.path, "meta.json")) as f:
        assert json.load(f)["provenance"] == ["t1", "t2", "t3"]


def test_merge_stub_from_base_session(tmp_path):
    cfg = StreamConfig(trainer="stub")
    reg = Registry(str(tmp_path / "reg"))
    cand = make_candidate(tmp_path, "cand-c", ["t7"])
    merged = merge(reg.base(), cand, cfg, reg)
    assert (merged.name, merged.parent, merged.provenance) == ("v0001", "v0000", ["t7"])
