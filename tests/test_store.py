from pathlib import Path

from akd.store import Store
from akd.selector import Decision


def test_roundtrip(tmp_path):
    store = Store(tmp_path)
    dkey = ("site", ("id",), ("dev",), (("args",),))
    decision = Decision(
        winner=1,
        candidates=("a", "b"),
        timings=((0, 1.5), (1, 1.2)),
        pruned=((2, "boom"),),
        compile_ms=((0, 12.0),),
        unstable=True,
        reps=4,
        benched_at=123.0,
    )
    store.save(dkey, decision)
    loaded = store.load(dkey)
    assert loaded == decision


def test_load_missing_returns_none(tmp_path):
    assert Store(tmp_path).load(("nope",)) is None


def test_version_mismatch_invalidates(tmp_path, monkeypatch):
    store = Store(tmp_path)
    dkey = ("k",)
    store.save(dkey, Decision(winner=0))
    import akd.store as store_mod

    monkeypatch.setattr(store_mod, "_versions", lambda: {"akd": "other"})
    assert store.load(dkey) is None


def test_corrupt_file_returns_none(tmp_path):
    store = Store(tmp_path)
    dkey = ("k",)
    store._path(dkey).write_text("{not json")
    assert store.load(dkey) is None
