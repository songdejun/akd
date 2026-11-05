import inspect

import pytest

import akd
from akd import problem, registry
from akd.chain import Launch
from conftest import FakeDevice, FakeTensor, StubTarget


def test_tensor_fingerprint_is_value_based():
    fp1 = problem.value_fingerprint(FakeTensor((8, 16)), [])
    fp2 = problem.value_fingerprint(FakeTensor((8, 16)), [])
    assert fp1 == fp2
    assert fp1[0] == "T" and fp1[-1] == 0   # aligned


def test_misaligned_tensor_differs():
    aligned = problem.value_fingerprint(FakeTensor((8,)), [])
    odd = FakeTensor((8,))
    odd.data_ptr = lambda: 4098   # 2 mod 16
    assert aligned != problem.value_fingerprint(odd, [])


def test_scalar_and_skipped_objects():
    skipped = []
    assert problem.value_fingerprint(42, skipped) == ("S", 42)
    assert problem.value_fingerprint("abc", skipped) == ("S", "abc")
    assert problem.value_fingerprint(object(), skipped) is None
    assert len(skipped) == 1


def test_launch_fingerprint_uses_arg_names_and_detects_changes():
    meta = registry.meta_for(StubTarget("k", arg_names=("x", "n", "eps")))
    x = FakeTensor((8,))
    fp1 = problem.launch_fingerprint(meta, (x, 8), {"eps": 1.0}, [], {})
    fp2 = problem.launch_fingerprint(meta, (x, 8), {"eps": 2.0}, [], {})
    assert fp1 != fp2
    assert [name for name, _ in fp1] == ["x", "n", "eps"]


def test_filter_ignored_drops_named_entries():
    meta = registry.meta_for(StubTarget("k", arg_names=("x", "n", "eps")))
    x = FakeTensor((8,))
    fps = problem.launch_fingerprint(meta, (x, 8), {"eps": 1.0}, [], {})
    filtered = problem.filter_ignored(fps, ["eps"], "k", has_arg_names=True)
    assert [name for name, _ in filtered] == ["x", "n"]
    assert problem.filter_ignored(fps, None, "k", True) == fps


def test_filter_ignored_requires_arg_names():
    with pytest.raises(akd.AkdError, match="arg_names"):
        problem.filter_ignored((("x", ("S", 1)),), ["x"], "no_signature", has_arg_names=False)


def test_candidate_identity_semantics():
    # tuple grids are deliberately not part of identity (their values derive from
    # keyed arguments); a callable grid formula is.
    k = StubTarget("k")
    meta = registry.meta_for(k)
    plain = problem.candidate_identity([Launch(k, (4,), (), {})], [meta])
    with_callable = problem.candidate_identity(
        [Launch(k, lambda meta: (1,), (), {})], [meta]
    )
    assert with_callable != plain
    same = problem.candidate_identity([Launch(k, (8,), (), {})], [meta])
    assert same == plain


def test_meta_for_is_cached_and_weak():
    k = StubTarget("k", arg_names=("a",))
    m1 = registry.meta_for(k)
    m2 = registry.meta_for(k)
    assert m1 is m2
    assert m1.arg_names == ("a",)
    assert m1.name == "k"


def test_site_identity_contains_qualname_and_offset():
    def _capture():
        return problem.site_identity(inspect.currentframe().f_back, None, "fb")

    def wrapper():
        return _capture()

    site = wrapper()
    qualname, rest = site.split("@", 1)
    assert qualname.endswith("wrapper")
    assert ":" in rest   # filename:lasti


def test_site_identity_explicit_wins():
    assert problem.site_identity(None, "my_site", "fb") == "my_site"


def test_device_fingerprint_is_deterministic():
    fp1 = problem.device_fingerprint(FakeDevice())
    fp2 = problem.device_fingerprint(FakeDevice())
    assert fp1 == fp2
    assert fp1[0] in ("cuda", "cuda-unavailable")
    assert problem.device_fingerprint(FakeDevice(type="cpu")) == ("cpu",)
