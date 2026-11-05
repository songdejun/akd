import pytest
import torch

import akd
from akd.errors import AkdError
from conftest import FakeDevice, FakeTensor, StubTarget


def test_bench_once_then_cache_hit(test_config, fresh_cache, stub_bench, interp_env):
    k0, k1 = StubTarget("k0"), StubTarget("k1")
    x, out = FakeTensor((8,)), FakeTensor((8,))

    def call():
        return akd.cand(k0, (1,), x, out, 8, BLOCK=4).cand(k1, (1,), x, out, 8, BLOCK=8).dispatch()

    ret = call()
    assert stub_bench["calls"] == 1
    assert ret.winner == 0                  # default times favor idx 0
    assert ret.benched is True
    assert k0.launches == 1                 # winner relaunch on real buffers
    assert k1.launches == 0
    assert len(ret.timings) == 2

    ret = call()
    assert stub_bench["calls"] == 1         # cache hit, no re-bench
    assert ret.benched is False
    assert ret.winner == 0
    assert k0.launches == 2


def test_fastest_candidate_wins(test_config, fresh_cache, stub_bench, interp_env):
    k0, k1 = StubTarget("k0"), StubTarget("k1")
    stub_bench["times"] = {0: 5.0, 1: 1.0}
    x, out = FakeTensor((8,)), FakeTensor((8,))
    ret = akd.cand(k0, (1,), x, out, 8).cand(k1, (1,), x, out, 8).dispatch()
    assert ret.winner == 1
    assert k1.launches == 1 and k0.launches == 0


def test_min_improvement_prefers_declared_order(test_config, fresh_cache, stub_bench, interp_env):
    k0, k1 = StubTarget("k0"), StubTarget("k1")
    x, out = FakeTensor((8,)), FakeTensor((8,))
    stub_bench["times"] = {0: 1.0, 1: 1.02}   # within the 3% margin
    assert akd.cand(k0, (1,), x, out, 8).cand(k1, (1,), x, out, 8).dispatch().winner == 0

    k2, k3 = StubTarget("k2"), StubTarget("k3")
    stub_bench["times"] = {0: 1.2, 1: 1.0}    # k3 clearly faster
    assert akd.cand(k2, (1,), x, out, 8).cand(k3, (1,), x, out, 8).dispatch().winner == 1


def test_single_candidate_skips_bench(test_config, fresh_cache, stub_bench, interp_env):
    k0 = StubTarget("k0")
    x, out = FakeTensor((8,)), FakeTensor((8,))
    ret = akd.cand(k0, (1,), x, out, 8).dispatch()
    assert stub_bench["calls"] == 0
    assert k0.launches == 1
    assert ret.winner == 0 and ret.benched is False


def test_skip_bench_flag(test_config, fresh_cache, stub_bench, interp_env):
    k0, k1 = StubTarget("k0"), StubTarget("k1")
    stub_bench["times"] = {0: 9.0, 1: 1.0}
    x, out = FakeTensor((8,)), FakeTensor((8,))
    ret = akd.cand(k0, (1,), x, out, 8).cand(k1, (1,), x, out, 8).dispatch(skip_bench=True)
    assert stub_bench["calls"] == 0
    assert ret.winner == 0


def test_pruned_candidate_recorded(test_config, fresh_cache, stub_bench, interp_env):
    k0, k1 = StubTarget("k0"), StubTarget("k1")
    stub_bench["pruned"] = [(1, "RuntimeError: stub kernel failure")]
    x, out = FakeTensor((8,)), FakeTensor((8,))
    ret = akd.cand(k0, (1,), x, out, 8).cand(k1, (1,), x, out, 8).dispatch()
    assert ret.winner == 0
    assert ret.pruned == ((1, "RuntimeError: stub kernel failure"),)


def test_all_candidates_pruned_raises(test_config, fresh_cache, stub_bench, interp_env):
    k0, k1 = StubTarget("k0"), StubTarget("k1")
    stub_bench["pruned"] = [(0, "boom"), (1, "boom")]
    x, out = FakeTensor((8,)), FakeTensor((8,))
    with pytest.raises(AkdError, match="every candidate failed"):
        akd.cand(k0, (1,), x, out, 8).cand(k1, (1,), x, out, 8).dispatch()


def test_ignore_increases_hit_rate(test_config, fresh_cache, stub_bench, interp_env):
    k0 = StubTarget("k0", arg_names=("x", "n", "eps"))
    k1 = StubTarget("k1", arg_names=("x", "n", "eps"))
    x, out = FakeTensor((8,)), FakeTensor((8,))

    def call(eps):
        return (
            akd.cand(k0, (1,), x, out, 8, eps=eps)
            .cand(k1, (1,), x, out, 8, eps=eps)
            .dispatch(ignore=["eps"])
        )

    call(eps=1.0)
    call(eps=2.0)
    assert stub_bench["calls"] == 1   # eps excluded by ignore=["eps"]


def test_outputs_clone_protects_accumulator(test_config, fresh_cache, stub_bench, interp_env):
    # Accumulating stub: out += 1 per launch. Bench launches must hit clones.
    k0 = StubTarget("k0", arg_names=("x", "out"), effect=lambda x, out: out.add_(1.0))
    k1 = StubTarget("k1", arg_names=("x", "out"), effect=lambda x, out: out.add_(1.0))
    stub_bench["launch_each"] = True
    x = torch.zeros(4)
    out = torch.zeros(4)
    akd.cand(k0, (1,), x, out).cand(k1, (1,), x, out).dispatch(outputs=["out"])
    # bench launched each candidate on clones; only the winner relaunch touched the real out
    assert torch.equal(out, torch.ones(4))


def test_verify_prunes_disagreeing_majority(test_config, fresh_cache, stub_bench, interp_env):
    def wrong(x, out):
        out.fill_(999.0)

    def right(x, out):
        out.fill_(1.0)

    k0 = StubTarget("k0", arg_names=("x", "out"), effect=wrong)
    k1 = StubTarget("k1", arg_names=("x", "out"), effect=right)
    k2 = StubTarget("k2", arg_names=("x", "out"), effect=right)
    stub_bench["times"] = {0: 0.5, 1: 1.0, 2: 1.5}   # the wrong one is fastest
    x = torch.zeros(4)
    out = torch.empty(4)
    ret = (
        akd.cand(k0, (1,), x, out)
        .cand(k1, (1,), x, out)
        .cand(k2, (1,), x, out)
        .dispatch(verify=True)
    )
    assert ret.winner == 1   # fastest of the agreeing majority
    assert torch.equal(out, torch.ones(4))   # winner relaunch wrote the real buffer
    assert any(idx == 0 for idx, _ in ret.pruned)


def test_two_call_points_are_distinct_sites(test_config, fresh_cache, stub_bench, interp_env):
    k0, k1, k2 = StubTarget("k0"), StubTarget("k1"), StubTarget("k2")
    x, out = FakeTensor((8,)), FakeTensor((8,))

    def wrapper():
        akd.cand(k0, (1,), x, out, 8).dispatch()
        akd.cand(k1, (1,), x, out, 8).cand(k2, (1,), x, out, 8).dispatch()

    wrapper()
    assert len(fresh_cache) == 2
    assert stub_bench["calls"] == 1   # only the two-candidate site benches
    for key, _ in fresh_cache.items():
        assert key[0].split("@")[0].endswith("wrapper")


def test_shape_change_rebenches(test_config, fresh_cache, stub_bench, interp_env):
    k0, k1 = StubTarget("k0"), StubTarget("k1")

    def call(n):
        x, out = FakeTensor((n,)), FakeTensor((n,))
        return akd.cand(k0, (1,), x, out, n).cand(k1, (1,), x, out, n).dispatch()

    call(8)
    call(16)
    assert stub_bench["calls"] == 2


def test_dispatch_inside_capture_uses_first_candidate(
    test_config, fresh_cache, stub_bench, interp_env, monkeypatch
):
    import akd.contract

    k0, k1 = StubTarget("k0"), StubTarget("k1")
    x, out = FakeTensor((8,)), FakeTensor((8,))
    monkeypatch.setattr(akd.contract, "is_capturing", lambda device: True)
    ret = akd.cand(k0, (1,), x, out, 8).cand(k1, (1,), x, out, 8).dispatch()
    assert ret.winner == 0
    assert ret.benched is False
    assert len(fresh_cache) == 0   # decision deliberately left uncached
    assert stub_bench["calls"] == 0
