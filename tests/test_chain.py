import gc
import warnings

import pytest

import akd
from akd.chain import Launch
from akd.errors import AkdError, AkdWarning
from conftest import FakeTensor, StubTarget


def test_cand_defers_until_dispatch():
    target = StubTarget("k")
    x = FakeTensor((8,))
    akd.cand(target, (4,), x, 8, BLOCK=4)
    assert target.launches == 0


def test_cand_dispatch_launches_and_returns_akd_ret():
    target = StubTarget("k", ret="handle")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ret = akd.cand(target, (4,), FakeTensor((8,)), 8, BLOCK=4).dispatch(skip_bench=True)
    assert isinstance(ret, akd.AKDRet)
    assert ret.winner == 0
    assert ret.name == "k"
    assert ret.kernel is target
    assert ret.return_value == "handle"      # winner's final launch return passes through
    assert ret.benched is False              # skip_bench: no bench ran
    assert target.launches == 1


def test_cands_grouped_form_with_spec_normalization():
    target = StubTarget("k")
    x = FakeTensor((8,))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ret = akd.cands(
            (target, (4,), (x, 8)),
            (target, (4,), [x, 8], {"BLOCK": 8}),
        ).dispatch(skip_bench=True)
    assert ret.winner == 0
    assert target.launches == 1              # only the winner is launched


def test_chain_mutates_and_supports_mixed_cand_cands():
    target = StubTarget("k")
    x = FakeTensor((8,))
    chain = akd.cand(target, (1,), x).cands((target, (2,), (x,))).cand(target, (3,), x, 4)
    assert len(chain.launches) == 3
    assert isinstance(chain.launches[0], Launch)
    assert chain.launches[0].args == (x,)
    assert chain.launches[2].args == (x, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chain.dispatch(skip_bench=True)


def test_undispatched_chain_warns():
    with pytest.warns(AkdWarning, match="without .dispatch"):
        chain = akd.cand(StubTarget("k"), (1,), FakeTensor((8,)))
        del chain
        gc.collect()


def test_dispatched_chain_does_not_warn(test_config, fresh_cache, stub_bench, interp_env):
    target = StubTarget("k")
    x, out = FakeTensor((8,)), FakeTensor((8,))
    akd.cand(target, (1,), x, out, 8).dispatch()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gc.collect()
    assert not [w for w in caught if issubclass(w.category, AkdWarning)]


def test_spec_validation_errors():
    x = FakeTensor((8,))
    with pytest.raises(AkdError, match="specs must be"):
        akd.cands((StubTarget("k"),))
    with pytest.raises(AkdError, match="args must be a tuple"):
        akd.cands((StubTarget("k"), (1,), 42))
    with pytest.raises(AkdError, match="kwargs must be a dict"):
        akd.cands((StubTarget("k"), (1,), (x,), {3}))


def test_empty_dispatch_raises(test_config):
    with pytest.raises(AkdError, match="at least one candidate"):
        akd.CandidateChain().dispatch()


def test_dispatch_without_interp_rejects_cpu(test_config, monkeypatch):
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    from conftest import FakeDevice, FakeTensor

    target = StubTarget("k")
    x = FakeTensor((8,), device=FakeDevice(type="cpu"))
    with pytest.raises(akd.ContractError, match="TRITON_INTERPRET"):
        akd.cand(target, (1,), x, 8).dispatch()
