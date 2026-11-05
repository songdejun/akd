import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

import akd
from akd import config as akd_config
from akd.selector import DecisionCache


class FakeDevice:
    def __init__(self, type="cuda", index=0):
        self.type = type
        self.index = index

    def __eq__(self, other):
        return isinstance(other, FakeDevice) and self.type == other.type and self.index == other.index

    def __hash__(self):
        return hash((self.type, self.index))


class FakeTensor:
    """Duck tensor for key/contract tests without torch CUDA."""

    def __init__(self, shape, dtype="float32", device=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device or FakeDevice()

    def stride(self):
        strides = []
        acc = 1
        for size in reversed(self.shape):
            strides.append(acc)
            acc *= size
        return tuple(reversed(strides))

    def data_ptr(self):
        return 4096

    def clone(self):
        return FakeTensor(self.shape, self.dtype, self.device)


class StubTarget:
    """Stands in for a JITFunction: counts launches, can fail or mutate a buffer."""

    def __init__(self, name, arg_names=(), effect=None, fail=False, ret=None):
        self.name = name
        self.__name__ = name   # JITFunctions carry __name__; metadata extraction relies on it
        self.arg_names = list(arg_names)
        self.effect = effect
        self.fail = fail
        self.ret = ret
        self.launches = 0

    def __getitem__(self, grid):
        target = self

        class _Callable:
            def __call__(self, *args, **kwargs):
                target.launches += 1
                if target.fail:
                    raise RuntimeError("stub kernel failure")
                if target.effect is not None:
                    target.effect(*args, **kwargs)
                return target.ret

        return _Callable()

    def __repr__(self):
        return f"<StubTarget {self.name} {id(self)}>"


@pytest.fixture
def interp_env(monkeypatch):
    monkeypatch.setenv("TRITON_INTERPRET", "1")


@pytest.fixture
def test_config():
    cfg = akd_config.Config(store=akd_config.StoreConfig(enabled=False))
    akd_config.set_config(cfg)
    yield cfg
    akd_config.reset_config()


@pytest.fixture
def fresh_cache(monkeypatch):
    cache = DecisionCache()
    monkeypatch.setattr(akd.selector, "CACHE", cache)
    yield cache


@pytest.fixture
def stub_bench(monkeypatch):
    """Replace the timing backend with a controllable fake.

    state["times"]: {candidate idx: median ms}, default favors idx 0.
    state["launch_each"]: run every candidate once during "bench" (exercise clone paths).
    state["pruned"]: [(idx, reason)] excluded from medians.
    """
    import akd.api
    import akd.bench

    state = {"times": {}, "launch_each": False, "pruned": [], "calls": 0}

    def fake_bench(cands, *, device, rounds, budget_ms, flush_l2):
        state["calls"] += 1
        pruned_idx = {idx for idx, _ in state["pruned"]}
        medians = []
        for c in cands:
            if c.idx in pruned_idx:
                continue
            if state["launch_each"]:
                c.target[c.grid](*c.args, **c.kwargs)
            medians.append((c.idx, state["times"].get(c.idx, 1.0 + 0.1 * c.idx)))
        return akd.bench.BenchResult(
            medians=medians,
            pruned=list(state["pruned"]),
            unstable=False,
            compile_ms=[(c.idx, 5.0) for c in cands],
            reps=1,
        )

    monkeypatch.setattr(akd.api, "_bench", fake_bench)
    return state
