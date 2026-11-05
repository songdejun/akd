"""Candidate chains: record Triton launches, then dispatch benchmarks and picks a winner.

Kernels are plain @triton.jit functions - akd never wraps them. ``akd.cand(fn, grid, *args,
**kwargs)`` starts a chain, ``.cand()``/``.cands()`` add candidates in place, and
``.dispatch(...)`` benches on first call, caches the decision, and replays the winner.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from .errors import AkdError, AkdWarning


@dataclass(frozen=True)
class AKDRet:
    """Result of a dispatch: which candidate won, and what its final launch returned."""

    winner: int                    # index in declaration order
    name: str                      # winner's kernel name
    kernel: object                 # winner's fn object
    return_value: object = None    # whatever the winner's final launch returned
    benched: bool = False          # True only when this call ran a fresh benchmark
    timings: tuple = ()            # ((candidate idx, median ms), ...) survivors in order
    pruned: tuple = ()             # ((candidate idx, reason), ...)


@dataclass(frozen=True)
class Launch:
    """One recorded candidate: everything needed to replay ``fn[grid](*args, **kwargs)``."""

    fn: object
    grid: object
    args: tuple
    kwargs: dict


def launch(fn, grid, args, kwargs):
    return fn[grid](*args, **kwargs)


def _normalize_spec(spec) -> Launch:
    if not isinstance(spec, tuple) or not 2 <= len(spec) <= 4:
        raise AkdError(
            f"cands() specs must be (fn, grid) / (fn, grid, args) / (fn, grid, args, kwargs)"
            f" tuples, got {spec!r}"
        )
    fn, grid, *rest = spec
    args = rest[0] if len(rest) >= 1 else ()
    kwargs = rest[1] if len(rest) >= 2 else {}
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if not isinstance(args, (tuple, list)):
        raise AkdError(f"spec args must be a tuple or list, got {type(args).__name__}")
    if not isinstance(kwargs, dict):
        raise AkdError(f"spec kwargs must be a dict, got {type(kwargs).__name__}")
    return Launch(fn, grid, tuple(args), dict(kwargs))


class CandidateChain:
    __slots__ = ("launches", "_dispatched")

    def __init__(self):
        self.launches: list[Launch] = []
        self._dispatched = False

    def cand(self, fn, grid, /, *args, **kwargs) -> "CandidateChain":
        """Add one candidate (spread form, mirrors the Triton launch statement)."""
        self.launches.append(Launch(fn, grid, args, kwargs))
        return self

    def cands(self, *specs) -> "CandidateChain":
        """Add candidates in grouped form; each spec is (fn, grid[, args[, kwargs]])."""
        for spec in specs:
            self.launches.append(_normalize_spec(spec))
        return self

    def dispatch(self, ignore=None, outputs=None, verify=False, min_improvement=None,
                 site=None, skip_bench=False, bench_budget_ms=None, rounds=None,
                 flush_l2=None) -> AKDRet:
        self._dispatched = True
        # Imported here: chain is the public entry, api owns the dispatch machinery.
        from . import api

        caller = inspect_current_caller()
        return api.dispatch(
            self, caller, ignore=ignore, outputs=outputs, verify=verify,
            min_improvement=min_improvement, site=site, skip_bench=skip_bench,
            bench_budget_ms=bench_budget_ms, rounds=rounds, flush_l2=flush_l2,
        )

    def __del__(self):
        # A chain that never reached dispatch means the user thinks kernels ran but none did.
        try:
            if not self._dispatched and self.launches:
                warnings.warn(
                    "akd: a candidate chain was collected without .dispatch();"
                    " no kernel was launched",
                    AkdWarning,
                    stacklevel=1,
                )
        except Exception:
            pass


def inspect_current_caller():
    import inspect

    return inspect.currentframe().f_back.f_back   # cand-chain.dispatch -> its caller


def cand(fn, grid, /, *args, **kwargs) -> CandidateChain:
    """Start a chain with one candidate (spread form)."""
    chain = CandidateChain()
    return chain.cand(fn, grid, *args, **kwargs)


def cands(*specs) -> CandidateChain:
    """Start a chain with grouped candidates: (fn, grid) / (fn, grid, args) / (fn, grid, args, kwargs)."""
    chain = CandidateChain()
    return chain.cands(*specs)
