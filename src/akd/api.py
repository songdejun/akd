"""Dispatch machinery: first call benches, later calls replay the cached winner."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from . import bench as bench_mod
from . import config, contract, problem, registry, selector
from .chain import AKDRet, launch as _launch
from .errors import AkdError
from .selector import Decision, pick_winner
from .store import Store

_log = logging.getLogger("akd")

VERIFY_RTOL = 1e-2
VERIFY_ATOL = 1e-2

_bench_lock = threading.Lock()
_bench = bench_mod.run_bench   # test seam: unit tests inject a fake timing backend
_store: Store | None = None


def warmup(fn):
    """Run fn once up front so first-call benches happen outside the hot path."""
    return fn()


def dispatch(chain, caller, *, ignore, outputs, verify, min_improvement, site, skip_bench,
             bench_budget_ms, rounds, flush_l2) -> AKDRet:
    cfg = config.get()
    launches = chain.launches
    if not launches:
        raise AkdError("dispatch needs at least one candidate")

    device = contract.check_devices(launches)
    skipped: list = []
    memo: dict = {}
    metas = [registry.meta_for(l.fn) for l in launches]
    names = tuple(m.name for m in metas)
    site_id = problem.site_identity(caller, site, fallback="akd")
    ident = problem.candidate_identity(launches, metas)
    dev_fp = problem.device_fingerprint(device)
    fps = tuple(
        problem.filter_ignored(
            problem.launch_fingerprint(m, l.args, l.kwargs, skipped, memo),
            ignore, m.name, m.arg_names is not None,
        )
        for m, l in zip(metas, launches)
    )
    dkey = (site_id, ident, dev_fp, fps)
    _log.debug("akd dispatch site=%s candidates=%s skipped_args=%d", site_id, names, len(skipped))

    decision = selector.CACHE.get(dkey)
    if decision is None and cfg.store.enabled:
        decision = _get_store().load(dkey)
    if decision is not None:
        selector.CACHE.put(dkey, decision)   # hydrate memory: hot path and report() must see it
        _log.debug("akd cache hit at %s -> %s", site_id, names[decision.winner])
        return _replay(launches, names, decision, benched=False)

    if contract.is_capturing(device):
        # Launches inside a captured graph cannot be event-timed; defer real benching.
        _log.warning("akd: CUDA graph capture in progress; launching first candidate at %s", site_id)
        return _replay(launches, names, Decision(winner=0), benched=False)

    if skip_bench or len(launches) == 1 or (
        contract.interpret_mode() and _bench is bench_mod.run_bench
    ):
        decision = Decision(winner=0, candidates=names, benched_at=time.time())
        selector.CACHE.put(dkey, decision)
        _save(cfg, dkey, decision)
        return _replay(launches, names, decision, benched=False)

    mi = cfg.dispatch.min_improvement if min_improvement is None else min_improvement
    budget = cfg.bench.budget_ms if bench_budget_ms is None else bench_budget_ms
    nrounds = cfg.bench.rounds if rounds is None else rounds
    flush = cfg.bench.flush_l2 if flush_l2 is None else flush_l2

    slots = _resolve_output_slots(launches, metas, outputs)
    cands = []
    for i, l in enumerate(launches):
        if slots[i] is not None:
            args, kwargs = contract.with_cloned_outputs(l, slots[i])
        else:
            args, kwargs = l.args, l.kwargs
        cands.append(bench_mod.Candidate(i, l.fn, l.grid, args, kwargs))

    benched_fresh = False
    with _bench_lock:
        # A thread that waited on the lock may find another thread's fresh decision.
        decision = selector.CACHE.get(dkey)
        if decision is None:
            decision = _bench_and_decide(cfg, launches, metas, cands, dkey, names,
                                         device, mi, nrounds, budget, flush, verify)
            benched_fresh = True
    return _replay(launches, names, decision, benched=benched_fresh)


def _bench_and_decide(cfg, launches, metas, cands, dkey, names, device, mi, rounds, budget,
                      flush, verify) -> Decision:
    result = _bench(cands, device=device, rounds=rounds, budget_ms=budget, flush_l2=flush)
    if not result.medians:
        detail = "; ".join(f"{names[i]}: {msg}" for i, msg in result.pruned)
        raise AkdError(f"akd: every candidate failed during bench: {detail}")
    medians = result.medians
    extra_pruned: list = []
    if verify:
        medians, extra_pruned = _verify_candidates(launches, metas, medians, device)
        if not medians:
            raise AkdError("akd: verify discarded every candidate (no majority agreement)")
    winner = pick_winner(medians, mi)
    decision = Decision(
        winner=winner,
        candidates=names,
        timings=tuple(result.medians),
        pruned=tuple(result.pruned) + tuple(extra_pruned),
        compile_ms=tuple(result.compile_ms),
        unstable=result.unstable,
        reps=result.reps,
        benched_at=time.time(),
    )
    selector.CACHE.put(dkey, decision)
    _save(cfg, dkey, decision)
    return decision


def _verify_candidates(launches, metas, medians, device):
    """Launch each candidate on cloned args; keep the largest group whose outputs agree.

    Protects against slow-but-wrong variants sneaking ahead. If the overall fastest is a
    minority (likely the buggy one), the majority group wins instead. With two candidates
    a disagreement keeps the faster one: akd has no oracle, only agreement.
    """
    order = [idx for idx, _ in medians]
    outputs: dict[int, list] = {}
    for idx in order:
        l = launches[idx]
        args, kwargs = contract.with_cloned_tensors(l)
        _launch(l.fn, l.grid, args, kwargs)
        outputs[idx] = contract.collect_tensors(args, kwargs)
    contract.synchronize(device)

    parent: dict[int, int] = {idx: idx for idx in order}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            if _agree(outputs[order[i]], outputs[order[j]]):
                parent[find(order[i])] = find(order[j])

    groups: dict[int, list[int]] = {}
    for idx in order:
        groups.setdefault(find(idx), []).append(idx)
    best_root = max(groups, key=lambda root: len(groups[root]))

    fastest = medians[0][0]
    if find(fastest) != best_root:
        _log.warning(
            "akd: verify: fastest candidate %r disagrees with the majority; keeping the majority",
            metas[fastest].name,
        )
    pruned = [
        (idx, "verify: outputs disagree with majority")
        for idx in order
        if find(idx) != best_root
    ]
    kept = [(idx, m) for idx, m in medians if find(idx) == best_root]
    return kept, pruned


def _agree(xs: list, ys: list) -> bool:
    if len(xs) != len(ys):
        return False
    import torch

    for x, y in zip(xs, ys):
        if x.shape != y.shape or x.dtype != y.dtype:
            return False
        if not torch.allclose(x, y, rtol=VERIFY_RTOL, atol=VERIFY_ATOL):
            return False
    return True


def _resolve_output_slots(launches, metas, outputs):
    if outputs is None:
        return [None] * len(launches)
    return [
        contract.resolve_output_slots(m, l.args, l.kwargs, outputs, m.name)
        for m, l in zip(metas, launches)
    ]


def _replay(launches, names, decision: Decision, *, benched: bool) -> AKDRet:
    winner = decision.winner
    l = launches[winner]
    ret = _launch(l.fn, l.grid, l.args, l.kwargs)
    return AKDRet(
        winner=winner,
        name=names[winner],
        kernel=l.fn,
        return_value=ret,
        benched=benched,
        timings=decision.timings,
        pruned=decision.pruned,
    )


def _get_store() -> Store:
    global _store
    if _store is None:
        _store = Store(Path(config.get().store.dir).expanduser())
    return _store


def _save(cfg, dkey, decision: Decision) -> None:
    if not cfg.store.enabled:
        return
    try:
        _get_store().save(dkey, decision)
    except OSError as e:
        _log.warning("akd: failed to persist decision: %s", e)
