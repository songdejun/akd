"""CUDA-event timing: warmup with failure pruning -> pilot -> interleaved shuffled rounds -> median.

Rounds are shuffled and interleaved because DVFS/thermal drift systematically favors
whichever candidate runs later; interleaving turns drift into spread-out noise.
"""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass

import torch

_L2_FLUSH_BYTES = 256 << 20
_l2_buffers: dict = {}


def _flush_l2(device) -> None:
    buffer = _l2_buffers.get(device)
    if buffer is None:
        buffer = torch.empty(_L2_FLUSH_BYTES // 4, dtype=torch.float32, device=device)
        _l2_buffers[device] = buffer
    buffer.zero_()


@dataclass(frozen=True)
class Candidate:
    idx: int
    target: object
    grid: object
    args: tuple
    kwargs: dict


@dataclass(frozen=True)
class BenchResult:
    medians: list          # [(candidate idx, median ms)] in candidate order
    pruned: list           # [(candidate idx, failure reason)]
    unstable: bool
    compile_ms: list       # [(candidate idx, first-launch wall ms)]
    reps: int


def _launch(c: Candidate) -> None:
    c.target[c.grid](*c.args, **c.kwargs)


def _time_burst(c: Candidate, device, reps: int, flush_l2: bool) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if flush_l2:
        _flush_l2(device)
    start.record()
    for _ in range(reps):
        _launch(c)
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end) / reps


def _cv(values: list[float]) -> float:
    mean = statistics.fmean(values)
    if mean == 0 or len(values) < 2:
        return 0.0
    return statistics.stdev(values) / mean


def run_bench(cands: list[Candidate], *, device, rounds: int = 3, budget_ms: float = 30.0,
              flush_l2: bool = True) -> BenchResult:
    rounds = max(1, rounds)
    survivors: list[Candidate] = []
    pruned: list[tuple[int, str]] = []
    compile_ms: list[tuple[int, float]] = []
    with torch.cuda.device(device):
        for c in cands:
            t0 = time.perf_counter()
            try:
                _launch(c)
                torch.cuda.synchronize(device)
            except Exception as e:
                # A variant that cannot compile/launch on this GPU is out; the decision
                # is made among survivors. This is error handling, not dispatch policy.
                pruned.append((c.idx, f"{type(e).__name__}: {e}"))
                continue
            compile_ms.append((c.idx, (time.perf_counter() - t0) * 1e3))
            survivors.append(c)
        if not survivors:
            return BenchResult([], pruned, False, compile_ms, 0)
        for _ in range(2):
            for c in survivors:
                _launch(c)
        torch.cuda.synchronize(device)

        est = {c.idx: _time_burst(c, device, 3, flush_l2) for c in survivors}
        worst = max(est.values())
        reps = max(1, min(256, int(budget_ms / (len(survivors) * worst + 1e-9))))

        rng = random.Random(0xA6D1)
        per_round: dict[int, list[float]] = {c.idx: [] for c in survivors}
        for _ in range(rounds):
            order = survivors[:]
            rng.shuffle(order)
            pairs = []
            for c in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                pairs.append((start, end, c))
                if flush_l2:
                    _flush_l2(device)
                start.record()
                for _ in range(reps):
                    _launch(c)
                end.record()
            torch.cuda.synchronize(device)
            for start, end, c in pairs:
                per_round[c.idx].append(start.elapsed_time(end) / reps)

    medians = [(c.idx, statistics.median(per_round[c.idx])) for c in survivors]
    unstable = any(_cv(per_round[c.idx]) > 0.15 for c in survivors)
    return BenchResult(medians, pruned, unstable, compile_ms, reps)
