"""Decision cache and winner policy. Pure Python: no torch, no triton."""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    winner: int
    candidates: tuple[str, ...] = ()
    timings: tuple[tuple[int, float], ...] = ()   # (candidate idx, median ms)
    pruned: tuple[tuple[int, str], ...] = ()
    compile_ms: tuple[tuple[int, float], ...] = ()
    unstable: bool = False
    reps: int = 0
    benched_at: float = 0.0


def pick_winner(medians: list[tuple[int, float]], min_improvement: float) -> int:
    """First candidate within (1 + min_improvement) of the best median wins.

    medians is in declared-candidate order, so ties resolve toward the earlier,
    presumably simpler variant.
    """
    best = min(m for _, m in medians)
    for idx, m in medians:
        if m <= best * (1.0 + min_improvement) + 1e-12:
            return idx
    return medians[0][0]


class DecisionCache:
    def __init__(self):
        self._entries: dict = {}
        self._lock = threading.Lock()

    def get(self, key):
        return self._entries.get(key)

    def put(self, key, decision: Decision) -> None:
        with self._lock:
            self._entries[key] = decision

    def reset(self, site: str | None = None) -> None:
        with self._lock:
            if site is None:
                self._entries.clear()
            else:
                self._entries = {
                    k: v for k, v in self._entries.items() if not str(k[0]).startswith(site)
                }

    def items(self) -> tuple:
        return tuple(self._entries.items())

    def __len__(self) -> int:
        return len(self._entries)


CACHE = DecisionCache()
