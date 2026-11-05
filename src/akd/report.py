"""Human-readable decision reports."""

from __future__ import annotations

import time

from . import selector




def _fmt_ms(ms: float) -> str:
    return f"{ms:.3f}ms"


def report() -> str:
    entries = selector.CACHE.items()
    if not entries:
        return "akd: no dispatch decisions yet"
    lines = [f"{'site':<50} {'winner':<24} {'timings':<40} {'flags':<10} {'age'}"]
    for key, d in entries:
        site = str(key[0])
        site = site if len(site) <= 48 else site[:45] + "..."
        winner = d.candidates[d.winner] if d.candidates else f"#{d.winner}"
        timings = ", ".join(
            f"{d.candidates[idx]}={_fmt_ms(ms)}" for idx, ms in d.timings if idx < len(d.candidates)
        )
        flags = []
        if d.unstable:
            flags.append("unstable")
        if d.pruned:
            flags.append(f"pruned={len(d.pruned)}")
        age = f"{time.time() - d.benched_at:.0f}s" if d.benched_at else "-"
        lines.append(f"{site:<50} {winner:<24} {timings:<40} {','.join(flags):<10} {age}")
    return "\n".join(lines)


def explain(pattern: str | None = None) -> str:
    """Verbose dump of matching decisions: useful when debugging 'why was this picked'."""
    out = []
    for key, d in selector.CACHE.items():
        site = str(key[0])
        if pattern and pattern not in site:
            continue
        out.append(f"site: {site}")
        out.append(f"  candidates: {', '.join(d.candidates)}")
        out.append(f"  winner: {d.candidates[d.winner] if d.candidates else d.winner}")
        for idx, ms in d.timings:
            mark = " <- winner" if idx == d.winner else ""
            out.append(f"    [{idx}] {d.candidates[idx]}: {_fmt_ms(ms)}{mark}")
        for idx, reason in d.pruned:
            out.append(f"    [{idx}] {d.candidates[idx]}: pruned ({reason})")
        for idx, ms in d.compile_ms:
            out.append(f"    [{idx}] {d.candidates[idx]}: compile+warmup {_fmt_ms(ms)}")
        out.append(f"  reps/round: {d.reps}  unstable: {d.unstable}")
        if d.benched_at:
            out.append(f"  benched: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(d.benched_at))}")
    return "\n".join(out) if out else "akd: no matching decisions"


def reset(site: str | None = None) -> None:
    selector.CACHE.reset(site)
