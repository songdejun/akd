# akd — Auto Kernel Dispatcher

Benchmark-driven dispatch across hand-written Triton kernel variants. `akd.cand(...).dispatch()`
benches the candidates on first call, caches the decision per key, and replays the winner
afterwards — replacing hard-coded dispatch thresholds that go stale on the next GPU.
Kernels stay plain `@triton.jit`; akd never wraps them.

See [DESIGN.md](DESIGN.md) for the full design rationale.

## Install

```bash
conda run -n torch pip install -e ".[dev]"
```

## Quick start

```python
import akd
import triton
import triton.language as tl

@triton.jit
def norm_oneshot(x_ptr, out_ptr, n, BLOCK: tl.constexpr): ...
@triton.jit
def norm_chunked(x_ptr, out_ptr, n, BLOCK: tl.constexpr): ...

def norm(x):
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out
    if n < 512:
        norm_oneshot[(1,)](x, out, n, BLOCK=512)   # fixed path: plain Triton, akd never sees it
        return out
    ret = akd.cand(
        norm_oneshot, (1,), x, out, n, BLOCK=8192,
      ).cand(
        norm_chunked, (1,), x, out, n, BLOCK=2048,
      ).dispatch(ignore=["eps"])
    # ret: AKDRet(winner=..., name=..., kernel=..., return_value=..., benched=..., timings=...)
    return out
```

- **First call** benches the candidates (interleaved shuffled rounds, CUDA events, median),
  relaunches the winner on the real buffers, and caches the decision.
- **Later calls** hit the decision cache and replay only the winner.
- Declaration order is the preference order: within `min_improvement` (default 3%), the
  earlier, simpler candidate wins.
- `akd.cands(*specs)` is the grouped form — `(fn, grid)` / `(fn, grid, args)` /
  `(fn, grid, args, kwargs)` — for forwarding wrappers and table-driven candidate sets.

Contracts to know:

- Candidates must be semantically equivalent; accumulating or in-place kernels must declare
  `outputs=["out"]` so bench runs write to clones. Tensor args other than declared outputs
  are treated as read-only.
- Decision keys are computed from the launch arguments at dispatch time (shapes, dtypes,
  strides, pointer alignment, scalars, device, candidate source hashes). `ignore=[...]`
  drops named parameters from the key — losing at most performance, never correctness.
- Build the chain inside the wrapper on every call: candidates capture the current call's
  tensor references.
- A chain that is garbage-collected without `.dispatch()` warns — no kernel was launched.
- Dynamic `BLOCK=min(n, 8192)` inside a candidate makes the key change every call
  (constant re-benching); keep candidate constexprs static or add one candidate line each.

## Config

Lookup order: project `akd.toml` (or `[tool.akd]` in `pyproject.toml`) > `~/.config/akd/config.toml`
> defaults. `AKD_DEBUG` / `AKD_CACHE_DIR` override everything.

```toml
[dispatch]
min_improvement = 0.03

[bench]
budget_ms = 30.0
rounds = 3
flush_l2 = true

[store]
enabled = true
dir = "~/.cache/akd"

[log]
debug = false
```

## Observability

```python
print(akd.report())        # decision table: site -> winner, timings, flags
print(akd.explain())       # verbose: per-candidate timings, pruned entries, compile cost
akd.reset()                # drop in-memory decisions, force a re-bench
```

## Example

```bash
conda run -n torch python examples/rowsum.py
```

Row-sum reduction with a whole-row one-shot variant vs a loop-chunked variant (the chunked
one exists for very long rows where a whole-row BLOCK would blow registers).

## Tests

```bash
conda run -n torch python -m pytest            # unit + interpreter; GPU tests auto-skip
conda run -n torch python -m pytest tests/test_rowsum_gpu.py
```
