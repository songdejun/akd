"""Row-sum reduction with two variants: whole-row one-shot vs loop-chunked.

The one-shot variant needs BLOCK >= n_cols; for very long rows that means register
overflow, so the wrapper hard-routes past ONESHOT_MAX (plain Python before dispatch)
and lets akd benchmark the two variants below the threshold.

    conda run -n torch python examples/rowsum.py [--rows 256] [--cols 4096]
"""

from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl

import akd

ONESHOT_MAX = 1 << 14


@triton.jit
def rowsum_oneshot(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * n_cols + offs, mask=offs < n_cols, other=0.0).to(tl.float32)
    tl.store(out_ptr + row, tl.sum(x, axis=0))


@triton.jit
def rowsum_chunked(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = x_ptr + row * n_cols
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, n_cols, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        acc += tl.load(base + offs, mask=offs < n_cols, other=0.0).to(tl.float32)
    tl.store(out_ptr + row, tl.sum(acc, axis=0))


def rowsum(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2 and x.is_contiguous()
    rows, n = x.shape
    out = torch.empty(rows, dtype=torch.float32, device=x.device)
    if n == 0:
        out.zero_()
        return out
    if n > ONESHOT_MAX:
        # fixed path: a plain Triton launch, akd never sees it
        rowsum_chunked[(rows,)](x, out, n, BLOCK=2048, num_warps=8)
        return out
    akd.cand(
        rowsum_oneshot, (rows,), x, out, n, BLOCK=triton.next_power_of_2(n),
      ).cand(
        rowsum_chunked, (rows,), x, out, n, BLOCK=1024,
      ).dispatch()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    x = torch.randn(args.rows, args.cols, device="cuda", dtype=dtype)

    out = rowsum(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x.sum(dim=1, dtype=torch.float32), rtol=1e-4, atol=1e-3)

    out = rowsum(x)  # cache hit: no bench, winner replay
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x.sum(dim=1, dtype=torch.float32), rtol=1e-4, atol=1e-3)
    print("correct at both calls; decision table:")
    print(akd.report())


if __name__ == "__main__":
    main()
