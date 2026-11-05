"""Test kernels: whole-row one-shot reduction vs loop-chunked reduction.

Plain @triton.jit functions - akd never wraps kernels. The one-shot variant needs
BLOCK >= n_cols; past ONESHOT_MAX that means register overflow, so the wrapper
hard-routes to the chunked variant and lets akd compare the two below the threshold.
"""

from __future__ import annotations

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
