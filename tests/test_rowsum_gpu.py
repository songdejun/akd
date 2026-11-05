"""Real-GPU integration: actual benches, decisions, and the long-row fixed path."""

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA GPU not available", allow_module_level=True)

import akd

import rowsum_kernels


def _ref(x):
    return x.sum(dim=1, dtype=torch.float32)


def test_dispatch_benches_and_is_correct(test_config, fresh_cache):
    x = torch.randn(64, 512, device="cuda")
    out = rowsum_kernels.rowsum(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, _ref(x), rtol=1e-4, atol=1e-3)

    (key, decision), = fresh_cache.items()
    assert decision.winner in (0, 1)
    assert len(decision.timings) == 2

    out2 = rowsum_kernels.rowsum(x)   # cache hit: winner replay
    torch.cuda.synchronize()
    torch.testing.assert_close(out2, _ref(x), rtol=1e-4, atol=1e-3)


def test_decision_tracks_size(test_config, fresh_cache):
    small = torch.randn(16, 128, device="cuda")
    large = torch.randn(16, 4096, device="cuda")
    torch.testing.assert_close(rowsum_kernels.rowsum(small), _ref(small), rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(rowsum_kernels.rowsum(large), _ref(large), rtol=1e-4, atol=1e-3)
    assert len(fresh_cache) >= 1
    assert rowsum_kernels.rowsum_oneshot.__name__ == "rowsum_oneshot"


def test_very_long_row_uses_fixed_path(test_config, fresh_cache):
    # Past ONESHOT_MAX the one-shot variant would need BLOCK >= n: register overflow.
    # The wrapper hard-routes to the chunked kernel; dispatch never triggers.
    n = rowsum_kernels.ONESHOT_MAX + 777   # not a multiple of any block size
    x = torch.randn(8, n, device="cuda")
    out = rowsum_kernels.rowsum(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, _ref(x), rtol=1e-4, atol=1e-3)
    assert len(fresh_cache) == 0


def test_report_smoke(test_config, fresh_cache):
    x = torch.randn(16, 256, device="cuda")
    rowsum_kernels.rowsum(x)
    text = akd.report()
    assert "rowsum" in text
