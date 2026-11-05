"""Interpreter-mode end to end on real Triton kernels.

Runs in a subprocess because TRITON_INTERPRET is read when JITFunctions are constructed,
and other test modules import triton first. Uses akd's real dispatch flow: no stub bench
here, so the auto-first-candidate interpret path (decision cached, winner replayed) is
what gets exercised.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("triton")

_SCRIPT = """
import os
assert os.environ["TRITON_INTERPRET"] == "1"
import sys
from pathlib import Path
tests_dir = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(tests_dir))
sys.path.insert(0, str(tests_dir.parent / "src"))

import torch
import akd
from akd import config as akd_config

akd_config.set_config(akd_config.Config(store=akd_config.StoreConfig(enabled=False)))

import rowsum_kernels

def ref(x):
    return x.sum(dim=1, dtype=torch.float32)

x = torch.randn(8, 128)
out = rowsum_kernels.rowsum(x)
assert torch.allclose(out, ref(x), rtol=1e-3, atol=1e-3)
assert len(akd.selector.CACHE) == 1

out = rowsum_kernels.rowsum(x)   # cache hit: same call point, winner replay
assert torch.allclose(out, ref(x), rtol=1e-3, atol=1e-3)
assert len(akd.selector.CACHE) == 1

long = torch.randn(4, rowsum_kernels.ONESHOT_MAX + 5)
out = rowsum_kernels.rowsum(long)
assert torch.allclose(out, ref(long), rtol=1e-3, atol=1e-3)
print("interpret e2e OK")
"""


def test_interpret_mode_end_to_end(tmp_path):
    script = tmp_path / "interpret_e2e.py"
    script.write_text(_SCRIPT)
    result = subprocess.run(
        [sys.executable, str(script), str(Path(__file__).parent)],
        env={**os.environ, "TRITON_INTERPRET": "1"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "interpret e2e OK" in result.stdout
