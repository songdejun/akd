"""Decision keys: dispatch-time fingerprints of kernel arguments, triton-style.

The key is computed from the arguments recorded in the candidate chain at the moment
dispatch runs, so everything the wrapper did beforehand (input replacement, fresh output
buffers, sliced views) is captured. Fingerprints only include pointer *alignment*, never
the pointer value: fresh allocations must not cause misses. ``ignore=[...]`` removes named
parameters from the key (lossy in performance, never in correctness).
"""

from __future__ import annotations

from functools import lru_cache

from .errors import AkdError
from .registry import code_hash

_SCALARS = (bool, int, float, str, type(None))


def value_fingerprint(value, skipped: list | None) -> tuple | None:
    if isinstance(value, _SCALARS):
        return ("S", value)
    if (
        hasattr(value, "data_ptr")
        and hasattr(value, "shape")
        and hasattr(value, "dtype")
        and hasattr(value, "stride")
    ):
        try:
            return (
                "T",
                str(value.dtype),
                tuple(value.shape),
                tuple(value.stride()),
                int(value.data_ptr()) % 16,
            )
        except Exception:
            pass
    if skipped is not None:
        skipped.append(value)
    return None


def launch_fingerprint(meta, args, kwargs, skipped: list, memo: dict) -> tuple:
    """All arguments of one candidate as (arg name, fingerprint) pairs, order-stable."""
    out: list = []

    def fp(value):
        key = id(value)
        if key not in memo:
            memo[key] = value_fingerprint(value, skipped)
        return memo[key]

    names = meta.arg_names
    for i, arg in enumerate(args):
        name = names[i] if names and i < len(names) else f"#{i}"
        out.append((name, fp(arg)))
    for name in sorted(kwargs):
        out.append((name, fp(kwargs[name])))
    return tuple(out)


def filter_ignored(fps: tuple, ignore, label: str, has_arg_names: bool) -> tuple:
    if not ignore:
        return fps
    if not has_arg_names:
        raise AkdError(
            f"ignore={list(ignore)!r} needs kernel signatures to match names,"
            f" but {label!r} exposes no arg_names"
        )
    wanted = set(ignore)
    return tuple(entry for entry in fps if entry[0] not in wanted)


def candidate_identity(launches, metas) -> tuple:
    """Which kernels (and which grid formulas) make up this dispatch site."""
    parts = []
    for launch_spec, meta in zip(launches, metas):
        grid_id = code_hash(launch_spec.grid) if callable(launch_spec.grid) else None
        parts.append((meta.source_hash, grid_id))
    return tuple(parts)


def site_identity(caller_frame, explicit: str | None, fallback: str) -> str:
    if explicit is not None:
        return explicit
    if caller_frame is not None:
        code = caller_frame.f_code
        # f_lasti pins the call point, so one wrapper can hold several dispatch sites.
        return f"{code.co_qualname}@{code.co_filename}:{caller_frame.f_lasti}"
    return fallback


@lru_cache(maxsize=8)
def _cuda_fingerprint(index: int) -> tuple:
    import torch

    props = torch.cuda.get_device_properties(index)
    return (
        "cuda",
        props.name,
        (props.major, props.minor),
        props.multi_processor_count,
        props.total_memory,
    )


def device_fingerprint(device) -> tuple:
    kind = getattr(device, "type", "cpu")
    if kind != "cuda":
        return ("cpu",)
    index = device.index if device.index is not None else 0
    try:
        return _cuda_fingerprint(index)
    except Exception:
        # Duck tensors in tests, or CUDA not initialized: stable sentinel per index.
        return ("cuda-unavailable", index)
