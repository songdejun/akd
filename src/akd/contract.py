"""Correctness contracts: device consistency, graph-capture guard, output cloning."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from .errors import ContractError
from .chain import Launch


def interpret_mode() -> bool:
    return os.environ.get("TRITON_INTERPRET") == "1"


def _is_tensor(value) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    return (
        hasattr(value, "data_ptr")
        and hasattr(value, "shape")
        and hasattr(value, "dtype")
        and hasattr(value, "device")
    )


def _iter_tensors(value):
    if _is_tensor(value):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def iter_tensors(launch: Launch):
    for arg in launch.args:
        yield from _iter_tensors(arg)
    for name in sorted(launch.kwargs):
        yield from _iter_tensors(launch.kwargs[name])


def check_devices(launches) -> "torch.device":
    device = None
    interp = interpret_mode()
    for launch in launches:
        for tensor in iter_tensors(launch):
            where = tensor.device
            if where.type != "cuda" and not interp:
                raise ContractError(
                    f"akd dispatches GPU kernels; got a tensor on {where}."
                    " CPU execution needs TRITON_INTERPRET=1."
                )
            if device is None:
                device = where
            elif where != device:
                raise ContractError(f"tensor arguments span devices {device} and {where}")
    if device is None:
        raise ContractError("no tensor arguments found in any candidate")
    return device


def is_capturing(device) -> bool:
    if interpret_mode():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing(device))
    except Exception:
        return False


def synchronize(device) -> None:
    if not interpret_mode():
        torch.cuda.synchronize(device)


@dataclass(frozen=True)
class Slot:
    kind: str            # "arg" | "kwarg"
    pos: int | None = None
    name: str | None = None


def resolve_output_slots(meta, args, kwargs, spec, label: str) -> list[Slot]:
    """Resolve `outputs=` entries (arg name or positional index) against one candidate."""
    items = spec if isinstance(spec, (list, tuple)) else (spec,)
    slots: list[Slot] = []
    arg_names = meta.arg_names
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (int, str)):
            raise ContractError(f"outputs entries must be str or int, got {item!r}")
        if isinstance(item, int):
            if item >= len(args):
                raise ContractError(f"outputs index {item} out of range for {label}")
            slots.append(Slot("arg", pos=item))
            continue
        if item in kwargs:
            slots.append(Slot("kwarg", name=item))
            continue
        if arg_names and item in arg_names:
            pos = arg_names.index(item)
            if pos < len(args):
                slots.append(Slot("arg", pos=pos))
                continue
        raise ContractError(
            f"outputs={item!r}: cannot resolve on {label}; pass it as a kwarg or positional index"
        )
    return slots


def with_cloned_outputs(launch: Launch, slots: list[Slot]) -> tuple[tuple, dict]:
    args = list(launch.args)
    kwargs = dict(launch.kwargs)
    for slot in slots:
        value = args[slot.pos] if slot.kind == "arg" else kwargs[slot.name]
        if not _is_tensor(value):
            raise ContractError(f"outputs slot {slot} on {launch.fn!r} is not a tensor")
        if slot.kind == "arg":
            args[slot.pos] = value.clone()
        else:
            kwargs[slot.name] = value.clone()
    return tuple(args), kwargs


def with_cloned_tensors(launch: Launch) -> tuple[tuple, dict]:
    """Clone every tensor argument; used by verify mode to compare full argument sets."""
    args = tuple(t.clone() if _is_tensor(t) else t for t in launch.args)
    kwargs = {k: (v.clone() if _is_tensor(v) else v) for k, v in launch.kwargs.items()}
    return args, kwargs


def collect_tensors(args, kwargs) -> list:
    out: list = []
    for arg in args:
        out.extend(_iter_tensors(arg))
    for name in sorted(kwargs):
        out.extend(_iter_tensors(kwargs[name]))
    return out
