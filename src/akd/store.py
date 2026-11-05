"""Disk persistence for dispatch decisions (json, one file per key hash)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path

from ._version import __version__
from .selector import Decision


def _versions() -> dict:
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = "n/a"
    try:
        import triton
        triton_version = triton.__version__
    except ImportError:
        triton_version = "n/a"
    return {"akd": __version__, "torch": torch_version, "triton": triton_version}


def _encode(decision: Decision) -> dict:
    return {f.name: getattr(decision, f.name) for f in fields(Decision)}


def _decode(data: dict) -> Decision:
    known = {f.name for f in fields(Decision)}
    kwargs = {k: v for k, v in data.items() if k in known}
    kwargs["candidates"] = tuple(kwargs.get("candidates", []))
    kwargs["timings"] = tuple(tuple(t) for t in kwargs.get("timings", []))
    kwargs["pruned"] = tuple(tuple(t) for t in kwargs.get("pruned", []))
    kwargs["compile_ms"] = tuple(tuple(t) for t in kwargs.get("compile_ms", []))
    return Decision(**kwargs)


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, dkey) -> Path:
        digest = hashlib.sha256(repr(dkey).encode()).hexdigest()
        return self.root / f"{digest}.json"

    def load(self, dkey) -> Decision | None:
        try:
            data = json.loads(self._path(dkey).read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if data.get("versions") != _versions():
            return None
        return _decode(data.get("decision", {}))

    def save(self, dkey, decision: Decision) -> None:
        path = self._path(dkey)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"versions": _versions(), "decision": _encode(decision)}))
        tmp.replace(path)
