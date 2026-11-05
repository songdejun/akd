"""Kernel metadata: stable source hashing and per-fn metadata caching (no wrapping)."""

from __future__ import annotations

import hashlib
import inspect
import weakref
from dataclasses import dataclass


@dataclass(frozen=True)
class KernelMeta:
    name: str
    source_hash: str
    arg_names: tuple[str, ...] | None


def source_hash(target) -> str:
    """Hash the target's source so editing a kernel invalidates its cached decisions."""
    fn = getattr(target, "fn", target)
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = repr(target)
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def code_hash(fn) -> str:
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = repr(getattr(fn, "__code__", fn))
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def make_meta(target, name: str | None = None) -> KernelMeta:
    if name is None:
        name = (
            getattr(target, "__name__", None)
            or getattr(getattr(target, "fn", None), "__name__", None)
            or type(target).__name__
        )
    arg_names = getattr(target, "arg_names", None)
    return KernelMeta(
        name=name,
        source_hash=source_hash(target),
        arg_names=tuple(arg_names) if arg_names else None,
    )


_meta_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def meta_for(target) -> KernelMeta:
    """Cached metadata for a launchable; keyed weakly so nothing keeps kernels alive."""
    try:
        meta = _meta_cache.get(target)
    except TypeError:
        meta = None   # unhashable: just recompute
    if meta is None:
        meta = make_meta(target)
        try:
            _meta_cache[target] = meta
        except TypeError:
            pass
    return meta
