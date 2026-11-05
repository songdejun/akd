"""akd: benchmark-driven dispatch across hand-written Triton kernel variants.

``akd.cand(fn, grid, *args, **kwargs).cand(...).dispatch()`` records candidate launches,
benches them on first call, caches the decision per key, and replays the winner afterwards.
Kernels stay plain @triton.jit; akd never wraps them. Fixed dispatch paths belong in plain
Python before the dispatch chain.
"""

import logging

from ._version import __version__
from .errors import AkdError, AkdWarning, ContractError
from . import config

from .chain import AKDRet, CandidateChain, Launch, cand, cands
from .api import warmup
from .report import explain, report, reset

_cfg = config.get()
_logger = logging.getLogger("akd")
_logger.setLevel(logging.DEBUG if _cfg.log.debug else logging.INFO)
if _cfg.log.debug and not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    _logger.addHandler(_handler)

__all__ = [
    "__version__",
    "AkdError",
    "AkdWarning",
    "ContractError",
    "Config",
    "AKDRet",
    "CandidateChain",
    "Launch",
    "cand",
    "cands",
    "warmup",
    "report",
    "explain",
    "reset",
]
