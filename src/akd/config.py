"""Configuration: project akd.toml (or pyproject [tool.akd]) > ~/.config/akd/config.toml > defaults.

Environment variables (AKD_DEBUG / AKD_CACHE_DIR) override everything.
"""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path

_log = logging.getLogger("akd")


@dataclass(frozen=True)
class DispatchConfig:
    min_improvement: float = 0.03


@dataclass(frozen=True)
class BenchConfig:
    budget_ms: float = 30.0
    rounds: int = 3
    flush_l2: bool = True


@dataclass(frozen=True)
class StoreConfig:
    enabled: bool = True
    dir: str = "~/.cache/akd"


@dataclass(frozen=True)
class LogConfig:
    debug: bool = False


@dataclass(frozen=True)
class Config:
    dispatch: DispatchConfig = DispatchConfig()
    bench: BenchConfig = BenchConfig()
    store: StoreConfig = StoreConfig()
    log: LogConfig = LogConfig()


_SECTIONS = {
    "dispatch": DispatchConfig,
    "bench": BenchConfig,
    "store": StoreConfig,
    "log": LogConfig,
}


def _read_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _akd_table(path: Path) -> dict:
    data = _read_toml(path)
    if path.name == "pyproject.toml":
        table = data.get("tool", {}).get("akd", {})
        return table if isinstance(table, dict) else {}
    return data


def _project_file() -> Path | None:
    for directory in (Path.cwd(), *Path.cwd().parents):
        candidate = directory / "akd.toml"
        if candidate.is_file():
            return candidate
        candidate = directory / "pyproject.toml"
        if candidate.is_file():
            try:
                if _akd_table(candidate):
                    return candidate
            except tomllib.TOMLDecodeError:
                _log.warning("akd: ignoring unparseable %s", candidate)
    return None


def _section(base, cls, raw: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        _log.debug("akd config: unknown keys %s ignored", sorted(unknown))
    values = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items() if k in known}
    return replace(base, **values)


def _env_flag(name: str) -> bool | None:
    value = os.environ.get(name)
    return None if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def load() -> Config:
    cfg = Config()
    paths: list[Path | None] = [
        Path.home() / ".config" / "akd" / "config.toml",
        _project_file(),
    ]
    for path in paths:
        if path is None or not path.is_file():
            continue
        try:
            table = _akd_table(path)
        except tomllib.TOMLDecodeError as e:
            _log.warning("akd: ignoring unparseable config %s: %s", path, e)
            continue
        for section, cls in _SECTIONS.items():
            raw = table.get(section)
            if isinstance(raw, dict):
                cfg = replace(cfg, **{section: _section(getattr(cfg, section), cls, raw)})
    return _apply_env(cfg)


def _apply_env(cfg: Config) -> Config:
    debug = _env_flag("AKD_DEBUG")
    cache_dir = os.environ.get("AKD_CACHE_DIR")
    if debug is not None:
        cfg = replace(cfg, log=replace(cfg.log, debug=debug))
    if cache_dir:
        cfg = replace(cfg, store=replace(cfg.store, dir=cache_dir))
    return cfg


_config: Config | None = None


def get() -> Config:
    global _config
    if _config is None:
        _config = load()
    return _config


def set_config(cfg: Config) -> None:
    global _config
    _config = cfg


def reset_config() -> None:
    global _config
    _config = None
