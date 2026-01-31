"""Configuration loader utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).with_name("default.yaml")
ENV_CONFIG = "BITBAT_CONFIG"

_ACTIVE_CONFIG: dict[str, Any] | None = None
_ACTIVE_PATH: Path | None = None


def _resolve_path(path: str | Path | None = None) -> Path:
    if path is not None:
        candidate = Path(path)
    else:
        env_path = os.environ.get(ENV_CONFIG)
        candidate = Path(env_path) if env_path else DEFAULT_CONFIG_PATH
    return candidate.expanduser().resolve()


def load_config(path: str | Path | None = None, *, cache: bool = False) -> dict[str, Any]:
    """Load a YAML configuration file from disk."""
    try:
        import yaml  # type: ignore[import-not-found, import-untyped]
    except ImportError as exc:  # pragma: no cover - dependency guard
        msg = "PyYAML is required to load configuration files."
        raise RuntimeError(msg) from exc

    config_path = _resolve_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at root of config {config_path}")

    if cache:
        global _ACTIVE_CONFIG, _ACTIVE_PATH
        _ACTIVE_CONFIG = data
        _ACTIVE_PATH = config_path

    return data


def set_runtime_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and cache configuration for use throughout the application."""
    return load_config(path, cache=True)


def get_runtime_config() -> dict[str, Any]:
    """Return the active configuration, loading defaults if necessary."""
    global _ACTIVE_CONFIG
    if _ACTIVE_CONFIG is None:
        load_config(cache=True)
    # Return a shallow copy to prevent accidental mutation.
    return dict(_ACTIVE_CONFIG or {})


def get_runtime_config_path() -> Path:
    """Return the path to the active configuration file."""
    global _ACTIVE_PATH
    if _ACTIVE_PATH is None:
        _ACTIVE_PATH = _resolve_path()
    return _ACTIVE_PATH
