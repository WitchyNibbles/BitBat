"""Configuration loader utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).with_name("default.yaml")
ENV_CONFIG = "BITBAT_CONFIG"

_ACTIVE_CONFIG: dict[str, Any] | None = None
_ACTIVE_PATH: Path | None = None
_ACTIVE_SOURCE: str | None = None


def _default_user_config_path() -> Path:
    """Return the repo-local user override path paired with the packaged default config."""
    return DEFAULT_CONFIG_PATH.parents[3] / "config" / "user_config.yaml"


def _load_yaml_mapping(config_path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found, import-untyped]
    except ImportError as exc:  # pragma: no cover - dependency guard
        msg = "PyYAML is required to load configuration files."
        raise RuntimeError(msg) from exc

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at root of config {config_path}")

    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _resolve_path_with_source(path: str | Path | None = None) -> tuple[Path, str]:
    if path is not None:
        candidate = Path(path)
        source = "explicit"
    else:
        env_path = os.environ.get(ENV_CONFIG)
        if env_path:
            candidate = Path(env_path)
            source = "env"
        else:
            candidate = DEFAULT_CONFIG_PATH
            source = "default"
    return candidate.expanduser().resolve(), source


def _resolve_path(path: str | Path | None = None) -> Path:
    resolved_path, _ = _resolve_path_with_source(path)
    return resolved_path


def load_config(path: str | Path | None = None, *, cache: bool = False) -> dict[str, Any]:
    """Load a YAML configuration file from disk."""
    config_path, source = _resolve_path_with_source(path)
    default_data = _load_yaml_mapping(DEFAULT_CONFIG_PATH)

    if source == "default":
        data = default_data
        active_path = DEFAULT_CONFIG_PATH
        user_path = _default_user_config_path()
        if user_path.exists():
            data = _deep_merge(default_data, _load_yaml_mapping(user_path))
            source = "default+user"
            active_path = user_path
    elif config_path == DEFAULT_CONFIG_PATH:
        data = default_data
        active_path = config_path
    else:
        data = _deep_merge(default_data, _load_yaml_mapping(config_path))
        active_path = config_path

    if cache:
        global _ACTIVE_CONFIG, _ACTIVE_PATH, _ACTIVE_SOURCE
        _ACTIVE_CONFIG = data
        _ACTIVE_PATH = active_path
        _ACTIVE_SOURCE = source

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
        _ACTIVE_PATH, _ = _resolve_path_with_source()
    return _ACTIVE_PATH


def get_runtime_config_source() -> str:
    """Return where the active config came from: explicit, env, or default."""
    global _ACTIVE_SOURCE
    if _ACTIVE_SOURCE is None:
        load_config(cache=True)
    return str(_ACTIVE_SOURCE or "default")


def reset_runtime_config() -> None:
    """Reset all cached runtime configuration state.

    Intended for use in test teardown to prevent config state from leaking
    between tests. After calling this, the next call to get_runtime_config()
    will reload from disk.
    """
    global _ACTIVE_CONFIG, _ACTIVE_PATH, _ACTIVE_SOURCE
    _ACTIVE_CONFIG = None
    _ACTIVE_PATH = None
    _ACTIVE_SOURCE = None


def resolve_models_dir(config: dict[str, Any] | None = None) -> Path:
    """Return the canonical models directory."""
    cfg = config if config is not None else get_runtime_config()
    raw = str(cfg.get("models_dir", "models"))
    return Path(raw).expanduser()


def resolve_metrics_dir(config: dict[str, Any] | None = None) -> Path:
    """Return the canonical metrics directory."""
    cfg = config if config is not None else get_runtime_config()
    raw = str(cfg.get("metrics_dir", "metrics"))
    return Path(raw).expanduser()
