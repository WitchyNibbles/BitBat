"""Config-sourced default values for API route parameters."""

from __future__ import annotations

from bitbat.config.loader import load_config


def _default_freq() -> str:
    """Return the default bar frequency from config."""
    config = load_config()
    return str(config.get("freq", "1h"))


def _default_horizon() -> str:
    """Return the default prediction horizon from config."""
    config = load_config()
    return str(config.get("horizon", "4h"))
