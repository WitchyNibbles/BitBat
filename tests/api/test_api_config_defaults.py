"""Tests verifying API route defaults match config values (CORR-06).

These tests ensure freq/horizon defaults in API routes are sourced from
default.yaml rather than hardcoded, and guard against re-hardcoding.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bitbat.api.defaults import _default_freq, _default_horizon
from bitbat.config.loader import load_config

pytestmark = pytest.mark.behavioral


def test_api_defaults_match_config() -> None:
    """The defaults helper must return the values from default.yaml."""
    config = load_config()
    assert _default_freq() == str(config["freq"]), (
        f"_default_freq() returned {_default_freq()!r}, "
        f"but config has freq={config['freq']!r}"
    )
    assert _default_horizon() == str(config["horizon"]), (
        f"_default_horizon() returned {_default_horizon()!r}, "
        f"but config has horizon={config['horizon']!r}"
    )


def test_prediction_routes_use_config_defaults() -> None:
    """Module-level _FREQ/_HORIZON in predictions.py must match config."""
    config = load_config()
    from bitbat.api.routes.predictions import _FREQ, _HORIZON

    assert str(config["freq"]) == _FREQ, (
        f"predictions._FREQ is {_FREQ!r}, expected {config['freq']!r}"
    )
    assert str(config["horizon"]) == _HORIZON, (
        f"predictions._HORIZON is {_HORIZON!r}, expected {config['horizon']!r}"
    )


def test_no_hardcoded_1h_4h_in_api_routes() -> None:
    """No API route file should contain Query("1h" or Query("4h" defaults."""
    routes_dir = Path(__file__).resolve().parents[2] / "src" / "bitbat" / "api" / "routes"
    route_files = ["predictions.py", "analytics.py", "health.py"]

    for filename in route_files:
        source = (routes_dir / filename).read_text(encoding="utf-8")
        assert 'Query("1h"' not in source, (
            f"{filename} still contains hardcoded Query(\"1h\" default"
        )
        assert 'Query("4h"' not in source, (
            f"{filename} still contains hardcoded Query(\"4h\" default"
        )
        assert "= \"1h\"" not in source or "load_config" in source or "_default_" in source, (
            f"{filename} may contain hardcoded \"1h\" default not sourced from config"
        )
        assert "= \"4h\"" not in source or "load_config" in source or "_default_" in source, (
            f"{filename} may contain hardcoded \"4h\" default not sourced from config"
        )
