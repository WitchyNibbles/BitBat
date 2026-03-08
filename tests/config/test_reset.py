"""Tests for reset_runtime_config in config/loader.py."""

from __future__ import annotations

import pytest

import bitbat.config.loader as loader

pytestmark = pytest.mark.behavioral


@pytest.fixture(autouse=True)
def _reset_after_each() -> None:
    """Ensure loader state is clean after every test in this module."""
    loader.reset_runtime_config()
    yield  # type: ignore[misc]
    loader.reset_runtime_config()


def test_reset_clears_cached_config() -> None:
    """reset_runtime_config sets all three module globals to None."""
    # Populate the cache via the public API
    loader.set_runtime_config()
    assert loader._ACTIVE_CONFIG is not None

    loader.reset_runtime_config()

    assert loader._ACTIVE_CONFIG is None
    assert loader._ACTIVE_PATH is None
    assert loader._ACTIVE_SOURCE is None


def test_get_after_reset_reloads() -> None:
    """After reset, get_runtime_config() lazy-loads a fresh config dict."""
    loader.set_runtime_config()
    loader.reset_runtime_config()

    # _ACTIVE_CONFIG is now None — get_runtime_config must reload
    result = loader.get_runtime_config()

    assert isinstance(result, dict)
    assert len(result) > 0


def test_reset_is_idempotent() -> None:
    """Calling reset_runtime_config() twice in a row raises no errors."""
    loader.reset_runtime_config()
    loader.reset_runtime_config()  # should not raise
