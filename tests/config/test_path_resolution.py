"""Tests for config-driven models/metrics path resolution."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import bitbat.config.loader as loader
from bitbat.config.loader import resolve_metrics_dir, resolve_models_dir

pytestmark = pytest.mark.behavioral


@pytest.fixture(autouse=True)
def _reset_after_each() -> None:
    """Reset loader state before and after each test."""
    loader.reset_runtime_config()
    yield  # type: ignore[misc]
    loader.reset_runtime_config()


def test_resolve_models_dir_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default models dir remains cwd-relative for test compatibility."""
    monkeypatch.setattr(loader, "_ACTIVE_CONFIG", None)

    assert resolve_models_dir() == Path("models")


def test_resolve_metrics_dir_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default metrics dir remains cwd-relative for test compatibility."""
    monkeypatch.setattr(loader, "_ACTIVE_CONFIG", None)

    assert resolve_metrics_dir() == Path("metrics")


def test_config_redirect_models_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """models_dir config redirects model artifacts."""
    custom_models_dir = tmp_path / "custom_models"
    monkeypatch.setattr(loader, "_ACTIVE_CONFIG", {"models_dir": str(custom_models_dir)})

    assert resolve_models_dir() == custom_models_dir

    loader.reset_runtime_config()


def test_config_redirect_metrics_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """metrics_dir config redirects metrics artifacts."""
    custom_metrics_dir = tmp_path / "custom_metrics"
    monkeypatch.setattr(loader, "_ACTIVE_CONFIG", {"metrics_dir": str(custom_metrics_dir)})

    assert resolve_metrics_dir() == custom_metrics_dir

    loader.reset_runtime_config()


def test_no_hardcoded_models_path() -> None:
    """No literal Path("models") remains in src/."""
    result = subprocess.run(
        ["grep", "-r", "--include=*.py", 'Path("models")', "src/"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.stdout == ""


def test_no_hardcoded_metrics_path() -> None:
    """No literal Path("metrics") remains in src/."""
    result = subprocess.run(
        ["grep", "-r", "--include=*.py", 'Path("metrics")', "src/"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.stdout == ""
