"""Tests proving regression_metrics() is a pure function with no I/O side effects."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bitbat.model.evaluate import regression_metrics, write_regression_metrics

pytestmark = pytest.mark.behavioral


def test_regression_metrics_no_file_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """regression_metrics() must not create any files or directories."""
    monkeypatch.chdir(tmp_path)

    rng = np.random.default_rng(42)
    y_true = pd.Series(rng.normal(0.0, 0.01, size=50))
    y_pred = y_true + rng.normal(0.0, 0.005, size=50)

    regression_metrics(y_true, y_pred)

    # No files should exist in the working directory
    created_files = list(tmp_path.rglob("*"))
    assert created_files == [], (
        f"regression_metrics() created files as side effects: {created_files}"
    )


def test_regression_metrics_returns_expected_keys() -> None:
    """regression_metrics() must return a dict with the documented metric keys."""
    y_true = np.array([0.01, -0.02, 0.03, -0.01, 0.005])
    y_pred = np.array([0.008, -0.015, 0.025, -0.012, 0.006])

    result = regression_metrics(y_true, y_pred)

    expected_keys = {"mae", "rmse", "r2", "directional_accuracy", "correlation", "n_samples"}
    assert set(result.keys()) == expected_keys
    assert result["n_samples"] == 5
    assert result["rmse"] > 0
    assert 0.0 <= result["directional_accuracy"] <= 1.0


def test_write_regression_metrics_creates_files(tmp_path: Path) -> None:
    """write_regression_metrics() must create both JSON and PNG outputs."""
    y_true = np.array([0.01, -0.02, 0.03, -0.01, 0.005])
    y_pred = np.array([0.008, -0.015, 0.025, -0.012, 0.006])

    metrics = regression_metrics(y_true, y_pred)
    result_path = write_regression_metrics(metrics, y_true, y_pred, output_dir=tmp_path)

    json_path = tmp_path / "regression_metrics.json"
    png_path = tmp_path / "prediction_scatter.png"

    assert json_path.exists(), "regression_metrics.json was not created"
    assert png_path.exists(), "prediction_scatter.png was not created"
    assert result_path == json_path

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["mae"] == metrics["mae"]
    assert data["rmse"] == metrics["rmse"]
