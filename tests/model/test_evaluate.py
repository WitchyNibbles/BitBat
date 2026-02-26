from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

if pytest.importorskip("matplotlib"):
    import matplotlib.pyplot as plt  # noqa: F401

from bitbat.model.evaluate import regression_metrics, window_diagnostics, write_window_diagnostics


def test_regression_metrics_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(42)
    y_true = pd.Series(rng.normal(0.0, 0.01, size=100))
    y_pred = y_true + rng.normal(0.0, 0.005, size=100)

    monkeypatch.chdir(tmp_path)
    metrics = regression_metrics(y_true, y_pred)

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "directional_accuracy" in metrics
    assert "correlation" in metrics
    assert metrics["rmse"] > 0
    assert 0.0 <= metrics["directional_accuracy"] <= 1.0

    metrics_path = Path("metrics") / "regression_metrics.json"
    assert metrics_path.exists()
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "mae" in data

    png_path = Path("metrics") / "prediction_scatter.png"
    assert png_path.exists()


def test_regression_metrics_perfect_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    y_true = pd.Series([0.01, -0.02, 0.03, -0.01, 0.005])
    y_pred = y_true.copy()

    monkeypatch.chdir(tmp_path)
    metrics = regression_metrics(y_true, y_pred)

    assert metrics["mae"] < 1e-10
    assert metrics["rmse"] < 1e-10
    assert metrics["directional_accuracy"] == 1.0


def test_window_diagnostics_are_deterministic() -> None:
    y_true = pd.Series([0.01, -0.02, 0.015, -0.005, 0.012, -0.008], dtype="float64")
    y_pred = pd.Series([0.009, -0.018, 0.014, -0.006, 0.011, -0.007], dtype="float64")

    first = window_diagnostics(y_true, y_pred, window_id="fold-1", family="xgb")
    second = window_diagnostics(y_true, y_pred, window_id="fold-1", family="xgb")

    assert first == second
    assert first["window_id"] == "fold-1"
    assert first["family"] == "xgb"
    assert first["regime"] in {"low_volatility", "medium_volatility", "high_volatility"}
    assert first["n_samples"] == len(y_true)


def test_write_window_diagnostics_outputs_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    diagnostics = {
        "window_id": "fold-2",
        "regime": "medium_volatility",
        "drift_score": 0.002,
    }

    monkeypatch.chdir(tmp_path)
    output = write_window_diagnostics(diagnostics)

    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["window_id"] == "fold-2"
    assert payload["regime"] == "medium_volatility"
