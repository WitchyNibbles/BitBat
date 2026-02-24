from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

if pytest.importorskip("matplotlib"):
    import matplotlib.pyplot as plt  # noqa: F401

from bitbat.model.evaluate import regression_metrics


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
