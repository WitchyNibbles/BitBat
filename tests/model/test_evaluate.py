from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

if pytest.importorskip("matplotlib"):
    import matplotlib.pyplot as plt  # noqa: F401

from bitbat.model.evaluate import (
    build_candidate_report,
    compute_multiple_testing_safeguards,
    regression_metrics,
    window_diagnostics,
    write_window_diagnostics,
)


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


def test_candidate_report_contains_regression_directional_and_risk_metrics() -> None:
    fold_metrics = [
        {
            "rmse": 0.01,
            "mae": 0.008,
            "directional_accuracy": 0.58,
            "net_sharpe": 1.1,
            "gross_sharpe": 1.3,
            "max_drawdown": -0.12,
            "net_return": 0.14,
            "gross_return": 0.16,
            "total_costs": 0.02,
        },
        {
            "rmse": 0.012,
            "mae": 0.009,
            "directional_accuracy": 0.56,
            "net_sharpe": 1.0,
            "gross_sharpe": 1.2,
            "max_drawdown": -0.11,
            "net_return": 0.12,
            "gross_return": 0.15,
            "total_costs": 0.018,
        },
    ]

    report = build_candidate_report(
        candidate_id="xgb",
        family="xgb",
        fold_metrics=fold_metrics,
    )

    assert report["candidate_id"] == "xgb"
    assert report["family"] == "xgb"
    assert report["metrics"]["regression"]["mean_rmse"] > 0
    assert report["metrics"]["directional"]["mean_directional_accuracy"] > 0
    assert report["metrics"]["risk"]["mean_net_sharpe"] > 0
    assert report["metrics"]["risk"]["total_costs"] > 0


def test_candidate_report_is_deterministic_for_same_input() -> None:
    fold_metrics = [
        {
            "rmse": 0.01,
            "mae": 0.008,
            "directional_accuracy": 0.58,
            "net_sharpe": 1.1,
            "gross_sharpe": 1.3,
            "max_drawdown": -0.12,
            "net_return": 0.14,
            "gross_return": 0.16,
            "total_costs": 0.02,
        }
    ]

    first = build_candidate_report(candidate_id="rf", family="random_forest", fold_metrics=fold_metrics)
    second = build_candidate_report(candidate_id="rf", family="random_forest", fold_metrics=fold_metrics)
    assert first == second


def test_multiple_testing_safeguards_are_deterministic() -> None:
    outer_folds = [
        {"outer_score": 0.0062},
        {"outer_score": 0.0060},
        {"outer_score": 0.0059},
        {"outer_score": 0.0061},
    ]

    first = compute_multiple_testing_safeguards(
        outer_folds,
        trial_count=30,
        min_deflated_sharpe=-0.2,
        max_overfit_probability=0.60,
    )
    second = compute_multiple_testing_safeguards(
        outer_folds,
        trial_count=30,
        min_deflated_sharpe=-0.2,
        max_overfit_probability=0.60,
    )

    assert first == second
    assert first["trial_count"] == 30
    assert "deflated_sharpe" in first
    assert "overfit_probability" in first
    assert "pass" in first
    assert isinstance(first["reasons"], list)


def test_multiple_testing_safeguards_fail_for_unstable_results() -> None:
    unstable_outer_folds = [
        {"outer_score": 0.0210},
        {"outer_score": 0.0075},
        {"outer_score": 0.0300},
        {"outer_score": 0.0040},
    ]

    safeguards = compute_multiple_testing_safeguards(
        unstable_outer_folds,
        trial_count=80,
        min_deflated_sharpe=0.0,
        max_overfit_probability=0.35,
    )

    assert safeguards["pass"] is False
    assert safeguards["overfit_probability"] >= 0.35
    assert len(safeguards["reasons"]) >= 1
