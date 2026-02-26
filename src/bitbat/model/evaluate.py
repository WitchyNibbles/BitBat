"""Model evaluation routines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def window_diagnostics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    window_id: str | None = None,
    family: str | None = None,
) -> dict[str, Any]:
    """Compute deterministic regime/drift diagnostics for one evaluation window."""
    y_t = np.asarray(y_true, dtype="float64")
    y_p = np.asarray(y_pred, dtype="float64")
    if y_t.shape != y_p.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    error = y_p - y_t
    mae = float(np.mean(np.abs(error))) if y_t.size else 0.0
    rmse = float(np.sqrt(np.mean(error**2))) if y_t.size else 0.0
    bias = float(np.mean(error)) if y_t.size else 0.0

    realized_volatility = float(np.std(y_t)) if y_t.size else 0.0
    predicted_volatility = float(np.std(y_p)) if y_p.size else 0.0
    volatility_ratio = (
        float(predicted_volatility / realized_volatility)
        if realized_volatility > 0
        else 0.0
    )

    directional_accuracy = (
        float(np.mean(np.sign(y_t) == np.sign(y_p)))
        if y_t.size
        else 0.0
    )
    directional_stability = (
        float(np.mean(np.sign(y_p[1:]) == np.sign(y_p[:-1])))
        if y_p.size > 1
        else 1.0
    )

    if realized_volatility >= 0.01:
        regime = "high_volatility"
    elif realized_volatility >= 0.003:
        regime = "medium_volatility"
    else:
        regime = "low_volatility"

    diagnostics = {
        "window_id": window_id,
        "family": family,
        "regime": regime,
        "drift_score": float(abs(bias) + mae),
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "realized_volatility": realized_volatility,
        "predicted_volatility": predicted_volatility,
        "volatility_ratio": volatility_ratio,
        "directional_accuracy": directional_accuracy,
        "directional_stability": directional_stability,
        "n_samples": int(y_t.size),
    }
    return diagnostics


def write_window_diagnostics(
    diagnostics: dict[str, Any] | list[dict[str, Any]],
    *,
    output_path: str | Path = Path("metrics") / "window_diagnostics.json",
) -> Path:
    """Persist diagnostics payload for downstream retraining analysis."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return target


def regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Compute regression metrics from true and predicted returns.

    Writes `metrics/regression_metrics.json` and
    `metrics/prediction_scatter.png` as side effects.

    Args:
        y_true: Ground-truth forward returns for each sample.
        y_pred: Predicted forward returns.

    Returns:
        Mapping of MAE, RMSE, R-squared, directional accuracy, and correlation.
    """
    y_t = np.asarray(y_true, dtype="float64")
    y_p = np.asarray(y_pred, dtype="float64")

    if y_t.shape != y_p.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    residuals = y_t - y_p
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Directional accuracy: sign match
    sign_match = np.sign(y_t) == np.sign(y_p)
    directional_accuracy = float(np.mean(sign_match))

    # Correlation
    correlation = (
        float(np.corrcoef(y_t, y_p)[0, 1])
        if y_t.std() > 0 and y_p.std() > 0
        else 0.0
    )

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2),
        "directional_accuracy": directional_accuracy,
        "correlation": correlation,
        "n_samples": int(len(y_t)),
    }

    # Scatter plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_t, y_p, alpha=0.3, s=10)
    lims = [min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual Return")
    ax.set_ylabel("Predicted Return")
    ax.set_title(f"Predicted vs Actual (R²={r2:.3f})")

    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "regression_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fig_path = metrics_dir / "prediction_scatter.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    return metrics


def _fold_mean(folds: list[dict[str, Any]], key: str) -> float:
    values = [float(fold[key]) for fold in folds if key in fold]
    if not values:
        return 0.0
    return float(np.mean(values))


def _fold_sum(folds: list[dict[str, Any]], key: str) -> float:
    values = [float(fold[key]) for fold in folds if key in fold]
    if not values:
        return 0.0
    return float(np.sum(values))


def build_candidate_report(
    *,
    candidate_id: str,
    family: str,
    fold_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic multi-metric candidate report for model selection."""
    folds = list(fold_metrics)
    report = {
        "candidate_id": candidate_id,
        "family": family,
        "n_folds": len(folds),
        "metrics": {
            "regression": {
                "mean_rmse": round(_fold_mean(folds, "rmse"), 6),
                "mean_mae": round(_fold_mean(folds, "mae"), 6),
            },
            "directional": {
                "mean_directional_accuracy": round(
                    _fold_mean(folds, "directional_accuracy"), 6
                ),
                "mean_correlation": round(_fold_mean(folds, "correlation"), 6),
            },
            "risk": {
                "mean_net_sharpe": round(_fold_mean(folds, "net_sharpe"), 6),
                "mean_gross_sharpe": round(_fold_mean(folds, "gross_sharpe"), 6),
                "mean_net_return": round(_fold_mean(folds, "net_return"), 6),
                "mean_gross_return": round(_fold_mean(folds, "gross_return"), 6),
                "mean_max_drawdown": round(_fold_mean(folds, "max_drawdown"), 6),
                "total_costs": round(_fold_sum(folds, "total_costs"), 6),
                "total_fee_costs": round(_fold_sum(folds, "total_fee_costs"), 6),
                "total_slippage_costs": round(_fold_sum(folds, "total_slippage_costs"), 6),
            },
        },
    }
    return report
