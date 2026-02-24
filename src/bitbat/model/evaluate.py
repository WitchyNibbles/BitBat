"""Model evaluation routines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
