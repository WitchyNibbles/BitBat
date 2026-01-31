"""Backtest metrics calculations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summary(equity_curve: pd.Series, trades: pd.DataFrame | None = None) -> dict[str, float]:
    """Compute backtest metrics and persist summary artifacts.

    This function writes JSON metrics, an equity curve plot, and (optionally)
    trade details into a ``metrics/`` directory under the current working
    directory.

    Args:
        equity_curve: Equity curve indexed by timestamp.
        trades: Optional trades DataFrame with a ``position`` column for
            turnover calculation and CSV export.

    Returns:
        Dictionary containing sharpe ratio, max drawdown, hit rate, average
        return, and turnover.
    """
    returns = equity_curve.pct_change().fillna(0.0)
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0.0
    drawdown = equity_curve / equity_curve.cummax() - 1
    max_dd = drawdown.min()

    hits = (returns > 0).sum()
    total = (returns != 0).sum()
    hit_rate = hits / total if total else 0.0
    avg_ret = returns.mean()

    turnover = 0.0
    if trades is not None and "position" in trades.columns:
        turnover = trades["position"].diff().abs().sum()

    metrics = {
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "hit_rate": float(hit_rate),
        "avg_return": float(avg_ret),
        "turnover": float(turnover),
    }

    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "backtest_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    equity_curve.plot(ax=ax, title="Equity Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    fig.savefig(metrics_dir / "equity_curve.png", bbox_inches="tight")
    plt.close(fig)

    if trades is not None:
        trades.to_csv(metrics_dir / "trades.csv", index=True)

    return metrics
