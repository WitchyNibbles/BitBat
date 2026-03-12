"""Backtest metrics calculations."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bitbat.config.loader import resolve_metrics_dir


def _sharpe(returns: pd.Series, annualization: float = 252.0) -> float:
    """Annualised Sharpe ratio from a return series."""
    if returns.std() == 0:
        return 0.0
    return float(np.sqrt(annualization) * returns.mean() / returns.std())


def summary(  # noqa: C901
    equity_curve: pd.Series,
    trades: pd.DataFrame | None = None,
    predicted_returns: pd.Series | None = None,
    actual_returns: pd.Series | None = None,
) -> dict[str, float]:
    """Compute backtest metrics and persist summary artifacts.

    This function writes JSON metrics, an equity curve plot, and (optionally)
    trade details into a ``metrics/`` directory under the current working
    directory.

    Args:
        equity_curve: Equity curve indexed by timestamp.
        trades: Optional trades DataFrame with ``position``, ``costs``,
            ``gross_pnl``, and ``pnl`` columns.
        predicted_returns: Optional series of predicted returns (aligned with
            ``actual_returns``). When both are provided, MAE and correlation
            are included in the output.
        actual_returns: Optional series of actual returns.

    Returns:
        Dictionary containing net/gross sharpe, max drawdown, hit rate,
        average return, turnover, total costs, and optionally prediction_mae
        and prediction_correlation.
    """
    returns = equity_curve.pct_change().fillna(0.0)
    net_sharpe = _sharpe(returns)
    drawdown = equity_curve / equity_curve.cummax() - 1
    max_dd = drawdown.min()

    hits = (returns > 0).sum()
    total = (returns != 0).sum()
    hit_rate = hits / total if total else 0.0
    avg_ret = returns.mean()

    turnover = 0.0
    total_costs = 0.0
    total_fee_costs = 0.0
    total_slippage_costs = 0.0
    gross_sharpe = 0.0
    net_return = float(equity_curve.iloc[-1] - 1) if len(equity_curve) > 0 else 0.0
    gross_return = net_return

    if trades is not None:
        if "position" in trades.columns:
            turnover = trades["position"].diff().abs().sum()
        if "fee_costs" in trades.columns:
            total_fee_costs = float(trades["fee_costs"].sum())
        if "slippage_costs" in trades.columns:
            total_slippage_costs = float(trades["slippage_costs"].sum())
        if "costs" in trades.columns:
            total_costs = float(trades["costs"].sum())
        elif total_fee_costs > 0 or total_slippage_costs > 0:
            total_costs = float(total_fee_costs + total_slippage_costs)
        if "gross_pnl" in trades.columns:
            gross_sharpe = _sharpe(trades["gross_pnl"])
            gross_curve = (1.0 + trades["gross_pnl"]).cumprod()
            if len(gross_curve) > 0:
                gross_return = float(gross_curve.iloc[-1] - 1.0)

    metrics: dict[str, float] = {
        "sharpe": float(net_sharpe),
        "net_sharpe": float(net_sharpe),
        "gross_sharpe": float(gross_sharpe),
        "max_drawdown": float(max_dd),
        "hit_rate": float(hit_rate),
        "avg_return": float(avg_ret),
        "net_return": float(net_return),
        "gross_return": float(gross_return),
        "total_costs": float(total_costs),
        "total_fee_costs": float(total_fee_costs),
        "total_slippage_costs": float(total_slippage_costs),
        "turnover": float(turnover),
    }

    if predicted_returns is not None and actual_returns is not None:
        # Align on shared index and drop NaNs
        aligned = pd.DataFrame({"pred": predicted_returns, "actual": actual_returns}).dropna()
        if len(aligned) > 0:
            errors = aligned["pred"] - aligned["actual"]
            metrics["prediction_mae"] = float(errors.abs().mean())
            corr = aligned["pred"].corr(aligned["actual"])
            metrics["prediction_correlation"] = float(corr) if pd.notna(corr) else 0.0

    metrics_dir = resolve_metrics_dir()
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
