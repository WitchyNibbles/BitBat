"""Backtest engine implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def run(
    prices: pd.Series,
    predicted_returns: pd.Series | np.ndarray,
    *,
    min_signal: float = 0.0005,
    allow_short: bool = True,
    cost_bps: float = 4.0,
    fee_bps: float | None = None,
    slippage_bps: float | None = None,
    scale_factor: float = 0.01,
) -> tuple[pd.DataFrame, pd.Series]:
    """Run a regression-based backtest and return trades plus equity.

    Position sizing is proportional to predicted return magnitude:
    ``position = clamp(predicted_return / scale_factor, -1, 1)``.
    Predictions smaller than ``min_signal`` are treated as flat.
    """
    close = pd.Series(prices, dtype="float64")
    pred = pd.Series(predicted_returns, index=close.index, dtype="float64")

    raw_position = pred / scale_factor
    position = np.clip(raw_position, -1.0, 1.0)
    position = position.where(pred.abs() >= min_signal, 0.0)

    if not allow_short:
        position = position.clip(lower=0.0)

    position_series = pd.Series(position, index=close.index, name="position")
    returns = close.pct_change().fillna(0.0)

    trade_changes = position_series.diff().fillna(position_series.iloc[0])
    if fee_bps is None and slippage_bps is None:
        resolved_fee_bps = float(cost_bps)
        resolved_slippage_bps = 0.0
    elif fee_bps is None:
        resolved_slippage_bps = float(slippage_bps or 0.0)
        resolved_fee_bps = max(float(cost_bps) - resolved_slippage_bps, 0.0)
    elif slippage_bps is None:
        resolved_fee_bps = float(fee_bps)
        resolved_slippage_bps = max(float(cost_bps) - resolved_fee_bps, 0.0)
    else:
        resolved_fee_bps = float(fee_bps)
        resolved_slippage_bps = float(slippage_bps)

    fee_costs = np.abs(trade_changes) * (resolved_fee_bps / 10000.0)
    slippage_costs = np.abs(trade_changes) * (resolved_slippage_bps / 10000.0)
    costs = fee_costs + slippage_costs

    pnl = position_series.shift(1).fillna(0.0) * returns - costs
    equity_curve = (1 + pnl).cumprod()
    gross_pnl = position_series.shift(1).fillna(0.0) * returns

    trades = pd.DataFrame(
        {
            "close": close,
            "position": position_series,
            "returns": returns,
            "fee_costs": fee_costs,
            "slippage_costs": slippage_costs,
            "costs": costs,
            "gross_pnl": gross_pnl,
            "pnl": pnl,
        },
        index=close.index,
    )
    equity_curve.name = "equity"
    return trades, equity_curve
