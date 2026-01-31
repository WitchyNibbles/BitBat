"""Backtest engine implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def run(
    prices: pd.Series,
    proba_up: pd.Series | np.ndarray,
    proba_down: pd.Series | np.ndarray,
    *,
    enter: float = 0.6,
    allow_short: bool = False,
    cost_bps: float = 4.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Run a simple probability threshold strategy."""
    close = pd.Series(prices, dtype="float64")
    up = pd.Series(proba_up, index=close.index, dtype="float64")
    down = pd.Series(proba_down, index=close.index, dtype="float64")

    position = np.zeros(len(close))
    for i in range(len(close)):
        if up.iloc[i] >= enter:
            position[i] = 1.0
        elif allow_short and down.iloc[i] >= enter:
            position[i] = -1.0
        else:
            position[i] = position[i - 1] if i > 0 else 0.0

    position_series = pd.Series(position, index=close.index, name="position")
    returns = close.pct_change().fillna(0.0)

    trade_changes = position_series.diff().fillna(position_series.iloc[0])
    costs = np.abs(trade_changes) * (cost_bps / 10000.0)

    pnl = position_series.shift(1).fillna(0.0) * returns - costs
    equity_curve = (1 + pnl).cumprod()

    trades = pd.DataFrame(
        {
            "close": close,
            "position": position_series,
            "returns": returns,
            "pnl": pnl,
        },
        index=close.index,
    )
    equity_curve.name = "equity"

    return trades, equity_curve


def run_backtest(strategy: Any, data: Any) -> Any:  # pragma: no cover - legacy stub
    """Execute a backtest for the provided strategy."""
    raise NotImplementedError("run_backtest is not implemented yet.")
