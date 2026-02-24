from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from bitbat.backtest.engine import run


def test_backtest_long_only() -> None:
    idx = pd.date_range(datetime(2024, 1, 1), periods=5, freq="1h")
    prices = pd.Series([100, 101, 102, 101, 103], index=idx, dtype=float)
    predicted_returns = pd.Series([0.001, 0.01, 0.02, -0.01, 0.02], index=idx)

    trades, equity = run(prices, predicted_returns, allow_short=False)

    # Positive predictions -> long, negative -> flat (no short allowed)
    assert trades["position"].iloc[0] >= 0
    # First bar may have a small cost from opening the position
    assert equity.iloc[0] == pytest.approx(1.0, abs=1e-3)


def test_backtest_allows_short() -> None:
    idx = pd.date_range(datetime(2024, 1, 1), periods=5, freq="1h")
    prices = pd.Series([100, 98, 97, 99, 95], index=idx, dtype=float)
    predicted_returns = pd.Series([-0.02, -0.03, -0.04, -0.01, -0.03], index=idx)

    trades, equity = run(prices, predicted_returns, allow_short=True)

    # Negative predicted returns -> short positions
    assert (trades["position"] < 0).any()


def test_trades_contain_cost_columns() -> None:
    idx = pd.date_range(datetime(2024, 1, 1), periods=10, freq="1h")
    prices = pd.Series(range(100, 110), index=idx, dtype=float)
    predicted_returns = pd.Series([0.01] * 10, index=idx)

    trades, _ = run(prices, predicted_returns, cost_bps=4.0)

    assert "costs" in trades.columns
    assert "gross_pnl" in trades.columns
    assert "pnl" in trades.columns


def test_zero_cost_gives_equal_pnl() -> None:
    idx = pd.date_range(datetime(2024, 1, 1), periods=10, freq="1h")
    prices = pd.Series(range(100, 110), index=idx, dtype=float)
    predicted_returns = pd.Series([0.01] * 10, index=idx)

    trades, _ = run(prices, predicted_returns, cost_bps=0.0)

    pd.testing.assert_series_equal(
        trades["pnl"], trades["gross_pnl"], check_names=False
    )
    assert (trades["costs"] == 0.0).all()


def test_high_cost_reduces_equity() -> None:
    idx = pd.date_range(datetime(2024, 1, 1), periods=10, freq="1h")
    prices = pd.Series(range(100, 110), index=idx, dtype=float)
    predicted_returns = pd.Series([0.01] * 10, index=idx)

    _, equity_low = run(prices, predicted_returns, cost_bps=0.0)
    _, equity_high = run(prices, predicted_returns, cost_bps=100.0)

    assert equity_low.iloc[-1] > equity_high.iloc[-1]


def test_min_signal_filters_noise() -> None:
    idx = pd.date_range(datetime(2024, 1, 1), periods=5, freq="1h")
    prices = pd.Series([100, 101, 102, 103, 104], index=idx, dtype=float)
    # Very small predicted returns below min_signal
    predicted_returns = pd.Series(
        [0.0001, 0.0001, 0.0001, 0.0001, 0.0001], index=idx
    )

    trades, _ = run(prices, predicted_returns, min_signal=0.0005)

    # All positions should be 0 since predictions are below min_signal
    assert (trades["position"] == 0.0).all()
