from __future__ import annotations

from datetime import datetime

import pandas as pd

from bitbat.backtest.engine import run


def test_backtest_long_only() -> None:
    idx = pd.date_range(datetime(2024, 1, 1), periods=5, freq="1h")
    prices = pd.Series([100, 101, 102, 101, 103], index=idx)
    proba_up = pd.Series([0.5, 0.7, 0.8, 0.4, 0.9], index=idx)
    proba_down = 1 - proba_up

    trades, equity = run(prices, proba_up, proba_down, enter=0.6, allow_short=False)

    assert trades["position"].iloc[1] == 1
    assert trades["position"].iloc[-1] == 1
    assert equity.iloc[0] == 1
    assert equity.iloc[-1] > 1


def test_backtest_allows_short() -> None:
    idx = pd.date_range(datetime(2024, 1, 1), periods=5, freq="1h")
    prices = pd.Series([100, 98, 97, 99, 95], index=idx)
    proba_up = pd.Series([0.4, 0.3, 0.2, 0.4, 0.3], index=idx)
    proba_down = 1 - proba_up

    trades, equity = run(prices, proba_up, proba_down, enter=0.6, allow_short=True)

    assert trades["position"].iloc[1] == -1
    assert equity.iloc[-1] > 1


def test_trades_contain_cost_columns() -> None:
    idx = pd.date_range(datetime(2024, 1, 1), periods=10, freq="1h")
    prices = pd.Series(range(100, 110), index=idx, dtype=float)
    proba_up = pd.Series([0.7] * 10, index=idx)
    proba_down = 1 - proba_up

    trades, _ = run(prices, proba_up, proba_down, enter=0.6, cost_bps=4.0)

    assert "costs" in trades.columns
    assert "gross_pnl" in trades.columns
    assert "pnl" in trades.columns


def test_zero_cost_gives_equal_pnl() -> None:
    idx = pd.date_range(datetime(2024, 1, 1), periods=10, freq="1h")
    prices = pd.Series(range(100, 110), index=idx, dtype=float)
    proba_up = pd.Series([0.7] * 10, index=idx)
    proba_down = 1 - proba_up

    trades, _ = run(prices, proba_up, proba_down, enter=0.6, cost_bps=0.0)

    pd.testing.assert_series_equal(trades["pnl"], trades["gross_pnl"], check_names=False)
    assert (trades["costs"] == 0.0).all()


def test_high_cost_reduces_equity() -> None:
    idx = pd.date_range(datetime(2024, 1, 1), periods=10, freq="1h")
    prices = pd.Series(range(100, 110), index=idx, dtype=float)
    proba_up = pd.Series([0.7] * 10, index=idx)
    proba_down = 1 - proba_up

    _, equity_low = run(prices, proba_up, proba_down, enter=0.6, cost_bps=0.0)
    _, equity_high = run(prices, proba_up, proba_down, enter=0.6, cost_bps=100.0)

    assert equity_low.iloc[-1] > equity_high.iloc[-1]
