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
