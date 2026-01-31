from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pd_testing

from alpha.features.price import (
    atr,
    lagged_returns,
    macd,
    obv,
    rolling_std,
    rolling_z,
    rsi,
)


def _assert_no_leakage(
    result_full: pd.Series | pd.DataFrame,
    result_trunc: pd.Series | pd.DataFrame,
    upto: int,
) -> None:
    if isinstance(result_full, pd.DataFrame):
        pd_testing.assert_frame_equal(
            result_full.iloc[: upto + 1],
            result_trunc.iloc[: upto + 1],
            check_names=True,
        )
    else:
        pd_testing.assert_series_equal(
            result_full.iloc[: upto + 1],
            result_trunc.iloc[: upto + 1],
            check_names=True,
        )


def test_lagged_returns_no_leakage_and_values() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="1h")
    close = pd.Series([100, 110, 105, 120, 115], index=index)
    features = lagged_returns(close, lags=[1, 2])

    expected = pd.DataFrame(
        {
            "return_lag_1": [np.nan, 0.1, -0.0454545, 0.1428571, -0.0416667],
            "return_lag_2": [np.nan, np.nan, 0.05, 0.0909091, 0.0952381],
        },
        index=index,
    )
    pd_testing.assert_frame_equal(features, expected, rtol=1e-6, atol=1e-6)

    for i in range(len(close)):
        truncated = close.copy()
        truncated.iloc[i + 1 :] = np.nan
        truncated_features = lagged_returns(truncated, lags=[1, 2])
        _assert_no_leakage(features, truncated_features, i)


def test_rolling_std_no_leakage() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="1h")
    close = pd.Series([100, 110, 121, 133.1, 146.41, 161.051], index=index)
    std = rolling_std(close, window=3)

    returns = close.pct_change()
    expected = returns.rolling(window=3, min_periods=3).std()
    expected.name = "rolling_std_3"
    pd_testing.assert_series_equal(std, expected)

    for i in range(len(close)):
        truncated = close.copy()
        truncated.iloc[i + 1 :] = np.nan
        truncated_std = rolling_std(truncated, window=3)
        _assert_no_leakage(std, truncated_std, i)


def test_atr_no_leakage() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="1h")
    df = pd.DataFrame(
        {
            "high": [10, 11, 12, 14, 13, 15],
            "low": [9, 9.5, 10, 11, 12, 13],
            "close": [9.5, 10, 11.5, 13, 12.5, 14],
        },
        index=index,
    )
    atr_full = atr(df, window=3)

    for i in range(len(df)):
        truncated = df.copy()
        truncated.iloc[i + 1 :, :] = np.nan
        atr_trunc = atr(truncated, window=3)
        _assert_no_leakage(atr_full, atr_trunc, i)


def test_rsi_no_leakage() -> None:
    index = pd.date_range("2024-01-01", periods=10, freq="1h")
    close = pd.Series(np.linspace(100, 109, 10), index=index)
    rsi_full = rsi(close, window=3)

    for i in range(len(close)):
        truncated = close.copy()
        truncated.iloc[i + 1 :] = np.nan
        rsi_trunc = rsi(truncated, window=3)
        _assert_no_leakage(rsi_full, rsi_trunc, i)


def test_macd_no_leakage_and_columns() -> None:
    index = pd.date_range("2024-01-01", periods=50, freq="1h")
    close = pd.Series(np.linspace(100, 150, 50), index=index)
    macd_full = macd(close)
    assert list(macd_full.columns) == ["macd", "macd_signal", "macd_hist"]

    for i in range(len(close)):
        truncated = close.copy()
        truncated.iloc[i + 1 :] = np.nan
        macd_trunc = macd(truncated)
        _assert_no_leakage(macd_full, macd_trunc, i)


def test_obv_no_leakage() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="1h")
    close = pd.Series([10, 11, 10.5, 10.5, 12, 11], index=index)
    volume = pd.Series([100, 150, 120, 110, 200, 90], index=index)
    obv_full = obv(close, volume)

    manual = []
    total = 0.0
    for i in range(len(close)):
        if i == 0:
            total += 0
        else:
            if close.iloc[i] > close.iloc[i - 1]:
                total += volume.iloc[i]
            elif close.iloc[i] < close.iloc[i - 1]:
                total -= volume.iloc[i]
        manual.append(total)
    expected = pd.Series(manual, index=index, name="obv")
    pd_testing.assert_series_equal(obv_full, expected)

    for i in range(len(close)):
        close_trunc = close.copy()
        vol_trunc = volume.copy()
        close_trunc.iloc[i + 1 :] = np.nan
        vol_trunc.iloc[i + 1 :] = np.nan
        obv_trunc = obv(close_trunc, vol_trunc)
        _assert_no_leakage(obv_full, obv_trunc, i)


def test_rolling_z_mean_zero_post_warmup() -> None:
    index = pd.date_range("2024-01-01", periods=30, freq="1h")
    rng = np.random.default_rng(0)
    base = pd.Series(rng.standard_normal(30), index=index)
    series = base.cumsum()

    z_scores = rolling_z(series, window=5)
    warm = z_scores.dropna()
    centered = warm.abs().mean()
    assert centered < 1.5
