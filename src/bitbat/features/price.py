"""Price-based feature generation."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def lagged_returns(close: pd.Series, lags: Iterable[int] | range = range(1, 25)) -> pd.DataFrame:
    """Compute lagged percent returns for the provided close prices."""
    close_series = pd.Series(close, dtype="float64")
    features = {
        f"return_lag_{lag}": close_series.pct_change(periods=lag, fill_method=None) for lag in lags
    }
    return pd.DataFrame(features, index=close_series.index)


def rolling_std(close: pd.Series, window: int = 24) -> pd.Series:
    """Rolling standard deviation of percent returns."""
    close_series = pd.Series(close, dtype="float64")
    returns = close_series.pct_change(fill_method=None)
    std = returns.rolling(window=window, min_periods=window).std()
    std.name = f"rolling_std_{window}"
    return std


def rolling_z(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling z-score over the given window."""
    data = pd.Series(series, dtype="float64")
    mean = data.rolling(window=window, min_periods=window).mean()
    std = data.rolling(window=window, min_periods=window).std()
    z = (data - mean) / std
    z.name = f"rolling_z_{window}"
    return z


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range over the specified window."""
    required = {"high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns for ATR calculation: {missing}")

    high = pd.Series(df["high"], dtype="float64")
    low = pd.Series(df["low"], dtype="float64")
    close = pd.Series(df["close"], dtype="float64")

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_series = tr.rolling(window=window, min_periods=window).mean()
    atr_series.name = f"atr_{window}"
    return atr_series


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing."""
    close_series = pd.Series(close, dtype="float64")
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))
    rsi_series.name = f"rsi_{window}"
    return rsi_series


def macd(close: pd.Series) -> pd.DataFrame:
    """Moving Average Convergence Divergence with default spans (12, 26, 9)."""
    close_series = pd.Series(close, dtype="float64")

    ema_fast = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = close_series.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = ema_fast - ema_slow
    signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    hist = macd_line - signal

    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal,
            "macd_hist": hist,
        },
        index=close_series.index,
    )


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume indicator."""
    close_series = pd.Series(close, dtype="float64")
    volume_series = pd.Series(volume, dtype="float64").fillna(0.0)

    price_change = close_series.diff().fillna(0.0)

    direction = price_change.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv_contrib = volume_series * direction
    obv_series = obv_contrib.cumsum()
    obv_series.name = "obv"
    return obv_series
