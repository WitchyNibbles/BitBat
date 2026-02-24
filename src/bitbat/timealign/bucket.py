"""Bucketing utilities for aligning timeseries."""

from __future__ import annotations

import pandas as pd

_SUPPORTED_FREQUENCIES: set[str] = {
    "1m", "5m", "15m", "30m", "1h", "4h", "24h",
}

# pandas >= 2.2 deprecated 'm' for minutes; map to 'min'.
_PANDAS_FREQ_MAP: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
}


def to_bar(ts: pd.Series, freq: str) -> pd.Series:
    """Floor timestamps to the requested bar size in UTC to prevent lookahead.

    Leakage guarantee: flooring (never ceiling) ensures each timestamp maps to
    its current bar, so events are not assigned to a future bar.
    """
    if not isinstance(ts, pd.Series):
        raise TypeError("`ts` must be a pandas Series.")

    if freq not in _SUPPORTED_FREQUENCIES:
        raise ValueError(f"Unsupported frequency '{freq}'.")

    pd_freq = _PANDAS_FREQ_MAP.get(freq, freq)
    dt_series = pd.to_datetime(ts, utc=True, errors="raise")
    floored = dt_series.dt.floor(pd_freq)
    return floored.dt.tz_localize(None)


def bars_for_duration(duration: str, freq: str) -> int:
    """Convert a time duration to bar count at the given frequency.

    E.g., ``bars_for_duration("24h", "5m")`` returns 288.
    """
    dur_td = pd.to_timedelta(duration)
    freq_td = pd.to_timedelta(freq)
    if freq_td <= pd.Timedelta(0):
        raise ValueError(f"Frequency must be positive: {freq}")
    return max(1, int(dur_td / freq_td))
