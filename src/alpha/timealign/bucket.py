"""Bucketing utilities for aligning timeseries."""

from __future__ import annotations

from typing import Literal

import pandas as pd

_FREQUENCY_MAP: dict[str, str] = {
    "1h": "1h",
    "4h": "4h",
    "24h": "24h",
}


def to_bar(ts: pd.Series, freq: Literal["1h", "4h", "24h"]) -> pd.Series:
    """Floor timestamps to the requested bar size in UTC to prevent lookahead.

    Leakage guarantee: flooring (never ceiling) ensures each timestamp maps to
    its current bar, so events are not assigned to a future bar.
    """
    if not isinstance(ts, pd.Series):
        raise TypeError("`ts` must be a pandas Series.")

    if freq not in _FREQUENCY_MAP:
        raise ValueError(f"Unsupported frequency '{freq}'.")

    dt_series = pd.to_datetime(ts, utc=True, errors="raise")
    floored = dt_series.dt.floor(_FREQUENCY_MAP[freq])
    return floored.dt.tz_localize(None)
