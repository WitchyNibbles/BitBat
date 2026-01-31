"""Bucketing utilities for aligning timeseries."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import pandas as pd

_FREQUENCY_MAP: dict[str, str] = {
    "1h": "1h",
    "4h": "4h",
    "24h": "24h",
}


def to_bar(ts: pd.Series, freq: Literal["1h", "4h", "24h"]) -> pd.Series:
    """Floor timestamps to the requested bar size ensuring UTC-normalised output."""
    if not isinstance(ts, pd.Series):
        raise TypeError("`ts` must be a pandas Series.")

    if freq not in _FREQUENCY_MAP:
        raise ValueError(f"Unsupported frequency '{freq}'.")

    dt_series = pd.to_datetime(ts, utc=True, errors="raise")
    floored = dt_series.dt.floor(_FREQUENCY_MAP[freq])
    return floored.dt.tz_localize(None)


def bucketize(
    records: Iterable[Any],
    interval_seconds: int,
) -> list[Any]:  # pragma: no cover - legacy stub
    """Group records into uniform time buckets."""
    raise NotImplementedError("bucketize is not implemented yet.")
