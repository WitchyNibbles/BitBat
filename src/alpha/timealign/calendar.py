"""Trading calendar utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date

import pandas as pd


def build_trading_days(
    start: date,
    end: date,
    holidays: Iterable[date] | None = None,
) -> Sequence[date]:
    """Build a sequence of trading days between start and end."""
    raise NotImplementedError("build_trading_days is not implemented yet.")


def ensure_utc(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Ensure a DataFrame column is UTC-normalised and tz-naive."""
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    normalised = df.copy()
    series = pd.to_datetime(normalised[column], utc=True, errors="coerce")
    if series.isna().any():
        invalid_rows = normalised[series.isna()]
        raise ValueError(
            f"Unable to convert all values to UTC datetimes for column '{column}'. "
            f"Invalid rows: {len(invalid_rows)}"
        )

    normalised[column] = series.dt.tz_localize(None)
    return normalised
