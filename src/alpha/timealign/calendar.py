"""Trading calendar utilities."""

from __future__ import annotations

import pandas as pd


def ensure_utc(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return a copy with `column` coerced to UTC and tz-naive to avoid leakage.

    Leakage guarantee: timestamps are normalised to a single UTC frame and made
    tz-naive, so downstream alignment uses consistent absolute times and avoids
    lookahead caused by mixed or ambiguous timezone assumptions.
    """
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
