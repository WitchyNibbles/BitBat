"""FRED macroeconomic data ingestion."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from bitbat.io.fs import write_parquet

logger = logging.getLogger(__name__)

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

DEFAULT_SERIES: dict[str, str] = {
    "DFF": "fed_funds_rate",
    "DGS10": "treasury_10y",
    "DTWEXBGS": "usd_index",
    "VIXCLS": "vix",
    "T5YIE": "inflation_5y_breakeven",
}


def _target_path(root: Path | str | None = None) -> Path:
    """Build the parquet target path for FRED macro data."""
    base = Path(root) if root is not None else Path("data") / "raw" / "macro"
    return base / "fred.parquet"


def _fetch_series(
    series_id: str,
    start: datetime,
    end: datetime,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch a single FRED series as a DataFrame."""
    key = api_key or os.environ.get("FRED_API_KEY", "")
    params: dict[str, str] = {
        "series_id": series_id,
        "observation_start": start.strftime("%Y-%m-%d"),
        "observation_end": end.strftime("%Y-%m-%d"),
        "file_type": "json",
    }
    if key:
        params["api_key"] = key

    resp = requests.get(FRED_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    observations = data.get("observations", [])
    if not observations:
        return pd.DataFrame(columns=["date", "value"])

    rows = []
    for obs in observations:
        val = obs.get("value", ".")
        rows.append({
            "date": obs["date"],
            "value": float(val) if val != "." else None,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_fred(
    series_ids: dict[str, str] | None = None,
    start: datetime = datetime(2017, 1, 1),
    end: datetime | None = None,
    *,
    api_key: str | None = None,
    output_root: Path | str | None = None,
) -> pd.DataFrame:
    """Fetch FRED macro series and persist to parquet.

    Parameters
    ----------
    series_ids:
        Mapping of FRED series ID to column name.  Defaults to
        ``DEFAULT_SERIES``.
    start:
        Start date for observations.
    end:
        End date (defaults to today).
    api_key:
        FRED API key.  Falls back to ``FRED_API_KEY`` env var.
    output_root:
        Root directory for output.  Parquet is written to
        ``{output_root}/fred.parquet``.

    Returns
    -------
    DataFrame with ``date`` index and one column per series.
    """
    if end is None:
        end = datetime.now()
    if series_ids is None:
        series_ids = DEFAULT_SERIES

    merged: pd.DataFrame | None = None
    for fred_id, col_name in series_ids.items():
        logger.info("Fetching FRED series %s → %s", fred_id, col_name)
        df = _fetch_series(fred_id, start, end, api_key=api_key)
        if df.empty:
            logger.warning("No data returned for FRED series %s", fred_id)
            continue
        df = df.rename(columns={"value": col_name}).set_index("date")
        merged = df if merged is None else merged.join(df, how="outer")

    if merged is None or merged.empty:
        raise ValueError("No data returned from FRED for any series.")

    # Forward-fill missing daily observations (weekends, holidays)
    merged = merged.sort_index().ffill()

    target = _target_path(output_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(merged.reset_index(), target)
    logger.info("Wrote FRED macro data to %s (%d rows)", target, len(merged))

    return merged
