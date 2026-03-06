"""Shared price loading utilities consolidating divergent implementations.

Previously three independent price-loading functions existed in:
- bitbat.cli (_load_prices_indexed)
- bitbat.autonomous.predictor (_load_ingested_prices)
- bitbat.autonomous.continuous_trainer (ContinuousTrainer._load_prices)

This module provides a single canonical implementation that callers delegate to.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from bitbat.timealign.calendar import ensure_utc

logger = logging.getLogger(__name__)


def load_prices(data_dir: Path, freq: str) -> pd.DataFrame:
    """Load price bars from the date-partitioned ingestion directory.

    Scans ``data_dir / "raw" / "prices"`` for all ``**/*.parquet`` files
    (date-partitioned files written by PriceIngestionService and the legacy
    flat file written by the CLI ingest command).  Filters frames that contain
    both ``timestamp_utc`` and ``close`` columns, concatenates, deduplicates on
    ``timestamp_utc`` (keep last), normalises timestamps to tz-naive UTC, sets
    the index to ``timestamp_utc``, and sorts.

    Parameters
    ----------
    data_dir:
        Root data directory (e.g. ``Path("data")``).
    freq:
        Bar frequency string (e.g. ``"1h"``).  Currently informational; may be
        used for future per-frequency path resolution.

    Returns
    -------
    pd.DataFrame
        Price DataFrame indexed by tz-naive UTC ``timestamp_utc``, sorted
        ascending.

    Raises
    ------
    RuntimeError
        If no price data is found under ``data_dir / "raw" / "prices"``.
    """
    prices_root = data_dir / "raw" / "prices"
    frames: list[pd.DataFrame] = []

    if prices_root.exists():
        for parquet_file in sorted(prices_root.glob("**/*.parquet")):
            try:
                df = pd.read_parquet(parquet_file)
                if "timestamp_utc" in df.columns and "close" in df.columns:
                    frames.append(df)
            except Exception as exc:
                logger.warning("Skipping unreadable price file %s: %s", parquet_file, exc)

    if not frames:
        raise RuntimeError(f"No price data found under {prices_root}")

    merged = pd.concat(frames, ignore_index=True)
    merged["timestamp_utc"] = pd.to_datetime(
        merged["timestamp_utc"], utc=True, errors="coerce"
    ).dt.tz_localize(None)
    merged = (
        merged.sort_values("timestamp_utc")
        .drop_duplicates(subset=["timestamp_utc"], keep="last")
        .reset_index(drop=True)
    )
    return merged.set_index("timestamp_utc").sort_index()


def load_prices_for_cli(freq: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Load a single specific price file using the CLI's flat-file convention.

    The CLI ingest command writes to a single flat file:
    ``data/raw/prices/btcusd_yf_{freq}.parquet``.  This wrapper preserves that
    explicit single-file contract for callers that need it (e.g. batch inference,
    backtest commands).

    Parameters
    ----------
    freq:
        Bar frequency string (e.g. ``"1h"``).
    data_dir:
        Root data directory.  Defaults to ``Path("data")``.

    Returns
    -------
    pd.DataFrame
        Price DataFrame indexed by tz-naive UTC ``timestamp_utc``, sorted
        ascending.

    Raises
    ------
    FileNotFoundError
        If the expected price file does not exist.
    """
    base = data_dir if data_dir is not None else Path("data")
    prices_path = base / "raw" / "prices" / f"btcusd_yf_{freq}.parquet"
    if not prices_path.exists():
        raise FileNotFoundError(f"Prices parquet not found: {prices_path}")
    return (
        ensure_utc(pd.read_parquet(prices_path), "timestamp_utc")
        .set_index("timestamp_utc")
        .sort_index()
    )
