"""Shared utilities for autonomous ingestion."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd


def get_last_parquet_date(
    parquet_file: Path, date_col: str, logger: logging.Logger
) -> datetime | None:
    """Return the latest date found in the stored parquet file."""
    if not parquet_file.exists():
        return None
    try:
        df = pd.read_parquet(parquet_file, columns=[date_col])
        if df.empty or date_col not in df.columns:
            return None
        max_date = pd.to_datetime(df[date_col]).max()
        return max_date.to_pydatetime()
    except Exception as exc:
        logger.warning("Could not read existing parquet %s: %s", parquet_file, exc)
        return None


def run_with_retry(
    fetch_func: Callable[[], int],
    logger: logging.Logger,
    name: str,
    max_retries: int = 3,
) -> int:
    """Run a fetch function with exponential backoff on failure."""
    for attempt in range(max_retries):
        try:
            return fetch_func()
        except Exception as exc:
            if attempt >= max_retries - 1:
                logger.error("All %d %s fetch attempts failed: %s", max_retries, name, exc)
                raise
            wait = 2**attempt
            logger.warning(
                "%s fetch attempt %d/%d failed: %s — retrying in %ds",
                name.capitalize(),
                attempt + 1,
                max_retries,
                exc,
                wait,
            )
            time.sleep(wait)
    return 0  # unreachable, but satisfies type checker
