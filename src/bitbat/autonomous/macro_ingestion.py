"""Autonomous FRED macro data ingestion service."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from bitbat.ingest.macro_fred import fetch_fred

logger = logging.getLogger(__name__)


class MacroIngestionService:
    """Incrementally fetch FRED macro data and append to local parquet store."""

    def __init__(self, data_dir: Path | None = None) -> None:
        if data_dir is None:
            data_dir = Path("data")
        self.macro_dir = data_dir / "raw" / "macro"
        self.macro_dir.mkdir(parents=True, exist_ok=True)
        logger.info("MacroIngestionService initialised: %s", self.macro_dir)

    def _get_last_date(self) -> datetime | None:
        """Return the latest date found in the stored parquet file."""
        parquet_file = self.macro_dir / "fred.parquet"
        if not parquet_file.exists():
            return None
        try:
            df = pd.read_parquet(parquet_file)
            if df.empty or "date" not in df.columns:
                return None
            max_date = pd.to_datetime(df["date"]).max()
            return max_date.to_pydatetime()
        except Exception:
            logger.warning("Could not read existing macro parquet")
            return None

    def fetch_latest(self) -> int:
        """Fetch macro data from the last stored date to today.

        Returns the number of new rows fetched.
        """
        last = self._get_last_date()
        start = datetime(2017, 1, 1) if last is None else last - timedelta(days=5)

        logger.info("Fetching FRED macro data from %s", start.date())
        frame = fetch_fred(start=start, output_root=self.macro_dir)
        new_rows = len(frame) if frame is not None else 0
        logger.info("Fetched %d macro rows", new_rows)
        return new_rows

    def fetch_with_retry(self, max_retries: int = 3) -> int:
        """Fetch with exponential backoff on failure."""
        for attempt in range(max_retries):
            try:
                return self.fetch_latest()
            except Exception as exc:
                if attempt >= max_retries - 1:
                    logger.error("All %d macro fetch attempts failed: %s", max_retries, exc)
                    raise
                wait = 2**attempt
                logger.warning(
                    "Macro fetch attempt %d/%d failed: %s — retrying in %ds",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
        return 0  # unreachable, satisfies type checker
