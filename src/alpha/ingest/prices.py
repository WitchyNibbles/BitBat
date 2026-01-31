"""Market price ingestion pipelines."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from alpha.contracts import ensure_prices_contract
from alpha.io.fs import write_parquet

# Yahoo restricts intraday downloads to roughly two years per request.
_DEFAULT_CHUNK_DAYS = 700

_INTERVAL_STEP = {
    "1m": timedelta(minutes=1),
    "2m": timedelta(minutes=2),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "60m": timedelta(hours=1),
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
}


def _ensure_utc_start(start: datetime) -> datetime:
    """Normalise the start datetime to UTC."""
    if start.tzinfo is None:
        return start.replace(tzinfo=UTC)
    return start.astimezone(UTC)


def _target_path(symbol: str, interval: str, root: Path | str | None = None) -> Path:
    """Build the parquet target path for a symbol/interval combination."""
    base = Path(root) if root is not None else Path("data") / "raw" / "prices"
    sanitized_symbol = symbol.replace("-", "").replace("/", "").lower()
    sanitized_interval = interval.lower()
    return base / f"{sanitized_symbol}_yf_{sanitized_interval}.parquet"


def _resolve_interval_step(interval: str) -> timedelta:
    """Get the sampling resolution for the given interval."""
    if interval not in _INTERVAL_STEP:
        raise ValueError(f"Unsupported interval '{interval}'.")
    return _INTERVAL_STEP[interval]


def _flatten_columns(columns: pd.Index, symbol: str) -> list[str]:
    """Flatten yfinance columns, dropping ticker suffixes."""
    flat_index = columns.to_flat_index() if hasattr(columns, "to_flat_index") else columns
    symbol_lower = symbol.lower()
    flattened: list[str] = []
    for label in flat_index:
        if isinstance(label, tuple):
            candidate = None
            for part in label:
                if part is None:
                    continue
                part_str = str(part)
                if part_str and part_str.lower() != symbol_lower:
                    candidate = part_str
                    break
            if candidate is None and label:
                candidate = str(label[-1])
            flattened.append(candidate.lower() if candidate else "")
        else:
            flattened.append(str(label).lower())
    return flattened


def _download_chunk(
    symbol: str,
    interval: str,
    chunk_start: datetime,
    chunk_end: datetime,
) -> pd.DataFrame:
    """Download a chunk of price data from yfinance."""
    return yf.download(
        symbol,
        interval=interval,
        start=chunk_start,
        end=chunk_end,
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )


def fetch_yf(
    symbol: str,
    interval: str,
    start: datetime,
    *,
    output_root: Path | str | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV bars from yfinance and persist them to partitioned parquet."""
    start_utc = _ensure_utc_start(start)
    end_utc = datetime.now(UTC)
    if start_utc >= end_utc:
        raise ValueError("Start time must be earlier than now.")

    step = _resolve_interval_step(interval)
    chunk_span = timedelta(days=_DEFAULT_CHUNK_DAYS)

    frames: list[pd.DataFrame] = []
    cursor = start_utc
    while cursor < end_utc:
        upper_bound = min(cursor + chunk_span, end_utc + step)
        chunk = _download_chunk(symbol, interval, cursor, upper_bound)
        if not chunk.empty:
            frames.append(chunk)
            last_index = chunk.index[-1]
            if isinstance(last_index, pd.Timestamp):
                last_dt = last_index.to_pydatetime()
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=UTC)
                else:
                    last_dt = last_dt.astimezone(UTC)
                cursor = max(last_dt + step, cursor + step)
            else:
                cursor = upper_bound
        else:
            cursor = upper_bound

    if not frames:
        raise ValueError(f"No data returned from yfinance for {symbol} ({interval}).")

    price_frame = pd.concat(frames, axis=0)

    frame = price_frame.copy()
    frame.columns = _flatten_columns(frame.columns, symbol)
    frame = frame.reset_index(drop=False)
    frame.columns = _flatten_columns(frame.columns, symbol)

    timestamp_column = frame.columns[0]
    frame = frame.rename(columns={timestamp_column: "timestamp_utc"})

    allowed_cols = {"timestamp_utc", "open", "high", "low", "close", "adj close", "volume"}
    frame = frame.drop(columns=[col for col in frame.columns if col not in allowed_cols])
    frame = frame.drop(columns=["adj close"], errors="ignore")

    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame["timestamp_utc"] = timestamps.dt.tz_localize(None)

    frame = frame.sort_values("timestamp_utc").drop_duplicates(subset="timestamp_utc")

    start_bound = start_utc.replace(tzinfo=None)
    end_bound = end_utc.replace(tzinfo=None)
    frame = frame[(frame["timestamp_utc"] >= start_bound) & (frame["timestamp_utc"] <= end_bound)]

    ordered_columns = ["timestamp_utc", "open", "high", "low", "close", "volume"]
    missing_columns = sorted(set(ordered_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing expected columns from yfinance payload: {missing_columns}")

    frame = frame[ordered_columns]
    frame["source"] = "yfinance"
    frame = ensure_prices_contract(frame)

    target_dir = _target_path(symbol, interval, output_root)
    if target_dir.exists():
        if target_dir.is_dir():
            shutil.rmtree(target_dir)
        else:
            target_dir.unlink()
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    partitioned = frame.copy()
    partitioned["year"] = partitioned["timestamp_utc"].dt.year
    write_parquet(partitioned, target_dir, partition_cols=["year"], engine="pyarrow")

    return frame
