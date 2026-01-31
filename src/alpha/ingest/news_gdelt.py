"""GDELT news ingestion pipelines."""

from __future__ import annotations

import logging
import random
import shutil
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import pandas as pd

try:
    import requests  # type: ignore[import-not-found, import-untyped]
except ImportError:  # pragma: no cover - defer import errors for optional dependency
    requests = None  # type: ignore[assignment]

from alpha.contracts import ensure_news_contract
from alpha.io.fs import read_parquet, write_parquet

LOGGER = logging.getLogger(__name__)

GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
KEYWORDS = "(bitcoin OR btc OR crypto OR cryptocurrency)"
RESULT_COLUMNS = ["published_utc", "title", "url", "source", "lang", "sentiment_score"]
WINDOW = timedelta(hours=1)


class GdeltError(RuntimeError):
    """Raised when the GDELT API returns an unexpected response."""


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _to_gdelt_datetime(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def _iter_windows(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    windows: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end:
        next_cursor = min(cursor + WINDOW, end)
        windows.append((cursor, next_cursor))
        cursor = next_cursor
    return windows


def _target_path(root: Path | str | None = None) -> Path:
    base = Path(root) if root is not None else Path("data") / "raw" / "news" / "gdelt_1h"
    return base / "gdelt_crypto_1h.parquet"


@runtime_checkable
class SessionProtocol(Protocol):
    def get(  # pragma: no cover - protocol
        self,
        url: str,
        **kwargs: Any,
    ) -> Any:
        """Perform a GET request."""

    def close(self) -> None:  # pragma: no cover - protocol
        """Close the session."""


def _fetch_chunk(
    session: SessionProtocol,
    start: datetime,
    end: datetime,
    *,
    retries: int,
    backoff_base: float,
) -> list[dict[str, Any]]:
    payload = {
        "query": KEYWORDS,
        "mode": "artlist",
        "format": "json",
        "maxrecords": 250,
        "sort": "DateAsc",
        "startdatetime": _to_gdelt_datetime(start),
        "enddatetime": _to_gdelt_datetime(end),
    }
    attempt = 0
    delay = max(backoff_base, 0.0)
    while True:
        response = session.get(GDELT_ENDPOINT, params=payload, timeout=30)
        if response.status_code == 429:
            if attempt >= retries:
                raise GdeltError("GDELT request failed with status 429 after retries")
            jitter = random.uniform(0, max(delay, backoff_base))  # noqa: S311
            sleep_for = max(delay + jitter, backoff_base)
            LOGGER.warning(
                "Rate limited by GDELT (429) for %s-%s; sleeping %.2fs before retry",
                start,
                end,
                sleep_for,
            )
            time.sleep(sleep_for)
            attempt += 1
            delay = max(delay * 2, backoff_base)
            continue

        if response.status_code >= 400:
            raise GdeltError(f"GDELT request failed with status {response.status_code}")

        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - defensive
            snippet: str | None = None
            try:
                snippet = response.text[:200]
            except Exception:  # pragma: no cover - best effort only
                snippet = None
            message = "Failed to decode GDELT JSON response"
            if snippet:
                message += f"; payload preview: {snippet!r}"
            raise GdeltError(message) from exc
        break

    articles = data.get("articles") or data.get("docs") or []
    if not isinstance(articles, list):
        raise GdeltError("Unexpected GDELT response structure")
    return articles


def _articles_to_frame(articles: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for article in articles:
        published = article.get("seendate") or article.get("date")
        title = article.get("title")
        url = article.get("url")
        source = article.get("sourceCommonName") or article.get("source")
        lang = article.get("language")
        records.append(
            {
                "published_utc": published,
                "title": title,
                "url": url,
                "source": source,
                "lang": lang,
            }
        )

    frame = pd.DataFrame.from_records(records, columns=RESULT_COLUMNS)
    if frame.empty:
        return frame

    frame["published_utc"] = pd.to_datetime(frame["published_utc"], utc=True, errors="coerce")
    frame["published_utc"] = frame["published_utc"].dt.tz_localize(None)
    frame = frame.dropna(subset=["published_utc", "url"])
    frame = frame[frame["url"].str.startswith("http", na=False)]
    frame["title"] = frame["title"].fillna("")
    frame["source"] = frame["source"].fillna("unknown")
    frame["lang"] = frame["lang"].fillna("unknown")
    from alpha.features.sentiment import score_vader

    frame["sentiment_score"] = score_vader(frame["title"])
    return frame


def _load_existing(target: Path) -> pd.DataFrame | None:
    if target.exists():
        try:
            existing = read_parquet(target)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to read existing GDELT dataset: %s", exc)
            return None
        missing_cols = [col for col in RESULT_COLUMNS if col not in existing.columns]
        for col in missing_cols:
            existing[col] = pd.NA
        return existing[RESULT_COLUMNS]
    return None


def fetch(
    from_dt: datetime,
    to_dt: datetime,
    *,
    session: SessionProtocol | None = None,
    output_root: Path | str | None = None,
    throttle_seconds: float = 0.0,
    retry_limit: int = 3,
) -> pd.DataFrame:
    """Fetch GDELT news excerpts for the BTC keyword set and persist to parquet."""
    if from_dt >= to_dt:
        raise ValueError("`from_dt` must be earlier than `to_dt`.")

    start = _ensure_utc(from_dt)
    end = _ensure_utc(to_dt)

    created_session = False
    if session is not None:
        active_session = session
    else:
        if requests is None:  # pragma: no cover - dependency guard
            raise RuntimeError("The `requests` package is required to fetch GDELT data.")
        active_session = cast(SessionProtocol, requests.Session())
        created_session = True
    windows = _iter_windows(start, end)

    all_frames: list[pd.DataFrame] = []
    for window_start, window_end in windows:
        try:
            articles = _fetch_chunk(
                active_session,
                window_start,
                window_end,
                retries=retry_limit,
                backoff_base=max(throttle_seconds, 1.0),
            )
        except GdeltError as exc:
            LOGGER.warning("Chunk fetch failed for %s-%s: %s", window_start, window_end, exc)
        else:
            frame = _articles_to_frame(articles)
            if not frame.empty:
                all_frames.append(frame)
        finally:
            if throttle_seconds > 0:
                time.sleep(throttle_seconds)

    if created_session:
        active_session.close()

    if not all_frames:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    merged = pd.concat(all_frames, axis=0, ignore_index=True)
    merged = merged.sort_values("published_utc").drop_duplicates(subset=["url"])

    target_path = _target_path(output_root)
    existing = _load_existing(target_path)
    if existing is not None and not existing.empty:
        merged = (
            pd.concat([existing, merged], axis=0, ignore_index=True)
            .sort_values("published_utc")
            .drop_duplicates(subset=["url"])
        )

    merged = ensure_news_contract(merged)

    partitions = merged.copy()
    partitions["year"] = partitions["published_utc"].dt.year
    partitions["month"] = partitions["published_utc"].dt.month
    partitions["day"] = partitions["published_utc"].dt.day
    partitions["hour"] = partitions["published_utc"].dt.hour

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        if target_path.is_dir():
            shutil.rmtree(target_path)
        else:
            target_path.unlink()

    write_parquet(partitions, target_path, partition_cols=["year", "month", "day", "hour"])

    return merged.reset_index(drop=True)
