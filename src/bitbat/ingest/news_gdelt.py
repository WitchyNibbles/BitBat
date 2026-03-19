"""GDELT news ingestion pipelines."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd

try:
    import requests  # type: ignore[import-not-found, import-untyped]
except ImportError:  # pragma: no cover - defer import errors for optional dependency
    requests = None  # type: ignore[assignment]

from bitbat.ingest.http_helpers import (
    SessionProtocol,
    ensure_utc,
    fetch_json_with_backoff,
    merge_and_save_news_parquet,
)

LOGGER = logging.getLogger(__name__)

GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
KEYWORDS = "(bitcoin OR btc OR crypto OR cryptocurrency)"
RESULT_COLUMNS = ["published_utc", "title", "url", "source", "lang", "sentiment_score"]
WINDOW = timedelta(hours=1)


class GdeltError(RuntimeError):
    """Raised when the GDELT API returns an unexpected response."""


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


def _fetch_chunk(
    session: SessionProtocol,
    start: datetime,
    end: datetime,
    *,
    retries: int,
    throttle_seconds: float,
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

    data = fetch_json_with_backoff(
        session=session,
        url=GDELT_ENDPOINT,
        params=payload,
        retries=retries,
        throttle_seconds=throttle_seconds,
        backoff_base=max(5.0, throttle_seconds),
        api_name="GDELT",
        context_msg=f"for {start}-{end}",
        error_class=GdeltError,
    )

    if not isinstance(data, dict):
        raise GdeltError("Unexpected GDELT response structure")

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
        records.append({
            "published_utc": published,
            "title": title,
            "url": url,
            "source": source,
            "lang": lang,
        })

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
    from bitbat.features.sentiment import score_vader

    frame["sentiment_score"] = score_vader(frame["title"])
    return frame


def fetch(  # noqa: C901
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

    start = ensure_utc(from_dt)
    end = ensure_utc(to_dt)
    if end - start > timedelta(days=30):
        raise ValueError("Use incremental for realtime; max 30d historical")

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
                throttle_seconds=throttle_seconds,
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

    target_path = _target_path(output_root)
    return merge_and_save_news_parquet(all_frames, target_path, RESULT_COLUMNS)
