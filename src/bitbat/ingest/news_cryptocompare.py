"""CryptoCompare news ingestion pipelines for historical backfill."""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime
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

CRYPTOCOMPARE_ENDPOINT = "https://min-api.cryptocompare.com/data/v2/news/"
RESULT_COLUMNS = ["published_utc", "title", "url", "source", "lang", "sentiment_score"]
_DEFAULT_CATEGORIES = "BTC"


class CryptoCompareError(RuntimeError):
    """Raised when the CryptoCompare API returns an unexpected response."""


def _target_path(root: Path | str | None = None) -> Path:
    base = Path(root) if root is not None else Path("data") / "raw" / "news" / "cryptocompare_1h"
    return base / "cryptocompare_btc_1h.parquet"


def _fetch_page(  # noqa: C901
    session: SessionProtocol,
    *,
    lts: int,
    categories: str,
    language: str,
    retries: int,
    throttle_seconds: float,
) -> list[dict[str, Any]]:
    backoff_base = max(1.0, throttle_seconds)
    payload = {
        "lang": language,
        "categories": categories,
        "lTs": int(lts),
    }

    attempt = 0
    delay = max(backoff_base, 0.0)

    while True:
        try:
            payload_json = fetch_json_with_backoff(
                session=session,
                url=CRYPTOCOMPARE_ENDPOINT,
                params=payload,
                retries=retries,
                throttle_seconds=throttle_seconds,
                backoff_base=backoff_base,
                api_name="CryptoCompare",
                context_msg=f"at lTs={lts}",
                error_class=CryptoCompareError,
            )
        except CryptoCompareError:
            raise

        if not isinstance(payload_json, dict):
            raise CryptoCompareError("Unexpected CryptoCompare response structure")

        response_type = str(payload_json.get("Response", "") or "").lower()
        if response_type == "error":
            message = str(payload_json.get("Message", "CryptoCompare returned an error"))
            if "rate" in message.lower() and "limit" in message.lower() and attempt < retries:
                jitter = random.uniform(0, max(delay, backoff_base))  # noqa: S311
                sleep_for = max(delay + jitter, backoff_base)
                LOGGER.warning(
                    "CryptoCompare payload rate-limit at lTs=%s; sleeping %.2fs before retry: %s",
                    lts,
                    sleep_for,
                    message,
                )
                time.sleep(sleep_for)
                attempt += 1
                delay = max(delay * 2, backoff_base)
                continue
            raise CryptoCompareError(message)

        records = payload_json.get("Data") or []
        if not isinstance(records, list):
            raise CryptoCompareError(
                "Unexpected CryptoCompare response payload: Data is not a list"
            )
        return records


def _articles_to_frame(articles: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for article in articles:
        published = article.get("published_on")
        title = article.get("title")
        url = article.get("url")
        source_info = article.get("source_info") or {}
        source = None
        if isinstance(source_info, dict):
            source = source_info.get("name")
        if source is None:
            source = article.get("source")
        lang = article.get("lang") or article.get("language") or "en"
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

    frame["published_utc"] = pd.to_datetime(
        frame["published_utc"], unit="s", utc=True, errors="coerce"
    )
    frame["published_utc"] = frame["published_utc"].dt.tz_localize(None)
    frame = frame.dropna(subset=["published_utc", "url"])
    frame = frame[frame["url"].str.startswith("http", na=False)]
    frame["title"] = frame["title"].fillna("")
    frame["source"] = frame["source"].fillna("cryptocompare")
    frame["lang"] = frame["lang"].fillna("en")

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
    categories: str = _DEFAULT_CATEGORIES,
    language: str = "EN",
    max_pages: int | None = None,
) -> pd.DataFrame:
    """Fetch historical CryptoCompare BTC news and persist to partitioned parquet."""
    if from_dt >= to_dt:
        raise ValueError("`from_dt` must be earlier than `to_dt`.")

    start = ensure_utc(from_dt)
    end = ensure_utc(to_dt)
    start_naive = start.replace(tzinfo=None)
    end_naive = end.replace(tzinfo=None)

    created_session = False
    if session is not None:
        active_session = session
    else:
        if requests is None:  # pragma: no cover - dependency guard
            raise RuntimeError("The `requests` package is required to fetch CryptoCompare data.")
        active_session = cast(SessionProtocol, requests.Session())
        created_session = True

    all_frames: list[pd.DataFrame] = []
    cursor = int(end.timestamp())
    min_timestamp = int(start.timestamp())
    page_count = 0

    while cursor >= min_timestamp:
        if max_pages is not None and page_count >= max_pages:
            break

        try:
            articles = _fetch_page(
                active_session,
                lts=cursor,
                categories=categories,
                language=language,
                retries=retry_limit,
                throttle_seconds=throttle_seconds,
            )
        except CryptoCompareError as exc:
            LOGGER.warning("CryptoCompare page fetch failed at lTs=%s: %s", cursor, exc)
            break

        if not articles:
            break

        frame = _articles_to_frame(articles)
        if not frame.empty:
            frame = frame[
                (frame["published_utc"] >= start_naive) & (frame["published_utc"] <= end_naive)
            ]
            if not frame.empty:
                all_frames.append(frame)

        timestamps = [
            int(item.get("published_on"))
            for item in articles
            if isinstance(item, dict) and item.get("published_on") is not None
        ]
        if not timestamps:
            break

        oldest = min(timestamps)
        next_cursor = oldest - 1
        if next_cursor >= cursor:
            break
        cursor = next_cursor
        page_count += 1

        if throttle_seconds > 0:
            time.sleep(throttle_seconds)

    if created_session:
        active_session.close()

    target_path = _target_path(output_root)
    return merge_and_save_news_parquet(all_frames, target_path, RESULT_COLUMNS)
