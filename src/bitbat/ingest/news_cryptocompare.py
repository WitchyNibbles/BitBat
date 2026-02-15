"""CryptoCompare news ingestion pipelines for historical backfill."""

from __future__ import annotations

import logging
import random
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import pandas as pd

try:
    import requests  # type: ignore[import-not-found, import-untyped]
except ImportError:  # pragma: no cover - defer import errors for optional dependency
    requests = None  # type: ignore[assignment]

from bitbat.contracts import ensure_news_contract
from bitbat.io.fs import read_parquet, write_parquet

LOGGER = logging.getLogger(__name__)

CRYPTOCOMPARE_ENDPOINT = "https://min-api.cryptocompare.com/data/v2/news/"
RESULT_COLUMNS = ["published_utc", "title", "url", "source", "lang", "sentiment_score"]
_DEFAULT_CATEGORIES = "BTC"


class CryptoCompareError(RuntimeError):
    """Raised when the CryptoCompare API returns an unexpected response."""


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _target_path(root: Path | str | None = None) -> Path:
    base = (
        Path(root)
        if root is not None
        else Path("data") / "raw" / "news" / "cryptocompare_1h"
    )
    return base / "cryptocompare_btc_1h.parquet"


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


def _response_content_type(response: Any) -> str:
    headers = getattr(response, "headers", None)
    if headers is None:
        return "unknown"
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return "unknown"
    value = getter("Content-Type") or getter("content-type")
    return str(value) if value else "unknown"


def _response_preview(response: Any, limit: int = 200) -> str | None:
    try:
        text = str(getattr(response, "text", "") or "")
    except Exception:  # pragma: no cover - best effort only
        return None
    compact = " ".join(text.split())
    if not compact:
        return None
    return compact[:limit]


def _fetch_page(
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
        response = session.get(
            CRYPTOCOMPARE_ENDPOINT,
            params=payload,
            timeout=30,
            headers={"Accept": "application/json"},
        )

        if response.status_code == 429:
            if attempt >= retries:
                raise CryptoCompareError(
                    "CryptoCompare request failed with status 429 after retries"
                )
            jitter = random.uniform(0, max(delay, backoff_base))  # noqa: S311
            sleep_for = max(delay + jitter, backoff_base)
            LOGGER.warning(
                "Rate limited by CryptoCompare (429) at lTs=%s; sleeping %.2fs before retry",
                lts,
                sleep_for,
            )
            time.sleep(sleep_for)
            attempt += 1
            delay = max(delay * 2, backoff_base)
            continue

        if response.status_code >= 500:
            if attempt >= retries:
                raise CryptoCompareError(
                    f"CryptoCompare request failed with status {response.status_code} after retries"
                )
            jitter = random.uniform(0, max(delay, backoff_base))  # noqa: S311
            sleep_for = max(delay + jitter, backoff_base)
            LOGGER.warning(
                "Transient CryptoCompare server error (%s) at lTs=%s; sleeping %.2fs before retry",
                response.status_code,
                lts,
                sleep_for,
            )
            time.sleep(sleep_for)
            attempt += 1
            delay = max(delay * 2, backoff_base)
            continue

        if response.status_code >= 400:
            raise CryptoCompareError(
                f"CryptoCompare request failed with status {response.status_code}"
            )

        try:
            payload_json = response.json()
        except ValueError as exc:  # pragma: no cover - defensive
            content_type = _response_content_type(response)
            snippet = _response_preview(response)
            if attempt >= retries:
                message = (
                    "Failed to decode CryptoCompare JSON response after retries "
                    f"(status={response.status_code}, content-type={content_type})"
                )
                if snippet:
                    message += f"; payload preview: {snippet!r}"
                raise CryptoCompareError(message) from exc

            jitter = random.uniform(0, max(delay, backoff_base))  # noqa: S311
            sleep_for = max(delay + jitter, backoff_base)
            LOGGER.warning(
                "Non-JSON CryptoCompare response at lTs=%s "
                "(status=%s, content-type=%s); sleeping %.2fs before retry",
                lts,
                response.status_code,
                content_type,
                sleep_for,
            )
            time.sleep(sleep_for)
            attempt += 1
            delay = max(delay * 2, backoff_base)
            continue

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
            raise CryptoCompareError("Unexpected CryptoCompare response payload: Data is not a list")
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


def _load_existing(target: Path) -> pd.DataFrame | None:
    if target.exists():
        try:
            existing = read_parquet(target)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to read existing CryptoCompare dataset: %s", exc)
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
    categories: str = _DEFAULT_CATEGORIES,
    language: str = "EN",
    max_pages: int | None = None,
) -> pd.DataFrame:
    """Fetch historical CryptoCompare BTC news and persist to partitioned parquet."""
    if from_dt >= to_dt:
        raise ValueError("`from_dt` must be earlier than `to_dt`.")

    start = _ensure_utc(from_dt)
    end = _ensure_utc(to_dt)
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
                (frame["published_utc"] >= start_naive)
                & (frame["published_utc"] <= end_naive)
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

    if not all_frames:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    merged = (
        pd.concat(all_frames, axis=0, ignore_index=True)
        .sort_values("published_utc")
        .drop_duplicates(subset=["url"])
    )

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
