"""Shared HTTP and I/O utilities for ingestors."""

import logging
import random
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from bitbat.contracts import ensure_news_contract
from bitbat.io.fs import read_parquet, write_parquet

LOGGER = logging.getLogger(__name__)


def ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware in UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class SessionProtocol(Protocol):
    """Protocol for HTTP sessions."""

    def get(self, url: str, **kwargs: Any) -> Any:  # pragma: no cover
        """Perform a GET request."""

    def close(self) -> None:  # pragma: no cover
        """Close the session."""


def response_content_type(response: Any) -> str:
    """Extract content type safely."""
    headers = getattr(response, "headers", None)
    if headers is None:
        return "unknown"
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return "unknown"
    value = getter("Content-Type") or getter("content-type")
    return str(value) if value else "unknown"


def response_preview(response: Any, limit: int = 200) -> str | None:
    """Preview response text safely."""
    try:
        text = str(getattr(response, "text", "") or "")
    except Exception:  # pragma: no cover
        return None
    compact = " ".join(text.split())
    if not compact:
        return None
    return compact[:limit]


def fetch_json_with_backoff(
    session: SessionProtocol,
    url: str,
    params: dict[str, Any],
    retries: int,
    throttle_seconds: float,
    backoff_base: float,
    api_name: str,
    context_msg: str,
    error_class: type[Exception],
) -> dict[str, Any] | list[Any]:
    """Generic fetch loop handling 429, 500, and JSON decoding."""
    attempt = 0
    delay = max(backoff_base, 0.0)

    while True:
        response = session.get(
            url,
            params=params,
            timeout=30,
            headers={"Accept": "application/json"},
        )

        if response.status_code == 429:
            if attempt >= retries:
                raise error_class(f"{api_name} request failed with status 429 after retries")
            jitter = random.uniform(0, max(delay, backoff_base))  # noqa: S311
            sleep_for = max(delay + jitter, backoff_base)
            LOGGER.warning(
                "Rate limited by %s (429) %s; sleeping %.2fs before retry",
                api_name,
                context_msg,
                sleep_for,
            )
            time.sleep(sleep_for)
            attempt += 1
            delay = max(delay * 2, backoff_base)
            continue

        if response.status_code >= 500:
            if attempt >= retries:
                raise error_class(
                    f"{api_name} request failed with status {response.status_code} after retries"
                )
            jitter = random.uniform(0, max(delay, backoff_base))  # noqa: S311
            sleep_for = max(delay + jitter, backoff_base)
            LOGGER.warning(
                "Transient %s server error (%s) %s; sleeping %.2fs before retry",
                api_name,
                response.status_code,
                context_msg,
                sleep_for,
            )
            time.sleep(sleep_for)
            attempt += 1
            delay = max(delay * 2, backoff_base)
            continue

        if response.status_code >= 400:
            raise error_class(f"{api_name} request failed with status {response.status_code}")

        try:
            payload = response.json()
            if not isinstance(payload, (dict, list)):
                raise ValueError("Payload is not a dict or list")
            return payload
        except ValueError as exc:
            content_type = response_content_type(response)
            snippet = response_preview(response)
            if attempt >= retries:
                message = (
                    f"Failed to decode {api_name} JSON response after retries "
                    f"(status={response.status_code}, content-type={content_type})"
                )
                if snippet:
                    message += f"; payload preview: {snippet!r}"
                raise error_class(message) from exc

            jitter = random.uniform(0, max(delay, backoff_base))  # noqa: S311
            sleep_for = max(delay + jitter, backoff_base)
            LOGGER.warning(
                "Non-JSON %s response %s (status=%s, content-type=%s); sleeping %.2fs before retry",
                api_name,
                context_msg,
                response.status_code,
                content_type,
                sleep_for,
            )
            time.sleep(sleep_for)
            attempt += 1
            delay = max(delay * 2, backoff_base)
            continue


def load_existing_parquet(target: Path, result_columns: list[str]) -> pd.DataFrame | None:
    """Load existing dataset securely."""
    if target.exists():
        try:
            existing = read_parquet(target)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to read existing dataset: %s", exc)
            return None
        missing_cols = [col for col in result_columns if col not in existing.columns]
        for col in missing_cols:
            existing[col] = pd.NA
        return existing[result_columns]
    return None


def merge_and_save_news_parquet(
    all_frames: list[pd.DataFrame], target_path: Path, result_columns: list[str]
) -> pd.DataFrame:
    """Merge arrays of DataFrames with an existing parquet file, partitioning properly."""
    if not all_frames:
        return pd.DataFrame(columns=result_columns)

    merged = (
        pd.concat(all_frames, axis=0, ignore_index=True)
        .sort_values("published_utc")
        .drop_duplicates(subset=["url"])
    )

    existing = load_existing_parquet(target_path, result_columns)
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

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        if target_path.is_dir():
            shutil.rmtree(target_path)
        else:
            target_path.unlink()

    write_parquet(partitions, target_path, partition_cols=["year", "month"])
    return merged.reset_index(drop=True)
