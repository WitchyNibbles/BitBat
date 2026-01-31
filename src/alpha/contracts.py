"""Data contract validators for persisted datasets."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype


class ContractError(ValueError):
    """Raised when a dataset violates its declared contract."""


def _ensure_datetime(column: pd.Series, name: str) -> pd.Series:
    converted = pd.to_datetime(column, utc=True, errors="raise")
    return converted.dt.tz_localize(None)


def ensure_prices_contract(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the prices parquet contract."""
    expected = [
        "timestamp_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
    ]
    missing = set(expected) - set(frame.columns)
    if missing:
        raise ContractError(f"Prices frame missing columns: {sorted(missing)}")

    validated = frame.copy()
    validated["timestamp_utc"] = _ensure_datetime(validated["timestamp_utc"], "timestamp_utc")

    for col in ["open", "high", "low", "close", "volume"]:
        validated[col] = validated[col].astype("float64")

    validated["source"] = validated["source"].astype("string")
    return validated[expected]


def ensure_news_contract(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the news parquet contract."""
    expected = [
        "published_utc",
        "title",
        "url",
        "source",
        "lang",
        "sentiment_score",
    ]
    missing = set(expected) - set(frame.columns)
    if missing:
        raise ContractError(f"News frame missing columns: {sorted(missing)}")

    validated = frame.copy()
    validated["published_utc"] = _ensure_datetime(validated["published_utc"], "published_utc")
    validated["title"] = validated["title"].astype("string")
    validated["url"] = validated["url"].astype("string")
    validated["source"] = validated["source"].astype("string")
    validated["lang"] = validated["lang"].astype("string")
    validated["sentiment_score"] = validated["sentiment_score"].astype("float64")
    return validated[expected]


def _expected_sentiment_features(
    windows: tuple[str, ...] = ("1h", "4h", "24h"),
    suffixes: tuple[str, ...] = ("mean", "median", "pos", "neg", "neu", "count", "decay"),
) -> set[str]:
    expected: set[str] = set()
    for window in windows:
        prefix = f"feat_sent_{window}".replace("h", "h_")
        for suffix in suffixes:
            expected.add(f"{prefix}{suffix}")
    return expected


def ensure_feature_contract(
    frame: pd.DataFrame,
    *,
    require_label: bool,
    require_forward_return: bool,
    require_features_full: bool = False,
) -> pd.DataFrame:
    """Validate and normalize the feature parquet contract."""
    if "timestamp_utc" not in frame.columns:
        raise ContractError("Feature frame missing 'timestamp_utc'.")

    validated = frame.copy()
    validated["timestamp_utc"] = _ensure_datetime(validated["timestamp_utc"], "timestamp_utc")

    feature_cols = [col for col in validated.columns if col.startswith("feat_")]
    if not feature_cols:
        raise ContractError("Feature frame must contain at least one 'feat_' column.")

    for col in feature_cols:
        if not is_numeric_dtype(validated[col]):
            validated[col] = pd.to_numeric(validated[col], errors="raise")

    if require_features_full:
        expected_sentiment = _expected_sentiment_features()
        missing_sentiment = sorted(expected_sentiment - set(feature_cols))
        if missing_sentiment:
            raise ContractError(
                "Feature frame missing sentiment columns: "
                + ", ".join(missing_sentiment)
            )

    ordered = ["timestamp_utc", *feature_cols]

    if require_label:
        if "label" not in validated.columns:
            raise ContractError("Feature frame missing 'label'.")
        validated["label"] = validated["label"].astype("string")
        ordered.append("label")

    if require_forward_return:
        if "r_forward" not in validated.columns:
            raise ContractError("Feature frame missing 'r_forward'.")
        validated["r_forward"] = validated["r_forward"].astype("float64")
        if "r_forward" not in ordered:
            ordered.append("r_forward")

    return validated[ordered]


def ensure_predictions_contract(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the predictions parquet contract."""
    expected = [
        "timestamp_utc",
        "p_up",
        "p_down",
        "horizon",
        "freq",
        "model_version",
        "realized_r",
        "realized_label",
    ]
    missing = set(expected) - set(frame.columns)
    if missing:
        raise ContractError(f"Predictions frame missing columns: {sorted(missing)}")

    validated = frame.copy()
    validated["timestamp_utc"] = _ensure_datetime(validated["timestamp_utc"], "timestamp_utc")
    validated["p_up"] = pd.to_numeric(validated["p_up"], errors="raise").astype("float64")
    validated["p_down"] = pd.to_numeric(validated["p_down"], errors="raise").astype("float64")
    validated["horizon"] = validated["horizon"].astype("string")
    validated["freq"] = validated["freq"].astype("string")
    validated["model_version"] = validated["model_version"].astype("string")
    validated["realized_r"] = pd.to_numeric(validated["realized_r"], errors="coerce").astype(
        "float64"
    )
    validated["realized_label"] = validated["realized_label"].astype("string")
    return validated[expected]
