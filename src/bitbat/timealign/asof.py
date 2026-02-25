"""As-of alignment helpers for leakage-safe feature joins."""

from __future__ import annotations

import pandas as pd


def _normalize_datetime_index(index: pd.Index, *, label: str) -> pd.DatetimeIndex:
    """Normalize an index to UTC-naive datetimes and validate monotonicity."""
    normalized = pd.DatetimeIndex(pd.to_datetime(index, utc=True, errors="raise")).tz_localize(None)
    if not normalized.is_monotonic_increasing:
        raise ValueError(f"{label} timestamps must be sorted ascending for as-of alignment.")
    if normalized.has_duplicates:
        raise ValueError(f"{label} timestamps must be unique for as-of alignment.")
    return normalized


def ensure_no_future_matches(
    target_timestamps: pd.Series,
    matched_source_timestamps: pd.Series,
    *,
    source_name: str = "source",
) -> None:
    """Raise when an as-of match points to a source timestamp in the future."""
    comparison = pd.concat(
        [
            pd.to_datetime(target_timestamps, utc=True, errors="raise"),
            pd.to_datetime(matched_source_timestamps, utc=True, errors="coerce"),
        ],
        axis=1,
    )
    comparison.columns = ["target", "source"]

    invalid = comparison["source"].notna() & (comparison["source"] > comparison["target"])
    if invalid.any():
        raise ValueError(
            f"Future {source_name} values detected during as-of alignment "
            f"({int(invalid.sum())} row(s))."
        )


def align_features_asof(
    target_index: pd.Index,
    source_features: pd.DataFrame,
    *,
    source_name: str = "source",
) -> pd.DataFrame:
    """Align source feature rows to target timestamps using backward as-of semantics."""
    target = _normalize_datetime_index(target_index, label="target")
    source = source_features.copy()
    source.index = _normalize_datetime_index(source.index, label=source_name)

    left = pd.DataFrame({"target_timestamp_utc": target})
    right = source.reset_index(names="source_timestamp_utc")

    merged = pd.merge_asof(
        left.sort_values("target_timestamp_utc"),
        right.sort_values("source_timestamp_utc"),
        left_on="target_timestamp_utc",
        right_on="source_timestamp_utc",
        direction="backward",
        allow_exact_matches=True,
    )

    ensure_no_future_matches(
        merged["target_timestamp_utc"],
        merged["source_timestamp_utc"],
        source_name=source_name,
    )

    aligned = merged.drop(columns=["source_timestamp_utc"]).set_index("target_timestamp_utc")
    aligned.index.name = target_index.name
    return aligned[source.columns]
