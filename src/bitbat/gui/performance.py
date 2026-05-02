"""Helpers for the supported Performance page."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

_BOOL_TRUE = {"1", "true", "t", "yes", "y"}
_BOOL_FALSE = {"0", "false", "f", "no", "n"}


def resolve_performance_scope(
    session_state: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
) -> tuple[str, str]:
    """Return the active freq/horizon pair for Performance views."""
    freq = str(session_state.get("active_freq") or runtime_config.get("freq", "1h"))
    horizon = str(session_state.get("active_horizon") or runtime_config.get("horizon", "4h"))
    return freq, horizon


def _coerce_nullable_bool(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):  # noqa: UP038
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _BOOL_TRUE:
            return True
        if normalized in _BOOL_FALSE:
            return False
    return None


def _format_percent(value: Any) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _format_currency(value: Any) -> str:
    if pd.isna(value):
        return "—"
    return f"${float(value):,.2f}"


def normalize_performance_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce prediction-outcome rows into a consistent typed frame."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "created_at",
                "predicted_direction",
                "actual_direction",
                "actual_return",
                "correct",
                "p_up",
                "p_down",
                "p_flat",
                "predicted_price",
                "start_price",
                "end_price",
                "confidence",
            ]
        )

    normalized = df.copy()
    normalized["timestamp_utc"] = pd.to_datetime(
        normalized["timestamp_utc"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None)
    normalized = normalized.dropna(subset=["timestamp_utc"])
    if normalized.empty:
        return normalize_performance_rows(pd.DataFrame())

    if "created_at" in normalized.columns:
        normalized["created_at"] = pd.to_datetime(
            normalized["created_at"],
            errors="coerce",
            utc=True,
        ).dt.tz_convert(None)
    else:
        normalized["created_at"] = normalized["timestamp_utc"]

    for column in (
        "p_up",
        "p_down",
        "p_flat",
        "actual_return",
        "predicted_price",
        "start_price",
        "end_price",
    ):
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        else:
            normalized[column] = pd.NA

    if "predicted_direction" not in normalized.columns:
        normalized["predicted_direction"] = "flat"
    if "actual_direction" not in normalized.columns:
        normalized["actual_direction"] = pd.NA

    normalized["predicted_direction"] = (
        normalized["predicted_direction"].astype("string").str.lower().fillna("flat")
    )
    normalized["actual_direction"] = normalized["actual_direction"].astype("string").str.lower()

    if "correct" in normalized.columns:
        normalized["correct"] = normalized["correct"].map(_coerce_nullable_bool).astype("boolean")
    else:
        normalized["correct"] = pd.Series(pd.NA, index=normalized.index, dtype="boolean")

    direction_known = normalized["actual_direction"].notna()
    derived = normalized["predicted_direction"].eq(normalized["actual_direction"])
    normalized["correct"] = normalized["correct"].where(
        ~(normalized["correct"].isna() & direction_known),
        derived,
    )

    confidence_inputs = normalized[["p_up", "p_down", "p_flat"]].astype("Float64")
    normalized["confidence"] = confidence_inputs.max(axis=1, skipna=True)
    normalized["confidence"] = normalized["confidence"].where(
        confidence_inputs.notna().any(axis=1),
        pd.NA,
    )

    normalized = normalized.sort_values(["timestamp_utc", "created_at"])
    normalized = normalized.drop_duplicates(subset=["timestamp_utc"], keep="last")
    return normalized.reset_index(drop=True)


def attach_price_evidence(
    predictions: pd.DataFrame,
    prices: pd.DataFrame | None,
    *,
    freq: str | None = None,
) -> pd.DataFrame:
    """Attach user-facing price evidence to normalized performance rows."""
    normalized = normalize_performance_rows(predictions)
    if normalized.empty:
        normalized["actual_price"] = pd.Series(dtype="Float64")
        normalized["price_gap_pct"] = pd.Series(dtype="Float64")
        return normalized

    enriched = normalized.copy()
    if "start_price" in enriched.columns:
        entry_price = enriched["start_price"].astype("Float64")
    else:
        entry_price = pd.Series(pd.NA, index=enriched.index, dtype="Float64")

    if prices is not None and not prices.empty:
        price_frame = prices.copy()
        if "timestamp_utc" in price_frame.columns and "close" in price_frame.columns:
            price_frame["timestamp_utc"] = pd.to_datetime(
                price_frame["timestamp_utc"],
                errors="coerce",
                utc=True,
            ).dt.tz_convert(None)
            price_frame["close"] = pd.to_numeric(price_frame["close"], errors="coerce")
            price_frame = price_frame.dropna(subset=["timestamp_utc", "close"]).sort_values(
                "timestamp_utc"
            )
            if not price_frame.empty:
                tolerance = None
                if freq:
                    try:
                        tolerance = pd.to_timedelta(freq) / 2
                    except Exception:
                        tolerance = None
                merged = pd.merge_asof(
                    enriched[["timestamp_utc"]].sort_values("timestamp_utc"),
                    price_frame[["timestamp_utc", "close"]],
                    on="timestamp_utc",
                    direction="nearest",
                    tolerance=tolerance,
                )
                merged = merged.rename(columns={"close": "entry_price"})
                merged = (
                    merged.set_index("timestamp_utc")
                    .reindex(enriched["timestamp_utc"])
                    .reset_index()
                )
                merged["entry_price"] = pd.to_numeric(
                    merged["entry_price"],
                    errors="coerce",
                ).astype("Float64")
                entry_price = entry_price.fillna(merged["entry_price"])

    actual_end = enriched["end_price"].astype("Float64")
    derived_actual = entry_price * (1.0 + enriched["actual_return"].astype("Float64"))
    actual_price = actual_end.fillna(derived_actual)

    enriched["entry_price"] = entry_price
    enriched["actual_price"] = actual_price.astype("Float64")
    gap_denominator = enriched["actual_price"].replace({0.0: pd.NA}).astype("Float64")
    enriched["price_gap_pct"] = (
        (enriched["predicted_price"].astype("Float64") - gap_denominator) / gap_denominator
    ).astype("Float64")
    return enriched


def summarize_performance_rows(predictions: pd.DataFrame) -> dict[str, float | int]:
    """Summarize total, completed, correct, pending, and accuracy from rows."""
    normalized = normalize_performance_rows(predictions)
    total = int(len(normalized))
    completed_mask = (
        normalized["correct"].notna()
        | normalized["actual_direction"].notna()
        | normalized["actual_return"].notna()
    )
    completed = int(completed_mask.sum())
    correct = int(normalized["correct"].eq(True).sum())
    pending = total - completed
    accuracy = (correct / completed * 100.0) if completed else 0.0

    return {
        "total": total,
        "completed": completed,
        "correct": correct,
        "pending": pending,
        "accuracy": accuracy,
    }


def build_accuracy_history(predictions: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Build a rolling market-time accuracy series from realized rows."""
    normalized = normalize_performance_rows(predictions)
    realized = normalized[normalized["correct"].notna()].copy()
    if realized.empty:
        return pd.DataFrame(columns=["Accuracy %"])

    realized["correct_flag"] = realized["correct"].astype("int64")
    realized["Accuracy %"] = (
        realized["correct_flag"].rolling(window=window, min_periods=1).mean() * 100.0
    )
    return realized.set_index("timestamp_utc")[["Accuracy %"]]


def summarize_current_streak(
    predictions: pd.DataFrame,
    *,
    limit: int = 20,
) -> dict[str, Any]:
    """Return the current realized streak from the latest market-time rows."""
    normalized = normalize_performance_rows(predictions)
    recent = normalized[normalized["correct"].notna()].sort_values(
        "timestamp_utc",
        ascending=False,
    )
    recent = recent.head(limit)
    if recent.empty:
        return {"count": 0, "type": "none", "recent_flags": []}

    flags = [bool(value) for value in recent["correct"].tolist()]
    first = flags[0]
    count = 0
    for flag in flags:
        if flag is first:
            count += 1
        else:
            break

    return {
        "count": count,
        "type": "win" if first else "loss",
        "recent_flags": flags,
    }


def summarize_recent_mix(
    predictions: pd.DataFrame,
    *,
    limit: int = 20,
) -> dict[str, float | int]:
    """Summarize recent class mix to prevent flat-heavy streak overclaiming."""
    normalized = normalize_performance_rows(predictions)
    recent = normalized[normalized["correct"].notna()].sort_values(
        "timestamp_utc",
        ascending=False,
    )
    recent = recent.head(limit)
    if recent.empty:
        return {
            "window": 0,
            "predicted_flat_rate": 0.0,
            "actual_flat_rate": 0.0,
        }

    return {
        "window": int(len(recent)),
        "predicted_flat_rate": float(recent["predicted_direction"].eq("flat").mean() * 100.0),
        "actual_flat_rate": float(recent["actual_direction"].eq("flat").mean() * 100.0),
    }


def format_recent_predictions(
    predictions: pd.DataFrame,
    *,
    limit: int = 20,
    prices: pd.DataFrame | None = None,
    freq: str | None = None,
) -> pd.DataFrame:
    """Return a user-facing recent-predictions table."""
    normalized = attach_price_evidence(predictions, prices, freq=freq)
    recent = normalized[normalized["correct"].notna()].sort_values(
        "timestamp_utc",
        ascending=False,
    )
    recent = recent.head(limit)
    if recent.empty:
        return pd.DataFrame(
            columns=["Time", "Prediction", "Confidence %", "Actual Outcome", "Correct?"]
        )

    display = pd.DataFrame({
        "Time": recent["timestamp_utc"],
        "Prediction": recent["predicted_direction"].map({
            "up": "📈 UP",
            "down": "📉 DOWN",
            "flat": "➡️ FLAT",
        }),
        "Confidence %": recent["confidence"].map(_format_percent),
        "Predicted Price": recent["predicted_price"].map(_format_currency),
        "Actual Price": recent["actual_price"].map(_format_currency),
        "Price Gap %": recent["price_gap_pct"].map(_format_percent),
        "Actual Outcome": recent["actual_direction"].map({
            "up": "📈 UP",
            "down": "📉 DOWN",
            "flat": "➡️ FLAT",
        }),
        "Actual Return": recent["actual_return"].map(_format_percent),
        "Correct?": recent["correct"].map({True: "✅ Yes", False: "❌ No"}),
    })
    return display.fillna("—").reset_index(drop=True)
