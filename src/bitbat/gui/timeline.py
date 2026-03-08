"""Prediction timeline chart for the BitBat dashboard."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

_DIRECTION_STYLES = {
    "up": {"color": "#10B981", "symbol": "triangle-up"},
    "down": {"color": "#EF4444", "symbol": "triangle-down"},
    "flat": {"color": "#6B7280", "symbol": "circle"},
}

_STATUS_STYLES = {
    "pending": {"opacity": 0.75, "size": 10, "label": "Pending"},
    "realized_correct": {"opacity": 1.0, "size": 14, "label": "Realized (Correct)"},
    "realized_wrong": {"opacity": 0.4, "size": 12, "label": "Realized (Wrong)"},
}

_BOOL_TRUE = {"1", "true", "t", "yes", "y"}
_BOOL_FALSE = {"0", "false", "f", "no", "n"}
_DATE_WINDOW_DELTAS = {
    "24h": pd.Timedelta(hours=24),
    "7d": pd.Timedelta(days=7),
    "30d": pd.Timedelta(days=30),
    "all": None,
}

_TIMELINE_COLUMNS = [
    "timestamp_utc",
    "predicted_direction",
    "p_up",
    "p_down",
    "predicted_return",
    "predicted_price",
    "actual_return",
    "actual_direction",
    "correct",
    "confidence",
    "is_realized",
    "prediction_status",
]


def _empty_timeline_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_TIMELINE_COLUMNS)


def _coerce_direction(series: pd.Series, *, default: str | None) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    valid = values.isin({"up", "down", "flat"})
    if default is None:
        return values.where(valid, pd.NA)
    return values.where(valid, default).fillna(default)


def _coerce_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")
    return pd.to_numeric(df[column], errors="coerce").astype("Float64")


def _duration_sort_key(value: str) -> float:
    try:
        return pd.to_timedelta(value).total_seconds()
    except Exception:
        return float("inf")


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


def _direction_from_return(value: Any) -> str | None:
    if pd.isna(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed > 0:
        return "up"
    if parsed < 0:
        return "down"
    return "flat"


def _normalize_timeline_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_timeline_frame()

    normalized = df.copy()

    if "timestamp_utc" not in normalized.columns:
        return _empty_timeline_frame()

    normalized["timestamp_utc"] = pd.to_datetime(
        normalized["timestamp_utc"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None)
    normalized = normalized.dropna(subset=["timestamp_utc"])
    if normalized.empty:
        return _empty_timeline_frame()

    normalized["predicted_direction"] = _coerce_direction(
        normalized.get("predicted_direction", pd.Series("flat", index=normalized.index)),
        default="flat",
    )
    normalized["actual_direction"] = _coerce_direction(
        normalized.get("actual_direction", pd.Series(pd.NA, index=normalized.index)),
        default=None,
    )

    normalized["p_up"] = _coerce_numeric(normalized, "p_up")
    normalized["p_down"] = _coerce_numeric(normalized, "p_down")
    normalized["predicted_return"] = _coerce_numeric(normalized, "predicted_return")
    normalized["predicted_price"] = _coerce_numeric(normalized, "predicted_price")
    normalized["actual_return"] = _coerce_numeric(normalized, "actual_return")

    raw_correct = normalized.get("correct", pd.Series(pd.NA, index=normalized.index))
    correct = raw_correct.map(_coerce_nullable_bool).astype("boolean")

    direction_known = normalized["actual_direction"].notna()
    derived_from_direction = normalized["predicted_direction"].eq(normalized["actual_direction"])
    correct = correct.where(~(correct.isna() & direction_known), derived_from_direction)

    realized_direction = normalized["actual_return"].map(_direction_from_return).astype("string")
    return_known = realized_direction.notna()
    derived_from_return = normalized["predicted_direction"].eq(realized_direction)
    correct = correct.where(~(correct.isna() & return_known), derived_from_return)
    normalized["correct"] = correct.astype("boolean")

    confidence_inputs = pd.concat([normalized["p_up"], normalized["p_down"]], axis=1)
    normalized["confidence"] = confidence_inputs.max(axis=1, skipna=True).astype("Float64")
    has_confidence = confidence_inputs.notna().any(axis=1)
    normalized["confidence"] = normalized["confidence"].where(has_confidence, pd.NA)

    normalized["is_realized"] = (
        normalized["correct"].notna()
        | normalized["actual_direction"].notna()
        | normalized["actual_return"].notna()
    )
    normalized["prediction_status"] = "pending"
    normalized.loc[
        normalized["is_realized"] & normalized["correct"].eq(True),
        "prediction_status",
    ] = "realized_correct"
    normalized.loc[
        normalized["is_realized"] & normalized["correct"].eq(False),
        "prediction_status",
    ] = "realized_wrong"

    normalized = normalized.sort_values("timestamp_utc").reset_index(drop=True)
    return normalized[_TIMELINE_COLUMNS]


def _prediction_columns(con: sqlite3.Connection) -> set[str]:
    table_exists = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='prediction_outcomes' LIMIT 1"
    ).fetchone()
    if table_exists is None:
        return set()
    rows = con.execute("PRAGMA table_info(prediction_outcomes)").fetchall()
    return {str(row[1]) for row in rows}


def _build_timeline_query(columns: set[str]) -> str | None:
    if not columns:
        return None

    if "timestamp_utc" in columns:
        timestamp_expr = "timestamp_utc"
    elif "prediction_timestamp" in columns:
        timestamp_expr = "prediction_timestamp AS timestamp_utc"
    else:
        return None

    if "freq" not in columns or "horizon" not in columns:
        return None

    select_exprs = [
        timestamp_expr,
        "predicted_direction"
        if "predicted_direction" in columns
        else "'flat' AS predicted_direction",  # noqa: E501
        "p_up" if "p_up" in columns else "NULL AS p_up",
        "p_down" if "p_down" in columns else "NULL AS p_down",
        "predicted_return" if "predicted_return" in columns else "NULL AS predicted_return",
        "predicted_price" if "predicted_price" in columns else "NULL AS predicted_price",
        "actual_return" if "actual_return" in columns else "NULL AS actual_return",
        "actual_direction" if "actual_direction" in columns else "NULL AS actual_direction",
        "correct" if "correct" in columns else "NULL AS correct",
    ]

    order_clause = "ORDER BY timestamp_utc DESC"
    if "created_at" in columns:
        order_clause = "ORDER BY timestamp_utc DESC, created_at DESC"
    elif "id" in columns:
        order_clause = "ORDER BY timestamp_utc DESC, id DESC"

    return (
        "SELECT "  # noqa: S608
        + ", ".join(select_exprs)
        + " FROM prediction_outcomes "
        "WHERE freq = ? AND horizon = ? " + order_clause + " "
        "LIMIT ?"
    )


def _format_percent(value: Any, *, signed: bool = False, decimals: int = 1) -> str:
    if pd.isna(value):
        return "n/a"
    parsed = float(value)
    if signed:
        return f"{parsed:+.{decimals}%}"
    return f"{parsed:.{decimals}%}"


def _resolve_marker_price(ts: pd.Timestamp, prices: pd.DataFrame, row: pd.Series) -> float | None:
    if not prices.empty and "close" in prices.columns:
        tolerance = pd.Timedelta(0)
        if len(prices.index) >= 2:
            deltas = prices.index.to_series().sort_values().diff().dropna()
            if not deltas.empty:
                tolerance = deltas.median() / 2
        try:
            idx = prices.index.get_indexer([ts], method="nearest", tolerance=tolerance)
        except Exception:
            idx = [-1]
        if len(idx) > 0 and idx[0] >= 0:
            close = prices["close"].iloc[idx[0]]
            if pd.notna(close):
                return float(close)

    predicted_price = row.get("predicted_price")
    if pd.notna(predicted_price):
        return float(predicted_price)
    return None


def _marker_trace_name(direction: str, status_label: str) -> str:
    direction_label = direction.upper()
    return f"{direction_label} - {status_label}"


def _build_marker_frame(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in predictions.iterrows():
        ts = pd.Timestamp(row["timestamp_utc"])
        price_at_ts = _resolve_marker_price(ts, prices, row)
        if price_at_ts is None:
            continue

        direction = str(row.get("predicted_direction", "flat"))
        if direction not in _DIRECTION_STYLES:
            direction = "flat"

        status = str(row.get("prediction_status", "pending"))
        if status not in _STATUS_STYLES:
            status = "pending"

        status_label = _STATUS_STYLES[status]["label"]
        rows.append({
            "timestamp_utc": ts,
            "marker_price": price_at_ts,
            "predicted_direction": direction,
            "prediction_status": status,
            "trace_name": _marker_trace_name(direction, status_label),
            "confidence_text": _format_percent(row.get("confidence"), decimals=2),
            "predicted_return_text": _format_percent(row.get("predicted_return"), signed=True),
            "actual_return_text": _format_percent(row.get("actual_return"), signed=True),
            "status_label": status_label,
        })

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "marker_price",
                "predicted_direction",
                "prediction_status",
                "trace_name",
                "confidence_text",
                "predicted_return_text",
                "actual_return_text",
                "status_label",
            ]
        )

    return pd.DataFrame(rows).sort_values("timestamp_utc").reset_index(drop=True)


def summarize_timeline_status(predictions: pd.DataFrame) -> dict[str, float]:
    """Summarize total/completed/correct/pending counts from normalized status fields."""
    normalized = _normalize_timeline_rows(predictions)

    total = int(len(normalized))
    completed = int(normalized["prediction_status"].ne("pending").sum())
    correct = int(normalized["prediction_status"].eq("realized_correct").sum())
    pending = int(normalized["prediction_status"].eq("pending").sum())
    accuracy = (correct / completed * 100) if completed else 0.0

    return {
        "total": total,
        "completed": completed,
        "correct": correct,
        "pending": pending,
        "accuracy": accuracy,
    }


def summarize_timeline_insights(predictions: pd.DataFrame) -> dict[str, float | None]:
    """Summarize compact timeline insight-strip metrics."""
    normalized = _normalize_timeline_rows(predictions)
    status = summarize_timeline_status(normalized)

    confidence = normalized["confidence"].dropna()
    average_confidence = float(confidence.mean() * 100) if not confidence.empty else None

    return {
        **status,
        "average_confidence": average_confidence,
        "up_count": int(normalized["predicted_direction"].eq("up").sum()),
        "down_count": int(normalized["predicted_direction"].eq("down").sum()),
        "flat_count": int(normalized["predicted_direction"].eq("flat").sum()),
    }


def _sanitize_limit(limit: int) -> int:
    parsed = int(limit)
    return max(parsed, 1)


def get_timeline_data(
    db_path: Path,
    freq: str,
    horizon: str,
    limit: int = 168,
) -> pd.DataFrame:
    """Query predictions from the autonomous DB for the timeline chart.

    Returns a normalized DataFrame sorted by timestamp with explicit status
    semantics (`prediction_status` and `is_realized`) so timeline consumers
    do not depend on ad-hoc null checks.
    """
    if not db_path.exists():
        return _empty_timeline_frame()

    safe_limit = _sanitize_limit(limit)

    try:
        with sqlite3.connect(str(db_path)) as con:
            columns = _prediction_columns(con)
            query = _build_timeline_query(columns)
            if query is None:
                return _empty_timeline_frame()
            raw_df = pd.read_sql_query(query, con, params=(freq, horizon, safe_limit))
    except Exception:
        return _empty_timeline_frame()
    return _normalize_timeline_rows(raw_df)


def list_timeline_filter_options(
    db_path: Path,
    default_freq: str,
    default_horizon: str,
) -> tuple[list[str], list[str]]:
    """List available freq/horizon filter options from prediction history."""
    freqs = {default_freq}
    horizons = {default_horizon}

    if db_path.exists():
        try:
            with sqlite3.connect(str(db_path)) as con:
                columns = _prediction_columns(con)
                if {"freq", "horizon"}.issubset(columns):
                    rows = con.execute(
                        """
                        SELECT DISTINCT freq, horizon
                        FROM prediction_outcomes
                        WHERE freq IS NOT NULL AND horizon IS NOT NULL
                        """
                    ).fetchall()
                    for freq, horizon in rows:
                        freqs.add(str(freq))
                        horizons.add(str(horizon))
        except Exception:  # noqa: S110
            pass

    sorted_freqs = sorted(freqs, key=_duration_sort_key)
    sorted_horizons = sorted(horizons, key=_duration_sort_key)
    return sorted_freqs, sorted_horizons


def apply_timeline_filters(
    predictions: pd.DataFrame,
    *,
    date_window: str = "7d",
) -> pd.DataFrame:
    """Apply date-window filter to normalized timeline rows."""
    normalized = _normalize_timeline_rows(predictions)
    if normalized.empty:
        return normalized

    if date_window not in _DATE_WINDOW_DELTAS:
        date_window = "7d"

    delta = _DATE_WINDOW_DELTAS[date_window]
    if delta is None:
        return normalized

    max_ts = normalized["timestamp_utc"].max()
    min_ts = max_ts - delta
    return normalized.loc[normalized["timestamp_utc"] >= min_ts].reset_index(drop=True)


def format_timeline_empty_state(freq: str, horizon: str, date_window: str) -> str:
    """Build explicit no-result timeline message for current filter set."""
    window_label = date_window if date_window in _DATE_WINDOW_DELTAS else "7d"
    return (
        "No timeline events match the current filters "
        f"({freq} / {horizon} / {window_label}). "
        "Try adjusting freq, horizon, or date window."
    )


def build_timeline_overlay_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build predicted-vs-realized overlay dataset from normalized rows."""
    normalized = _normalize_timeline_rows(predictions)
    if normalized.empty:
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "predicted_return",
                "actual_return",
                "prediction_status",
                "mismatch_abs",
                "upper_return",
                "lower_return",
            ]
        )

    overlay = normalized.loc[
        :,
        ["timestamp_utc", "predicted_return", "actual_return", "prediction_status"],
    ].copy()
    overlay["actual_return"] = overlay["actual_return"].where(
        overlay["prediction_status"].ne("pending"),
        pd.NA,
    )
    overlay["mismatch_abs"] = (overlay["predicted_return"] - overlay["actual_return"]).abs()

    aligned = overlay["actual_return"].notna()
    overlay["upper_return"] = pd.NA
    overlay["lower_return"] = pd.NA
    overlay.loc[aligned, "upper_return"] = overlay.loc[
        aligned, ["predicted_return", "actual_return"]
    ].max(  # noqa: E501
        axis=1
    )
    overlay.loc[aligned, "lower_return"] = overlay.loc[
        aligned, ["predicted_return", "actual_return"]
    ].min(  # noqa: E501
        axis=1
    )
    return overlay


def _add_overlay_traces(
    fig: Any,
    overlay: pd.DataFrame,
    *,
    yaxis: str,
) -> None:
    predicted_percent = overlay["predicted_return"].astype("Float64") * 100.0
    realized_percent = overlay["actual_return"].astype("Float64") * 100.0
    upper_percent = overlay["upper_return"].astype("Float64") * 100.0
    lower_percent = overlay["lower_return"].astype("Float64") * 100.0

    axis_kwargs = {"yaxis": yaxis} if yaxis != "y" else {}
    fig.add_trace({
        "type": "scatter",
        "x": overlay["timestamp_utc"],
        "y": predicted_percent,
        "mode": "lines",
        "name": "Predicted Return",
        "line": {"color": "#22C55E", "width": 2},
        **axis_kwargs,
    })
    fig.add_trace({
        "type": "scatter",
        "x": overlay["timestamp_utc"],
        "y": realized_percent,
        "mode": "lines",
        "name": "Realized Return",
        "line": {"color": "#F97316", "width": 2},
        "connectgaps": False,
        **axis_kwargs,
    })
    fig.add_trace({
        "type": "scatter",
        "x": overlay["timestamp_utc"],
        "y": lower_percent,
        "mode": "lines",
        "line": {"color": "rgba(0,0,0,0)", "width": 0},
        "showlegend": False,
        "hoverinfo": "skip",
        **axis_kwargs,
    })
    fig.add_trace({
        "type": "scatter",
        "x": overlay["timestamp_utc"],
        "y": upper_percent,
        "mode": "lines",
        "name": "Mismatch Band",
        "line": {"color": "rgba(251,191,36,0.40)", "width": 1},
        "fill": "tonexty",
        "fillcolor": "rgba(251,191,36,0.15)",
        **axis_kwargs,
    })


def get_price_series(
    data_dir: Path,
    freq: str,
    start_ts: pd.Timestamp,
) -> pd.DataFrame:
    """Load BTC close prices from the ingested parquet starting at *start_ts*.

    Returns a DataFrame indexed by ``timestamp_utc`` with a ``close`` column.
    """
    legacy_path = data_dir / "raw" / "prices" / f"btcusd_yf_{freq}.parquet"
    if not legacy_path.exists():
        return pd.DataFrame(columns=["close"])

    prices = pd.read_parquet(legacy_path)
    if "timestamp_utc" not in prices.columns or "close" not in prices.columns:
        return pd.DataFrame(columns=["close"])

    prices["timestamp_utc"] = pd.to_datetime(prices["timestamp_utc"])
    prices = prices.set_index("timestamp_utc").sort_index()
    return prices.loc[prices.index >= start_ts, ["close"]]


def build_timeline_figure(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    show_overlay: bool = False,
) -> object:
    """Build a Plotly figure showing BTC price with prediction markers.

    Markers are color-coded by direction (green=up, red=down, gray=flat)
    and styled by explicit prediction status semantics.

    Returns a ``plotly.graph_objects.Figure``.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    normalized = _normalize_timeline_rows(predictions)
    marker_prices = prices.sort_index() if not prices.empty else prices

    # Price line
    if not marker_prices.empty:
        fig.add_trace(
            go.Scatter(
                x=marker_prices.index,
                y=marker_prices["close"],
                mode="lines",
                name="BTC Price",
                line={"color": "#60A5FA", "width": 2},
            )
        )

    # Prediction markers grouped by direction/status to avoid one-trace-per-point clutter.
    marker_frame = _build_marker_frame(normalized, marker_prices)
    marker_order = (
        ("up", "realized_correct"),
        ("up", "realized_wrong"),
        ("up", "pending"),
        ("down", "realized_correct"),
        ("down", "realized_wrong"),
        ("down", "pending"),
        ("flat", "realized_correct"),
        ("flat", "realized_wrong"),
        ("flat", "pending"),
    )
    for direction, status in marker_order:
        grouped = marker_frame.loc[
            marker_frame["predicted_direction"].eq(direction)
            & marker_frame["prediction_status"].eq(status)
        ]
        if grouped.empty:
            continue

        direction_style = _DIRECTION_STYLES[direction]
        status_style = _STATUS_STYLES[status]

        fig.add_trace(
            go.Scatter(
                x=grouped["timestamp_utc"],
                y=grouped["marker_price"],
                mode="markers",
                name=grouped["trace_name"].iloc[0],
                marker={
                    "color": direction_style["color"],
                    "size": status_style["size"],
                    "symbol": direction_style["symbol"],
                    "opacity": status_style["opacity"],
                },
                customdata=grouped[
                    [
                        "confidence_text",
                        "predicted_return_text",
                        "actual_return_text",
                        "status_label",
                    ]
                ],
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d %H:%M}</b><br>"
                    f"Prediction: {direction}<br>"
                    "Confidence: %{customdata[0]}<br>"
                    "Predicted Return: %{customdata[1]}<br>"
                    "Actual Return: %{customdata[2]}<br>"
                    "Status: %{customdata[3]}"
                    "<extra></extra>"
                ),
            )
        )

    if show_overlay:
        overlay = build_timeline_overlay_frame(normalized)
        if not overlay.empty:
            _add_overlay_traces(fig, overlay, yaxis="y2")

    fig.update_layout(
        title="Prediction Timeline",
        xaxis_title="Time",
        yaxis_title="BTC Price (USD)",
        xaxis={"gridcolor": "rgba(148,163,184,0.16)"},
        yaxis={"gridcolor": "rgba(148,163,184,0.16)"},
        yaxis2={
            "title": "Return (%)",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
        height=520,
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1f2e",
        legend={
            "font": {"color": "white", "size": 11},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
        },
    )

    return fig


def build_timeline_comparison_figure(predictions: pd.DataFrame) -> object:
    """Build a focused comparison figure for predicted vs realized returns.

    This is an opt-in companion view used by the dashboard to keep the primary
    timeline readable while still exposing return comparison analytics.
    """
    import plotly.graph_objects as go

    normalized = _normalize_timeline_rows(predictions)
    overlay = build_timeline_overlay_frame(normalized)

    fig = go.Figure()
    if not overlay.empty:
        _add_overlay_traces(fig, overlay, yaxis="y")

    fig.update_layout(
        title="Predicted vs Realized Return Comparison",
        xaxis_title="Time",
        yaxis_title="Return (%)",
        xaxis={"gridcolor": "rgba(148,163,184,0.16)"},
        yaxis={"gridcolor": "rgba(148,163,184,0.16)"},
        height=330,
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1f2e",
        legend={
            "font": {"color": "white", "size": 11},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
        },
    )

    return fig
