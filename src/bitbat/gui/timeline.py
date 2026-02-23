"""Prediction timeline chart for the BitBat dashboard."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def get_timeline_data(
    db_path: Path,
    freq: str,
    horizon: str,
    limit: int = 168,
) -> pd.DataFrame:
    """Query predictions from the autonomous DB for the timeline chart.

    Returns a DataFrame sorted by timestamp with columns:
    ``timestamp_utc``, ``predicted_direction``, ``p_up``, ``p_down``,
    ``actual_direction``, ``correct``.
    """
    if not db_path.exists():
        return pd.DataFrame()

    con = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(
            """
            SELECT timestamp_utc, predicted_direction, p_up, p_down,
                   actual_return, actual_direction, correct
            FROM prediction_outcomes
            WHERE freq = ? AND horizon = ?
            ORDER BY timestamp_utc DESC
            LIMIT ?
            """,
            con,
            params=(freq, horizon, limit),
        )
    except Exception:
        return pd.DataFrame()
    finally:
        con.close()

    if df.empty:
        return df

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    return df.sort_values("timestamp_utc").reset_index(drop=True)


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
) -> object:
    """Build a Plotly figure showing BTC price with prediction markers.

    Markers are color-coded by direction (green=up, red=down, gray=flat)
    and opacity indicates correctness (bright=correct, faded=wrong,
    medium=pending).

    Returns a ``plotly.graph_objects.Figure``.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # Price line
    if not prices.empty:
        fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=prices["close"],
                mode="lines",
                name="BTC Price",
                line={"color": "#6366f1", "width": 2},
            )
        )

    # Prediction markers
    direction_style = {
        "up": {"color": "#10B981", "symbol": "triangle-up"},
        "down": {"color": "#EF4444", "symbol": "triangle-down"},
        "flat": {"color": "#6B7280", "symbol": "circle"},
    }

    for _, row in predictions.iterrows():
        ts = pd.Timestamp(row["timestamp_utc"])
        style = direction_style.get(row["predicted_direction"], direction_style["flat"])

        # Find the closest price for marker y-position
        price_at_ts = None
        if not prices.empty:
            idx = prices.index.get_indexer([ts], method="nearest")
            if len(idx) > 0 and idx[0] >= 0:
                price_at_ts = float(prices["close"].iloc[idx[0]])

        if price_at_ts is None:
            continue

        # Correctness determines opacity and size
        correct = row.get("correct")
        if correct is True or correct == 1:
            opacity, size = 1.0, 14
            result_label = "Correct"
        elif correct is False or correct == 0:
            opacity, size = 0.4, 12
            result_label = "Wrong"
        else:
            opacity, size = 0.7, 10
            result_label = "Pending"

        confidence = max(float(row.get("p_up", 0)), float(row.get("p_down", 0)))

        fig.add_trace(
            go.Scatter(
                x=[ts],
                y=[price_at_ts],
                mode="markers",
                marker={
                    "color": style["color"],
                    "size": size,
                    "symbol": style["symbol"],
                    "opacity": opacity,
                },
                showlegend=False,
                hovertemplate=(
                    f"<b>{ts:%Y-%m-%d %H:%M}</b><br>"
                    f"Prediction: {row['predicted_direction']}<br>"
                    f"Confidence: {confidence:.1%}<br>"
                    f"Result: {result_label}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Prediction Timeline",
        xaxis_title="Time",
        yaxis_title="BTC Price (USD)",
        height=500,
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1f2e",
        legend={"font": {"color": "white"}},
    )

    return fig
