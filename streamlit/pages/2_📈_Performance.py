"""Performance metrics page — user-friendly view of prediction accuracy."""
# ruff: noqa: E402, I001

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
STREAMLIT_DIR = ROOT / "streamlit"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(STREAMLIT_DIR))

from bitbat.config.loader import get_runtime_config, load_config
from bitbat.gui.performance import (
    build_accuracy_history,
    format_recent_predictions,
    resolve_performance_scope,
    summarize_current_streak,
    summarize_performance_rows,
    summarize_recent_mix,
)
from bitbat.io.prices import load_prices
from style import inject_css

st.set_page_config(page_title="Performance — BitBat", page_icon="📈", layout="wide")
inject_css()

st.title("Performance")
st.markdown(
    "Read the signal ledger: accuracy, retraining, and whether the watcher still deserves trust."
)

_DB = ROOT / "data" / "autonomous.db"


def _query(sql: str, params: tuple = ()) -> list:
    if not _DB.exists():
        return []
    try:
        con = sqlite3.connect(str(_DB))
        rows = con.execute(sql, params).fetchall()
        con.close()
        return rows
    except Exception:
        return []


def _df(sql: str, cols: list[str], params: tuple = ()) -> pd.DataFrame:
    rows = _query(sql, params=params)
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


def _latest_snapshot(freq: str, horizon: str) -> tuple | None:
    rows = _query(
        "SELECT snapshot_time, hit_rate, realized_predictions, window_days "
        "FROM performance_snapshots "
        "WHERE freq = ? AND horizon = ? "
        "ORDER BY snapshot_time DESC LIMIT 1",
        (freq, horizon),
    )
    return rows[0] if rows else None


def _table_columns(table: str) -> set[str]:
    rows = _query(f"PRAGMA table_info({table})")
    columns: set[str] = set()
    for row in rows:
        if len(row) > 1:
            columns.add(str(row[1]))
    return columns


def _prediction_rows(freq: str, horizon: str) -> pd.DataFrame:
    columns = _table_columns("prediction_outcomes")
    if not columns:
        return pd.DataFrame()

    select_exprs = [
        "timestamp_utc",
        "created_at" if "created_at" in columns else "timestamp_utc AS created_at",
        (
            "predicted_direction"
            if "predicted_direction" in columns
            else "'flat' AS predicted_direction"
        ),
        "p_up" if "p_up" in columns else "NULL AS p_up",
        "p_down" if "p_down" in columns else "NULL AS p_down",
        "p_flat" if "p_flat" in columns else "NULL AS p_flat",
        "predicted_price" if "predicted_price" in columns else "NULL AS predicted_price",
        "actual_return" if "actual_return" in columns else "NULL AS actual_return",
        "actual_direction" if "actual_direction" in columns else "NULL AS actual_direction",
        "start_price" if "start_price" in columns else "NULL AS start_price",
        "end_price" if "end_price" in columns else "NULL AS end_price",
        "correct" if "correct" in columns else "NULL AS correct",
    ]

    sql = (
        f"SELECT {', '.join(select_exprs)} FROM prediction_outcomes "  # noqa: S608
        "WHERE freq = ? AND horizon = ? "
        "ORDER BY timestamp_utc ASC"
    )
    return _df(
        sql,
        cols=[
            "timestamp_utc",
            "created_at",
            "predicted_direction",
            "p_up",
            "p_down",
            "p_flat",
            "predicted_price",
            "actual_return",
            "actual_direction",
            "start_price",
            "end_price",
            "correct",
        ],
        params=(freq, horizon),
    )


def _load_prices_for_freq(freq: str) -> pd.DataFrame:
    try:
        config = get_runtime_config() or load_config()
        data_dir = Path(str(config.get("data_dir", "data"))).expanduser()
        prices = load_prices(data_dir, freq)
    except Exception:
        return pd.DataFrame(columns=["timestamp_utc", "close"])

    frame = prices.reset_index()[["timestamp_utc", "close"]].copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True).dt.tz_localize(None)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    return frame.dropna(subset=["timestamp_utc", "close"])


def _display_timezone() -> object | None:
    return datetime.now().astimezone().tzinfo


# ------------------------------------------------------------------
# No data guard
# ------------------------------------------------------------------
if not _DB.exists():
    st.warning(
        "No performance data yet.  \n"
        "Start the monitoring system and wait for predictions to arrive."
    )
    st.stop()

config = get_runtime_config() or load_config()
freq, horizon = resolve_performance_scope(st.session_state, config)
tau = float(config.get("tau", 0.01) or 0.01)

st.caption(
    f"Scope: `{freq} / {horizon}`. All-time metrics use realized rows for this pair only. "
    "Rolling accuracy follows market time, and "
    f"`flat` means the realized return stayed within ±{tau:.2%}."
)

prediction_rows = _prediction_rows(freq, horizon)
price_rows = _load_prices_for_freq(freq)
summary = summarize_performance_rows(prediction_rows)

# ------------------------------------------------------------------
# Overall statistics
# ------------------------------------------------------------------
st.header("Overall Statistics")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Predictions", f"{summary['total']:,}")
with c2:
    st.metric(
        "Completed", f"{summary['completed']:,}", help="Predictions where the outcome is known"
    )
with c3:
    st.metric("Correct", f"{summary['correct']:,}")
with c4:
    st.metric("All-Time Accuracy", f"{summary['accuracy']:.1f}%")

if summary["completed"] > 0:
    if summary["accuracy"] >= 65:
        st.success(
            "Strong scoped ledger: "
            f"{summary['accuracy']:.1f}% across "
            f"{summary['completed']:,} realized predictions."
        )
    elif summary["accuracy"] >= 55:
        st.info(
            "Moderate scoped ledger: "
            f"{summary['accuracy']:.1f}% across "
            f"{summary['completed']:,} realized predictions."
        )
    else:
        st.warning(
            "Low scoped ledger: "
            f"{summary['accuracy']:.1f}% across "
            f"{summary['completed']:,} realized predictions."
        )

# ------------------------------------------------------------------
# Rolling accuracy chart
# ------------------------------------------------------------------
st.header("Accuracy Over Time")

history_df = build_accuracy_history(prediction_rows, window=20)
if not history_df.empty:
    st.line_chart(history_df["Accuracy %"], width="stretch")
    st.caption("20-signal rolling accuracy by prediction time.")
else:
    st.info("Not enough data yet for a chart. Check back after more predictions are made.")

snapshot = _latest_snapshot(freq, horizon)
if snapshot is not None:
    snapshot_time, hit_rate, realized_predictions, window_days = snapshot
    st.caption(
        "Latest watcher snapshot: "
        f"{float(hit_rate or 0.0) * 100:.1f}% over {int(realized_predictions or 0)} realized "
        f"predictions in the last {int(window_days or 0)} days "
        f"(snapshot recorded {snapshot_time})."
    )

# ------------------------------------------------------------------
# Win / lose streak
# ------------------------------------------------------------------
st.header("Recent Streak")

streak = summarize_current_streak(prediction_rows, limit=20)
mix = summarize_recent_mix(prediction_rows, limit=20)

if streak["count"] > 0:
    emoji = "🏆" if streak["type"] == "win" else "📉"
    st.metric(
        f"Current {str(streak['type']).title()} Streak",
        f"{emoji} {int(streak['count'])}",
    )

    streak_display = "".join("✅" if r else "❌" for r in list(streak["recent_flags"])[:10])
    st.markdown(f"**Last 10 predictions:** {streak_display}")
    if mix["window"] > 0:
        st.caption(
            f"Recent class mix: {mix['predicted_flat_rate']:.0f}% predicted flat, "
            f"{mix['actual_flat_rate']:.0f}% realized flat across the last "
            f"{mix['window']} completed predictions."
        )
    if mix["predicted_flat_rate"] >= 60.0:
        st.info(
            f"This streak is flat-heavy under the current ±{tau:.2%} threshold. "
            "Treat it as low-volatility stability, not directional edge."
        )
else:
    st.info("No completed predictions yet.")

# ------------------------------------------------------------------
# Recent predictions table
# ------------------------------------------------------------------
st.header("Recent Predictions")
st.caption(
    "Signal Time is when the model made the call. Forecast For is the target time when the "
    f"`{horizon}` horizon resolves. Price Gap compares the target against the market price known "
    "for that row: entry price while pending, realized price once completed."
)

recent_df = format_recent_predictions(
    prediction_rows,
    limit=20,
    prices=price_rows,
    freq=freq,
    horizon=horizon,
    tau=tau,
    display_timezone=_display_timezone(),
)
if not recent_df.empty:
    st.dataframe(recent_df, width="stretch", hide_index=True)
else:
    st.info("No completed predictions to display yet.")

# ------------------------------------------------------------------
# Current model info
# ------------------------------------------------------------------
st.header("Current Model")

model_rows = _query(
    "SELECT version, cv_score, training_start, training_end, training_samples "
    "FROM model_versions "
    "WHERE is_active=1 AND freq = ? AND horizon = ? "
    "ORDER BY deployed_at DESC LIMIT 1",
    (freq, horizon),
)

if model_rows:
    ver, cv_score, t_start, t_end, n_samples = model_rows[0]
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Model Version", ver)
    with m2:
        st.metric(
            "Quality Score",
            f"{cv_score:.2f}" if cv_score is not None else "N/A",
            help="Cross-validation score — higher is better (max 1.0)",
        )
    with m3:
        st.metric("Training Samples", f"{n_samples:,}" if n_samples else "N/A")
    st.caption(f"Trained on data from {t_start} to {t_end}")
else:
    st.info("No model version information available yet.")

# ------------------------------------------------------------------
# Retraining history
# ------------------------------------------------------------------
with st.expander("🔄 Retraining History"):
    retrain_df = _df(
        "SELECT started_at, trigger_reason, status, cv_improvement, new_model_version "
        "FROM retraining_events ORDER BY started_at DESC LIMIT 10",
        cols=["Started", "Reason", "Status", "CV Improvement", "New Model"],
    )
    if not retrain_df.empty:
        retrain_df["Reason"] = (
            retrain_df["Reason"]
            .map({
                "drift_detected": "Performance degraded",
                "scheduled": "Scheduled",
                "manual": "Manual trigger",
                "poor_performance": "Poor performance",
            })
            .fillna(retrain_df["Reason"])
        )
        retrain_df["Status"] = (
            retrain_df["Status"]
            .map({"completed": "✅ Done", "failed": "❌ Failed", "started": "⏳ Running"})
            .fillna(retrain_df["Status"])
        )
        st.dataframe(retrain_df, width="stretch", hide_index=True)
    else:
        st.info("No retraining events recorded yet.")

# ------------------------------------------------------------------
# Retraining impact (pre/post metrics)
# ------------------------------------------------------------------
with st.expander("📐 Retraining Impact (Accuracy & Sharpe)"):
    impact_rows = _query(
        "SELECT started_at, new_model_version FROM retraining_events "
        "WHERE status='completed' AND new_model_version IS NOT NULL "
        "ORDER BY started_at DESC LIMIT 5"
    )
    if impact_rows:
        impact_data: list[dict[str, object]] = []
        for started_at, new_version in impact_rows:
            started_str = str(started_at)
            version_str = str(new_version)
            pre = _query(
                "SELECT snapshot_time, hit_rate, sharpe_ratio "
                "FROM performance_snapshots "
                "WHERE model_version=? AND snapshot_time < ? "
                "ORDER BY snapshot_time DESC LIMIT 1",
                (version_str, started_str),
            )
            post = _query(
                "SELECT snapshot_time, hit_rate, sharpe_ratio "
                "FROM performance_snapshots "
                "WHERE model_version=? AND snapshot_time >= ? "
                "ORDER BY snapshot_time ASC LIMIT 1",
                (version_str, started_str),
            )

            pre_hit = pre_sharpe = None
            pre_time = None
            if pre:
                pre_time, pre_hit, pre_sharpe = pre[0]
            post_hit = post_sharpe = None
            post_time = None
            if post:
                post_time, post_hit, post_sharpe = post[0]

            impact_data.append({
                "Retrain Started": started_str,
                "Model": version_str,
                "Pre Hit-Rate": f"{pre_hit * 100:.1f}%" if pre_hit is not None else "—",
                "Post Hit-Rate": f"{post_hit * 100:.1f}%" if post_hit is not None else "—",
                "Pre Sharpe": f"{pre_sharpe:.2f}" if pre_sharpe is not None else "—",
                "Post Sharpe": f"{post_sharpe:.2f}" if post_sharpe is not None else "—",
            })

        impact_df = pd.DataFrame(impact_data)
        st.dataframe(impact_df, width="stretch", hide_index=True)
        st.caption(
            "Pre metrics are the last snapshot before retraining; "
            "post metrics are the first snapshot after."
        )
    else:
        st.info("No completed retraining events with impact metrics available yet.")
