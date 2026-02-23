"""Performance metrics page — user-friendly view of prediction accuracy."""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="Performance — BitBat", page_icon="📈", layout="wide")

st.title("📈 Performance")
st.markdown("See how well BitBat is predicting Bitcoin prices.")

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


def _df(sql: str, cols: list[str]) -> pd.DataFrame:
    rows = _query(sql)
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


# ------------------------------------------------------------------
# No data guard
# ------------------------------------------------------------------
if not _DB.exists():
    st.warning(
        "No performance data yet.  \n"
        "Start the monitoring system and wait for predictions to arrive."
    )
    st.stop()

# ------------------------------------------------------------------
# Overall statistics
# ------------------------------------------------------------------
st.header("Overall Statistics")

total_rows = _query("SELECT COUNT(*) FROM prediction_outcomes")
total = total_rows[0][0] if total_rows else 0

realized_rows = _query(
    "SELECT COUNT(*) FROM prediction_outcomes WHERE actual_return IS NOT NULL"
)
realized = realized_rows[0][0] if realized_rows else 0

correct_rows = _query(
    "SELECT COUNT(*) FROM prediction_outcomes WHERE correct=1"
)
correct = correct_rows[0][0] if correct_rows else 0

accuracy = (correct / realized * 100) if realized > 0 else 0.0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Predictions", f"{total:,}")
with c2:
    st.metric("Completed", f"{realized:,}", help="Predictions where the outcome is known")
with c3:
    st.metric("Correct", f"{correct:,}")
with c4:
    delta_color = "normal" if accuracy >= 55 else "inverse"
    st.metric("Accuracy", f"{accuracy:.1f}%")

if realized > 0:
    if accuracy >= 65:
        st.success(f"Great performance! {accuracy:.1f}% accuracy on {realized:,} completed predictions.")
    elif accuracy >= 55:
        st.info(f"Decent performance: {accuracy:.1f}% accuracy on {realized:,} predictions.")
    else:
        st.warning(f"Below average: {accuracy:.1f}% accuracy. The model may need retraining.")

# ------------------------------------------------------------------
# Rolling accuracy chart
# ------------------------------------------------------------------
st.header("Accuracy Over Time")

hist_rows = _query(
    "SELECT snapshot_time, hit_rate, total_predictions "
    "FROM performance_snapshots "
    "ORDER BY snapshot_time ASC"
)

if hist_rows:
    snap_df = pd.DataFrame(hist_rows, columns=["Time", "Accuracy %", "Predictions"])
    snap_df["Accuracy %"] = snap_df["Accuracy %"] * 100
    snap_df["Time"] = pd.to_datetime(snap_df["Time"])
    snap_df = snap_df.set_index("Time")
    st.line_chart(snap_df["Accuracy %"], width="stretch")
    st.caption("Accuracy % per monitoring snapshot")
else:
    st.info("Not enough data yet for a chart. Check back after more predictions are made.")

# ------------------------------------------------------------------
# Win / lose streak
# ------------------------------------------------------------------
st.header("Recent Streak")

streak_rows = _query(
    "SELECT correct FROM prediction_outcomes "
    "WHERE actual_return IS NOT NULL "
    "ORDER BY created_at DESC LIMIT 20"
)

if streak_rows:
    recent = [r[0] for r in streak_rows]
    # Count current streak from the most recent prediction
    current_streak = 0
    streak_type = "win" if recent[0] else "loss"
    for val in recent:
        if bool(val) == (streak_type == "win"):
            current_streak += 1
        else:
            break
    emoji = "🏆" if streak_type == "win" else "📉"
    st.metric(
        f"Current {streak_type.title()} Streak",
        f"{emoji} {current_streak}",
    )

    # Visual streak bar
    streak_display = "".join("✅" if r else "❌" for r in recent[:10])
    st.markdown(f"**Last 10 predictions:** {streak_display}")
else:
    st.info("No completed predictions yet.")

# ------------------------------------------------------------------
# Recent predictions table
# ------------------------------------------------------------------
st.header("Recent Predictions")

recent_df = _df(
    "SELECT timestamp_utc, predicted_direction, "
    "ROUND(MAX(p_up, p_down)*100,1) as confidence_pct, "
    "actual_direction, correct "
    "FROM prediction_outcomes "
    "WHERE actual_return IS NOT NULL "
    "ORDER BY created_at DESC LIMIT 20",
    cols=["Time", "Prediction", "Confidence %", "Actual Outcome", "Correct?"],
)

if not recent_df.empty:
    # Translate technical values to plain language
    recent_df["Prediction"] = recent_df["Prediction"].map(
        {"up": "📈 UP", "down": "📉 DOWN", "flat": "➡️ FLAT"}
    )
    recent_df["Actual Outcome"] = recent_df["Actual Outcome"].map(
        {"up": "📈 UP", "down": "📉 DOWN", "flat": "➡️ FLAT"}
    ).fillna("Pending")
    recent_df["Correct?"] = recent_df["Correct?"].map({1: "✅ Yes", 0: "❌ No"}).fillna("—")
    st.dataframe(recent_df, width="stretch", hide_index=True)
else:
    st.info("No completed predictions to display yet.")

# ------------------------------------------------------------------
# Current model info
# ------------------------------------------------------------------
st.header("Current Model")

model_rows = _query(
    "SELECT version, cv_score, training_start, training_end, training_samples "
    "FROM model_versions WHERE is_active=1 ORDER BY deployed_at DESC LIMIT 1"
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
        retrain_df["Reason"] = retrain_df["Reason"].map(
            {
                "drift_detected": "Performance degraded",
                "scheduled": "Scheduled",
                "manual": "Manual trigger",
                "poor_performance": "Poor performance",
            }
        ).fillna(retrain_df["Reason"])
        retrain_df["Status"] = retrain_df["Status"].map(
            {"completed": "✅ Done", "failed": "❌ Failed", "started": "⏳ Running"}
        ).fillna(retrain_df["Status"])
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

            impact_data.append(
                {
                    "Retrain Started": started_str,
                    "Model": version_str,
                    "Pre Hit-Rate": f"{pre_hit*100:.1f}%" if pre_hit is not None else "—",
                    "Post Hit-Rate": f"{post_hit*100:.1f}%" if post_hit is not None else "—",
                    "Pre Sharpe": f"{pre_sharpe:.2f}" if pre_sharpe is not None else "—",
                    "Post Sharpe": f"{post_sharpe:.2f}" if post_sharpe is not None else "—",
                }
            )

        impact_df = pd.DataFrame(impact_data)
        st.dataframe(impact_df, width="stretch", hide_index=True)
        st.caption(
            "Pre metrics are the last snapshot before retraining; post metrics are the first snapshot after."
        )
    else:
        st.info("No completed retraining events with impact metrics available yet.")
