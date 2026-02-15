"""Historical Performance page — calendar heatmap, accuracy trends, streaks."""

from __future__ import annotations

import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bitbat.gui.widgets import db_query

st.set_page_config(page_title="History — BitBat", page_icon="📅", layout="wide")

st.title("📅 Historical Performance")
st.markdown("See how prediction accuracy has changed over time.")

_DB = ROOT / "data" / "autonomous.db"

if not _DB.exists():
    st.warning("No historical data yet. Start the monitoring system to begin collecting data.")
    st.stop()

# ------------------------------------------------------------------
# Load prediction history
# ------------------------------------------------------------------
rows = db_query(
    _DB,
    "SELECT date(timestamp_utc) as date, correct "
    "FROM prediction_outcomes "
    "WHERE actual_return IS NOT NULL "
    "ORDER BY timestamp_utc ASC",
)

if not rows:
    st.info(
        "No completed predictions yet. Come back after the monitoring system has been running for a while."
    )
    st.stop()

pred_df = pd.DataFrame(rows, columns=["date", "correct"])
pred_df["date"] = pd.to_datetime(pred_df["date"])
pred_df["correct"] = pred_df["correct"].astype(int)

# ------------------------------------------------------------------
# Overall summary
# ------------------------------------------------------------------
st.header("Summary")

total = len(pred_df)
correct = pred_df["correct"].sum()
accuracy = correct / total * 100 if total > 0 else 0.0
first_date = pred_df["date"].min().strftime("%Y-%m-%d")
last_date = pred_df["date"].max().strftime("%Y-%m-%d")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("All-Time Accuracy", f"{accuracy:.1f}%")
with c2:
    st.metric("Total Predictions", f"{total:,}")
with c3:
    st.metric("Correct", f"{correct:,}")
with c4:
    days_active = (pred_df["date"].max() - pred_df["date"].min()).days + 1
    st.metric("Days of History", f"{days_active:,}")

st.caption(f"Data from {first_date} to {last_date}")

# ------------------------------------------------------------------
# Calendar heatmap
# ------------------------------------------------------------------
st.header("Accuracy Calendar")
st.markdown("Daily prediction accuracy — green is good, red is poor:")

daily = (
    pred_df.groupby("date")
    .agg(accuracy=("correct", "mean"), count=("correct", "count"))
    .reset_index()
)
daily["accuracy_pct"] = (daily["accuracy"] * 100).round(1)

# Build calendar grid for the most recent 12 months
latest_month = daily["date"].max().to_period("M")
earliest_month = (daily["date"].min()).to_period("M")
months_to_show = min(12, (latest_month - earliest_month).n + 1)

fig, axes = plt.subplots(
    nrows=(months_to_show + 3) // 4,
    ncols=min(4, months_to_show),
    figsize=(16, max(4, ((months_to_show + 3) // 4) * 3)),
)
fig.patch.set_facecolor("#0e1117")

if months_to_show == 1:
    axes = np.array([[axes]])
elif months_to_show <= 4:
    axes = np.array([axes]) if axes.ndim == 1 else axes

daily_dict = daily.set_index("date")["accuracy_pct"].to_dict()
daily_count = daily.set_index("date")["count"].to_dict()

current_period = latest_month
for i in range(months_to_show):
    row_idx = (months_to_show - 1 - i) // 4
    col_idx = (months_to_show - 1 - i) % 4
    try:
        ax = axes.flat[months_to_show - 1 - i]
    except (IndexError, AttributeError):
        break

    period = current_period - i
    month_start = period.to_timestamp()
    month_name = month_start.strftime("%b %Y")

    # Build 6×7 grid (weeks × days)
    grid = np.full((6, 7), np.nan)
    first_weekday = month_start.weekday()  # 0=Monday
    days_in_month = (period + 1).to_timestamp() - pd.Timedelta(days=1)
    days_in_month = days_in_month.day

    for day in range(1, days_in_month + 1):
        date_key = month_start + pd.Timedelta(days=day - 1)
        cell = first_weekday + day - 1
        row, col = divmod(cell, 7)
        if row < 6:
            acc = daily_dict.get(date_key)
            grid[row, col] = acc if acc is not None else np.nan

    masked = np.ma.masked_invalid(grid)
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="#1a1f2e")
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_title(month_name, color="white", fontsize=10)
    ax.set_xticks(range(7))
    ax.set_xticklabels(["M", "T", "W", "T", "F", "S", "S"], color="gray", fontsize=7)
    ax.set_yticks([])
    ax.set_facecolor("#1a1f2e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

# Hide unused axes
for j in range(months_to_show, len(axes.flat)):
    try:
        axes.flat[j].set_visible(False)
    except (IndexError, AttributeError):
        pass

plt.tight_layout()
st.pyplot(fig, clear_figure=True)
st.caption("Color scale: 🔴 0% → 🟡 50% → 🟢 100% accuracy. Gray = no data.")

# ------------------------------------------------------------------
# Weekly accuracy chart
# ------------------------------------------------------------------
st.header("Weekly Accuracy Trend")

weekly = (
    pred_df.set_index("date")
    .resample("W")["correct"]
    .agg(accuracy="mean", count="count")
    .reset_index()
)
weekly["accuracy_pct"] = weekly["accuracy"] * 100

if len(weekly) >= 2:
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(
        weekly["date"],
        weekly["accuracy_pct"],
        color="#10B981",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax2.axhline(50, color="#EF4444", linewidth=1, linestyle="--", alpha=0.6, label="50% (random)")
    ax2.axhline(accuracy, color="#F59E0B", linewidth=1, linestyle="--", alpha=0.6, label=f"All-time avg {accuracy:.1f}%")
    ax2.fill_between(weekly["date"], weekly["accuracy_pct"], 50, alpha=0.15, color="#10B981")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Accuracy %", color="white")
    ax2.set_title("Weekly Prediction Accuracy", color="white")
    ax2.set_facecolor("#1a1f2e")
    fig2.patch.set_facecolor("#0e1117")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444")
    ax2.legend(facecolor="#1a1f2e", labelcolor="white")
    plt.tight_layout()
    st.pyplot(fig2, clear_figure=True)
else:
    st.info("Not enough weekly data yet.")

# ------------------------------------------------------------------
# Day-of-week and hour-of-day accuracy
# ------------------------------------------------------------------
st.header("When Is BitBat Most Accurate?")

hour_rows = db_query(
    _DB,
    "SELECT strftime('%H', timestamp_utc) as hour, "
    "AVG(CASE WHEN correct=1 THEN 100.0 ELSE 0.0 END) as accuracy, "
    "COUNT(*) as count "
    "FROM prediction_outcomes WHERE actual_return IS NOT NULL "
    "GROUP BY hour ORDER BY hour",
)

dow_rows = db_query(
    _DB,
    "SELECT strftime('%w', timestamp_utc) as dow, "
    "AVG(CASE WHEN correct=1 THEN 100.0 ELSE 0.0 END) as accuracy, "
    "COUNT(*) as count "
    "FROM prediction_outcomes WHERE actual_return IS NOT NULL "
    "GROUP BY dow ORDER BY dow",
)

time_col, dow_col = st.columns(2)

with time_col:
    if hour_rows:
        hour_df = pd.DataFrame(hour_rows, columns=["Hour (UTC)", "Accuracy %", "Count"])
        hour_df["Hour (UTC)"] = hour_df["Hour (UTC)"].astype(int)
        hour_df["Accuracy %"] = hour_df["Accuracy %"].round(1)
        st.subheader("By Hour of Day")
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        colors = ["#10B981" if v >= 55 else ("#F59E0B" if v >= 50 else "#EF4444") for v in hour_df["Accuracy %"]]
        ax3.bar(hour_df["Hour (UTC)"], hour_df["Accuracy %"], color=colors)
        ax3.axhline(50, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
        ax3.set_xlabel("Hour (UTC)", color="white")
        ax3.set_ylabel("Accuracy %", color="white")
        ax3.set_facecolor("#1a1f2e")
        fig3.patch.set_facecolor("#0e1117")
        ax3.tick_params(colors="white")
        for spine in ax3.spines.values():
            spine.set_edgecolor("#444")
        plt.tight_layout()
        st.pyplot(fig3, clear_figure=True)
    else:
        st.info("Not enough data for hour-of-day analysis.")

with dow_col:
    if dow_rows:
        dow_df = pd.DataFrame(dow_rows, columns=["dow", "Accuracy %", "Count"])
        dow_labels = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"}
        dow_df["Day"] = dow_df["dow"].astype(int).map(dow_labels)
        dow_df["Accuracy %"] = dow_df["Accuracy %"].round(1)
        st.subheader("By Day of Week")
        fig4, ax4 = plt.subplots(figsize=(7, 4))
        colors = ["#10B981" if v >= 55 else ("#F59E0B" if v >= 50 else "#EF4444") for v in dow_df["Accuracy %"]]
        ax4.bar(dow_df["Day"], dow_df["Accuracy %"], color=colors)
        ax4.axhline(50, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
        ax4.set_ylabel("Accuracy %", color="white")
        ax4.set_facecolor("#1a1f2e")
        fig4.patch.set_facecolor("#0e1117")
        ax4.tick_params(colors="white")
        for spine in ax4.spines.values():
            spine.set_edgecolor("#444")
        plt.tight_layout()
        st.pyplot(fig4, clear_figure=True)
    else:
        st.info("Not enough data for day-of-week analysis.")

# ------------------------------------------------------------------
# Prediction table (paginated)
# ------------------------------------------------------------------
with st.expander("📋 Full Prediction History"):
    hist_rows = db_query(
        _DB,
        "SELECT timestamp_utc, predicted_direction, "
        "ROUND(MAX(p_up,p_down)*100,1) as confidence, "
        "actual_direction, correct "
        "FROM prediction_outcomes WHERE actual_return IS NOT NULL "
        "ORDER BY timestamp_utc DESC LIMIT 100",
    )
    if hist_rows:
        hist_df = pd.DataFrame(
            hist_rows,
            columns=["Time", "Prediction", "Confidence %", "Actual", "Correct?"],
        )
        hist_df["Prediction"] = hist_df["Prediction"].map({"up": "📈 UP", "down": "📉 DOWN", "flat": "➡️ FLAT"})
        hist_df["Actual"] = hist_df["Actual"].map({"up": "📈 UP", "down": "📉 DOWN", "flat": "➡️ FLAT"})
        hist_df["Correct?"] = hist_df["Correct?"].map({1: "✅", 0: "❌"})
        st.dataframe(hist_df, width="stretch", hide_index=True)
    else:
        st.info("No prediction history available.")
