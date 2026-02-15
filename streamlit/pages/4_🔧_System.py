"""System Health page — ingestion status, monitoring state, recent logs."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bitbat.gui.widgets import (
    db_query,
    get_ingestion_status,
    get_recent_events,
    get_system_status,
)

st.set_page_config(page_title="System Health — BitBat", page_icon="🔧", layout="wide")

st.title("🔧 System Health")
st.markdown("Monitor data ingestion, agent status, and system logs.")

_DB = ROOT / "data" / "autonomous.db"
_DATA_DIR = ROOT / "data"

# ------------------------------------------------------------------
# Monitoring agent status
# ------------------------------------------------------------------
st.header("Monitoring Agent")
sys_info = get_system_status(_DB)

col_status, col_last = st.columns(2)
with col_status:
    st.metric("Agent Status", sys_info["label"])
with col_last:
    if sys_info["hours_ago"] is not None:
        st.metric("Last Check-In", f"{sys_info['hours_ago']:.1f} hours ago")
    else:
        st.metric("Last Check-In", "Never")

if sys_info["status"] == "not_started":
    st.info(
        "The monitoring agent is not running.  \n"
        "Start it with: `docker-compose up -d` or "
        "`poetry run python scripts/run_monitoring_agent.py`"
    )
elif sys_info["status"] == "idle":
    st.warning(
        f"Agent appears idle — last activity {sys_info['hours_ago']:.1f}h ago.  \n"
        "Check the agent logs for errors."
    )
else:
    st.success("Monitoring agent is running normally.")

# ------------------------------------------------------------------
# Data ingestion status
# ------------------------------------------------------------------
st.header("Data Ingestion")
ingest_info = get_ingestion_status(_DATA_DIR)

c1, c2 = st.columns(2)
with c1:
    st.metric("Price Data", ingest_info["prices"])
    if ingest_info["prices_mtime"]:
        st.caption(f"Last updated: {ingest_info['prices_mtime'].strftime('%Y-%m-%d %H:%M')}")
with c2:
    st.metric("News Data", ingest_info["news"])
    if ingest_info["news_mtime"]:
        st.caption(f"Last updated: {ingest_info['news_mtime'].strftime('%Y-%m-%d %H:%M')}")

# Data directory contents
with st.expander("📂 Data Directory Overview"):
    dirs_to_check = [
        ("Price data", _DATA_DIR / "raw" / "prices"),
        ("News data", _DATA_DIR / "raw" / "news"),
        ("Features", _DATA_DIR / "features"),
        ("Predictions", _DATA_DIR / "predictions"),
    ]
    rows = []
    for label, d in dirs_to_check:
        if d.exists():
            files = list(d.glob("**/*.parquet"))
            total_mb = sum(f.stat().st_size for f in files) / 1e6
            rows.append({"Directory": label, "Files": len(files), "Size (MB)": f"{total_mb:.1f}"})
        else:
            rows.append({"Directory": label, "Files": 0, "Size (MB)": "—"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# ------------------------------------------------------------------
# Performance snapshot history
# ------------------------------------------------------------------
st.header("Performance Snapshot History")

snap_rows = db_query(
    _DB,
    "SELECT snapshot_time, model_version, hit_rate, total_predictions, "
    "sharpe_ratio, max_drawdown "
    "FROM performance_snapshots ORDER BY snapshot_time DESC LIMIT 20",
)

if snap_rows:
    snap_df = pd.DataFrame(
        snap_rows,
        columns=["Time", "Model", "Accuracy %", "Predictions", "Sharpe", "Max Drawdown"],
    )
    snap_df["Accuracy %"] = snap_df["Accuracy %"].apply(
        lambda x: f"{x*100:.1f}%" if x is not None else "—"
    )
    snap_df["Sharpe"] = snap_df["Sharpe"].apply(
        lambda x: f"{x:.2f}" if x is not None else "—"
    )
    snap_df["Max Drawdown"] = snap_df["Max Drawdown"].apply(
        lambda x: f"{x:.2f}" if x is not None else "—"
    )
    st.dataframe(snap_df, width="stretch", hide_index=True)
else:
    st.info("No performance snapshots yet. The monitoring agent records these hourly.")

# ------------------------------------------------------------------
# System logs
# ------------------------------------------------------------------
st.header("System Logs")

log_limit = st.slider("Number of log entries", 10, 100, 20, step=10)
events = get_recent_events(_DB, limit=log_limit)

if events:
    level_icons = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌", "DEBUG": "🔍"}
    for ev in events:
        icon = level_icons.get(str(ev["level"]).upper(), "•")
        color = {"ERROR": "🔴", "WARNING": "🟡"}.get(str(ev["level"]).upper(), "")
        st.markdown(f"{color}{icon} `{ev['time']}` — {ev['message']}")
else:
    if not _DB.exists():
        st.info("Database not found. Start the monitoring system to generate logs.")
    else:
        st.info("No log entries found in database.")

# Log files on disk
with st.expander("📄 Log Files on Disk"):
    logs_dir = ROOT / "logs"
    if logs_dir.exists():
        log_files = sorted(logs_dir.glob("*.log"))
        if log_files:
            for lf in log_files:
                size_kb = lf.stat().st_size / 1024
                st.markdown(f"- `{lf.name}` — {size_kb:.1f} KB")
        else:
            st.info("No log files found in logs/ directory.")
    else:
        st.info("logs/ directory does not exist yet.")

# ------------------------------------------------------------------
# Retraining events
# ------------------------------------------------------------------
st.header("Retraining Events")

retrain_rows = db_query(
    _DB,
    "SELECT started_at, trigger_reason, status, "
    "old_model_version, new_model_version, cv_improvement, training_duration_seconds "
    "FROM retraining_events ORDER BY started_at DESC LIMIT 10",
)

if retrain_rows:
    retrain_df = pd.DataFrame(
        retrain_rows,
        columns=["Started", "Trigger", "Status", "Old Model", "New Model", "CV Improvement", "Duration (s)"],
    )
    retrain_df["Trigger"] = retrain_df["Trigger"].map(
        {
            "drift_detected": "Performance degraded",
            "scheduled": "Scheduled",
            "manual": "Manual trigger",
            "poor_performance": "Poor performance",
        }
    ).fillna(retrain_df["Trigger"])
    retrain_df["Status"] = retrain_df["Status"].map(
        {"completed": "✅ Done", "failed": "❌ Failed", "started": "⏳ Running"}
    ).fillna(retrain_df["Status"])
    st.dataframe(retrain_df, width="stretch", hide_index=True)
else:
    st.info("No retraining events recorded yet.")

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.divider()
st.caption(f"Refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
if st.button("🔄 Refresh Now"):
    st.rerun()
