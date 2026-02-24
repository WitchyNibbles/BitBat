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
_HEARTBEAT = _DATA_DIR / "monitoring_agent_heartbeat.json"

# ------------------------------------------------------------------
# Monitoring agent status
# ------------------------------------------------------------------
st.header("Monitoring Agent")
sys_info = get_system_status(_DB)

col_status, col_last, col_interval = st.columns(3)
with col_status:
    st.metric("Agent Status", sys_info["label"])
with col_last:
    if sys_info["hours_ago"] is not None:
        st.metric("Last Check-In", f"{sys_info['hours_ago']:.1f} hours ago")
    else:
        st.metric("Last Check-In", "Never")
with col_interval:
    if _HEARTBEAT.exists():
        import json

        try:
            hb = json.loads(_HEARTBEAT.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            st.metric("Schedule", "Unknown")
        else:
            interval = hb.get("interval_seconds")
            status = hb.get("status", "unknown")
            if interval is not None:
                st.metric("Schedule", f"Every {int(interval)} s")
            else:
                st.metric("Schedule", status)
    else:
        st.metric("Schedule", "Not available")

# ------------------------------------------------------------------
# Autonomous configuration (drift detection & retraining)
# ------------------------------------------------------------------
st.header("Autonomous Configuration")

try:
    import yaml  # type: ignore[import-not-found, import-untyped]
except Exception:  # pragma: no cover - defensive import
    yaml = None  # type: ignore[assignment]


def _load_user_config() -> dict:
    cfg_path = ROOT / "config" / "user_config.yaml"
    if not cfg_path.exists() or yaml is None:
        return {}
    try:
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))  # type: ignore[call-arg]
    except Exception:
        return {}
    return data or {}


def _save_user_config(data: dict) -> None:
    cfg_dir = ROOT / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "user_config.yaml"
    if yaml is None:
        return
    cfg_path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")  # type: ignore[call-arg]


user_cfg = _load_user_config()
auto_cfg = user_cfg.get("autonomous", {}) or {}
drift_cfg = auto_cfg.get("drift_detection", {}) or {}
retrain_cfg = auto_cfg.get("retraining", {}) or {}

dc1, dc2, dc3 = st.columns(3)
with dc1:
    window_days = st.number_input(
        "Drift window (days)",
        min_value=7,
        max_value=365,
        value=int(drift_cfg.get("window_days", 30) or 30),
        step=1,
        help="Number of days of realized predictions to use for drift checks.",
    )
with dc2:
    min_preds = st.number_input(
        "Min realized predictions",
        min_value=10,
        max_value=1000,
        value=int(drift_cfg.get("min_predictions_required", 30) or 30),
        step=5,
        help="Require at least this many realized predictions before evaluating drift.",
    )
with dc3:
    hit_drop = st.slider(
        "Hit-rate drop threshold",
        min_value=0.01,
        max_value=0.25,
        value=float(drift_cfg.get("hit_rate_drop_threshold", 0.05) or 0.05),
        step=0.01,
        help="Trigger drift when hit-rate falls this much below baseline.",
    )

rc1, rc2, rc3 = st.columns(3)
with rc1:
    sharpe_threshold = st.slider(
        "Sharpe ratio threshold",
        min_value=-1.0,
        max_value=1.0,
        value=float(drift_cfg.get("sharpe_threshold", 0.0) or 0.0),
        step=0.05,
        help="Trigger drift when Sharpe falls below this value.",
    )
with rc2:
    cooldown_hours = st.number_input(
        "Retrain cooldown (hours)",
        min_value=1,
        max_value=168,
        value=int(retrain_cfg.get("cooldown_hours", 24) or 24),
        step=1,
        help="Minimum hours between automatic retraining runs.",
    )
with rc3:
    cv_improvement = st.slider(
        "Min CV improvement",
        min_value=0.0,
        max_value=0.2,
        value=float(retrain_cfg.get("cv_improvement_threshold", 0.02) or 0.02),
        step=0.005,
        help="Only deploy a new model if CV score improves by at least this amount.",
    )

extra_col1, extra_col2 = st.columns(2)
with extra_col1:
    max_train_time = st.number_input(
        "Max training time (seconds)",
        min_value=60,
        max_value=24 * 3600,
        value=int(retrain_cfg.get("max_training_time_seconds", 3600) or 3600),
        step=60,
        help="Upper bound for automatic retraining duration.",
    )
with extra_col2:
    if st.button("Save Autonomous Settings", width="stretch"):
        user_cfg.setdefault("autonomous", {})
        user_cfg["autonomous"]["drift_detection"] = {
            "window_days": int(window_days),
            "min_predictions_required": int(min_preds),
            "hit_rate_drop_threshold": float(hit_drop),
            "sharpe_threshold": float(sharpe_threshold),
        }
        user_cfg["autonomous"]["retraining"] = {
            "cooldown_hours": int(cooldown_hours),
            "cv_improvement_threshold": float(cv_improvement),
            "max_training_time_seconds": int(max_train_time),
        }
        _save_user_config(user_cfg)
        st.success("Autonomous drift and retraining settings saved.")

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
