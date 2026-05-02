"""System Health page — ingestion status, monitoring state, recent logs."""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
STREAMLIT_DIR = ROOT / "streamlit"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(STREAMLIT_DIR))

from style import inject_css  # noqa: E402

from bitbat.gui.widgets import (  # noqa: E402
    db_query,
    format_relative_time,
    get_cycle_health,
    get_ingestion_status,
    get_monitor_heartbeat,
    get_recent_events,
    get_runtime_summary,
    get_system_status,
    normalize_cycle_health,
    normalize_runtime_summary,
    sanitize_heartbeat_payload,
)

st.set_page_config(page_title="System Health — BitBat", page_icon="🔧", layout="wide")
inject_css()

st.title("System Health")
st.markdown("Proof, cadence, and next action for the watcher before the deeper diagnostics.")

_DB = ROOT / "data" / "autonomous.db"
_DATA_DIR = ROOT / "data"
_HEARTBEAT = _DATA_DIR / "monitoring_agent_heartbeat.json"

try:
    import yaml  # type: ignore[import-not-found, import-untyped]
except Exception:  # pragma: no cover - defensive import
    yaml = None  # type: ignore[assignment]


def _render_story_cards(cards: list[tuple[str, str, str]]) -> None:
    markup = []
    for label, value, copy in cards:
        markup.append(
            "<div class='bb-card'>"
            f"<div class='bb-label'>{escape(label)}</div>"
            f"<div class='bb-value'>{escape(value)}</div>"
            f"<div class='bb-copy'>{escape(copy)}</div>"
            "</div>"
        )
    st.markdown(f"<div class='bb-grid'>{''.join(markup)}</div>", unsafe_allow_html=True)


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

heartbeat_interval_seconds = None
if _HEARTBEAT.exists():
    import json

    try:
        heartbeat_payload = json.loads(_HEARTBEAT.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        heartbeat_payload = {}
    heartbeat_interval_seconds = heartbeat_payload.get("interval_seconds")

summary_interval_minutes = 60
if heartbeat_interval_seconds is not None:
    try:
        summary_interval_minutes = max(1, int(heartbeat_interval_seconds) // 60)
    except (TypeError, ValueError):
        summary_interval_minutes = 60

runtime = normalize_runtime_summary(
    get_runtime_summary(_DB, _DATA_DIR, interval_minutes=summary_interval_minutes)
)
sys_info = get_system_status(_DB)
ingest_info = get_ingestion_status(_DATA_DIR)
cycle_health = normalize_cycle_health(
    get_cycle_health(_DB, interval_minutes=summary_interval_minutes)
)
heartbeat_payload = get_monitor_heartbeat(_DB)
safe_heartbeat_payload = sanitize_heartbeat_payload(heartbeat_payload)

st.markdown(
    (
        "<div class='bb-hero'>"
        "<div class='bb-kicker'>Watcher Status</div>"
        f"<div class='bb-state-chip' data-tone='{escape(str(runtime['tone']))}'>"
        f"{escape(str(runtime['title']))}</div>"
        f"<h2 class='bb-title'>{escape(str(runtime['proof']))}</h2>"
        f"<div class='bb-subtitle'>{escape(str(runtime['next']))}</div>"
        f"<div class='bb-subtitle'>{escape(str(runtime['detail']))}</div>"
        "</div>"
    ),
    unsafe_allow_html=True,
)

_render_story_cards([
    (
        "Cycle Health",
        str(cycle_health["title"]),
        str(cycle_health["summary"]),
    ),
    (
        "Watcher",
        sys_info["label"],
        "Durable health derived from stored snapshots, logs, and heartbeat proof.",
    ),
    (
        "Last Check-In",
        "Never" if sys_info["hours_ago"] is None else f"{sys_info['hours_ago']:.1f}h ago",
        "Long gaps usually mean the durable watcher process is down or blocked.",
    ),
    (
        "Price Candles",
        ingest_info["prices"],
        "Price freshness is the minimum requirement for useful signals.",
    ),
    (
        "News Signals",
        ingest_info["news"],
        "News can lag without fully blocking price-only predictions.",
    ),
])

st.header("Last Cycle")
if cycle_health["state"] in {"blocked", "error"}:
    st.error(str(cycle_health["summary"]))
elif cycle_health["state"] == "degraded":
    st.warning(str(cycle_health["summary"]))
elif cycle_health["state"] == "healthy":
    st.success(str(cycle_health["summary"]))
else:
    st.info(str(cycle_health["summary"]))

_render_story_cards([
    ("Cycle", str(cycle_health["title"]), "How the latest durable cycle ended."),
    (
        "Ingestion",
        "OK" if cycle_health["issue_count"] == 0 else f"{cycle_health['issue_count']} failed",
        "Required source failures block prediction; optional source failures degrade it.",
    ),
    (
        "Prediction",
        str((heartbeat_payload or {}).get("cycle_prediction_state", "unknown")).title(),
        "Whether the latest durable cycle generated, skipped, or failed a signal.",
    ),
    ("Operator Action", str(cycle_health["action"]), "What to do next if trust is reduced."),
])

if safe_heartbeat_payload and safe_heartbeat_payload.get("error_summary"):
    st.caption(f"Last error summary: {safe_heartbeat_payload['error_summary']}")

if cycle_health["issues"]:
    issue_rows = []
    for issue in cycle_health["issues"]:
        details = issue.get("details") or {}
        detail_text = "n/a"
        if isinstance(details, dict) and details:
            detail_text = "Hidden in this view; inspect watcher logs for full diagnostics."
        issue_rows.append({
            "Source": str(issue.get("source", "unknown")),
            "Required": "Yes" if issue.get("required") else "No",
            "Message": str(issue.get("message", "")),
            "Detail": detail_text,
        })
    st.dataframe(pd.DataFrame(issue_rows), width="stretch", hide_index=True)

with st.expander("Last cycle payload", expanded=False):
    if safe_heartbeat_payload is None:
        st.caption("No heartbeat payload found yet.")
    else:
        st.caption("Sensitive runtime fields are hidden in this view.")
        st.json(safe_heartbeat_payload)

st.header("Recent Proof")
events = get_recent_events(_DB, limit=6)
if events:
    for event in events:
        level = str(event.get("level", "INFO")).upper()
        age = format_relative_time(event.get("time")) or "unknown age"
        message = str(event.get("message", "No details recorded"))
        st.markdown(
            (
                "<div class='bb-event-row'>"
                f"<div class='bb-event-meta'>watcher • {escape(level)} • {escape(age)}</div>"
                f"<div class='bb-event-message'>{escape(message)}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
else:
    st.info(
        "No watcher events recorded yet. Start monitoring and the first durable cycle "
        "will appear here."
    )

if sys_info["status"] == "not_started":
    st.warning(
        "No durable watcher proof yet. Train a model, then start monitoring from "
        "Quick Start or your service runner."
    )
elif sys_info["status"] == "idle":
    st.warning(
        f"Watcher looks stale. Last durable activity was about "
        f"{sys_info['hours_ago']:.1f} hours ago."
    )
else:
    st.success("Watcher proof looks current.")

with st.expander("Autonomous Controls", expanded=False):
    st.caption("Use these only after the health summary above looks correct.")

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

st.header("Data Footprint")

with st.expander("Data Directory Overview", expanded=False):
    dirs_to_check = [
        ("Price data", _DATA_DIR / "raw" / "prices"),
        ("News data", _DATA_DIR / "raw" / "news"),
        ("Features", _DATA_DIR / "features"),
        ("Predictions", _DATA_DIR / "predictions"),
    ]
    rows = []
    for label, directory in dirs_to_check:
        if directory.exists():
            files = list(directory.glob("**/*.parquet"))
            total_mb = sum(path.stat().st_size for path in files) / 1e6
            rows.append({"Directory": label, "Files": len(files), "Size (MB)": f"{total_mb:.1f}"})
        else:
            rows.append({"Directory": label, "Files": 0, "Size (MB)": "—"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

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
        lambda value: f"{value * 100:.1f}%" if value is not None else "—"
    )
    snap_df["Sharpe"] = snap_df["Sharpe"].apply(
        lambda value: f"{value:.2f}" if value is not None else "—"
    )
    snap_df["Max Drawdown"] = snap_df["Max Drawdown"].apply(
        lambda value: f"{value:.2f}" if value is not None else "—"
    )
    st.dataframe(snap_df, width="stretch", hide_index=True)
else:
    st.info(
        "No performance snapshots yet. The watcher records these over time once monitoring is live."
    )

st.header("Retraining Events")
retraining_rows = db_query(
    _DB,
    "SELECT started_at, trigger_reason, status, new_model_version "
    "FROM retraining_events ORDER BY started_at DESC LIMIT 10",
)

if retraining_rows:
    retraining_df = pd.DataFrame(
        retraining_rows,
        columns=["Started", "Reason", "Status", "New Model"],
    )
    st.dataframe(retraining_df, width="stretch", hide_index=True)
else:
    st.info("No retraining events recorded yet.")
