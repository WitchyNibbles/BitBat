"""
Reusable Streamlit UI widgets for the BitBat dashboard.

All widgets accept a Streamlit container (or None to render in the main area)
and render consistently across pages.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from bitbat.autonomous.db import AutonomousDB, MonitorDatabaseError
from bitbat.common.ingestion_status import get_ingestion_status  # noqa: F401
from bitbat.gui.shared import db_query, get_db

# ---------------------------------------------------------------------------
# Data helpers (no Streamlit dependency — pure Python)
# ---------------------------------------------------------------------------


def _table_columns(db_path: Path, table: str) -> set[str]:
    """Return available column names for a table, or an empty set."""
    rows = db_query(db_path, f"PRAGMA table_info({table})")  # noqa: S608
    columns: set[str] = set()
    for row in rows:
        if len(row) > 1:
            columns.add(str(row[1]))
    return columns


def _to_float(value: Any) -> float | None:
    """Convert common numeric inputs to float when possible."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_available_column(columns: set[str], candidates: tuple[str, ...]) -> str | None:
    """Pick the first column name that exists in the table."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse common DB/file timestamp values into naive UTC datetimes."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is not None:
        return dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def _latest_timestamp(
    db_path: Path,
    table: str,
    column: str,
    where_sql: str | None = None,
    params: tuple = (),
) -> datetime | None:
    """Fetch latest timestamp value from a table/column, if available."""
    sql = f"SELECT {column} FROM {table}"  # noqa: S608
    if where_sql:
        sql += f" WHERE {where_sql}"
    sql += f" ORDER BY {column} DESC LIMIT 1"
    rows = db_query(db_path, sql, params)
    if not rows:
        return None
    return _parse_timestamp(rows[0][0])


def _latest_monitor_heartbeat(db_path: Path) -> datetime | None:
    """Read monitoring heartbeat file timestamp from the shared data directory."""
    heartbeat_path = db_path.parent / "monitoring_agent_heartbeat.json"
    if not heartbeat_path.exists():
        return None

    try:
        payload = json.loads(heartbeat_path.read_text())
        parsed = _parse_timestamp(payload.get("updated_at"))
        if parsed is not None:
            return parsed
    except Exception:  # noqa: S110
        pass

    try:
        return datetime.fromtimestamp(heartbeat_path.stat().st_mtime)
    except Exception:
        return None


def get_system_status(db_path: Path) -> dict[str, Any]:
    """Return a dict with system status derived from recent monitoring activity."""
    db = get_db(db_path)
    latest_snapshot = None
    latest_monitor_log = None
    latest_retraining = None
    if db is not None:
        try:
            activity = db.get_system_activity_summary()
        except MonitorDatabaseError:
            activity = {}
        latest_snapshot = _parse_timestamp(activity.get("latest_snapshot"))
        latest_monitor_log = _parse_timestamp(activity.get("latest_monitor_log"))
        latest_retraining = _parse_timestamp(activity.get("latest_retraining"))
    latest_heartbeat = _latest_monitor_heartbeat(db_path)

    candidates = [
        dt
        for dt in (
            latest_snapshot,
            latest_monitor_log,
            latest_retraining,
            latest_heartbeat,
        )
        if dt is not None
    ]
    if not candidates:
        return {"status": "not_started", "label": "⚪ Not Started", "hours_ago": None}

    latest_activity = max(candidates)
    now = datetime.now(UTC).replace(tzinfo=None)
    try:
        hours_ago = max(0.0, (now - latest_activity).total_seconds() / 3600)
        if hours_ago < 2:
            return {"status": "active", "label": "🟢 Active", "hours_ago": hours_ago}
        return {"status": "idle", "label": "🟡 Idle", "hours_ago": hours_ago}
    except Exception:
        return {"status": "unknown", "label": "❓ Unknown", "hours_ago": None}


def get_latest_prediction(db_path: Path) -> dict[str, Any] | None:
    """Return the most recent prediction row, or None."""
    db = get_db(db_path)
    if db is None:
        return None
    try:
        return db.get_latest_prediction_payload()
    except MonitorDatabaseError:
        return None


def get_recent_events(db_path: Path, limit: int = 10) -> list[dict[str, Any]]:
    """Return recent system events from system_logs."""
    db = get_db(db_path)
    if db is None:
        return []
    try:
        return db.list_recent_system_events(limit=limit)
    except MonitorDatabaseError:
        return []


def minutes_until_next_prediction(
    last_created_at: str | None, interval_hours: int = 1
) -> int | None:
    """Return minutes until the next expected prediction, or None."""
    if last_created_at is None:
        return None
    try:
        last_dt = datetime.fromisoformat(last_created_at)
        next_dt = last_dt + timedelta(hours=interval_hours)
        remaining = (next_dt - datetime.now(UTC).replace(tzinfo=None)).total_seconds() / 60
        return max(0, int(remaining))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Streamlit render helpers (import st lazily to keep module importable in tests)
# ---------------------------------------------------------------------------


def render_status_card(db_path: Path) -> None:
    """Render a compact status card (active/idle/not started)."""
    import streamlit as st  # noqa: PLC0415

    info = get_system_status(db_path)
    st.metric("System Status", info["label"])
    if info["hours_ago"] is not None:
        st.caption(f"Last snapshot {info['hours_ago']:.1f}h ago")


def render_prediction_card(pred: dict[str, Any]) -> None:
    """Render a compact prediction result card."""
    import streamlit as st  # noqa: PLC0415

    direction = pred["direction"]
    predicted_return = pred.get("predicted_return", 0.0)
    predicted_price = pred.get("predicted_price")

    price_str = f"${predicted_price:,.0f} " if predicted_price is not None else ""
    sign = "+" if predicted_return >= 0 else ""
    ret_str = f"({sign}{predicted_return:.2%})"

    if direction == "up":
        st.success(f"Predicted: {price_str}{ret_str}")
    elif direction == "down":
        st.error(f"Predicted: {price_str}{ret_str}")
    else:
        st.info(f"Predicted: {price_str}{ret_str}")


def render_countdown(minutes: int | None) -> None:
    """Render a 'Next prediction in X minutes' widget."""
    import streamlit as st  # noqa: PLC0415

    if minutes is None:
        st.caption("Next prediction: unknown")
    elif minutes == 0:
        st.caption("Next prediction: imminent")
    else:
        hrs, mins = divmod(minutes, 60)
        if hrs > 0:
            st.caption(f"Next prediction in: {hrs}h {mins}m")
        else:
            st.caption(f"Next prediction in: {mins}m")


def render_activity_feed(db_path: Path, limit: int = 10) -> None:
    """Render a list of recent system events."""
    import streamlit as st  # noqa: PLC0415

    events = get_recent_events(db_path, limit=limit)
    if not events:
        st.info("No recent activity recorded yet.")
        return

    level_icons = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌", "DEBUG": "🔍"}
    for ev in events:
        icon = level_icons.get(str(ev["level"]).upper(), "•")
        st.markdown(f"{icon} `{ev['time']}` — {ev['message']}")
