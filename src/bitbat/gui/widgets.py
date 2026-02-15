"""
Reusable Streamlit UI widgets for the BitBat dashboard.

All widgets accept a Streamlit container (or None to render in the main area)
and render consistently across pages.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data helpers (no Streamlit dependency — pure Python)
# ---------------------------------------------------------------------------


def db_query(db_path: Path, sql: str, params: tuple = ()) -> list:
    """Run a SQL SELECT against the autonomous DB, returning rows or []."""
    if not db_path.exists():
        return []
    try:
        con = sqlite3.connect(str(db_path))
        rows = con.execute(sql, params).fetchall()
        con.close()
        return rows
    except Exception:
        return []


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
    sql = f"SELECT {column} FROM {table}"
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
    except Exception:
        pass

    try:
        return datetime.fromtimestamp(heartbeat_path.stat().st_mtime)
    except Exception:
        return None


def get_system_status(db_path: Path) -> dict[str, Any]:
    """Return a dict with system status derived from recent monitoring activity."""
    latest_snapshot = _latest_timestamp(db_path, "performance_snapshots", "snapshot_time")
    latest_monitor_log = _latest_timestamp(
        db_path,
        "system_logs",
        "timestamp",
        where_sql="service = ?",
        params=("monitoring_agent",),
    )
    if latest_monitor_log is None:
        latest_monitor_log = _latest_timestamp(db_path, "system_logs", "timestamp")
    if latest_monitor_log is None:
        latest_monitor_log = _latest_timestamp(db_path, "system_logs", "created_at")
    latest_retraining = _latest_timestamp(db_path, "retraining_events", "started_at")
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
    rows = db_query(
        db_path,
        "SELECT timestamp_utc, predicted_direction, p_up, p_down, model_version, created_at "
        "FROM prediction_outcomes ORDER BY created_at DESC LIMIT 1",
    )
    if not rows:
        return None
    ts, direction, p_up, p_down, model_ver, created_at = rows[0]
    confidence = max(float(p_up), float(p_down))
    return {
        "timestamp_utc": ts,
        "direction": direction,
        "p_up": float(p_up),
        "p_down": float(p_down),
        "confidence": confidence,
        "model_version": model_ver,
        "created_at": created_at,
    }


def get_recent_events(db_path: Path, limit: int = 10) -> list[dict[str, Any]]:
    """Return recent system events from system_logs."""
    rows = db_query(
        db_path,
        "SELECT timestamp, level, message FROM system_logs ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    )
    if not rows:
        # Backward-compat with earlier local/test schemas.
        rows = db_query(
            db_path,
            "SELECT created_at, level, message FROM system_logs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
    return [
        {"time": r[0], "level": r[1], "message": r[2]}
        for r in rows
    ]


def get_ingestion_status(data_dir: Path) -> dict[str, Any]:
    """Check freshness of price and news data on disk."""
    prices_dir = data_dir / "raw" / "prices"
    news_dir = data_dir / "raw" / "news"

    def _latest_mtime(d: Path) -> datetime | None:
        if not d.exists():
            return None
        files = list(d.glob("**/*.parquet"))
        if not files:
            return None
        return datetime.fromtimestamp(max(f.stat().st_mtime for f in files))

    prices_mtime = _latest_mtime(prices_dir)
    news_mtime = _latest_mtime(news_dir)
    now = datetime.now(UTC).replace(tzinfo=None)

    def _freshness(mtime: datetime | None) -> str:
        if mtime is None:
            return "⚪ No data"
        hours = (now - mtime).total_seconds() / 3600
        if hours < 2:
            return "🟢 Fresh"
        elif hours < 24:
            return f"🟡 {int(hours)}h ago"
        else:
            return f"🔴 {int(hours // 24)}d ago"

    return {
        "prices": _freshness(prices_mtime),
        "news": _freshness(news_mtime),
        "prices_mtime": prices_mtime,
        "news_mtime": news_mtime,
    }


def minutes_until_next_prediction(last_created_at: str | None, interval_hours: int = 1) -> int | None:
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
    confidence = pred["confidence"]

    if direction == "up":
        st.success(f"📈 **UP** — {confidence:.0%} confidence")
    elif direction == "down":
        st.error(f"📉 **DOWN** — {confidence:.0%} confidence")
    else:
        st.info(f"➡️ **FLAT** — {confidence:.0%} confidence")

    st.progress(min(confidence, 1.0))


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
