"""
Reusable Streamlit UI widgets for the BitBat dashboard.

All widgets accept a Streamlit container (or None to render in the main area)
and render consistently across pages.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from bitbat.autonomous.db import MonitorDatabaseError
from bitbat.common.ingestion_status import get_ingestion_status  # noqa: F401
from bitbat.gui.shared import db_query, get_db

_FREQ_TO_MINUTES = {
    "1m": 1,
    "2m": 2,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "24h": 1440,
    "1d": 1440,
}

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


def format_local_timestamp(
    value: Any,
    *,
    display_timezone: Any | None = None,
) -> str | None:
    """Format a stored UTC timestamp in local time with an explicit zone label."""
    parsed = _parse_timestamp(value)
    if parsed is None:
        return None
    zone = display_timezone or datetime.now().astimezone().tzinfo
    if zone is None:
        return parsed.strftime("%Y-%m-%d %H:%M UTC")
    return parsed.replace(tzinfo=UTC).astimezone(zone).strftime("%Y-%m-%d %H:%M %Z")


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
    payload = get_monitor_heartbeat(db_path)
    if payload is not None:
        parsed = _parse_timestamp(payload.get("updated_at"))
        if parsed is not None:
            return parsed

    try:
        heartbeat_path = db_path.parent / "monitoring_agent_heartbeat.json"
        return datetime.fromtimestamp(heartbeat_path.stat().st_mtime)
    except Exception:
        return None


def get_monitor_heartbeat(db_path: Path) -> dict[str, Any] | None:
    """Return the latest monitor heartbeat payload, if available."""
    heartbeat_path = db_path.parent / "monitoring_agent_heartbeat.json"
    if not heartbeat_path.exists():
        return None
    try:
        payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def get_cycle_health(  # noqa: C901
    db_path: Path,
    *,
    interval_minutes: int = 60,
) -> dict[str, Any]:
    """Return UI-ready health for the latest durable monitoring cycle."""
    heartbeat = get_monitor_heartbeat(db_path)
    if heartbeat is None:
        return {
            "state": "unknown",
            "tone": "info",
            "title": "No Cycle Proof",
            "summary": (
                "No latest cycle payload yet. Start monitoring or wait for the next heartbeat."
            ),
            "action": (
                "Use Quick Start or the monitor service to produce the first durable cycle."
            ),
            "issues": [],
            "issue_count": 0,
            "payload": None,
        }

    updated_at = heartbeat.get("updated_at")
    updated_at_dt = _parse_timestamp(updated_at)
    heartbeat_status = str(heartbeat.get("status", "unknown"))
    heartbeat_interval_seconds = heartbeat.get("interval_seconds")
    heartbeat_interval_minutes = _coerce_positive_minutes(
        heartbeat_interval_seconds,
        int(interval_minutes),
    )

    max_age_minutes = max(5, heartbeat_interval_minutes * 2)
    if updated_at_dt is not None:
        age_minutes = (
            datetime.now(UTC).replace(tzinfo=None) - updated_at_dt
        ).total_seconds() / 60.0
        if age_minutes > max_age_minutes:
            return {
                "state": "degraded",
                "tone": "warning",
                "title": "Watcher Stale",
                "summary": "Latest durable heartbeat is older than the expected watcher cadence.",
                "action": (
                    "Check the durable monitor process and restart it if the cadence is broken."
                ),
                "issues": [],
                "issue_count": 0,
                "updated_at": updated_at,
                "payload": heartbeat,
            }

    raw_failures = heartbeat.get("cycle_ingestion_failures", [])
    issues: list[dict[str, Any]] = []
    if isinstance(raw_failures, list):
        for item in raw_failures:
            if not isinstance(item, dict):
                continue
            issues.append({
                "source": str(item.get("source", "unknown")),
                "required": bool(item.get("required", False)),
                "status": str(item.get("status", "unknown")),
                "message": str(item.get("message", "")),
                "details": item.get("details") if isinstance(item.get("details"), dict) else {},
            })

    if heartbeat_status == "error":
        return {
            "state": "error",
            "tone": "danger",
            "title": "Monitor Error",
            "summary": "Latest cycle failed before completion.",
            "action": (
                "Inspect System logs, then restart the watcher after the runtime issue is fixed."
            ),
            "issues": issues,
            "issue_count": len(issues),
            "updated_at": updated_at,
            "payload": heartbeat,
        }

    ingestion_state = str(heartbeat.get("cycle_ingestion_state", "unknown"))
    if ingestion_state == "degraded":
        blocking = any(bool(item.get("required")) for item in issues)
        if blocking:
            return {
                "state": "blocked",
                "tone": "danger",
                "title": "Prediction Blocked",
                "summary": "Cycle degraded: price ingestion failed; prediction was skipped.",
                "action": (
                    "Restore price candles and let the next cycle retry before trusting signals."
                ),
                "issues": issues,
                "issue_count": len(issues),
                "updated_at": updated_at,
                "payload": heartbeat,
            }
        return {
            "state": "degraded",
            "tone": "warning",
            "title": "Cycle Degraded",
            "summary": (
                "Cycle degraded: optional news or auxiliary data failed; "
                "price-only signals may continue."
            ),
            "action": (
                "Review the failed sources, but price-only prediction flow may still be alive."
            ),
            "issues": issues,
            "issue_count": len(issues),
            "updated_at": updated_at,
            "payload": heartbeat,
        }

    if heartbeat_status == "starting":
        return {
            "state": "warming",
            "tone": "info",
            "title": "Cycle Warming",
            "summary": "Watcher started and is waiting to seal its first current cycle payload.",
            "action": "Wait for the next cycle to complete, then refresh System.",
            "issues": [],
            "issue_count": 0,
            "updated_at": updated_at,
            "payload": heartbeat,
        }

    if heartbeat_status == "stopped":
        return {
            "state": "stopped",
            "tone": "warning",
            "title": "Watcher Stopped",
            "summary": "The latest durable watcher heartbeat says monitoring is stopped.",
            "action": "Restart monitoring before expecting new proof or new signals.",
            "issues": [],
            "issue_count": 0,
            "updated_at": updated_at,
            "payload": heartbeat,
        }

    return {
        "state": "healthy",
        "tone": "success",
        "title": "Cycle Healthy",
        "summary": "Latest cycle completed with ingestion healthy.",
        "action": "Use Performance or System if you want deeper diagnostics.",
        "issues": [],
        "issue_count": 0,
        "updated_at": updated_at,
        "payload": heartbeat,
    }


def normalize_cycle_health(
    cycle_health: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a UI-safe cycle-health payload with stable keys."""
    fallback = {
        "state": "unknown",
        "tone": "info",
        "title": "Cycle Status Unavailable",
        "summary": "Cycle health details are unavailable right now.",
        "action": "Refresh the page or inspect System for the latest watcher proof.",
        "issues": [],
        "issue_count": 0,
        "updated_at": None,
        "payload": None,
    }
    if not isinstance(cycle_health, Mapping):
        return fallback

    issues = cycle_health.get("issues")
    if not isinstance(issues, list):
        issues = []

    issue_count = cycle_health.get("issue_count")
    if not isinstance(issue_count, int):
        issue_count = len(issues)

    normalized = fallback.copy()
    normalized.update({
        "state": str(cycle_health.get("state") or fallback["state"]),
        "tone": str(cycle_health.get("tone") or fallback["tone"]),
        "title": str(cycle_health.get("title") or fallback["title"]),
        "summary": str(cycle_health.get("summary") or fallback["summary"]),
        "action": str(cycle_health.get("action") or fallback["action"]),
        "issues": issues,
        "issue_count": issue_count,
        "updated_at": cycle_health.get("updated_at"),
        "payload": cycle_health.get("payload"),
    })
    return normalized


def normalize_runtime_summary(
    runtime: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a UI-safe runtime summary with stable top-level and nested keys."""
    fallback = {
        "status": "unknown",
        "tone": "info",
        "title": "Watcher Status Unavailable",
        "proof": "Last proof is unavailable right now.",
        "next": "Open System for the latest watcher details.",
        "detail": "Runtime summary data is incomplete.",
        "action": "Refresh the page or inspect System for the latest durable proof.",
        "countdown_minutes": None,
        "cycle_health": normalize_cycle_health(None),
    }
    if not isinstance(runtime, Mapping):
        return fallback

    normalized = fallback.copy()
    normalized.update({
        "status": str(runtime.get("status") or fallback["status"]),
        "tone": str(runtime.get("tone") or fallback["tone"]),
        "title": str(runtime.get("title") or fallback["title"]),
        "proof": str(runtime.get("proof") or fallback["proof"]),
        "next": str(runtime.get("next") or fallback["next"]),
        "detail": str(runtime.get("detail") or fallback["detail"]),
        "action": str(runtime.get("action") or fallback["action"]),
        "countdown_minutes": runtime.get("countdown_minutes"),
        "cycle_health": normalize_cycle_health(runtime.get("cycle_health")),
    })
    return normalized


def sanitize_heartbeat_payload(
    payload: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return a heartbeat payload safe for UI display."""
    if not isinstance(payload, Mapping):
        return None

    safe_payload: dict[str, Any] = {}
    for key in (
        "status",
        "updated_at",
        "freq",
        "horizon",
        "config_source",
        "interval_seconds",
        "cycle_prediction_state",
        "cycle_prediction_reason",
        "cycle_realization_state",
        "cycle_diagnostic",
        "cycle_ingestion_state",
    ):
        if payload.get(key) is not None:
            safe_payload[key] = payload.get(key)

    raw_failures = payload.get("cycle_ingestion_failures")
    safe_failures: list[dict[str, Any]] = []
    if isinstance(raw_failures, list):
        for failure in raw_failures:
            if not isinstance(failure, Mapping):
                continue
            safe_failures.append({
                "source": str(failure.get("source", "unknown")),
                "required": bool(failure.get("required", False)),
                "status": str(failure.get("status", "unknown")),
                "message": str(failure.get("message", "")),
            })
    if safe_failures:
        safe_payload["cycle_ingestion_failures"] = safe_failures

    raw_error = payload.get("error")
    if raw_error is not None:
        safe_payload["error_summary"] = _redact_runtime_text(str(raw_error))

    if any(payload.get(key) is not None for key in ("database_url", "config_path", "error")):
        safe_payload["sensitive_fields_hidden"] = True

    return safe_payload


def _redact_runtime_text(value: str) -> str:
    """Remove obvious secrets and paths from runtime error text before display."""
    cleaned = re.sub(r"(database_url=)\S+", r"\1<redacted>", value)
    cleaned = re.sub(r"(config_path=)\S+", r"\1<redacted>", cleaned)
    cleaned = re.sub(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://\S+", "<redacted-url>", cleaned)
    if len(cleaned) > 160:
        return f"{cleaned[:157].rstrip()}..."
    return cleaned


def cadence_minutes(freq: str | None) -> int:
    """Translate a preset/runtime frequency into minutes."""
    if freq is None:
        return 60
    return _FREQ_TO_MINUTES.get(str(freq).strip().lower(), 60)


def format_relative_time(value: Any) -> str | None:
    """Render a timestamp as a user-facing age label."""
    dt = _parse_timestamp(value)
    if dt is None:
        return None

    now = datetime.now(UTC).replace(tzinfo=None)
    delta = now - dt
    seconds = max(0, int(delta.total_seconds()))

    if seconds < 90:
        return "just now"

    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"

    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"

    days = hours // 24
    return f"{days}d ago"


def _format_minutes_window(minutes: int | None) -> str:
    """Format minutes into a compact user-facing window."""
    if minutes is None:
        return "unknown"
    if minutes <= 0:
        return "imminent"

    hours, remainder = divmod(minutes, 60)
    if hours > 0 and remainder > 0:
        return f"{hours}h {remainder}m"
    if hours > 0:
        return f"{hours}h"
    return f"{remainder}m"


def _confidence_label(confidence: float | None) -> str:
    """Convert confidence into a short plain-language label."""
    if confidence is None:
        return "unscored confidence"
    if confidence >= 0.75:
        return "very high confidence"
    if confidence >= 0.65:
        return "high confidence"
    if confidence >= 0.55:
        return "medium confidence"
    return "low confidence"


def _coerce_positive_minutes(value: Any, fallback: int) -> int:
    """Convert raw cadence values into a safe positive minute count."""
    try:
        return max(1, int(value) // 60)
    except (TypeError, ValueError):
        return max(1, fallback)


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
    heartbeat_payload = get_monitor_heartbeat(db_path)

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
        allowed_hours = 2.0
        if isinstance(heartbeat_payload, Mapping):
            interval_seconds = heartbeat_payload.get("interval_seconds")
            interval_minutes = _coerce_positive_minutes(interval_seconds, 60)
            allowed_hours = max(5, interval_minutes * 2) / 60.0

        if hours_ago < allowed_hours:
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
    last_created_at: str | None,
    interval_hours: int = 1,
    interval_minutes: int | None = None,
) -> int | None:
    """Return minutes until the next expected prediction, or None."""
    if last_created_at is None:
        return None
    try:
        last_dt = datetime.fromisoformat(last_created_at)
        if interval_minutes is None:
            delta = timedelta(hours=interval_hours)
        else:
            delta = timedelta(minutes=interval_minutes)
        next_dt = last_dt + delta
        remaining = (next_dt - datetime.now(UTC).replace(tzinfo=None)).total_seconds() / 60
        return max(0, int(remaining))
    except Exception:
        return None


def get_runtime_summary(
    db_path: Path,
    data_dir: Path,
    *,
    interval_minutes: int = 60,
) -> dict[str, Any]:
    """Build a user-facing runtime summary for Home and System surfaces."""
    system = get_system_status(db_path)
    ingestion = get_ingestion_status(data_dir)
    latest_prediction = get_latest_prediction(db_path)
    cycle_health = normalize_cycle_health(
        get_cycle_health(db_path, interval_minutes=interval_minutes)
    )

    proof_age = None
    if latest_prediction is not None:
        proof_age = format_relative_time(latest_prediction.get("created_at"))
    if proof_age is None and system["hours_ago"] is not None:
        proof_age = format_relative_time(
            datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=system["hours_ago"])
        )

    countdown = minutes_until_next_prediction(
        latest_prediction.get("created_at") if latest_prediction else None,
        interval_minutes=interval_minutes,
    )
    prediction_exists = latest_prediction is not None
    price_ready = "Fresh" in ingestion["prices"]
    news_ready = "Fresh" in ingestion["news"] or "ago" in ingestion["news"]

    if system["status"] == "active" and cycle_health["state"] in {"blocked", "error", "degraded"}:
        return normalize_runtime_summary({
            "status": str(cycle_health["state"]),
            "tone": str(cycle_health["tone"]),
            "title": str(cycle_health["title"]),
            "proof": f"Last proof: watcher checked in {proof_age or 'recently'}.",
            "next": str(cycle_health["action"]),
            "detail": str(cycle_health["summary"]),
            "action": "Open System for the failure list, raw cycle payload, and recent proof.",
            "countdown_minutes": countdown,
            "cycle_health": cycle_health,
        })

    if system["status"] == "active" and prediction_exists:
        latest = latest_prediction or {}
        direction = str(latest.get("direction", "flat")).upper()
        confidence = _confidence_label(_to_float(latest.get("confidence")))
        return normalize_runtime_summary({
            "status": "active",
            "tone": "success",
            "title": "Watcher Active",
            "proof": f"Last proof: signal sealed {proof_age or 'recently'}.",
            "next": f"Next cycle: {_format_minutes_window(countdown)}.",
            "detail": f"Latest signal: {direction} with {confidence}.",
            "action": "Use System for raw logs and watcher diagnostics.",
            "countdown_minutes": countdown,
            "cycle_health": cycle_health,
        })

    if system["status"] == "active":
        return normalize_runtime_summary({
            "status": "warming",
            "tone": "info",
            "title": "Watcher Warming",
            "proof": f"Last proof: watcher checked in {proof_age or 'recently'}.",
            "next": "Waiting for the first completed prediction cycle.",
            "detail": "Training or monitoring is alive, but no durable signal is sealed yet.",
            "action": "Keep Quick Start open for progress or check System for recent events.",
            "countdown_minutes": countdown,
            "cycle_health": cycle_health,
        })

    if system["status"] == "idle":
        return normalize_runtime_summary({
            "status": "idle",
            "tone": "warning",
            "title": "Watcher Idle",
            "proof": f"Last proof: activity seen {proof_age or 'a while ago'}.",
            "next": "Open System to inspect logs or restart the watcher.",
            "detail": "BitBat has durable history, but the live watcher does not look current.",
            "action": "Confirm the watcher process is still running.",
            "countdown_minutes": countdown,
            "cycle_health": cycle_health,
        })

    if db_path.exists():
        return normalize_runtime_summary({
            "status": "warming",
            "tone": "warning",
            "title": "Watcher Unproven",
            "proof": "Last proof: database exists, but no durable watcher activity was found.",
            "next": "Start monitoring after training so the first cycle can write proof.",
            "detail": (
                f"Price candles: {ingestion['prices']} • News signals: {ingestion['news']}."
            ),
            "action": "Use Quick Start to train or restart the watcher.",
            "countdown_minutes": countdown,
            "cycle_health": cycle_health,
        })

    data_detail = (
        "Fresh market data found." if price_ready or news_ready else "No runtime data yet."
    )
    return normalize_runtime_summary({
        "status": "dormant",
        "tone": "warning",
        "title": "Watcher Dormant",
        "proof": "Last proof: no database, prediction, or heartbeat was found.",
        "next": "Train a model on Quick Start, then start monitoring.",
        "detail": data_detail,
        "action": "BitBat has not completed its first durable run on this machine.",
        "countdown_minutes": countdown,
        "cycle_health": cycle_health,
    })


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
