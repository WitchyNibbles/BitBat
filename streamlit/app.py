"""
BitBat — Bitcoin Price Prediction System
Home dashboard (simplified for non-technical users).

Session 2 additions: auto-refresh, activity feed, countdown timer.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from html import escape
from pathlib import Path

import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from style import inject_css  # noqa: E402

from bitbat.gui.presets import DEFAULT_PRESET, get_preset  # noqa: E402
from bitbat.gui.widgets import (  # noqa: E402
    cadence_minutes,
    format_local_timestamp,
    format_relative_time,
    get_ingestion_status,
    get_latest_prediction,
    get_recent_events,
    get_runtime_summary,
    get_system_status,
    normalize_runtime_summary,
    render_prediction_card,
)

st.set_page_config(
    page_title="BitBat — Bitcoin Predictions",
    page_icon="🦇",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ------------------------------------------------------------------
# Home content with home-only auto-rerun (every 60 seconds)
# ------------------------------------------------------------------
_REFRESH_INTERVAL = 60  # seconds


def _render_runtime_hero(
    *,
    preset_label: str,
    runtime: dict[str, object],
    cadence_label: str,
) -> None:
    st.markdown(
        (
            "<div class='bb-hero'>"
            "<div class='bb-kicker'>BitBat Command Center</div>"
            f"<div class='bb-state-chip' data-tone='{escape(str(runtime['tone']))}'>"
            f"{escape(str(runtime['title']))}</div>"
            f"<h1 class='bb-title'>{escape(preset_label)}</h1>"
            f"<div class='bb-subtitle'>{escape(str(runtime['proof']))}</div>"
            f"<div class='bb-subtitle'>{escape(str(runtime['next']))}</div>"
            f"<div class='bb-subtitle'>{escape(str(runtime['detail']))}</div>"
            f"<div class='bb-subtitle'>Cadence: {escape(cadence_label)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_story_cards(cards: list[tuple[str, str, str]]) -> None:
    card_markup = []
    for label, value, copy in cards:
        card_markup.append(
            "<div class='bb-card'>"
            f"<div class='bb-label'>{escape(label)}</div>"
            f"<div class='bb-value'>{escape(value)}</div>"
            f"<div class='bb-copy'>{escape(copy)}</div>"
            "</div>"
        )
    st.markdown(f"<div class='bb-grid'>{''.join(card_markup)}</div>", unsafe_allow_html=True)


def _render_event_feed(events: list[dict[str, object]]) -> None:
    if not events:
        st.info("No omen feed yet. The watcher will log proof here after the first cycle.")
        return

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


@st.fragment(run_every=_REFRESH_INTERVAL)
def _render_home() -> None:  # noqa: C901
    # ------------------------------------------------------------------
    # Load user preset
    # ------------------------------------------------------------------
    _user_config_path = ROOT / "config" / "user_config.yaml"
    _current_preset_name = DEFAULT_PRESET
    if _user_config_path.exists():
        try:
            _cfg = yaml.safe_load(_user_config_path.read_text())
            _current_preset_name = _cfg.get("preset", DEFAULT_PRESET)
        except Exception:
            _current_preset_name = DEFAULT_PRESET

    current_preset = get_preset(_current_preset_name)
    cadence = cadence_minutes(current_preset.freq)
    forecast_horizon_minutes = cadence_minutes(current_preset.horizon)
    _DB = ROOT / "data" / "autonomous.db"
    _DATA_DIR = ROOT / "data"

    runtime = normalize_runtime_summary(
        get_runtime_summary(_DB, _DATA_DIR, interval_minutes=cadence)
    )
    cycle_health = runtime["cycle_health"]
    sys_status = get_system_status(_DB)
    ingest_status = get_ingestion_status(_DATA_DIR)
    latest_pred = get_latest_prediction(_DB)

    _render_runtime_hero(
        preset_label=f"{current_preset.icon} {current_preset.name} Watch",
        runtime=runtime,
        cadence_label=current_preset.to_display()["Update Frequency"],
    )

    _render_story_cards([
        ("State", str(runtime["title"]), str(runtime["action"])),
        (
            "Cycle Health",
            str(cycle_health["title"]),
            str(cycle_health["summary"]),
        ),
        ("Price Candles", ingest_status["prices"], "Fresh candles keep the watcher grounded."),
        (
            "News Signals",
            ingest_status["news"],
            "Sentiment data can lag without breaking price-only runs.",
        ),
        (
            "Durable Watcher",
            sys_status["label"],
            "This is the stored health signal, not just the browser session.",
        ),
    ])

    if cycle_health["state"] in {"blocked", "error"}:
        st.error(str(cycle_health["summary"]))
    elif cycle_health["state"] == "degraded":
        st.warning(str(cycle_health["summary"]))

    st.header("Latest Signal")

    if latest_pred:
        pred_col, meter_col = st.columns([2, 1])
        with pred_col:
            direction = latest_pred.get("direction", "flat")
            confidence = latest_pred.get("confidence")
            if direction == "up":
                st.success("### Signal points UP")
            elif direction == "down":
                st.error("### Signal points DOWN")
            else:
                st.info("### Signal points FLAT")
            if confidence is None:
                st.markdown("**Confidence:** n/a")
            else:
                st.markdown(f"**Confidence:** {confidence:.0%}")
            sealed_age = format_relative_time(latest_pred.get("created_at")) or "at an unknown time"
            signal_time = format_local_timestamp(latest_pred.get("timestamp_utc")) or "unknown"
            target_time = None
            raw_signal_time = latest_pred.get("timestamp_utc")
            if raw_signal_time is not None:
                try:
                    signal_dt = datetime.fromisoformat(str(raw_signal_time).replace("Z", "+00:00"))
                    target_time = format_local_timestamp(
                        signal_dt + timedelta(minutes=forecast_horizon_minutes)
                    )
                except ValueError:
                    target_time = None
            st.caption(
                f"Signal at {signal_time}"
                + (f" • Targets {target_time}" if target_time else "")
                + " • "
                f"Model {latest_pred.get('model_version', 'unknown')} • "
                f"Sealed {sealed_age}"
            )

        with meter_col:
            st.markdown("**Signal Meter**")
            render_prediction_card(latest_pred)
    else:
        if not _DB.exists():
            st.info("No signal ledger exists yet. Train a model first, then start the watcher.")
        else:
            st.info("No signal sealed yet. BitBat is still warming up its first durable cycle.")

    st.header("Operator Actions")

    act1, act2, act3, act4, act5 = st.columns(5)
    with act1:
        if st.button("Open Quick Start", width="stretch", type="primary"):
            st.switch_page("pages/0_Quick_Start.py")
    with act2:
        if st.button("Open Performance", width="stretch"):
            st.switch_page("pages/2_📈_Performance.py")
    with act3:
        if st.button("Open Settings", width="stretch"):
            st.switch_page("pages/1_⚙️_Settings.py")
    with act4:
        if st.button("Open Field Guide", width="stretch"):
            st.switch_page("pages/3_ℹ️_About.py")
    with act5:
        if st.button("Open System", width="stretch"):
            st.switch_page("pages/4_🔧_System.py")

    st.header("Omen Feed")
    events = get_recent_events(_DB, limit=10)
    _render_event_feed(events)

    if runtime["status"] in {"dormant", "warming"} or sys_status["status"] == "not_started":
        st.header("First Ritual")
        st.markdown(
            "Choose a preset, train once, then start the durable watcher. "
            "The browser preview helps with first-run checks, but 24/7 monitoring comes from "
            "the standalone runner."
        )
        if st.button("Begin in Quick Start", type="primary"):
            st.switch_page("pages/0_Quick_Start.py")

    st.divider()
    st.caption(
        f"BitBat • {current_preset.name} mode • refresh every {_REFRESH_INTERVAL}s • "
        f"last refreshed {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )


_render_home()
