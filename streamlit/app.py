"""
BitBat — Bitcoin Price Prediction System
Home dashboard (simplified for non-technical users).

Session 2 additions: auto-refresh, activity feed, countdown timer.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bitbat.gui.presets import DEFAULT_PRESET, get_preset
from bitbat.gui.widgets import (
    get_ingestion_status,
    get_latest_prediction,
    get_recent_events,
    get_system_status,
    minutes_until_next_prediction,
    render_countdown,
    render_prediction_card,
)

st.set_page_config(
    page_title="BitBat — Bitcoin Predictions",
    page_icon="🦇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Home content with home-only auto-rerun (every 60 seconds)
# ------------------------------------------------------------------
_REFRESH_INTERVAL = 60  # seconds


@st.fragment(run_every=_REFRESH_INTERVAL)
def _render_home() -> None:
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
            pass

    current_preset = get_preset(_current_preset_name)
    _DB = ROOT / "data" / "autonomous.db"
    _DATA_DIR = ROOT / "data"

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    st.title("🦇 BitBat")
    st.markdown(
        f"**Bitcoin Price Prediction System** &nbsp;•&nbsp; "
        f"{current_preset.icon} {current_preset.name} Mode"
    )

    # ------------------------------------------------------------------
    # System status
    # ------------------------------------------------------------------
    st.header("System Status")
    sys_status = get_system_status(_DB)
    ingest_status = get_ingestion_status(_DATA_DIR)

    latest_pred = get_latest_prediction(_DB)

    stat1, stat2, stat3, stat4 = st.columns(4)
    with stat1:
        st.metric("Monitoring", sys_status["label"])
    with stat2:
        st.metric("Price Data", ingest_status["prices"])
    with stat3:
        st.metric("News Data", ingest_status["news"])
    with stat4:
        mins = minutes_until_next_prediction(
            latest_pred["created_at"] if latest_pred else None
        )
        if mins is not None:
            render_countdown(mins)
        else:
            st.metric("Next Prediction", "—")

    # ------------------------------------------------------------------
    # Latest prediction
    # ------------------------------------------------------------------
    st.header("Latest Prediction")

    if latest_pred:
        pred_col, meter_col = st.columns([2, 1])
        with pred_col:
            direction = latest_pred["direction"]
            confidence = latest_pred["confidence"]
            if direction == "up":
                st.success("### 📈 Bitcoin will likely go **UP**")
            elif direction == "down":
                st.error("### 📉 Bitcoin will likely go **DOWN**")
            else:
                st.info("### ➡️ Bitcoin will likely stay **FLAT**")
            st.markdown(f"**Confidence:** {confidence:.0%}")
            st.caption(
                f"Forecast for: {latest_pred['timestamp_utc']}  |  "
                f"Model: {latest_pred['model_version']}"
            )

        with meter_col:
            st.markdown("**Confidence Meter**")
            render_prediction_card(latest_pred)
    else:
        if not _DB.exists():
            st.info(
                "Database not found. Have you started the monitoring system?  \n"
                "See **Getting Started** below."
            )
        else:
            st.info("No predictions yet — the system is still warming up.")

    # ------------------------------------------------------------------
    # Quick actions
    # ------------------------------------------------------------------
    st.header("Quick Actions")

    act1, act2, act3 = st.columns(3)
    with act1:
        if st.button("📈 View Performance", width="stretch"):
            st.switch_page("pages/2_📈_Performance.py")
    with act2:
        if st.button("⚙️ Change Settings", width="stretch"):
            st.switch_page("pages/1_⚙️_Settings.py")
    with act3:
        if st.button("❓ Help & About", width="stretch"):
            st.switch_page("pages/3_ℹ️_About.py")

    # ------------------------------------------------------------------
    # Recent activity feed
    # ------------------------------------------------------------------
    st.header("Recent Activity")
    events = get_recent_events(_DB, limit=10)

    if events:
        level_icons = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌", "DEBUG": "🔍"}
        for ev in events:
            icon = level_icons.get(str(ev["level"]).upper(), "•")
            st.markdown(f"{icon} `{ev['time']}` — {ev['message']}")
    else:
        st.caption("No recent activity yet. Activity will appear here once the monitoring system is running.")

    # ------------------------------------------------------------------
    # Getting started (shown when system isn't running)
    # ------------------------------------------------------------------
    if not _DB.exists() or sys_status["status"] == "not_started":
        st.header("🚀 Getting Started")
        st.markdown(
            """
Welcome to **BitBat**! Here's how to get started in 3 steps:

**1. Configure Settings** *(optional)*
- Visit the **Settings** page and choose a preset.
- **Balanced** is recommended for most users.

**2. Start the Monitoring System**
```bash
# Option A — Docker (recommended):
docker-compose up -d

# Option B — directly:
poetry run python scripts/run_monitoring_agent.py
```

**3. Wait for Predictions**
- The system collects data for ~1 hour before making its first prediction.
- This page refreshes automatically every 60 seconds.
- Come back here to check for updates!

*Need help? Visit the **About** page.*
"""
        )

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    st.divider()
    st.caption(
        f"BitBat • {current_preset.name} mode • "
        f"Auto-refresh every {_REFRESH_INTERVAL}s • "
        f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )


_render_home()
