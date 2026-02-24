"""Quick Start page — train a model and see predictions in one click."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="Quick Start — BitBat", page_icon="🦇", layout="wide")

from bitbat.gui.presets import PRESETS, get_preset  # noqa: E402

# Global stop event for the monitoring loop so we can start/stop it from the UI.
_MONITOR_STOP_EVENT = threading.Event()

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
_DATA_DIR = Path("data")
_DB_PATH = _DATA_DIR / "autonomous.db"


def _model_exists(freq: str, horizon: str) -> bool:
    return (Path("models") / f"{freq}_{horizon}" / "xgb.json").exists()


# ------------------------------------------------------------------
# State helpers
# ------------------------------------------------------------------
if "train_state" not in st.session_state:
    st.session_state["train_state"] = "INITIAL"
    # Only auto-detect on the very first page load
    for preset in PRESETS.values():
        if _model_exists(preset.freq, preset.horizon):
            st.session_state["train_state"] = "RUNNING"
            st.session_state["active_freq"] = preset.freq
            st.session_state["active_horizon"] = preset.horizon
            break

# ------------------------------------------------------------------
# Page header
# ------------------------------------------------------------------
st.title("BitBat Quick Start")

state = st.session_state["train_state"]

# ==================================================================
# INITIAL: Preset selection + Train button
# ==================================================================
if state == "INITIAL":
    st.markdown(
        "Train a Bitcoin prediction model in one click. "
        "Choose a trading style, press **Train**, and BitBat handles the rest."
    )

    preset_names = list(PRESETS.keys())
    labels = [f"{p.icon} {p.name}" for p in PRESETS.values()]

    choice = st.radio(
        "Choose a trading style:",
        labels,
        index=1,  # Balanced
        horizontal=True,
    )

    # Map label back to key
    selected_key = preset_names[labels.index(choice)]
    preset = get_preset(selected_key)

    # Show preset details
    with st.expander("Preset details", expanded=False):
        for label, value in preset.to_display().items():
            st.markdown(f"**{label}:** {value}")
        st.caption(preset.description)

    if st.button("Train Model", type="primary", width="stretch"):
        st.session_state["train_state"] = "TRAINING"
        st.session_state["selected_preset"] = selected_key
        st.rerun()

# ==================================================================
# TRAINING: Progress bar
# ==================================================================
elif state == "TRAINING":
    preset_key = st.session_state.get("selected_preset", "balanced")
    preset = get_preset(preset_key)

    st.markdown(f"Training **{preset.name}** model...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    def _update(msg: str, frac: float) -> None:
        progress_bar.progress(min(frac, 1.0))
        status_text.text(msg)

    from bitbat.autonomous.orchestrator import one_click_train

    result = one_click_train(
        preset_name=preset_key,
        progress_callback=_update,
    )

    st.session_state["train_result"] = result

    if result["status"] == "success":
        st.session_state["train_state"] = "RUNNING"
        st.session_state["active_freq"] = result["freq"]
        st.session_state["active_horizon"] = result["horizon"]
        st.rerun()
    else:
        st.session_state["train_state"] = "INITIAL"
        st.error(
            f"Training failed at step **{result.get('step', '?')}**: "
            f"{result.get('error', 'unknown error')}"
        )
        if st.button("Try Again"):
            st.rerun()

# ==================================================================
# RUNNING: Timeline + monitoring
# ==================================================================
elif state == "RUNNING":
    freq = st.session_state.get("active_freq", "1h")
    horizon = st.session_state.get("active_horizon", "4h")

    # Show training result banner (if just trained)
    result = st.session_state.get("train_result")
    if result and result.get("status") == "success":
        st.success(
            f"Model **{result['model_version']}** trained on "
            f"**{result['training_samples']:,}** samples in "
            f"**{result['duration_seconds']:.0f}s**."
        )

    # Show active model info
    model_path = Path("models") / f"{freq}_{horizon}" / "xgb.json"
    if model_path.exists():
        import datetime as _dt

        mtime = _dt.datetime.fromtimestamp(model_path.stat().st_mtime)
        st.info(
            f"Active model: **{freq}/{horizon}** "
            f"(last trained {mtime:%Y-%m-%d %H:%M})"
        )
    else:
        st.warning("No trained model found. Click **Retrain Model** below to train one.")

    # ------------------------------------------------------------------
    # Monitoring controls (start/stop + schedule info)
    # ------------------------------------------------------------------
    from bitbat.config.loader import get_runtime_config, load_config  # noqa: E402

    config = get_runtime_config() or load_config()
    autonomous_cfg = config.get("autonomous", {}) or {}
    db_url = str(autonomous_cfg.get("database_url", "sqlite:///data/autonomous.db"))
    interval = int(autonomous_cfg.get("validation_interval", 3600))

    status_cols = st.columns(3)
    with status_cols[0]:
        status_label = "Running" if st.session_state.get("_monitor_running", False) else "Stopped"
        st.metric("Monitoring Status", status_label)
    with status_cols[1]:
        st.metric("Validation Interval", f"{interval} s")
    with status_cols[2]:
        st.caption(f"DB URL: `{db_url}`")

    def _start_monitoring(f: str, h: str) -> None:
        """Start background monitoring loop if not already running."""
        if st.session_state.get("_monitor_running", False):
            return

        _MONITOR_STOP_EVENT.clear()

        def _monitor_loop(freq_value: str, horizon_value: str) -> None:
            from bitbat.autonomous.agent import MonitoringAgent
            from bitbat.autonomous.db import AutonomousDB
            from bitbat.autonomous.models import init_database
            from bitbat.config.loader import get_runtime_config, load_config

            cfg = get_runtime_config() or load_config()
            auto_cfg = cfg.get("autonomous", {}) or {}
            local_db_url = str(auto_cfg.get("database_url", "sqlite:///data/autonomous.db"))
            local_interval = int(auto_cfg.get("validation_interval", 3600))

            init_database(local_db_url)
            db = AutonomousDB(local_db_url)
            agent = MonitoringAgent(db, freq=freq_value, horizon=horizon_value)

            import contextlib

            while not _MONITOR_STOP_EVENT.is_set():
                with contextlib.suppress(Exception):
                    agent.run_once()
                # Use Event.wait so we can wake up promptly when stopping.
                _MONITOR_STOP_EVENT.wait(local_interval)

        thread = threading.Thread(
            target=_monitor_loop,
            args=(f, h),
            daemon=True,
            name="bitbat-monitor",
        )
        thread.start()
        st.session_state["_monitor_running"] = True

    control_cols = st.columns(2)
    with control_cols[0]:
        if not st.session_state.get("_monitor_running", False):
            if st.button("Start Monitoring", type="primary", width="stretch"):
                _start_monitoring(freq, horizon)
                st.success("Monitoring loop started.")
        else:
            st.caption("Monitoring loop is currently running.")

    with control_cols[1]:
        if st.session_state.get("_monitor_running", False):
            if st.button("Stop Monitoring", width="stretch"):
                _MONITOR_STOP_EVENT.set()
                st.session_state["_monitor_running"] = False
                st.info("Monitoring loop will stop after the current cycle.")
        else:
            st.caption("Press **Start Monitoring** to begin live predictions.")

    # ------------------------------------------------------------------
    # Auto-refreshing prediction timeline
    # ------------------------------------------------------------------
    @st.fragment(run_every=60)
    def _live_timeline() -> None:
        from bitbat.gui.timeline import (
            apply_timeline_filters,
            build_timeline_figure,
            format_timeline_empty_state,
            get_price_series,
            get_timeline_data,
            list_timeline_filter_options,
            summarize_timeline_insights,
        )

        timeline_freq_options, timeline_horizon_options = list_timeline_filter_options(
            _DB_PATH,
            freq,
            horizon,
        )
        date_window_options = ["24h", "7d", "30d", "all"]

        if "timeline_filter_freq" not in st.session_state:
            st.session_state["timeline_filter_freq"] = freq
        if "timeline_filter_horizon" not in st.session_state:
            st.session_state["timeline_filter_horizon"] = horizon
        if "timeline_filter_window" not in st.session_state:
            st.session_state["timeline_filter_window"] = "7d"
        if "timeline_show_overlay" not in st.session_state:
            st.session_state["timeline_show_overlay"] = True

        if st.session_state["timeline_filter_freq"] not in timeline_freq_options:
            st.session_state["timeline_filter_freq"] = freq
        if st.session_state["timeline_filter_horizon"] not in timeline_horizon_options:
            st.session_state["timeline_filter_horizon"] = horizon
        if st.session_state["timeline_filter_window"] not in date_window_options:
            st.session_state["timeline_filter_window"] = "7d"

        filter_cols = st.columns(4)
        with filter_cols[0]:
            selected_freq = st.selectbox(
                "Timeline Freq",
                options=timeline_freq_options,
                key="timeline_filter_freq",
            )
        with filter_cols[1]:
            selected_horizon = st.selectbox(
                "Timeline Horizon",
                options=timeline_horizon_options,
                key="timeline_filter_horizon",
            )
        with filter_cols[2]:
            selected_window = st.selectbox(
                "Date Window",
                options=date_window_options,
                key="timeline_filter_window",
            )
        with filter_cols[3]:
            show_overlay = st.toggle(
                "Overlay Compare",
                key="timeline_show_overlay",
            )

        predictions = get_timeline_data(_DB_PATH, selected_freq, selected_horizon, limit=2000)
        predictions = apply_timeline_filters(predictions, date_window=selected_window)

        if predictions.empty:
            st.info(format_timeline_empty_state(selected_freq, selected_horizon, selected_window))
            return

        first_ts = predictions["timestamp_utc"].min()
        prices = get_price_series(_DATA_DIR, selected_freq, first_ts)

        fig = build_timeline_figure(predictions, prices, show_overlay=show_overlay)
        st.plotly_chart(fig, width="stretch")

        # Legend
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("Green = predicted UP | Red = predicted DOWN | Gray = FLAT")
        with c2:
            st.caption("Status: Pending | Realized (Correct) | Realized (Wrong)")
        with c3:
            st.caption("Auto-refreshes every 60 seconds")

        # Summary metrics
        insights = summarize_timeline_insights(predictions)
        avg_confidence = insights["average_confidence"]

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Total Predictions", int(insights["total"]))
        with m2:
            st.metric("Completed", int(insights["completed"]))
        with m3:
            st.metric("Correct", int(insights["correct"]))
        with m4:
            st.metric("Accuracy", f"{insights['accuracy']:.1f}%")
        with m5:
            if avg_confidence is None:
                st.metric("Avg Confidence", "n/a")
            else:
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")

    _live_timeline()

    # ------------------------------------------------------------------
    # Retrain button
    # ------------------------------------------------------------------
    st.divider()
    if st.button("Retrain Model", type="primary", width="stretch"):
        # Reset monitor so it restarts with new model after retraining
        st.session_state["_monitor_running"] = False
        st.session_state["train_result"] = None
        st.session_state["train_state"] = "INITIAL"
        st.rerun()
