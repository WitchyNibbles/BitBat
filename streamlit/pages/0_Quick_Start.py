"""Quick Start page — train a model and see predictions in one click."""

from __future__ import annotations

import sys
import threading
from html import escape
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1].parent
STREAMLIT_DIR = ROOT / "streamlit"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(STREAMLIT_DIR))

st.set_page_config(page_title="Quick Start — BitBat", page_icon="🦇", layout="wide")

from style import inject_css  # noqa: E402

from bitbat.gui.presets import PRESETS, get_preset  # noqa: E402
from bitbat.gui.widgets import (  # noqa: E402
    cadence_minutes,
    get_runtime_summary,
    normalize_runtime_summary,
)

inject_css()

# Global stop event for the monitoring loop so we can start/stop it from the UI.
_MONITOR_STOP_EVENT = threading.Event()

_DATA_DIR = Path("data")
_DB_PATH = _DATA_DIR / "autonomous.db"
_TRAINING_STAGES = [
    "Loading configuration",
    "Gathering price candles",
    "Collecting news signals",
    "Binding features",
    "Training model",
    "Registering version",
    "Casting first prediction",
]


def _model_exists(freq: str, horizon: str) -> bool:
    return (Path("models") / f"{freq}_{horizon}" / "xgb.json").exists()


def _redact_db_locator(db_url: str) -> str:
    if ":///" in db_url:
        scheme, raw_path = db_url.split(":///", maxsplit=1)
        return f"{scheme}:///.../{Path(raw_path).name}"
    if "://" in db_url:
        scheme, _rest = db_url.split("://", maxsplit=1)
        return f"{scheme}://<redacted>"
    return db_url


def _render_hero(title: str, subtitle: str, badge: str) -> None:
    st.markdown(
        (
            "<div class='bb-hero'>"
            "<div class='bb-kicker'>Quick Start</div>"
            f"<div class='bb-state-chip' data-tone='info'>{escape(badge)}</div>"
            f"<h1 class='bb-title'>{escape(title)}</h1>"
            f"<div class='bb-subtitle'>{escape(subtitle)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


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


def _stage_for_message(message: str) -> str:
    lowered = message.lower()
    if "loading configuration" in lowered:
        return "Loading configuration"
    if "price data" in lowered:
        return "Gathering price candles"
    if "news data" in lowered or "ingestion complete" in lowered:
        return "Collecting news signals"
    if "features" in lowered:
        return "Binding features"
    if "training xgboost model" in lowered or "model trained" in lowered:
        return "Training model"
    if "registering model" in lowered or "model registered" in lowered:
        return "Registering version"
    if "first prediction" in lowered or "done!" in lowered:
        return "Casting first prediction"
    return "Preparing run"


def _render_training_ledger(completed: list[str], current: str) -> None:
    rows = []
    for stage in _TRAINING_STAGES:
        if stage in completed:
            state = "sealed"
        elif stage == current:
            state = "in motion"
        else:
            state = "waiting"
        rows.append(
            "<div class='bb-ledger-row'>"
            f"<div class='bb-ledger-step'>{escape(stage)}</div>"
            f"<div class='bb-ledger-state'>{escape(state)}</div>"
            "</div>"
        )
    st.markdown(f"<div class='bb-ledger'>{''.join(rows)}</div>", unsafe_allow_html=True)


def _find_active_preset_key(freq: str, horizon: str) -> str:
    for key, preset in PRESETS.items():
        if preset.freq == freq and preset.horizon == horizon:
            return key
    return "balanced"


if "train_state" not in st.session_state:
    st.session_state["train_state"] = "INITIAL"
    for preset in PRESETS.values():
        if _model_exists(preset.freq, preset.horizon):
            st.session_state["train_state"] = "RUNNING"
            st.session_state["active_freq"] = preset.freq
            st.session_state["active_horizon"] = preset.horizon
            st.session_state["selected_preset"] = _find_active_preset_key(
                preset.freq, preset.horizon
            )
            break

if "train_stage_history" not in st.session_state:
    st.session_state["train_stage_history"] = []

state = st.session_state["train_state"]

if state == "INITIAL":
    _render_hero(
        "Train your first watcher",
        (
            "Choose a trading style, train once, then let BitBat write durable proof "
            "into the signal ledger."
        ),
        "First Ritual",
    )

    preset_names = list(PRESETS.keys())
    labels = [f"{p.icon} {p.name}" for p in PRESETS.values()]

    choice = st.radio(
        "Choose a trading style:",
        labels,
        index=1,
        horizontal=True,
    )
    selected_key = preset_names[labels.index(choice)]
    preset = get_preset(selected_key)

    _render_story_cards([
        (
            "Update Frequency",
            preset.to_display()["Update Frequency"],
            "How often BitBat will attempt a new signal.",
        ),
        (
            "Forecast Period",
            preset.to_display()["Forecast Period"],
            "How far ahead the signal is aimed.",
        ),
        (
            "Confidence Gate",
            preset.to_display()["Confidence Required"],
            "Lower thresholds create more signals, but noisier ones.",
        ),
    ])

    with st.expander("Preset details", expanded=False):
        for label, value in preset.to_display().items():
            st.markdown(f"**{label}:** {value}")
        st.caption(preset.description)

    if st.button("Train Model", type="primary", width="stretch"):
        st.session_state["train_state"] = "TRAINING"
        st.session_state["selected_preset"] = selected_key
        st.session_state["train_stage_history"] = []
        st.rerun()

elif state == "TRAINING":
    preset_key = st.session_state.get("selected_preset", "balanced")
    preset = get_preset(preset_key)

    _render_hero(
        f"{preset.icon} {preset.name} training run",
        (
            "BitBat is stepping through data, features, model fitting, version "
            "registration, and a first prediction attempt."
        ),
        "Run in Progress",
    )

    progress_bar = st.progress(0)
    status_text = st.empty()
    ledger_placeholder = st.empty()
    completed_stages: list[str] = []
    current_stage = ["Loading configuration"]

    def _update(message: str, fraction: float) -> None:
        stage = _stage_for_message(message)
        current_stage[0] = stage
        if stage not in completed_stages and stage in _TRAINING_STAGES:
            completed_stages.append(stage)
        st.session_state["train_stage_history"] = completed_stages.copy()
        progress_bar.progress(min(fraction, 1.0))
        status_text.markdown(f"**Now:** {stage}  \n{message}")
        with ledger_placeholder.container():
            _render_training_ledger(completed_stages, current_stage[0])

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

    st.session_state["train_state"] = "INITIAL"
    st.error(
        f"Training failed at step **{result.get('step', '?')}**: "
        f"{result.get('error', 'unknown error')}"
    )
    st.info("Review the step ledger above, fix the underlying data/runtime issue, then try again.")
    if st.button("Try Again"):
        st.rerun()

elif state == "RUNNING":
    freq = st.session_state.get("active_freq", "1h")
    horizon = st.session_state.get("active_horizon", "4h")
    cadence = cadence_minutes(freq)
    runtime = normalize_runtime_summary(
        get_runtime_summary(_DB_PATH, _DATA_DIR, interval_minutes=cadence)
    )
    cycle_health = runtime["cycle_health"]
    preset_key = st.session_state.get("selected_preset", _find_active_preset_key(freq, horizon))
    preset = get_preset(preset_key)

    _render_hero(
        f"{preset.icon} {preset.name} watcher",
        f"{runtime['proof']} {runtime['next']}",
        str(runtime["title"]),
    )

    result = st.session_state.get("train_result")
    if result and result.get("status") == "success":
        st.success(
            f"Model {result['model_version']} trained on {result['training_samples']:,} samples in "
            f"{result['duration_seconds']:.0f}s."
        )

    model_path = Path("models") / f"{freq}_{horizon}" / "xgb.json"
    model_stamp = "No trained model found"
    if model_path.exists():
        import datetime as _dt

        model_stamp = _dt.datetime.fromtimestamp(model_path.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M"
        )

    from bitbat.config.loader import get_runtime_config, load_config  # noqa: E402

    config = get_runtime_config() or load_config()
    autonomous_cfg = config.get("autonomous", {}) or {}
    db_url = str(autonomous_cfg.get("database_url", "sqlite:///data/autonomous.db"))
    interval_seconds = int(autonomous_cfg.get("validation_interval", 3600))

    _render_story_cards([
        ("Model Window", f"{freq} / {horizon}", "The active frequency and forecast horizon."),
        (
            "Last Training",
            model_stamp,
            "When the current watcher model was last written to disk.",
        ),
        (
            "Watcher Interval",
            f"{interval_seconds}s",
            "How often the durable watcher aims to validate and predict.",
        ),
        (
            "Signal Ledger",
            _redact_db_locator(db_url),
            "Where BitBat stores durable predictions, logs, and model versions.",
        ),
        (
            "Cycle Health",
            str(cycle_health["title"]),
            str(cycle_health["summary"]),
        ),
    ])

    st.info(
        "The session preview below is temporary and ends with this browser session. "
        "For 24/7 monitoring, run `poetry run python scripts/run_monitoring_agent.py` "
        "or install the bundled service from `deployment/bitbat-monitor.service`."
    )

    session_running = st.session_state.get("_monitor_running", False)
    status_cols = st.columns(2)
    with status_cols[0]:
        st.metric("Session Preview", "Running" if session_running else "Stopped")
        if session_running:
            st.caption("This browser tab launched a preview loop and is not durable 24/7.")
        else:
            st.caption("No preview loop is attached to this browser session.")
    with status_cols[1]:
        st.metric("Durable Runtime", str(runtime["title"]))
        st.caption(str(runtime["action"]))

    if cycle_health["state"] in {"blocked", "error"}:
        st.error(str(cycle_health["summary"]))
        st.caption(str(cycle_health["action"]))
    elif cycle_health["state"] == "degraded":
        st.warning(str(cycle_health["summary"]))
        st.caption(str(cycle_health["action"]))

    def _start_monitoring(freq_value: str, horizon_value: str) -> None:
        if st.session_state.get("_monitor_running", False):
            return

        _MONITOR_STOP_EVENT.clear()

        def _monitor_loop(f_value: str, h_value: str) -> None:
            from bitbat.autonomous.agent import MonitoringAgent
            from bitbat.autonomous.db import AutonomousDB
            from bitbat.autonomous.heartbeat import (
                heartbeat_path_for_db_url,
                write_monitor_heartbeat,
            )
            from bitbat.autonomous.models import init_database
            from bitbat.config.loader import (
                get_runtime_config,
                get_runtime_config_path,
                get_runtime_config_source,
                load_config,
            )

            cfg = get_runtime_config() or load_config()
            auto_cfg = cfg.get("autonomous", {}) or {}
            local_db_url = str(auto_cfg.get("database_url", "sqlite:///data/autonomous.db"))
            local_interval = int(auto_cfg.get("validation_interval", 3600))
            heartbeat = heartbeat_path_for_db_url(local_db_url)
            config_source = get_runtime_config_source()
            config_path = str(get_runtime_config_path())

            init_database(local_db_url)
            db = AutonomousDB(local_db_url)
            agent = MonitoringAgent(db, freq=f_value, horizon=h_value)
            write_monitor_heartbeat(
                heartbeat,
                status="starting",
                freq=f_value,
                horizon=h_value,
                interval=local_interval,
                db_url=local_db_url,
                config_source=config_source,
                config_path=config_path,
            )

            while not _MONITOR_STOP_EVENT.is_set():
                try:
                    result = agent.run_once()
                    write_monitor_heartbeat(
                        heartbeat,
                        status="ok",
                        freq=f_value,
                        horizon=h_value,
                        interval=local_interval,
                        db_url=local_db_url,
                        config_source=config_source,
                        config_path=config_path,
                        cycle_prediction_state=(
                            str(result.get("prediction_state"))
                            if isinstance(result, dict)
                            and result.get("prediction_state") is not None
                            else None
                        ),
                        cycle_prediction_reason=(
                            str(result.get("prediction_reason"))
                            if isinstance(result, dict)
                            and result.get("prediction_reason") is not None
                            else None
                        ),
                        cycle_realization_state=(
                            str(result.get("realization_state"))
                            if isinstance(result, dict)
                            and result.get("realization_state") is not None
                            else None
                        ),
                        cycle_diagnostic=(
                            str(result.get("cycle_diagnostic"))
                            if isinstance(result, dict)
                            and result.get("cycle_diagnostic") is not None
                            else None
                        ),
                        cycle_ingestion_state=(
                            str(result.get("ingestion_state"))
                            if isinstance(result, dict)
                            and result.get("ingestion_state") is not None
                            else None
                        ),
                        cycle_ingestion_failures=(
                            result.get("ingestion_failures")
                            if isinstance(result, dict)
                            and isinstance(result.get("ingestion_failures"), list)
                            else None
                        ),
                    )
                except Exception as exc:
                    write_monitor_heartbeat(
                        heartbeat,
                        status="error",
                        freq=f_value,
                        horizon=h_value,
                        interval=local_interval,
                        db_url=local_db_url,
                        config_source=config_source,
                        config_path=config_path,
                        error=str(exc),
                    )
                _MONITOR_STOP_EVENT.wait(local_interval)

            write_monitor_heartbeat(
                heartbeat,
                status="stopped",
                freq=f_value,
                horizon=h_value,
                interval=local_interval,
                db_url=local_db_url,
                config_source=config_source,
                config_path=config_path,
            )

        thread = threading.Thread(
            target=_monitor_loop,
            args=(freq_value, horizon_value),
            daemon=True,
            name="bitbat-monitor",
        )
        thread.start()
        st.session_state["_monitor_running"] = True

    control_cols = st.columns(2)
    with control_cols[0]:
        if not session_running:
            if st.button("Start Session Preview", type="primary", width="stretch"):
                _start_monitoring(freq, horizon)
                st.success(
                    "Session preview started. Keep this tab open while the first proof is written."
                )
        else:
            st.caption("Session preview is already running.")

    with control_cols[1]:
        if session_running:
            if st.button("Stop Session Preview", width="stretch"):
                _MONITOR_STOP_EVENT.set()
                st.session_state["_monitor_running"] = False
                st.info("Session preview will stop after the current cycle closes cleanly.")
        else:
            st.caption(
                "Use System to inspect the durable watcher, or run the CLI watcher for 24/7 use."
            )

    with st.expander("24/7 Runner", expanded=False):
        st.caption(
            "Use the durable runner outside the browser when you want monitoring to survive "
            "tab closes, laptop sleep, or Streamlit restarts."
        )
        st.code("poetry run python scripts/run_monitoring_agent.py", language="bash")
        st.caption(
            "For service-managed uptime, install the bundled unit in "
            "`deployment/bitbat-monitor.service`."
        )

    @st.fragment(run_every=60)
    def _live_timeline() -> None:
        from bitbat.gui.timeline import (
            apply_timeline_filters,
            build_timeline_comparison_figure,
            build_timeline_figure,
            format_timeline_empty_state,
            get_price_series,
            get_timeline_data,
            list_timeline_filter_options,
            summarize_timeline_insights,
        )

        st.header("Signal Timeline")
        st.caption(
            "Use this ledger to verify whether BitBat is producing signals, "
            "realizing outcomes, and improving over time."
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
            st.session_state["timeline_show_overlay"] = False

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
                "Show Return Comparison",
                key="timeline_show_overlay",
            )

        predictions = get_timeline_data(_DB_PATH, selected_freq, selected_horizon, limit=2000)
        predictions = apply_timeline_filters(predictions, date_window=selected_window)

        if predictions.empty:
            st.info(format_timeline_empty_state(selected_freq, selected_horizon, selected_window))
            return

        first_ts = predictions["timestamp_utc"].min()
        prices = get_price_series(_DATA_DIR, selected_freq, first_ts)

        fig = build_timeline_figure(predictions, prices, show_overlay=False)
        st.plotly_chart(fig, width="stretch")

        if show_overlay:
            comparison_fig = build_timeline_comparison_figure(predictions)
            st.plotly_chart(comparison_fig, width="stretch")

        insights = summarize_timeline_insights(predictions)
        avg_confidence = insights["average_confidence"]

        _render_story_cards([
            (
                "Total Signals",
                str(int(insights["total"])),
                "Every durable prediction recorded in the filtered window.",
            ),
            (
                "Completed",
                str(int(insights["completed"])),
                "Signals whose outcome is already known.",
            ),
            (
                "Correct",
                str(int(insights["correct"])),
                "Completed signals that matched realized direction.",
            ),
            (
                "Accuracy",
                f"{insights['accuracy']:.1f}%",
                "A quick read on directional quality for this window.",
            ),
            (
                "Avg Confidence",
                "n/a" if avg_confidence is None else f"{avg_confidence:.1f}%",
                "High confidence with poor results is a drift warning.",
            ),
        ])

    _live_timeline()

    st.divider()
    if st.button("Retrain Model", type="primary", width="stretch"):
        st.session_state["_monitor_running"] = False
        st.session_state["train_result"] = None
        st.session_state["train_state"] = "INITIAL"
        st.session_state["train_stage_history"] = []
        st.rerun()
