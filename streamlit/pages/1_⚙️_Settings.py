"""Settings page — choose a prediction strategy preset."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bitbat.gui.presets import DEFAULT_PRESET, get_preset, list_presets

st.set_page_config(page_title="Settings — BitBat", page_icon="⚙️", layout="wide")

st.title("⚙️ Settings")
st.markdown("Choose how BitBat makes predictions for you.")

# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
if "current_preset" not in st.session_state:
    # Try to load from saved config
    config_path = ROOT / "config" / "user_config.yaml"
    if config_path.exists():
        try:
            saved = yaml.safe_load(config_path.read_text())
            st.session_state.current_preset = saved.get("preset", DEFAULT_PRESET)
        except Exception:
            st.session_state.current_preset = DEFAULT_PRESET
    else:
        st.session_state.current_preset = DEFAULT_PRESET

# ------------------------------------------------------------------
# Preset selector
# ------------------------------------------------------------------
st.header("Choose Your Strategy")
st.markdown("Select the approach that best matches your goals:")

presets = list_presets()
col1, col2, col3 = st.columns(3)

for col, key in zip([col1, col2, col3], ["conservative", "balanced", "aggressive"]):
    p = presets[key]
    is_active = st.session_state.current_preset == key
    with col:
        st.markdown(
            f"<div style='text-align:center; padding:12px; "
            f"border:2px solid {p.color}; border-radius:8px; "
            f"background:{p.color}20;'>"
            f"<h2>{p.icon}</h2>"
            f"<h3>{p.name}</h3>"
            f"<p style='font-size:0.9em;'>{p.description}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        btn_type = "primary" if is_active else "secondary"
        if st.button(
            f"{'✅ ' if is_active else ''}Select {p.name}",
            key=f"btn_{key}",
            width="stretch",
            type=btn_type,
        ):
            st.session_state.current_preset = key
            st.rerun()

# ------------------------------------------------------------------
# Current settings display
# ------------------------------------------------------------------
st.divider()
current = get_preset(st.session_state.current_preset)
st.header(f"Current Configuration: {current.icon} {current.name}")

disp_col, help_col = st.columns([1, 1])

with disp_col:
    st.markdown(f"*{current.description}*")
    display = current.to_display()
    for label, value in display.items():
        st.markdown(f"- **{label}:** {value}")

with help_col:
    st.markdown("**When to use each strategy:**")
    st.markdown(
        "🛡️ **Conservative** — Long-term holders, low risk  \n"
        "⚖️ **Balanced** — Most users, recommended starting point  \n"
        "🚀 **Aggressive** — Active traders, higher risk tolerance"
    )

# ------------------------------------------------------------------
# Advanced settings
# ------------------------------------------------------------------
with st.expander("🔧 Advanced Settings (for experienced users)"):
    st.warning(
        "⚠️ Changing these settings may affect prediction quality. "
        "Only modify if you understand the parameters."
    )
    cfg = current.to_dict()

    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        freq = st.selectbox(
            "Update Frequency",
            options=["1h", "4h", "1d"],
            index=["1h", "4h", "1d"].index(cfg["freq"]),
            help="How often to generate new predictions.",
        )
        horizon = st.selectbox(
            "Forecast Period",
            options=["1h", "4h", "24h"],
            index=["1h", "4h", "24h"].index(cfg["horizon"]),
            help="How far ahead to predict.",
        )
    with adv_col2:
        tau = st.slider(
            "Movement Sensitivity",
            min_value=0.001,
            max_value=0.05,
            value=float(cfg["tau"]),
            step=0.001,
            format="%.3f",
            help="Minimum price movement to classify as up/down.",
        )
        enter_threshold = st.slider(
            "Confidence Required",
            min_value=0.5,
            max_value=0.95,
            value=float(cfg["enter_threshold"]),
            step=0.05,
            format="%.0f%%",
            help="Minimum confidence required before making a prediction.",
        )

    if st.button("Apply Advanced Settings", type="primary"):
        config_path = ROOT / "config" / "user_config.yaml"
        config_path.parent.mkdir(exist_ok=True)
        custom = {
            "preset": "custom",
            "freq": freq,
            "horizon": horizon,
            "tau": tau,
            "enter_threshold": enter_threshold,
        }
        config_path.write_text(yaml.dump(custom))
        st.success("✅ Advanced settings saved!")
        st.info("Restart the monitoring system for changes to take effect.")

# ------------------------------------------------------------------
# Save / reset
# ------------------------------------------------------------------
st.divider()
save_col, reset_col = st.columns(2)

with save_col:
    if st.button("💾 Save and Apply", type="primary", width="stretch"):
        config_path = ROOT / "config" / "user_config.yaml"
        config_path.parent.mkdir(exist_ok=True)
        to_save = current.to_dict()
        to_save["preset"] = st.session_state.current_preset
        config_path.write_text(yaml.dump(to_save))
        st.success(f"✅ Saved {current.name} preset!")
        st.info("ℹ️ Restart the monitoring system for changes to take effect.")

with reset_col:
    if st.button("🔄 Reset to Default", width="stretch"):
        st.session_state.current_preset = DEFAULT_PRESET
        config_path = ROOT / "config" / "user_config.yaml"
        if config_path.exists():
            config_path.unlink()
        st.success("✅ Reset to Balanced preset!")
        st.rerun()
