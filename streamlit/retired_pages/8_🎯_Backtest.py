"""Retired Backtest page with safe legacy-route guidance."""

from __future__ import annotations

import streamlit as st

from _retired_notice import render_retired_page

st.set_page_config(page_title="Backtest (Retired) — BitBat", page_icon="🎯", layout="wide")
render_retired_page("Backtest")
