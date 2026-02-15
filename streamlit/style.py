"""
Shared CSS injected into Streamlit pages for mobile-friendly, consistent styling.

Usage:
    from style import inject_css
    inject_css()
"""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
/* ---- Mobile-first responsive layout ---- */
@media (max-width: 768px) {
    /* Stack columns vertically on small screens */
    [data-testid="column"] {
        width: 100% !important;
        flex: none !important;
    }
    /* Larger tap targets for buttons */
    .stButton button {
        min-height: 48px;
        font-size: 1rem;
    }
    /* Reduce padding on mobile */
    .block-container {
        padding: 1rem 0.75rem !important;
    }
}

/* ---- Touch-friendly button sizing ---- */
.stButton button {
    border-radius: 8px;
    font-weight: 500;
    transition: transform 0.1s ease;
}
.stButton button:active {
    transform: scale(0.97);
}

/* ---- Metric cards ---- */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 0.75rem;
}

/* ---- Confidence meter (progress bar) ---- */
.stProgress > div > div {
    border-radius: 4px;
}

/* ---- Prediction banners ---- */
.stSuccess, .stError, .stInfo, .stWarning {
    border-radius: 8px;
}

/* ---- Sidebar navigation links ---- */
[data-testid="stSidebarNav"] a {
    font-size: 1rem;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    display: block;
    margin-bottom: 2px;
}
[data-testid="stSidebarNav"] a:hover {
    background: rgba(255,255,255,0.08);
}

/* ---- Tables ---- */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
}

/* ---- Expander ---- */
[data-testid="stExpander"] {
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.08);
}
</style>
"""


def inject_css() -> None:
    """Inject shared mobile-friendly CSS into the current Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)
