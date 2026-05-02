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
:root {
    --bb-bg: #100f15;
    --bb-bg-2: #191624;
    --bb-panel: rgba(27, 23, 39, 0.88);
    --bb-panel-strong: rgba(20, 17, 30, 0.96);
    --bb-border: rgba(212, 188, 140, 0.22);
    --bb-bone: #f0e8d9;
    --bb-moon: #d8c08b;
    --bb-gold: #b9954b;
    --bb-moss: #8aa16b;
    --bb-ember: #d46a55;
    --bb-veil: rgba(121, 93, 163, 0.18);
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top, rgba(121, 93, 163, 0.18), transparent 34%),
        radial-gradient(circle at 85% 18%, rgba(212, 188, 140, 0.10), transparent 22%),
        linear-gradient(180deg, #17121f 0%, #0c0b11 100%);
    color: var(--bb-bone);
}

[data-testid="stAppViewContainer"] {
    font-family: "Trebuchet MS", "Segoe UI", sans-serif;
}

h1, h2, h3, h4, [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2 {
    font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif !important;
    letter-spacing: 0.02em;
    color: #f6efdf;
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(19, 16, 30, 0.98) 0%, rgba(11, 10, 18, 0.98) 100%);
    border-right: 1px solid rgba(216, 192, 139, 0.12);
}

[data-testid="stSidebar"] * {
    color: var(--bb-bone);
}

.block-container {
    padding-top: 1.4rem !important;
}

.bb-hero,
.bb-card,
.bb-ledger,
.bb-event-row {
    background: linear-gradient(180deg, var(--bb-panel) 0%, var(--bb-panel-strong) 100%);
    border: 1px solid var(--bb-border);
    border-radius: 20px;
    box-shadow: 0 18px 44px rgba(0, 0, 0, 0.24);
}

.bb-hero {
    padding: 1.25rem 1.3rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}

.bb-hero::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, transparent 0%, rgba(216, 192, 139, 0.08) 48%, transparent 100%);
    pointer-events: none;
}

.bb-kicker {
    text-transform: uppercase;
    letter-spacing: 0.22em;
    font-size: 0.72rem;
    color: var(--bb-moon);
    margin-bottom: 0.45rem;
}

.bb-title {
    font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
    font-size: 2.1rem;
    line-height: 1.05;
    margin: 0;
    color: #f7efde;
}

.bb-subtitle {
    margin-top: 0.5rem;
    color: rgba(240, 232, 217, 0.82);
}

.bb-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.9rem;
}

.bb-card {
    padding: 1rem;
    min-height: 100%;
}

.bb-label {
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.72rem;
    color: rgba(216, 192, 139, 0.88);
}

.bb-value {
    margin-top: 0.35rem;
    font-size: 1.18rem;
    font-weight: 600;
    color: #f8f0df;
}

.bb-copy {
    margin-top: 0.4rem;
    color: rgba(240, 232, 217, 0.78);
    font-size: 0.95rem;
}

.bb-state-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.32rem 0.78rem;
    border-radius: 999px;
    font-size: 0.84rem;
    font-weight: 600;
    border: 1px solid transparent;
}

.bb-state-chip[data-tone="success"] {
    background: rgba(138, 161, 107, 0.12);
    color: #d9eab7;
    border-color: rgba(138, 161, 107, 0.3);
}

.bb-state-chip[data-tone="warning"] {
    background: rgba(212, 188, 140, 0.12);
    color: #f0d9aa;
    border-color: rgba(212, 188, 140, 0.28);
}

.bb-state-chip[data-tone="danger"] {
    background: rgba(212, 106, 85, 0.14);
    color: #ffd9d0;
    border-color: rgba(212, 106, 85, 0.32);
}

.bb-state-chip[data-tone="info"] {
    background: rgba(121, 93, 163, 0.18);
    color: #decff5;
    border-color: rgba(121, 93, 163, 0.28);
}

.bb-ledger {
    padding: 0.9rem 1rem;
}

.bb-ledger-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    padding: 0.45rem 0;
    border-bottom: 1px solid rgba(216, 192, 139, 0.08);
}

.bb-ledger-row:last-child {
    border-bottom: none;
}

.bb-ledger-step {
    color: #f4ecdc;
    font-weight: 500;
}

.bb-ledger-state {
    color: rgba(240, 232, 217, 0.68);
    font-size: 0.9rem;
}

.bb-event-row {
    padding: 0.78rem 0.92rem;
    margin-bottom: 0.65rem;
}

.bb-event-meta {
    color: rgba(216, 192, 139, 0.85);
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.bb-event-message {
    margin-top: 0.28rem;
    color: #f3edde;
}

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

    .bb-title {
        font-size: 1.75rem;
    }
}

/* ---- Touch-friendly button sizing ---- */
.stButton button {
    border-radius: 999px;
    font-weight: 600;
    transition: transform 0.1s ease, border-color 0.2s ease, background 0.2s ease;
    border: 1px solid rgba(216, 192, 139, 0.22);
    background: linear-gradient(180deg, rgba(34, 29, 47, 0.96) 0%, rgba(19, 17, 28, 0.96) 100%);
    color: #f4edde;
}
.stButton button:active {
    transform: scale(0.97);
}
.stButton button:hover {
    border-color: rgba(216, 192, 139, 0.5);
}

/* ---- Metric cards ---- */
[data-testid="metric-container"] {
    background: linear-gradient(180deg, rgba(31, 26, 44, 0.96) 0%, rgba(18, 16, 26, 0.96) 100%);
    border: 1px solid rgba(216, 192, 139, 0.16);
    border-radius: 18px;
    padding: 0.85rem;
}

/* ---- Confidence meter (progress bar) ---- */
.stProgress > div > div {
    border-radius: 999px;
    background: linear-gradient(90deg, #8aa16b 0%, #d8c08b 100%);
}

/* ---- Prediction banners ---- */
.stSuccess, .stError, .stInfo, .stWarning {
    border-radius: 18px;
    border-width: 1px;
}

/* ---- Sidebar navigation links ---- */
[data-testid="stSidebarNav"] a {
    font-size: 1rem;
    padding: 0.6rem 0.78rem;
    border-radius: 12px;
    display: block;
    margin-bottom: 0.2rem;
}
[data-testid="stSidebarNav"] a:hover {
    background: rgba(121, 93, 163, 0.16);
}

/* ---- Tables ---- */
[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid rgba(216, 192, 139, 0.14);
}

/* ---- Expander ---- */
[data-testid="stExpander"] {
    border-radius: 18px;
    border: 1px solid rgba(216, 192, 139, 0.14);
    background: rgba(19, 17, 28, 0.72);
}
</style>
"""


def inject_css() -> None:
    """Inject shared mobile-friendly CSS into the current Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)
