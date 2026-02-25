"""Shared retirement notice rendering for legacy Streamlit pages."""

from __future__ import annotations

import streamlit as st

SUPPORTED_VIEWS: tuple[tuple[str, str], ...] = (
    ("Quick Start", "pages/0_Quick_Start.py"),
    ("Settings", "pages/1_⚙️_Settings.py"),
    ("Performance", "pages/2_📈_Performance.py"),
    ("About", "pages/3_ℹ️_About.py"),
    ("System", "pages/4_🔧_System.py"),
)


def render_retired_page(view_name: str) -> None:
    """Render a consistent retirement message and links to supported views."""
    st.title(f"🧭 {view_name} (Retired)")
    st.warning(
        "This advanced view was retired in v1.1 to keep the operator UI stable and focused."
    )
    st.markdown(
        "Use one of the supported pages below. If you arrived from a bookmarked link, "
        "update it to a supported route."
    )

    quick_start_col, settings_col, performance_col, about_col, system_col = st.columns(5)
    columns = (
        quick_start_col,
        settings_col,
        performance_col,
        about_col,
        system_col,
    )

    for index, ((label, target), column) in enumerate(zip(SUPPORTED_VIEWS, columns, strict=True)):
        with column:
            if st.button(label, width="stretch", key=f"retired-{view_name}-{index}"):
                st.switch_page(target)

    st.info("Need model operations? Use Quick Start for guided workflow and System for diagnostics.")
