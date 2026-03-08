"""Phase 10 completion gate: supported UI surface pruning contract."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
STREAMLIT_DIR = ROOT / "streamlit"
PAGES_DIR = STREAMLIT_DIR / "pages"
RETIRED_DIR = STREAMLIT_DIR / "retired_pages"

SUPPORTED_PAGES = {
    "0_Quick_Start.py",
    "1_⚙️_Settings.py",
    "2_📈_Performance.py",
    "3_ℹ️_About.py",
    "4_🔧_System.py",
}

RETIRED_PAGES = {
    "5_🔔_Alerts.py",
    "6_📊_Analytics.py",
    "7_📅_History.py",
    "8_🎯_Backtest.py",
    "9_🔬_Pipeline.py",
}


pytestmark = pytest.mark.structural


def test_phase10_supported_surface_active_pages_inventory() -> None:
    active = {p.name for p in PAGES_DIR.glob("*.py")}

    assert active == SUPPORTED_PAGES
    assert active.isdisjoint(RETIRED_PAGES)


def test_phase10_retired_pages_preserved_outside_runtime_surface() -> None:
    retired = {p.name for p in RETIRED_DIR.glob("*.py")}

    assert RETIRED_PAGES.issubset(retired)
    assert (RETIRED_DIR / "README.md").exists()


def test_phase10_home_navigation_targets_only_supported_views() -> None:
    app_source = (STREAMLIT_DIR / "app.py").read_text(encoding="utf-8")
    destinations = set(re.findall(r'st\.switch_page\("([^"]+)"\)', app_source))

    assert destinations == {
        "pages/0_Quick_Start.py",
        "pages/1_⚙️_Settings.py",
        "pages/2_📈_Performance.py",
        "pages/3_ℹ️_About.py",
        "pages/4_🔧_System.py",
    }


def test_phase10_about_copy_mentions_supported_pages_not_pipeline() -> None:
    about = (PAGES_DIR / "3_ℹ️_About.py").read_text(encoding="utf-8")

    assert "Supported Pages" in about
    assert "Advanced Pipeline" not in about
