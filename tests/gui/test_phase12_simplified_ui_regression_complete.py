"""Phase 12 completion gate: simplified UI regression and crash-guard contracts."""

from __future__ import annotations

import re
from pathlib import Path

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


def test_phase12_supported_surface_contract_remains_locked() -> None:
    active = {path.name for path in PAGES_DIR.glob("*.py")}
    retired = {path.name for path in RETIRED_DIR.glob("*.py")}

    assert active == SUPPORTED_PAGES
    assert RETIRED_PAGES.issubset(retired)
    assert active.isdisjoint(retired)


def test_phase12_app_navigation_targets_supported_pages_only() -> None:
    app_source = (STREAMLIT_DIR / "app.py").read_text(encoding="utf-8")
    destinations = set(re.findall(r'st\.switch_page\("([^"]+)"\)', app_source))

    assert destinations == {
        "pages/0_Quick_Start.py",
        "pages/1_⚙️_Settings.py",
        "pages/2_📈_Performance.py",
        "pages/3_ℹ️_About.py",
        "pages/4_🔧_System.py",
    }


def test_phase12_reported_failure_signatures_are_guarded_in_sources() -> None:
    app_source = (STREAMLIT_DIR / "app.py").read_text(encoding="utf-8")
    backtest_source = (RETIRED_DIR / "8_🎯_Backtest.py").read_text(encoding="utf-8")
    pipeline_source = (RETIRED_DIR / "9_🔬_Pipeline.py").read_text(encoding="utf-8")

    assert 'latest_pred["confidence"]' not in app_source
    assert "classification_metrics" not in pipeline_source
    assert "xgboost" not in pipeline_source
    assert "backtest_run" not in backtest_source
    assert "too many indices for array" not in backtest_source


def test_phase12_anchors_phase11_stability_gate_coverage() -> None:
    phase11_gate = (ROOT / "tests/gui/test_phase11_runtime_stability_complete.py").read_text(
        encoding="utf-8"
    )

    assert 'latest_pred["confidence"]' in phase11_gate
    assert "classification_metrics" in phase11_gate
    assert "backtest_run" in phase11_gate
