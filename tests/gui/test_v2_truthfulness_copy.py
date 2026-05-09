from __future__ import annotations

from pathlib import Path


def test_about_page_no_longer_claims_v2_self_retraining() -> None:
    source = Path("streamlit/pages/3_ℹ️_About.py").read_text(encoding="utf-8")

    assert "does **not** retrain itself yet" in source
    assert "legacy monitoring pipeline can detect drift and retrain models" in source


def test_oracle_page_no_longer_exposes_retrain_button() -> None:
    source = Path("dashboard/src/pages/Oracle.tsx").read_text(encoding="utf-8")

    assert "Request Retrain" not in source
    assert "signal source" in source


def test_legacy_dashboard_pages_are_marked_diagnostic_only() -> None:
    sidebar = Path("dashboard/src/components/Sidebar.tsx").read_text(encoding="utf-8")
    quickstart = Path("dashboard/src/pages/QuickStart.tsx").read_text(encoding="utf-8")
    system = Path("dashboard/src/pages/System.tsx").read_text(encoding="utf-8")

    assert "Legacy Quick Start" in sidebar
    assert "Legacy Performance" in sidebar
    assert "Legacy System" in sidebar
    assert "Oracle view" in quickstart
    assert "diagnostic-only" in system
