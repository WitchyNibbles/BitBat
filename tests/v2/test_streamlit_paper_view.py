from __future__ import annotations

from pathlib import Path


def test_performance_page_surfaces_v2_paper_trading_cockpit() -> None:
    source = Path("streamlit/pages/2_📈_Performance.py").read_text(encoding="utf-8")

    assert "Paper Trading Cockpit" in source
    assert "Legacy Signal Diagnostics" in source
    assert "_load_v2_views" in source
    assert "build_paper_cockpit_snapshot" in source
    assert "closed_trades_from_orders" in source


def test_performance_page_calls_out_paper_only_status() -> None:
    source = Path("streamlit/pages/2_📈_Performance.py").read_text(encoding="utf-8")

    assert "Paper-only loud and clear" in source
    assert "profit-first v2 ledger" in source.lower()
