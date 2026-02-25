"""Phase 11 completion gate: runtime stability and retired-route behavior."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from bitbat.gui.widgets import get_latest_prediction

ROOT = Path(__file__).resolve().parents[2]
STREAMLIT_DIR = ROOT / "streamlit"
RETIRED_DIR = STREAMLIT_DIR / "retired_pages"

SUPPORTED_DESTINATIONS = {
    "pages/0_Quick_Start.py",
    "pages/1_⚙️_Settings.py",
    "pages/2_📈_Performance.py",
    "pages/3_ℹ️_About.py",
    "pages/4_🔧_System.py",
}


def _seed_legacy_prediction_db(db_path: Path) -> None:
    con = sqlite3.connect(str(db_path))
    con.executescript(
        """
        CREATE TABLE prediction_outcomes (
            id INTEGER PRIMARY KEY,
            timestamp_utc TEXT,
            predicted_direction TEXT,
            predicted_return REAL,
            model_version TEXT
        );

        INSERT INTO prediction_outcomes
            (timestamp_utc, predicted_direction, predicted_return, model_version)
        VALUES
            (datetime('now'), 'up', 0.006, 'legacy-v1');
        """
    )
    con.commit()
    con.close()


def test_phase11_home_source_guards_confidence_rendering() -> None:
    app_source = (STREAMLIT_DIR / "app.py").read_text(encoding="utf-8")

    assert 'latest_pred.get("confidence")' in app_source
    assert 'latest_pred["confidence"]' not in app_source
    assert "**Confidence:** n/a" in app_source


def test_phase11_widgets_latest_prediction_handles_legacy_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    _seed_legacy_prediction_db(db_path)

    payload = get_latest_prediction(db_path)

    assert payload is not None
    assert payload["direction"] == "up"
    assert payload["predicted_return"] == 0.006
    assert payload["confidence"] is None
    assert payload["predicted_price"] is None
    assert payload["created_at"] == payload["timestamp_utc"]


def test_phase11_retired_notice_routes_only_to_supported_pages() -> None:
    helper = (RETIRED_DIR / "_retired_notice.py").read_text(encoding="utf-8")

    assert "SUPPORTED_VIEWS" in helper
    assert "st.switch_page(target)" in helper
    for destination in SUPPORTED_DESTINATIONS:
        assert f'"{destination}"' in helper


def test_phase11_retired_backtest_and_pipeline_are_safe_shells() -> None:
    backtest_source = (RETIRED_DIR / "8_🎯_Backtest.py").read_text(encoding="utf-8")
    pipeline_source = (RETIRED_DIR / "9_🔬_Pipeline.py").read_text(encoding="utf-8")

    assert "from _retired_notice import render_retired_page" in backtest_source
    assert "from _retired_notice import render_retired_page" in pipeline_source
    assert 'render_retired_page("Backtest")' in backtest_source
    assert 'render_retired_page("Pipeline")' in pipeline_source

    assert "classification_metrics" not in pipeline_source
    assert "xgboost" not in pipeline_source
    assert "backtest_run" not in backtest_source
    assert "too many indices for array" not in backtest_source


def test_phase11_known_failure_signatures_are_explicitly_blocked() -> None:
    app_source = (STREAMLIT_DIR / "app.py").read_text(encoding="utf-8")
    backtest_source = (RETIRED_DIR / "8_🎯_Backtest.py").read_text(encoding="utf-8")
    pipeline_source = (RETIRED_DIR / "9_🔬_Pipeline.py").read_text(encoding="utf-8")

    assert 'latest_pred["confidence"]' not in app_source
    assert "classification_metrics" not in pipeline_source
    assert "backtest_run" not in backtest_source
