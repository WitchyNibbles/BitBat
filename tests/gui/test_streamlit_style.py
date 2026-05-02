"""Structural tests for shared Streamlit CSS overrides."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STYLE_PATH = ROOT / "streamlit" / "style.py"


def test_app_font_rule_does_not_override_every_descendant() -> None:
    source = STYLE_PATH.read_text(encoding="utf-8")

    assert '[data-testid="stAppViewContainer"] * {' not in source
    assert '[data-testid="stAppViewContainer"] {' in source
