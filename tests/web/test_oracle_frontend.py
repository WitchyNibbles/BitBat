"""Structural regression tests for the static oracle frontend."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"

pytestmark = pytest.mark.structural


def test_oracle_frontend_explains_confidence_bar_purpose() -> None:
    source = (WEB_DIR / "index.html").read_text(encoding="utf-8")

    assert "Model confidence in this direction" in source
    assert "Probability split" in source
    assert "Grimoire Log" in source
    assert "Live updates every few seconds" in source


def test_oracle_frontend_uses_probability_contract_not_return_scaling() -> None:
    source = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    assert "predData.confidence" in source
    assert "predData.p_flat" in source
    assert "Probability split" in source
    assert "* 1000" not in source
