from __future__ import annotations

from pathlib import Path


def test_persistence_overview_exists() -> None:
    """Persistence overview doc should exist and mention key storage paths."""
    root = Path(__file__).resolve().parents[2]
    doc_path = root / "docs" / "persistence-overview.md"
    assert doc_path.exists(), "persistence-overview.md is missing"

    content = doc_path.read_text(encoding="utf-8")
    # Basic sanity checks so the document stays aligned with the code.
    assert "data/autonomous.db" in content
    assert "data/raw" in content
    assert "data/features" in content
    assert "models/" in content

