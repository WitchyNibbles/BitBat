from __future__ import annotations

from pathlib import Path


def test_runtime_code_has_no_raw_sqlite3_usage() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "bitbat"

    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "import sqlite3" in text or "sqlite3.connect" in text:
            offenders.append(str(path.relative_to(root.parent.parent)))

    assert offenders == []
