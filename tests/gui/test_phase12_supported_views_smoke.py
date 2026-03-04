"""Phase 12 smoke suite for supported Streamlit runtime views."""

from __future__ import annotations

import ast
import re
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
STREAMLIT_DIR = ROOT / "streamlit"
PAGES_DIR = STREAMLIT_DIR / "pages"

SUPPORTED_PAGE_FILES = {
    "0_Quick_Start.py",
    "1_⚙️_Settings.py",
    "2_📈_Performance.py",
    "3_ℹ️_About.py",
    "4_🔧_System.py",
}


pytestmark = pytest.mark.structural

def _read(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def test_phase12_smoke_supported_page_inventory_is_exact() -> None:
    active_pages = {path.name for path in PAGES_DIR.glob("*.py")}
    assert active_pages == SUPPORTED_PAGE_FILES


def test_phase12_smoke_supported_pages_are_parseable_python_modules() -> None:
    for page_name in SUPPORTED_PAGE_FILES:
        source = _read(PAGES_DIR / page_name)
        ast.parse(source, filename=page_name)


def test_phase12_smoke_supported_pages_define_streamlit_page_config() -> None:
    missing = []
    for page_name in SUPPORTED_PAGE_FILES:
        source = _read(PAGES_DIR / page_name)
        if "st.set_page_config(" not in source:
            missing.append(page_name)
    assert not missing, f"Missing st.set_page_config in: {missing}"


def test_phase12_smoke_app_navigation_covers_every_supported_page() -> None:
    app_source = _read(STREAMLIT_DIR / "app.py")
    destinations = set(re.findall(r'st\.switch_page\("([^"]+)"\)', app_source))

    expected = {
        "pages/0_Quick_Start.py",
        "pages/1_⚙️_Settings.py",
        "pages/2_📈_Performance.py",
        "pages/3_ℹ️_About.py",
        "pages/4_🔧_System.py",
    }
    assert destinations == expected
