"""Compatibility checks for Streamlit width API usage in runtime GUI files."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STREAMLIT_DIR = ROOT / "streamlit"
PAGES_DIR = STREAMLIT_DIR / "pages"
RETIRED_PAGES_DIR = STREAMLIT_DIR / "retired_pages"
ALLOWED_WIDTH_LITERALS = {"stretch", "content"}
RETIRED_PAGE_PATHS = {
    "pages/5_🔔_Alerts.py",
    "pages/6_📊_Analytics.py",
    "pages/7_📅_History.py",
    "pages/8_🎯_Backtest.py",
    "pages/9_🔬_Pipeline.py",
}


def _runtime_streamlit_files() -> list[Path]:
    files = [STREAMLIT_DIR / "app.py"]
    files.extend(sorted(PAGES_DIR.glob("*.py")))
    return files


def _iter_streamlit_calls(file_path: Path) -> list[ast.Call]:
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == "st":
                calls.append(node)
    return calls


def _call_target_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Attribute):
        return f"st.{call.func.attr}"
    return "st.<unknown>"


def _keyword_value(call: ast.Call, keyword_name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == keyword_name:
            return keyword.value
    return None


def test_runtime_scope_covers_primary_gui_entrypoints() -> None:
    names = {path.name for path in _runtime_streamlit_files()}
    expected = {
        "app.py",
        "0_Quick_Start.py",
        "1_⚙️_Settings.py",
        "2_📈_Performance.py",
        "3_ℹ️_About.py",
        "4_🔧_System.py",
    }
    assert names == expected


def test_runtime_scope_excludes_retired_pages_from_active_directory() -> None:
    active = {path.name for path in PAGES_DIR.glob("*.py")}
    retired = {path.name for path in RETIRED_PAGES_DIR.glob("*.py")}
    assert retired
    assert active.isdisjoint(retired)


def test_runtime_sources_do_not_reference_retired_page_routes() -> None:
    for file_path in _runtime_streamlit_files():
        source = file_path.read_text(encoding="utf-8")
        for retired_path in RETIRED_PAGE_PATHS:
            assert retired_path not in source


def test_deprecated_usage_absent_in_runtime_streamlit_sources() -> None:
    offenders: list[str] = []
    for file_path in _runtime_streamlit_files():
        for call in _iter_streamlit_calls(file_path):
            value = _keyword_value(call, "use_container_width")
            if value is not None:
                offenders.append(
                    f"{file_path.relative_to(ROOT)}:{call.lineno}:{_call_target_name(call)}"
                )

    assert not offenders, f"Deprecated use_container_width usage found: {offenders}"


def test_width_keyword_uses_supported_non_boolean_literals() -> None:
    boolean_offenders: list[str] = []
    literal_offenders: list[str] = []

    for file_path in _runtime_streamlit_files():
        for call in _iter_streamlit_calls(file_path):
            value = _keyword_value(call, "width")
            if value is None:
                continue

            if isinstance(value, ast.Constant) and isinstance(value.value, bool):
                boolean_offenders.append(
                    f"{file_path.relative_to(ROOT)}:{call.lineno}:{_call_target_name(call)}"
                )
                continue

            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                if value.value not in ALLOWED_WIDTH_LITERALS:
                    literal_offenders.append(
                        f"{file_path.relative_to(ROOT)}:{call.lineno}:{_call_target_name(call)}="
                        f"{value.value!r}"
                    )

    assert not boolean_offenders, f"Boolean width arguments found: {boolean_offenders}"
    assert not literal_offenders, f"Unsupported width literal values found: {literal_offenders}"


def test_runtime_scope_uses_modern_width_literals() -> None:
    literals: set[str] = set()

    for file_path in _runtime_streamlit_files():
        for call in _iter_streamlit_calls(file_path):
            value = _keyword_value(call, "width")
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                literals.add(value.value)

    assert literals
    assert literals.issubset(ALLOWED_WIDTH_LITERALS)
    assert "stretch" in literals
