"""Phase 8 release-readiness gate for D1/D2/D3 verification assumptions."""

from __future__ import annotations

import ast
from pathlib import Path

from tests.autonomous import test_phase8_d1_monitor_schema_complete as d1_gate
from tests.gui import test_phase8_d2_timeline_complete as d2_gate

ROOT = Path(__file__).resolve().parents[2]
STREAMLIT_DIR = ROOT / "streamlit"
PAGES_DIR = STREAMLIT_DIR / "pages"
REQUIRED_GATE_FILES = [
    "tests/autonomous/test_phase8_d1_monitor_schema_complete.py",
    "tests/gui/test_phase8_d2_timeline_complete.py",
    "tests/gui/test_phase10_supported_surface_complete.py",
    "tests/gui/test_streamlit_width_compat.py",
    "tests/gui/test_phase7_streamlit_compat_complete.py",
]


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


def _keyword_value(call: ast.Call, keyword_name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == keyword_name:
            return keyword.value
    return None


def test_phase8_release_required_gate_files_exist() -> None:
    for rel_path in REQUIRED_GATE_FILES:
        assert Path(rel_path).exists(), f"Missing release gate file: {rel_path}"


def test_phase8_release_depends_on_canonical_d1_and_d2_suite_contracts() -> None:
    assert "tests/autonomous/test_phase8_d1_monitor_schema_complete.py" in d1_gate.D1_CANONICAL_SUITE
    assert "tests/test_cli.py" in d1_gate.D1_CANONICAL_SUITE
    assert "tests/gui/test_phase8_d2_timeline_complete.py" in d2_gate.D2_CANONICAL_SUITES
    assert "tests/gui/test_phase6_timeline_ux_complete.py" in d2_gate.D2_CANONICAL_SUITES
    assert "tests/gui/test_phase9_timeline_readability_complete.py" in d2_gate.D2_CANONICAL_SUITES
    assert "tests/gui/test_phase10_supported_surface_complete.py" in d2_gate.D2_CANONICAL_SUITES


def test_phase8_release_runtime_streamlit_contract_has_no_deprecated_width_keyword() -> None:
    deprecated_offenders: list[str] = []
    boolean_width_offenders: list[str] = []

    for file_path in _runtime_streamlit_files():
        for call in _iter_streamlit_calls(file_path):
            deprecated_value = _keyword_value(call, "use_container_width")
            width_value = _keyword_value(call, "width")

            if deprecated_value is not None:
                deprecated_offenders.append(f"{file_path.relative_to(ROOT)}:{call.lineno}")

            if isinstance(width_value, ast.Constant) and isinstance(width_value.value, bool):
                boolean_width_offenders.append(f"{file_path.relative_to(ROOT)}:{call.lineno}")

    assert not deprecated_offenders, (
        f"Deprecated Streamlit width usage detected in runtime files: {deprecated_offenders}"
    )
    assert not boolean_width_offenders, (
        f"Boolean Streamlit width usage detected in runtime files: {boolean_width_offenders}"
    )


def test_phase8_release_makefile_target_covers_d1_d2_d3_commands() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "test-release:" in makefile
    assert "tests/autonomous/test_phase8_d1_monitor_schema_complete.py" in makefile
    assert "tests/gui/test_phase8_d2_timeline_complete.py" in makefile
    assert "tests/gui/test_phase10_supported_surface_complete.py" in makefile
    assert "tests/gui/test_phase8_release_verification_complete.py" in makefile
