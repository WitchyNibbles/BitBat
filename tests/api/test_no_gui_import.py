"""Structural guard: API layer must not import from the GUI layer."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.structural

_API_SRC = Path("src/bitbat/api")


def _gui_imports_in_file(path: Path) -> list[str]:
    """Return module names that are 'gui' imports found in *path*."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    return [
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and "gui" in node.module
    ]


def test_system_routes_no_gui_import() -> None:
    """api/routes/system.py must not import from any gui module."""
    system_py = _API_SRC / "routes" / "system.py"
    assert system_py.exists(), f"Expected file not found: {system_py}"

    gui_imports = _gui_imports_in_file(system_py)
    assert gui_imports == [], (
        f"Found gui imports in {system_py}: {gui_imports}. "
        "Move shared utilities to bitbat.common instead."
    )


def test_api_layer_no_gui_imports() -> None:
    """No .py file under src/bitbat/api/ may import from bitbat.gui."""
    violations: list[tuple[str, list[str]]] = []
    for py_file in _API_SRC.glob("**/*.py"):
        imports = _gui_imports_in_file(py_file)
        if imports:
            violations.append((str(py_file), imports))

    assert violations == [], (
        "Found api->gui import violations (move utilities to bitbat.common):\n"
        + "\n".join(f"  {f}: {mods}" for f, mods in violations)
    )
