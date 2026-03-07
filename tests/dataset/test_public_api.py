"""Structural tests confirming public API importability and backward-compat aliases.

These tests guard against regression to private (underscore-prefixed) imports
in external caller files.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.structural


def test_generate_price_features_importable() -> None:
    """Public generate_price_features is importable from bitbat.dataset.build."""
    from bitbat.dataset.build import generate_price_features

    assert generate_price_features is not None
    assert callable(generate_price_features)


def test_join_auxiliary_features_importable() -> None:
    """Public join_auxiliary_features is importable from bitbat.dataset.build."""
    from bitbat.dataset.build import join_auxiliary_features

    assert join_auxiliary_features is not None
    assert callable(join_auxiliary_features)


def test_package_reexports() -> None:
    """bitbat.dataset package re-exports both public feature pipeline functions."""
    from bitbat.dataset import generate_price_features, join_auxiliary_features  # noqa: F401

    assert generate_price_features is not None
    assert join_auxiliary_features is not None


def test_backward_compat_aliases() -> None:
    """Backward-compat underscore aliases still import correctly from bitbat.dataset.build."""
    from bitbat.dataset.build import (  # noqa: F401
        _generate_price_features,
        _join_auxiliary_features,
    )

    assert _generate_price_features is not None
    assert _join_auxiliary_features is not None


def test_load_prices_importable() -> None:
    """Shared load_prices function is importable from bitbat.io.prices."""
    from bitbat.io.prices import load_prices

    assert load_prices is not None
    assert callable(load_prices)


def _collect_private_feature_imports(source: str) -> list[str]:
    """Return any import names in source that are the old private feature function names."""
    tree = ast.parse(source)
    violations: list[str] = []
    private_names = {"_generate_price_features", "_join_auxiliary_features"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_name = alias.name
                if imported_name in private_names:
                    violations.append(imported_name)
    return violations


def test_no_private_imports_in_callers() -> None:
    """External caller files must not import private feature function names.

    This is a structural guard that prevents regression back to underscore-prefixed
    imports in cli.py, predictor.py, and continuous_trainer.py.
    """
    src_root = Path(__file__).parents[2] / "src"
    caller_paths = [
        src_root / "bitbat" / "cli.py",
        src_root / "bitbat" / "autonomous" / "predictor.py",
        src_root / "bitbat" / "autonomous" / "continuous_trainer.py",
    ]
    violations: dict[str, list[str]] = {}
    for path in caller_paths:
        assert path.exists(), f"Expected caller file not found: {path}"
        source = path.read_text(encoding="utf-8")
        bad_imports = _collect_private_feature_imports(source)
        if bad_imports:
            violations[str(path.name)] = bad_imports

    assert not violations, (
        "External callers must not import private feature function names. "
        f"Violations found: {violations}"
    )
