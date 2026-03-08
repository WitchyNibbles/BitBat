"""Tests proving isinstance guards survive python -O (optimized mode).

Verifies that production code uses if-not-isinstance-raise-TypeError instead
of assert isinstance, which would be stripped under python -O.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bitbat.model.train import fit_random_forest, fit_xgb

pytestmark = pytest.mark.behavioral


def _make_synthetic_data(
    n_samples: int = 50, n_features: int = 3, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Create a small synthetic dataset for training."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n_samples, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series(rng.normal(0.0, 0.01, size=n_samples), name="target")
    return X, y


def test_fit_xgb_returns_booster_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """fit_xgb() with valid data returns an xgb.Booster instance."""
    import xgboost as xgb

    monkeypatch.chdir(tmp_path)
    X, y = _make_synthetic_data()
    model, importance = fit_xgb(X, y, persist=False)

    assert isinstance(model, xgb.Booster)
    assert isinstance(importance, dict)
    assert len(importance) > 0


def test_fit_random_forest_returns_correct_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fit_random_forest() with valid data returns a RandomForestRegressor."""
    from sklearn.ensemble import RandomForestRegressor

    monkeypatch.chdir(tmp_path)
    X, y = _make_synthetic_data()
    model, importance = fit_random_forest(X, y, persist=False)

    assert isinstance(model, RandomForestRegressor)
    assert isinstance(importance, dict)
    assert len(importance) > 0


def test_no_assert_isinstance_in_production_code() -> None:
    """Structural regression test: no 'assert isinstance' in src/bitbat/.

    Uses AST parsing to detect assert statements with isinstance calls,
    ensuring they cannot be reintroduced. This guard prevents silent type
    check removal under python -O.
    """
    src_root = Path(__file__).resolve().parents[2] / "src" / "bitbat"
    violations: list[str] = []

    for py_file in sorted(src_root.rglob("*.py")):
        source = py_file.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                # Check if the test is a call to isinstance
                test = node.test
                if isinstance(test, ast.Call) and isinstance(test.func, ast.Name):  # noqa: SIM102
                    if test.func.id == "isinstance":
                        rel_path = py_file.relative_to(src_root.parent.parent)
                        violations.append(f"{rel_path}:{node.lineno}")

    assert violations == [], (
        f"Found 'assert isinstance' in production code (stripped by python -O): "
        f"{violations}"
    )
