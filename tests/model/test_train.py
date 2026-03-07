from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:  # pragma: no cover - dependency guard
    import xgboost  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("xgboost not installed", allow_module_level=True)

from bitbat.model.train import fit_baseline, fit_random_forest, fit_xgb

pytestmark = pytest.mark.behavioral

def test_fit_xgb_trains_and_saves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(size=(50, 6)),
        columns=[f"f{i}" for i in range(6)],
    )
    X.attrs["freq"] = "1h"
    X.attrs["horizon"] = "2h"
    y = pd.Series(rng.normal(0.0, 0.01, size=50))

    monkeypatch.chdir(tmp_path)
    booster, importance = fit_xgb(X, y, seed=0)

    assert booster is not None
    assert set(importance.keys()) == set(X.columns)
    model_path = Path("models") / "1h_2h" / "xgb.json"
    assert model_path.exists()


@pytest.mark.parametrize("family", ["xgb", "random_forest"])
def test_fit_baseline_supports_both_families(
    family: str,
) -> None:
    rng = np.random.default_rng(7)
    X = pd.DataFrame(
        rng.normal(size=(64, 5)),
        columns=[f"feat_{i}" for i in range(5)],
    )
    y = pd.Series(rng.normal(0.0, 0.01, size=64))

    model, importance = fit_baseline(X, y, family=family, seed=11, persist=False)

    assert model is not None
    assert set(importance.keys()) == set(X.columns)


def test_fit_random_forest_is_deterministic_for_fixed_seed() -> None:
    rng = np.random.default_rng(13)
    X = pd.DataFrame(
        rng.normal(size=(96, 4)),
        columns=[f"feat_{i}" for i in range(4)],
    )
    y = pd.Series(rng.normal(0.0, 0.01, size=96))

    model_a, _ = fit_random_forest(X, y, seed=99, persist=False)
    model_b, _ = fit_random_forest(X, y, seed=99, persist=False)

    pred_a = model_a.predict(X.astype(float))
    pred_b = model_b.predict(X.astype(float))
    assert np.allclose(pred_a, pred_b)
