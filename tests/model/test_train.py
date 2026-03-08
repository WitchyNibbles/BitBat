from __future__ import annotations

import json
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


def test_fit_xgb_trains_and_saves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(size=(50, 6)),
        columns=[f"f{i}" for i in range(6)],
    )
    X.attrs["freq"] = "1h"
    X.attrs["horizon"] = "2h"
    y = pd.Series(rng.choice(["up", "down", "flat"], size=50))

    monkeypatch.chdir(tmp_path)
    booster, importance = fit_xgb(X, y, seed=0)

    assert booster is not None
    assert set(importance.keys()) == set(X.columns)
    model_path = Path("models") / "1h_2h" / "xgb.json"
    assert model_path.exists()


@pytest.mark.parametrize("family", ["xgb", "random_forest"])
def test_fit_baseline_supports_both_families(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    family: str,
) -> None:
    rng = np.random.default_rng(7)
    X = pd.DataFrame(
        rng.normal(size=(64, 5)),
        columns=[f"feat_{i}" for i in range(5)],
    )
    if family == "xgb":
        # XGBoost path now expects direction labels
        y = pd.Series(rng.choice(["up", "down", "flat"], size=64))
    else:
        # RandomForest path remains float regression
        y = pd.Series(rng.normal(0.0, 0.01, size=64))

    monkeypatch.chdir(tmp_path)
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


def test_fit_xgb_uses_classification_objective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(60, 4)), columns=[f"f{i}" for i in range(4)])
    X.attrs["freq"] = "1h"
    X.attrs["horizon"] = "4h"
    y = pd.Series(rng.choice(["up", "down", "flat"], size=60))
    monkeypatch.chdir(tmp_path)
    booster, _ = fit_xgb(X, y, seed=0)
    cfg = json.loads(booster.save_config())
    objective = cfg["learner"]["objective"]["name"]
    assert objective == "multi:softprob"


def test_fit_xgb_classification_output_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import xgboost as xgb

    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(60, 4)), columns=[f"f{i}" for i in range(4)])
    X.attrs["freq"] = "1h"
    X.attrs["horizon"] = "4h"
    y = pd.Series(rng.choice(["up", "down", "flat"], size=60))
    monkeypatch.chdir(tmp_path)
    booster, _ = fit_xgb(X, y, seed=0)
    dmatrix = xgb.DMatrix(X.astype(float), feature_names=list(X.columns))
    probs = booster.predict(dmatrix)
    assert probs.shape == (60, 3), f"Expected (60, 3), got {probs.shape}"
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_direction_classes_consistent_across_modules() -> None:
    """DIRECTION_CLASSES must be identical in train and infer to prevent silent accuracy collapse."""  # noqa: E501
    from bitbat.model import infer as infer_mod
    from bitbat.model import train as train_mod

    assert train_mod.DIRECTION_CLASSES == infer_mod.DIRECTION_CLASSES, (
        f"DIRECTION_CLASSES mismatch: train={train_mod.DIRECTION_CLASSES}, "
        f"infer={infer_mod.DIRECTION_CLASSES}"
    )
