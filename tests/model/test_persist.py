from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from bitbat.model.persist import (
    default_model_artifact_path,
    load,
    load_baseline_artifact,
    save,
    save_baseline_artifact,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("seed", [0])
def test_save_load_roundtrip(tmp_path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(100, 4)), columns=list("abcd"))
    y = pd.Series(rng.integers(0, 3, size=100))

    dtrain = xgb.DMatrix(X, label=y, feature_names=list(X.columns))
    params = {"objective": "multi:softprob", "num_class": 3}
    booster = xgb.train(params, dtrain, num_boost_round=5)

    path = tmp_path / "model.json"
    save(booster, path)
    loaded = load(path)

    preds_orig = booster.predict(dtrain)
    preds_loaded = loaded.predict(dtrain)

    assert np.allclose(preds_orig, preds_loaded)


def test_save_load_roundtrip_random_forest(tmp_path: Path) -> None:
    rng = np.random.default_rng(12)
    X = pd.DataFrame(rng.normal(size=(120, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.normal(size=120))

    model = RandomForestRegressor(n_estimators=50, random_state=7)
    model.fit(X, y)

    path = tmp_path / "random_forest.pkl"
    save(model, path, family="random_forest", metadata={"family": "random_forest"})
    loaded = load(path, family="random_forest")

    assert isinstance(loaded, RandomForestRegressor)
    assert np.allclose(model.predict(X), loaded.predict(X))
    metadata_path = path.with_suffix(".meta.json")
    assert metadata_path.exists()


def test_baseline_artifact_helpers_use_stable_paths(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(size=(100, 4)), columns=list("abcd"))
    y = pd.Series(rng.normal(size=100))

    model = RandomForestRegressor(n_estimators=25, random_state=9)
    model.fit(X, y)

    artifact_path = save_baseline_artifact(
        model,
        family="random_forest",
        freq="1h",
        horizon="4h",
        root=tmp_path,
        metadata={"source": "unit-test"},
    )

    expected_path = default_model_artifact_path("1h", "4h", family="random_forest", root=tmp_path)
    assert artifact_path == expected_path
    assert artifact_path.exists()

    loaded = load_baseline_artifact(
        "1h",
        "4h",
        family="random_forest",
        root=tmp_path,
    )
    assert isinstance(loaded, RandomForestRegressor)
    assert np.allclose(model.predict(X), loaded.predict(X))
