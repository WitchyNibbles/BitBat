from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from bitbat.model.persist import load, save


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
