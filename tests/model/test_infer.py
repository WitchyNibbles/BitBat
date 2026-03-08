from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from bitbat.model.infer import predict_bar
from bitbat.model.train import fit_xgb

pytestmark = pytest.mark.behavioral


def _train_regression_model() -> tuple[xgb.Booster, pd.DataFrame]:
    """Helper: regression booster for backward-compat tests."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
    y = rng.normal(0.0, 0.01, size=120)
    dtrain = xgb.DMatrix(X, label=y, feature_names=list(X.columns))
    booster = xgb.train(
        {"objective": "reg:squarederror"},
        dtrain,
        num_boost_round=5,
    )
    return booster, X


def _train_classification_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[xgb.Booster, pd.DataFrame]:
    """Helper: multi:softprob booster for new inference tests."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(90, 4)), columns=list("abcd"))
    X.attrs["freq"] = "1h"
    X.attrs["horizon"] = "4h"
    y = pd.Series(rng.choice(["up", "down", "flat"], size=90))
    monkeypatch.chdir(tmp_path)
    booster, _ = fit_xgb(X, y, seed=0)
    return booster, X


def test_predict_bar_missing_feature_raises() -> None:
    model, X = _train_regression_model()
    row = X.iloc[-1]
    del row[row.index[0]]
    with pytest.raises(KeyError):
        predict_bar(model, row)


def test_predict_bar_returns_three_classes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bug 2 FIXED: predict_bar returns 3-class direction via argmax, not sign-based binary."""
    model, X = _train_classification_model(tmp_path, monkeypatch)
    row = X.iloc[-1]
    result = predict_bar(
        model,
        row,
        timestamp=datetime.now(UTC),
    )
    assert result["predicted_direction"] in {"up", "down", "flat"}
    assert isinstance(result["p_flat"], float)
    assert 0.0 <= result["p_flat"] <= 1.0
    assert result["predicted_return"] is None


def test_predict_bar_p_values_sum_to_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """p_up + p_down + p_flat must sum to 1.0 for a valid probability distribution."""
    model, X = _train_classification_model(tmp_path, monkeypatch)
    row = X.iloc[-1]
    result = predict_bar(model, row)
    p_sum = result["p_up"] + result["p_down"] + result["p_flat"]
    assert abs(p_sum - 1.0) < 1e-5, f"p_up+p_down+p_flat = {p_sum}, expected 1.0"
