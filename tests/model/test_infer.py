from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from bitbat.model.infer import predict_bar

pytestmark = pytest.mark.behavioral

def _train_model() -> tuple[xgb.Booster, pd.DataFrame]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(size=(120, 4)), columns=list("abcd")
    )
    y = rng.normal(0.0, 0.01, size=120)
    dtrain = xgb.DMatrix(
        X, label=y, feature_names=list(X.columns)
    )
    booster = xgb.train(
        {"objective": "reg:squarederror"},
        dtrain,
        num_boost_round=5,
    )
    return booster, X


def test_predict_bar_returns_regression_output() -> None:
    model, X = _train_model()
    row = X.iloc[-1]
    result = predict_bar(
        model,
        row,
        timestamp=datetime.now(UTC),
        current_price=50000.0,
    )
    assert "predicted_return" in result
    assert "predicted_price" in result
    assert "predicted_direction" in result
    assert isinstance(result["predicted_return"], float)
    assert result["predicted_direction"] in ("up", "down")


def test_predict_bar_missing_feature_raises() -> None:
    model, X = _train_model()
    row = X.iloc[-1]
    del row[row.index[0]]
    with pytest.raises(KeyError):
        predict_bar(model, row)
