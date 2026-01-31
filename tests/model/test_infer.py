from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from bitbat.model.infer import predict_bar


def _train_model() -> tuple[xgb.Booster, pd.DataFrame]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
    y = pd.Series(rng.integers(0, 3, size=120))
    dtrain = xgb.DMatrix(X, label=y, feature_names=list(X.columns))
    booster = xgb.train({"objective": "multi:softprob", "num_class": 3}, dtrain, num_boost_round=5)
    return booster, X


def test_predict_bar_probabilities_sum_to_one() -> None:
    model, X = _train_model()
    row = X.iloc[-1]
    result = predict_bar(model, row, timestamp=datetime.now(UTC))
    assert 0 <= result["p_up"] <= 1
    assert 0 <= result["p_down"] <= 1


def test_predict_bar_missing_feature_raises() -> None:
    model, X = _train_model()
    row = X.iloc[-1]
    del row[row.index[0]]
    with pytest.raises(KeyError):
        predict_bar(model, row)
