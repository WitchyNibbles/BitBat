from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from bitbat.model.infer import predict_bar, predict_classification, predict_with_metadata
from bitbat.model.persist import save_baseline_artifact
from bitbat.model.train import fit_random_forest, fit_xgb

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


def test_predict_with_metadata_maps_triple_barrier_outputs_to_direction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(31)
    X = pd.DataFrame(rng.normal(size=(90, 4)), columns=list("abcd"))
    X.attrs["freq"] = "5m"
    X.attrs["horizon"] = "30m"
    y = pd.Series(rng.choice(["take_profit", "stop_loss", "timeout"], size=90))
    monkeypatch.chdir(tmp_path)

    booster, _ = fit_xgb(
        X,
        y,
        seed=0,
        persist=False,
        class_labels=["take_profit", "stop_loss", "timeout"],
    )
    result = predict_with_metadata(
        booster,
        X.iloc[-1],
        metadata={
            "family": "xgb",
            "label_mode": "triple_barrier",
            "class_labels": ["take_profit", "stop_loss", "timeout"],
        },
        current_price=100.0,
        tau=0.003,
    )

    assert result["predicted_label"] in {"take_profit", "stop_loss", "timeout"}
    assert result["predicted_direction"] in {"up", "down", "flat"}
    assert abs(result["p_up"] + result["p_down"] + result["p_flat"] - 1.0) < 1e-5


def test_predict_bar_rejects_non_direction_artifact(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    X = pd.DataFrame(rng.normal(size=(80, 4)), columns=list("abcd"))
    y = pd.Series(rng.choice(["take_profit", "stop_loss", "timeout"], size=80))
    X.attrs["freq"] = "1h"
    X.attrs["horizon"] = "4h"

    model, _ = fit_xgb(X, y, label_mode="triple_barrier", persist=False)
    artifact_path = save_baseline_artifact(
        model,
        family="xgb",
        freq="1h",
        horizon="4h",
        label_mode="triple_barrier",
        root=tmp_path,
    )

    with pytest.raises(ValueError, match="only supports direction artifacts"):
        predict_bar(artifact_path, X.iloc[-1])


def test_predict_with_metadata_supports_random_forest_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(37)
    X = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
    y = pd.Series(0.01 + (X["a"] * 0.005) - (X["b"] * 0.002))
    monkeypatch.chdir(tmp_path)

    model, _ = fit_random_forest(X, y, seed=7, persist=False)
    row = X.iloc[-1]
    result = predict_with_metadata(
        model,
        row,
        metadata={
            "family": "random_forest",
            "label_mode": "return_direction",
        },
        current_price=100.0,
        tau=0.01,
    )

    assert result["predicted_return"] is not None
    assert result["predicted_price"] is not None
    assert result["predicted_direction"] in {"up", "down", "flat"}


def test_predict_classification_returns_label_probabilities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(23)
    X = pd.DataFrame(rng.normal(size=(90, 4)), columns=list("abcd"))
    X.attrs["freq"] = "5m"
    X.attrs["horizon"] = "30m"
    y = pd.Series(rng.choice(["act", "pass"], size=90))
    monkeypatch.chdir(tmp_path)

    booster, _ = fit_xgb(
        X,
        y,
        seed=0,
        persist=False,
        class_labels=["act", "pass"],
    )
    result = predict_classification(
        booster,
        X.iloc[-1],
        class_labels=["act", "pass"],
        label_mode="meta_label",
    )

    assert result["predicted_label"] in {"act", "pass"}
    assert set(result["probabilities"]) == {"act", "pass"}
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_classification_supports_meta_label_artifact(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.normal(size=(80, 4)), columns=list("abcd"))
    y = pd.Series(rng.choice(["act", "pass"], size=80))
    X.attrs["freq"] = "1h"
    X.attrs["horizon"] = "4h"

    model, _ = fit_xgb(X, y, label_mode="meta_label", persist=False, class_labels=["act", "pass"])
    artifact_path = save_baseline_artifact(
        model,
        family="xgb",
        freq="1h",
        horizon="4h",
        label_mode="meta_label",
        artifact_role="action",
        root=tmp_path,
    )

    result = predict_classification(artifact_path, X.iloc[-1])

    assert result["label_mode"] == "meta_label"
    assert result["predicted_label"] in {"act", "pass"}
    assert set(result["probabilities"]) == {"act", "pass"}
