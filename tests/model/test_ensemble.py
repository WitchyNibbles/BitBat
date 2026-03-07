"""Tests for MultiHorizonEnsemble (regression mode)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from bitbat.model.ensemble import (
    EnsemblePrediction,
    MultiHorizonEnsemble,
)

pytestmark = pytest.mark.integration

@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train tiny regression models for 15m, 30m, 1h."""
    root = tmp_path_factory.mktemp("models")
    rng = np.random.default_rng(42)

    for horizon in ("15m", "30m", "1h"):
        d = root / f"5m_{horizon}"
        d.mkdir()
        X = pd.DataFrame(
            {
                "feat_a": rng.normal(size=50),
                "feat_b": rng.normal(size=50),
            }
        )
        y = rng.normal(0.0, 0.01, size=50)
        dtrain = xgb.DMatrix(
            X, label=y, feature_names=list(X.columns)
        )
        booster = xgb.train(
            {
                "objective": "reg:squarederror",
                "max_depth": 2,
                "seed": int(hash(horizon) % 1000),
            },
            dtrain,
            num_boost_round=5,
        )
        booster.save_model(str(d / "xgb.json"))

    return root


@pytest.fixture()
def features() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"feat_a": [rng.normal()], "feat_b": [rng.normal()]}
    )


@pytest.fixture()
def ensemble(model_dir: Path) -> MultiHorizonEnsemble:
    return MultiHorizonEnsemble(
        model_dir,
        freq="5m",
        horizons=["15m", "30m", "1h"],
    )


class TestMultiHorizonEnsemble:
    def test_available_horizons(
        self, ensemble: MultiHorizonEnsemble
    ) -> None:
        avail = ensemble.available_horizons()
        assert set(avail) == {"15m", "30m", "1h"}

    def test_missing_horizon_excluded(
        self, model_dir: Path
    ) -> None:
        ens = MultiHorizonEnsemble(
            model_dir,
            freq="5m",
            horizons=["15m", "30m", "99h"],
        )
        assert "99h" not in ens.available_horizons()

    def test_predict_returns_ensemble(
        self,
        ensemble: MultiHorizonEnsemble,
        features: pd.DataFrame,
    ) -> None:
        pred = ensemble.predict(features)
        assert isinstance(pred, EnsemblePrediction)

    def test_predicted_return_is_float(
        self,
        ensemble: MultiHorizonEnsemble,
        features: pd.DataFrame,
    ) -> None:
        pred = ensemble.predict(features)
        assert isinstance(pred.predicted_return, float)

    def test_direction_is_valid(
        self,
        ensemble: MultiHorizonEnsemble,
        features: pd.DataFrame,
    ) -> None:
        pred = ensemble.predict(features)
        assert pred.predicted_direction in ("up", "down")

    def test_confidence_in_range(
        self,
        ensemble: MultiHorizonEnsemble,
        features: pd.DataFrame,
    ) -> None:
        pred = ensemble.predict(features)
        assert 0.0 <= pred.confidence <= 1.0

    def test_horizon_predictions_included(
        self,
        ensemble: MultiHorizonEnsemble,
        features: pd.DataFrame,
    ) -> None:
        pred = ensemble.predict(features)
        assert len(pred.horizon_predictions) == 3

    def test_custom_weights(
        self, model_dir: Path, features: pd.DataFrame
    ) -> None:
        ens = MultiHorizonEnsemble(
            model_dir,
            freq="5m",
            horizons=["15m", "30m", "1h"],
            weights={
                "15m": 1.0,
                "30m": 2.0,
                "1h": 3.0,
            },
        )
        pred = ens.predict(features)
        assert isinstance(pred.predicted_return, float)

    def test_summary_dict(
        self,
        ensemble: MultiHorizonEnsemble,
        features: pd.DataFrame,
    ) -> None:
        pred = ensemble.predict(features)
        s = pred.summary()
        assert "predicted_direction" in s
        assert "confidence" in s
        assert "horizons_used" in s
        assert s["horizons_used"] == 3

    def test_predict_batch(
        self, ensemble: MultiHorizonEnsemble
    ) -> None:
        rng = np.random.default_rng(1)
        batch_features = pd.DataFrame(
            {
                "feat_a": rng.normal(size=5),
                "feat_b": rng.normal(size=5),
            }
        )
        preds = ensemble.predict_batch(batch_features)
        assert len(preds) == 5
        assert all(
            isinstance(p, EnsemblePrediction)
            for p in preds
        )

    def test_no_models_raises(self, tmp_path: Path) -> None:
        ens = MultiHorizonEnsemble(
            tmp_path, horizons=["1h"]
        )
        feat = pd.DataFrame({"a": [1.0]})
        with pytest.raises(
            ValueError, match="No models available"
        ):
            ens.predict(feat)

    def test_booster_caching(
        self,
        ensemble: MultiHorizonEnsemble,
        features: pd.DataFrame,
    ) -> None:
        ensemble.predict(features)
        assert len(ensemble._boosters) == 3
        # Second call should reuse cached boosters
        ensemble.predict(features)
        assert len(ensemble._boosters) == 3
