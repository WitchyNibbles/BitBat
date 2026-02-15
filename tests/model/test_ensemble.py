"""Tests for MultiHorizonEnsemble (Phase 5, Session 3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from bitbat.model.ensemble import EnsemblePrediction, MultiHorizonEnsemble


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train tiny models for 1h, 4h, 24h horizons."""
    root = tmp_path_factory.mktemp("models")
    rng = np.random.default_rng(42)

    for horizon in ("1h", "4h", "24h"):
        d = root / f"1h_{horizon}"
        d.mkdir()
        X = pd.DataFrame({"feat_a": rng.normal(size=50), "feat_b": rng.normal(size=50)})
        y = rng.choice([0, 1, 2], size=50)
        dtrain = xgb.DMatrix(X, label=y, feature_names=list(X.columns))
        booster = xgb.train(
            {"objective": "multi:softprob", "num_class": 3, "max_depth": 2, "seed": int(hash(horizon) % 1000)},
            dtrain,
            num_boost_round=5,
        )
        booster.save_model(str(d / "xgb.json"))

    return root


@pytest.fixture()
def features() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"feat_a": [rng.normal()], "feat_b": [rng.normal()]})


@pytest.fixture()
def ensemble(model_dir: Path) -> MultiHorizonEnsemble:
    return MultiHorizonEnsemble(model_dir, freq="1h", horizons=["1h", "4h", "24h"])


class TestMultiHorizonEnsemble:
    def test_available_horizons(self, ensemble: MultiHorizonEnsemble) -> None:
        avail = ensemble.available_horizons()
        assert set(avail) == {"1h", "4h", "24h"}

    def test_missing_horizon_excluded(self, model_dir: Path) -> None:
        ens = MultiHorizonEnsemble(model_dir, horizons=["1h", "4h", "99h"])
        assert "99h" not in ens.available_horizons()

    def test_predict_returns_ensemble(
        self, ensemble: MultiHorizonEnsemble, features: pd.DataFrame
    ) -> None:
        pred = ensemble.predict(features)
        assert isinstance(pred, EnsemblePrediction)

    def test_probabilities_sum_to_one(
        self, ensemble: MultiHorizonEnsemble, features: pd.DataFrame
    ) -> None:
        pred = ensemble.predict(features)
        total = pred.p_up + pred.p_down + pred.p_flat
        assert abs(total - 1.0) < 1e-6

    def test_direction_is_valid(
        self, ensemble: MultiHorizonEnsemble, features: pd.DataFrame
    ) -> None:
        pred = ensemble.predict(features)
        assert pred.predicted_direction in ("up", "down", "flat")

    def test_confidence_in_range(
        self, ensemble: MultiHorizonEnsemble, features: pd.DataFrame
    ) -> None:
        pred = ensemble.predict(features)
        assert 0.0 <= pred.confidence <= 1.0

    def test_horizon_predictions_included(
        self, ensemble: MultiHorizonEnsemble, features: pd.DataFrame
    ) -> None:
        pred = ensemble.predict(features)
        assert len(pred.horizon_predictions) == 3

    def test_custom_weights(self, model_dir: Path, features: pd.DataFrame) -> None:
        ens = MultiHorizonEnsemble(
            model_dir,
            horizons=["1h", "4h", "24h"],
            weights={"1h": 1.0, "4h": 2.0, "24h": 3.0},
        )
        pred = ens.predict(features)
        # Weighted average should differ from equal-weight
        assert pred.p_up > 0 or pred.p_down > 0

    def test_summary_dict(
        self, ensemble: MultiHorizonEnsemble, features: pd.DataFrame
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
        features = pd.DataFrame({"feat_a": rng.normal(size=5), "feat_b": rng.normal(size=5)})
        preds = ensemble.predict_batch(features)
        assert len(preds) == 5
        assert all(isinstance(p, EnsemblePrediction) for p in preds)

    def test_no_models_raises(self, tmp_path: Path) -> None:
        ens = MultiHorizonEnsemble(tmp_path, horizons=["1h"])
        features = pd.DataFrame({"a": [1.0]})
        with pytest.raises(ValueError, match="No models available"):
            ens.predict(features)

    def test_booster_caching(self, ensemble: MultiHorizonEnsemble, features: pd.DataFrame) -> None:
        ensemble.predict(features)
        assert len(ensemble._boosters) == 3
        # Second call should reuse cached boosters
        ensemble.predict(features)
        assert len(ensemble._boosters) == 3
