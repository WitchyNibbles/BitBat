"""Tests for the prediction explainer module (Phase 3, Session 2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from bitbat.analytics.explainer import (
    PredictionExplainer,
    _build_plain_explanation,
    _readable_feature_name,
)

# ---------------------------------------------------------------------------
# Helpers — create a minimal trained model
# ---------------------------------------------------------------------------


def _train_tiny_model(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    """Train a tiny XGBoost model and return (model_path, test_X)."""
    rng = np.random.default_rng(99)
    n = 150
    feat_names = ["feat_ret_1", "feat_vol_24", "feat_rsi_14", "feat_sent_1h_mean"]
    X = pd.DataFrame(rng.normal(0, 1, (n, len(feat_names))), columns=feat_names)
    y_raw = rng.choice(["up", "down", "flat"], size=n)

    # Encode labels to ints
    label_enc = {"down": 0, "flat": 1, "up": 2}
    y = pd.Series([label_enc[v] for v in y_raw])

    dtrain = xgb.DMatrix(X.astype(float), label=y.to_numpy(), feature_names=feat_names)
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 2,
        "n_estimators": 5,
        "eta": 0.3,
        "seed": 42,
    }
    booster = xgb.train(params, dtrain, num_boost_round=5, verbose_eval=False)

    model_dir = tmp_path / "models" / "1h_4h"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "xgb.json"
    booster.save_model(str(model_path))

    return model_path, X


@pytest.fixture()
def trained_model(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    return _train_tiny_model(tmp_path)


@pytest.fixture()
def explainer(trained_model: tuple[Path, pd.DataFrame]) -> PredictionExplainer:
    model_path, _ = trained_model
    return PredictionExplainer(model_path)


# ---------------------------------------------------------------------------
# PredictionExplainer — model loading
# ---------------------------------------------------------------------------


class TestModelLoading:
    def test_loads_model_lazily(self, explainer: PredictionExplainer) -> None:
        assert explainer._booster is None
        explainer._load_model()
        assert explainer._booster is not None

    def test_model_cached(self, explainer: PredictionExplainer) -> None:
        b1 = explainer._load_model()
        b2 = explainer._load_model()
        assert b1 is b2

    def test_raises_on_missing_model(self, tmp_path: Path) -> None:
        ex = PredictionExplainer(tmp_path / "no_model.json")
        with pytest.raises(xgb.core.XGBoostError):  # noqa: B017
            ex._load_model()


# ---------------------------------------------------------------------------
# PredictionExplainer.shap_values
# ---------------------------------------------------------------------------


class TestShapValues:
    def test_returns_array(self, trained_model: tuple[Path, pd.DataFrame]) -> None:
        model_path, X = trained_model
        ex = PredictionExplainer(model_path)
        contribs = ex.shap_values(X.head(10))
        assert isinstance(contribs, np.ndarray)
        assert contribs.shape[0] == 10

    def test_feature_axis_matches_input(self, trained_model: tuple[Path, pd.DataFrame]) -> None:
        model_path, X = trained_model
        ex = PredictionExplainer(model_path)
        contribs = ex.shap_values(X.head(5))
        # Last axis or inner axis should be n_features + 1 (bias)
        n_feats = X.shape[1]
        if contribs.ndim == 3:
            assert contribs.shape[2] == n_feats + 1
        else:
            assert contribs.shape[1] == n_feats + 1


# ---------------------------------------------------------------------------
# PredictionExplainer.explain_row
# ---------------------------------------------------------------------------


class TestExplainRow:
    def test_returns_dict(self, trained_model: tuple[Path, pd.DataFrame]) -> None:
        model_path, X = trained_model
        ex = PredictionExplainer(model_path)
        result = ex.explain_row(X.iloc[0])
        assert isinstance(result, dict)

    def test_required_keys(self, trained_model: tuple[Path, pd.DataFrame]) -> None:
        model_path, X = trained_model
        ex = PredictionExplainer(model_path)
        result = ex.explain_row(X.iloc[0])
        for key in ("contributions", "top_positive", "top_negative", "plain_english"):
            assert key in result

    def test_contributions_is_series(self, trained_model: tuple[Path, pd.DataFrame]) -> None:
        model_path, X = trained_model
        ex = PredictionExplainer(model_path)
        result = ex.explain_row(X.iloc[0])
        assert isinstance(result["contributions"], pd.Series)

    def test_plain_english_is_string(self, trained_model: tuple[Path, pd.DataFrame]) -> None:
        model_path, X = trained_model
        ex = PredictionExplainer(model_path)
        result = ex.explain_row(X.iloc[0])
        assert isinstance(result["plain_english"], str)
        assert len(result["plain_english"]) > 10

    def test_works_with_dataframe_row(self, trained_model: tuple[Path, pd.DataFrame]) -> None:
        model_path, X = trained_model
        ex = PredictionExplainer(model_path)
        result = ex.explain_row(X.iloc[0:1])  # DataFrame row
        assert isinstance(result, dict)

    def test_with_label_map(self, trained_model: tuple[Path, pd.DataFrame]) -> None:
        model_path, X = trained_model
        ex = PredictionExplainer(model_path)
        label_map = {0: "down", 1: "flat", 2: "up"}
        result = ex.explain_row(X.iloc[0], label_map=label_map)
        assert result["predicted_direction"] in ("up", "down", "flat")


# ---------------------------------------------------------------------------
# PredictionExplainer.feature_importance_from_model
# ---------------------------------------------------------------------------


class TestFeatureImportance:
    def test_returns_series(self, explainer: PredictionExplainer) -> None:
        imp = explainer.feature_importance_from_model()
        assert isinstance(imp, pd.Series)

    def test_all_positive(self, explainer: PredictionExplainer) -> None:
        imp = explainer.feature_importance_from_model()
        assert (imp >= 0).all()

    def test_sorted_descending(self, explainer: PredictionExplainer) -> None:
        imp = explainer.feature_importance_from_model()
        vals = imp.tolist()
        assert vals == sorted(vals, reverse=True)


# ---------------------------------------------------------------------------
# PredictionExplainer.batch_mean_shap
# ---------------------------------------------------------------------------


class TestBatchMeanShap:
    def test_returns_dataframe(self, trained_model: tuple[Path, pd.DataFrame]) -> None:
        model_path, X = trained_model
        ex = PredictionExplainer(model_path)
        result = ex.batch_mean_shap(X.head(20))
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, trained_model: tuple[Path, pd.DataFrame]) -> None:
        model_path, X = trained_model
        ex = PredictionExplainer(model_path)
        result = ex.batch_mean_shap(X.head(20))
        assert "feature" in result.columns
        assert "mean_abs_shap" in result.columns
        assert "rank" in result.columns

    def test_all_features_present(self, trained_model: tuple[Path, pd.DataFrame]) -> None:
        model_path, X = trained_model
        ex = PredictionExplainer(model_path)
        result = ex.batch_mean_shap(X)
        assert set(result["feature"]) == set(X.columns)

    def test_values_non_negative(self, trained_model: tuple[Path, pd.DataFrame]) -> None:
        model_path, X = trained_model
        ex = PredictionExplainer(model_path)
        result = ex.batch_mean_shap(X)
        assert (result["mean_abs_shap"] >= 0).all()

    def test_rank_starts_at_one(self, trained_model: tuple[Path, pd.DataFrame]) -> None:
        model_path, X = trained_model
        ex = PredictionExplainer(model_path)
        result = ex.batch_mean_shap(X)
        assert result["rank"].iloc[0] == 1


# ---------------------------------------------------------------------------
# Plain-language helpers
# ---------------------------------------------------------------------------


class TestReadableFeatureName:
    def test_known_feature_translated(self) -> None:
        name = _readable_feature_name("feat_ret_1")
        assert name == "1-hour price return"

    def test_unknown_feature_humanized(self) -> None:
        name = _readable_feature_name("feat_custom_signal")
        assert "custom signal" in name.lower()

    def test_sentiment_feature_translated(self) -> None:
        name = _readable_feature_name("feat_sent_1h_mean")
        assert "sentiment" in name.lower()


class TestBuildPlainExplanation:
    def test_contains_direction(self) -> None:
        pos = pd.Series([0.1], index=["feat_ret_1"])
        neg = pd.Series([-0.05], index=["feat_vol_24"])
        text = _build_plain_explanation(pos, neg, "up")
        assert "UP" in text

    def test_contains_feature_name(self) -> None:
        pos = pd.Series([0.12], index=["feat_ret_1"])
        text = _build_plain_explanation(pos, pd.Series(dtype=float), "down")
        assert "price return" in text.lower()

    def test_handles_empty_contributions(self) -> None:
        text = _build_plain_explanation(pd.Series(dtype=float), pd.Series(dtype=float), "flat")
        assert isinstance(text, str)
        assert len(text) > 0
