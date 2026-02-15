"""
Phase 3 Complete Integration Test.

Exercises the full analytics pipeline end-to-end:
  feature_analysis → explainer (SHAP) → backtest_report → scenario comparison
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from bitbat.analytics.backtest_report import BacktestReport, compare_scenarios
from bitbat.analytics.explainer import PredictionExplainer
from bitbat.analytics.feature_analysis import FeatureAnalyzer
from bitbat.backtest.engine import run as backtest_run


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a realistic synthetic feature dataset parquet."""
    rng = np.random.default_rng(2024)
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")

    data = {
        "timestamp_utc": idx,
        "feat_ret_1": rng.normal(0, 0.01, n),
        "feat_ret_4": rng.normal(0, 0.015, n),
        "feat_vol_24": rng.uniform(0.01, 0.05, n),
        "feat_rsi_14": rng.uniform(20, 80, n),
        "feat_sent_1h_mean": rng.normal(0, 0.2, n),
        "feat_sent_4h_mean": rng.normal(0, 0.2, n),
        "label": rng.choice(["up", "down", "flat"], size=n, p=[0.35, 0.35, 0.30]),
        "r_forward": rng.normal(0, 0.01, n),
    }
    df = pd.DataFrame(data)
    path = tmp_path_factory.mktemp("data") / "dataset.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture(scope="module")
def analyzer(dataset_path: Path) -> FeatureAnalyzer:
    return FeatureAnalyzer(dataset_path)


@pytest.fixture(scope="module")
def model_path(tmp_path_factory: pytest.TempPathFactory, dataset_path: Path) -> Path:
    """Train a tiny XGBoost model on the synthetic dataset."""
    df = pd.read_parquet(dataset_path)
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    X = df[feature_cols].astype(float)
    y = pd.Categorical(df["label"]).codes  # down=0, flat=1, up=2

    dtrain = xgb.DMatrix(X, label=y, feature_names=list(X.columns))
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "seed": 42,
        "max_depth": 2,
    }
    booster = xgb.train(params, dtrain, num_boost_round=5)

    model_dir = tmp_path_factory.mktemp("models")
    path = model_dir / "xgb.json"
    booster.save_model(str(path))
    return path


@pytest.fixture(scope="module")
def explainer(model_path: Path) -> PredictionExplainer:
    return PredictionExplainer(model_path)


# ---------------------------------------------------------------------------
# Phase 3 end-to-end integration
# ---------------------------------------------------------------------------


class TestPhase3Integration:
    def test_feature_analysis_loads(self, analyzer: FeatureAnalyzer) -> None:
        df = analyzer.load()
        assert len(df) == 300
        assert len(analyzer.feature_cols) == 6

    def test_correlation_matrix_shape(self, analyzer: FeatureAnalyzer) -> None:
        mat = analyzer.correlation_matrix()
        assert "label_num" in mat.columns
        assert mat.shape[0] == mat.shape[1]

    def test_top_features_has_groups(self, analyzer: FeatureAnalyzer) -> None:
        top = analyzer.top_correlated_features(n=6)
        assert "group" in top.columns
        assert len(top) == 6

    def test_feature_groups_cover_all(self, analyzer: FeatureAnalyzer) -> None:
        groups = analyzer.feature_groups()
        all_grouped = {col for cols in groups.values() for col in cols}
        for feat in analyzer.feature_cols:
            assert feat in all_grouped

    def test_explainer_shap_shape(
        self, analyzer: FeatureAnalyzer, explainer: PredictionExplainer
    ) -> None:
        df = analyzer.load()
        feature_cols = analyzer.feature_cols
        X = df[feature_cols].dropna().astype(float).head(20)
        contribs = explainer.shap_values(X)
        # Multi-class: (n, 3, n_features+1) or (n, n_features+1)
        assert contribs.shape[0] == 20

    def test_explainer_feature_importance_nonempty(
        self, explainer: PredictionExplainer
    ) -> None:
        imp = explainer.feature_importance_from_model()
        assert isinstance(imp, pd.Series)
        assert len(imp) > 0

    def test_explainer_explain_row_keys(
        self, analyzer: FeatureAnalyzer, explainer: PredictionExplainer
    ) -> None:
        df = analyzer.load()
        row = df[analyzer.feature_cols].dropna().iloc[:1]
        result = explainer.explain_row(row)
        for key in ("contributions", "plain_english", "predicted_class"):
            assert key in result

    def test_batch_mean_shap_all_features(
        self, analyzer: FeatureAnalyzer, explainer: PredictionExplainer
    ) -> None:
        df = analyzer.load()
        X = df[analyzer.feature_cols].dropna().astype(float).head(50)
        shap_df = explainer.batch_mean_shap(X)
        assert set(analyzer.feature_cols) <= set(shap_df["feature"].tolist())

    def test_backtest_engine_runs(self, analyzer: FeatureAnalyzer) -> None:
        df = analyzer.load()
        n = len(df)
        rng = np.random.default_rng(42)
        idx = df.index[:n]
        close = pd.Series(np.cumprod(1 + df["feat_ret_1"].fillna(0)) * 100, index=idx)
        pu = pd.Series(rng.uniform(0.3, 0.7, n), index=idx)
        pd_ = pd.Series(rng.uniform(0.2, 0.5, n), index=idx)
        trades, equity = backtest_run(close, pu, pd_, enter=0.65)
        assert len(equity) == n
        assert "position" in trades.columns

    def test_backtest_report_from_engine(self, analyzer: FeatureAnalyzer) -> None:
        df = analyzer.load()
        n = len(df)
        rng = np.random.default_rng(1)
        idx = df.index
        close = pd.Series(np.cumprod(1 + df["feat_ret_1"].fillna(0)) * 100, index=idx)
        pu = pd.Series(rng.uniform(0.3, 0.7, n), index=idx)
        pd_ = pd.Series(rng.uniform(0.2, 0.5, n), index=idx)
        trades, equity = backtest_run(close, pu, pd_, enter=0.65)
        report = BacktestReport(equity, trades, preset_name="Integration")
        m = report.metrics()
        assert "sharpe" in m
        assert "total_return" in m
        assert isinstance(report.plain_summary(), str)
        assert report.rating() in ("Excellent", "Good", "Fair", "Poor")

    def test_scenario_comparison_three_presets(
        self, analyzer: FeatureAnalyzer
    ) -> None:
        df = analyzer.load()
        n = len(df)
        rng = np.random.default_rng(42)
        idx = df.index
        close = pd.Series(np.cumprod(1 + df["feat_ret_1"].fillna(0)) * 100, index=idx)
        pu = pd.Series(rng.uniform(0.3, 0.7, n), index=idx)
        pd_ = pd.Series(rng.uniform(0.2, 0.5, n), index=idx)

        thresholds = [0.75, 0.65, 0.55]
        names = ["Conservative", "Balanced", "Aggressive"]
        reports = []
        for thr, name in zip(thresholds, names):
            t, eq = backtest_run(close, pu, pd_, enter=thr)
            reports.append(BacktestReport(eq, t, preset_name=name, enter_threshold=thr))

        comparison = compare_scenarios(reports)
        assert len(comparison) == 3
        assert list(comparison["Scenario"]) == names

    def test_full_pipeline_no_exceptions(
        self, analyzer: FeatureAnalyzer, explainer: PredictionExplainer
    ) -> None:
        """End-to-end smoke test: feature analysis → SHAP → backtest → comparison."""
        # 1. Feature analysis
        df = analyzer.load()
        top = analyzer.top_correlated_features(n=5)
        assert len(top) == 5

        # 2. SHAP
        X_sample = df[analyzer.feature_cols].dropna().astype(float).head(30)
        shap_df = explainer.batch_mean_shap(X_sample)
        assert len(shap_df) > 0

        # 3. Backtest
        rng = np.random.default_rng(0)
        n = len(df)
        idx = df.index
        close = pd.Series(np.cumprod(1 + df["feat_ret_1"].fillna(0)) * 100, index=idx)
        pu = pd.Series(rng.uniform(0.4, 0.8, n), index=idx)
        pd_ = pd.Series(rng.uniform(0.1, 0.5, n), index=idx)
        trades, equity = backtest_run(close, pu, pd_, enter=0.65)

        # 4. Report
        report = BacktestReport(equity, trades, preset_name="Smoke Test")
        assert report.rating() in ("Excellent", "Good", "Fair", "Poor")
        summary = report.plain_summary()
        assert "Smoke Test" in summary
