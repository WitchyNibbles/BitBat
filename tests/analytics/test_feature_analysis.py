"""Tests for the feature analysis module (Phase 3, Session 1)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bitbat.analytics.feature_analysis import FeatureAnalyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_dataset(tmp_path: Path) -> Path:
    """Create a minimal feature dataset parquet file for testing."""
    rng = np.random.default_rng(42)
    n = 200

    # Simulate a datetime index
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
    }

    df = pd.DataFrame(data).set_index("timestamp_utc")
    path = tmp_path / "dataset.parquet"
    df.to_parquet(path)
    return path


@pytest.fixture()
def analyzer(sample_dataset: Path) -> FeatureAnalyzer:
    return FeatureAnalyzer(sample_dataset)


# ---------------------------------------------------------------------------
# FeatureAnalyzer.load
# ---------------------------------------------------------------------------


class TestLoad:
    def test_loads_dataframe(self, analyzer: FeatureAnalyzer) -> None:
        df = analyzer.load()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 200

    def test_cached_on_second_call(self, analyzer: FeatureAnalyzer) -> None:
        df1 = analyzer.load()
        df2 = analyzer.load()
        assert df1 is df2  # same object (cached)

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        a = FeatureAnalyzer(tmp_path / "nonexistent.parquet")
        with pytest.raises(Exception):
            a.load()

    def test_feature_cols_detected(self, analyzer: FeatureAnalyzer) -> None:
        cols = analyzer.feature_cols
        assert all(c.startswith("feat_") for c in cols)
        assert len(cols) == 6


# ---------------------------------------------------------------------------
# FeatureAnalyzer.correlation_matrix
# ---------------------------------------------------------------------------


class TestCorrelationMatrix:
    def test_pearson_shape(self, analyzer: FeatureAnalyzer) -> None:
        mat = analyzer.correlation_matrix(method="pearson")
        assert isinstance(mat, pd.DataFrame)
        # Should include feature cols + label_num
        assert "label_num" in mat.columns
        assert mat.shape[0] == mat.shape[1]

    def test_spearman_shape(self, analyzer: FeatureAnalyzer) -> None:
        mat = analyzer.correlation_matrix(method="spearman")
        assert "label_num" in mat.columns

    def test_diagonal_is_one(self, analyzer: FeatureAnalyzer) -> None:
        mat = analyzer.correlation_matrix()
        diag = np.diag(mat.values)
        np.testing.assert_allclose(diag, 1.0, atol=1e-6)

    def test_invalid_method_raises(self, analyzer: FeatureAnalyzer) -> None:
        with pytest.raises(ValueError, match="method must be"):
            analyzer.correlation_matrix(method="cosine")

    def test_values_in_range(self, analyzer: FeatureAnalyzer) -> None:
        mat = analyzer.correlation_matrix()
        assert (mat.values >= -1.0 - 1e-9).all()
        assert (mat.values <= 1.0 + 1e-9).all()


# ---------------------------------------------------------------------------
# FeatureAnalyzer.feature_label_correlations
# ---------------------------------------------------------------------------


class TestFeatureLabelCorrelations:
    def test_returns_series(self, analyzer: FeatureAnalyzer) -> None:
        corrs = analyzer.feature_label_correlations()
        assert isinstance(corrs, pd.Series)

    def test_all_feature_cols_present(self, analyzer: FeatureAnalyzer) -> None:
        corrs = analyzer.feature_label_correlations()
        for feat in analyzer.feature_cols:
            assert feat in corrs.index

    def test_sorted_by_absolute_value(self, analyzer: FeatureAnalyzer) -> None:
        corrs = analyzer.feature_label_correlations()
        abs_vals = corrs.abs().tolist()
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_label_num_not_in_result(self, analyzer: FeatureAnalyzer) -> None:
        corrs = analyzer.feature_label_correlations()
        assert "label_num" not in corrs.index


# ---------------------------------------------------------------------------
# FeatureAnalyzer.top_correlated_features
# ---------------------------------------------------------------------------


class TestTopCorrelatedFeatures:
    def test_returns_dataframe(self, analyzer: FeatureAnalyzer) -> None:
        df = analyzer.top_correlated_features(n=5)
        assert isinstance(df, pd.DataFrame)
        assert len(df) <= 5

    def test_required_columns(self, analyzer: FeatureAnalyzer) -> None:
        df = analyzer.top_correlated_features()
        assert "feature" in df.columns
        assert "correlation" in df.columns
        assert "abs_correlation" in df.columns
        assert "group" in df.columns

    def test_n_respected(self, analyzer: FeatureAnalyzer) -> None:
        df = analyzer.top_correlated_features(n=3)
        assert len(df) == 3


# ---------------------------------------------------------------------------
# FeatureAnalyzer.feature_groups
# ---------------------------------------------------------------------------


class TestFeatureGroups:
    def test_returns_dict(self, analyzer: FeatureAnalyzer) -> None:
        groups = analyzer.feature_groups()
        assert isinstance(groups, dict)

    def test_all_features_in_some_group(self, analyzer: FeatureAnalyzer) -> None:
        groups = analyzer.feature_groups()
        all_grouped = {col for cols in groups.values() for col in cols}
        for feat in analyzer.feature_cols:
            assert feat in all_grouped

    def test_no_empty_groups(self, analyzer: FeatureAnalyzer) -> None:
        groups = analyzer.feature_groups()
        for key, cols in groups.items():
            assert len(cols) > 0


# ---------------------------------------------------------------------------
# FeatureAnalyzer.feature_summary
# ---------------------------------------------------------------------------


class TestFeatureSummary:
    def test_returns_dataframe(self, analyzer: FeatureAnalyzer) -> None:
        summary = analyzer.feature_summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == len(analyzer.feature_cols)

    def test_has_mean_std(self, analyzer: FeatureAnalyzer) -> None:
        summary = analyzer.feature_summary()
        assert "mean" in summary.columns
        assert "std" in summary.columns
