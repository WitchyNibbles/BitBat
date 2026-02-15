"""
Feature correlation and importance analysis for BitBat.

Provides tools to understand which features drive predictions,
how correlated features are, and how importance changes over time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats


class FeatureAnalyzer:
    """Analyze correlations and importance of features in a BitBat dataset.

    Parameters
    ----------
    dataset_path:
        Path to a feature dataset parquet file (must have ``feat_*`` columns and a
        ``label`` column).
    """

    def __init__(self, dataset_path: Path | str) -> None:
        self.dataset_path = Path(dataset_path)
        self._df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Load the dataset, caching it in memory."""
        if self._df is None:
            self._df = pd.read_parquet(self.dataset_path)
        return self._df

    @property
    def feature_cols(self) -> list[str]:
        df = self.load()
        return [c for c in df.columns if c.startswith("feat_")]

    # ------------------------------------------------------------------
    # Correlation analysis
    # ------------------------------------------------------------------

    def correlation_matrix(self, method: str = "pearson") -> pd.DataFrame:
        """Compute a feature × feature (+ label) correlation matrix.

        Parameters
        ----------
        method:
            ``'pearson'`` or ``'spearman'``.

        Returns
        -------
        Correlation DataFrame (features + label encoded as ±1/0).
        """
        if method not in ("pearson", "spearman"):
            raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")

        df = self.load()
        cols = self.feature_cols.copy()

        if "label" in df.columns:
            label_map = {"up": 1, "flat": 0, "down": -1}
            df = df.copy()
            df["label_num"] = df["label"].map(label_map)
            cols.append("label_num")

        subset = df[cols].dropna()

        if method == "pearson":
            return subset.corr(method="pearson")
        return subset.corr(method="spearman")

    def feature_label_correlations(self, method: str = "spearman") -> pd.Series:
        """Return per-feature correlation with the label, sorted by absolute value.

        Returns
        -------
        Series indexed by feature name, values are correlation coefficients.
        """
        mat = self.correlation_matrix(method=method)
        if "label_num" not in mat.columns:
            return pd.Series(dtype=float)
        return mat["label_num"].drop("label_num").sort_values(key=abs, ascending=False)

    # ------------------------------------------------------------------
    # Temporal correlation study
    # ------------------------------------------------------------------

    def temporal_correlation(
        self,
        feature: str,
        window: str = "30D",
    ) -> pd.Series:
        """Rolling Spearman correlation between one feature and the label.

        Parameters
        ----------
        feature:
            Feature column name (e.g. ``'feat_ret_1'``).
        window:
            Pandas offset string for the rolling window (e.g. ``'30D'``, ``'7D'``).

        Returns
        -------
        Datetime-indexed Series of rolling correlations.
        """
        df = self.load()
        if feature not in df.columns:
            raise KeyError(f"Feature '{feature}' not found in dataset")
        if "label" not in df.columns:
            raise KeyError("Dataset must have a 'label' column")

        label_map = {"up": 1, "flat": 0, "down": -1}
        tmp = df[[feature, "label"]].copy().dropna()
        tmp["label_num"] = tmp["label"].map(label_map)

        # Ensure datetime index
        if not isinstance(tmp.index, pd.DatetimeIndex):
            if "timestamp_utc" in df.columns:
                tmp.index = pd.to_datetime(df.loc[tmp.index, "timestamp_utc"])
            else:
                return pd.Series(dtype=float)

        def _rolling_corr(x: pd.Series) -> float:
            if len(x) < 5:
                return float("nan")
            feat_vals = tmp.loc[x.index, feature]
            lbl_vals = tmp.loc[x.index, "label_num"]
            corr, _ = stats.spearmanr(feat_vals, lbl_vals, nan_policy="omit")
            return float(corr) if not np.isnan(corr) else float("nan")

        return tmp["label_num"].rolling(window).apply(lambda x: _rolling_corr(x), raw=False)

    # ------------------------------------------------------------------
    # Feature statistics
    # ------------------------------------------------------------------

    def feature_summary(self) -> pd.DataFrame:
        """Return descriptive stats for all feature columns."""
        df = self.load()
        return df[self.feature_cols].describe().T.round(4)

    def feature_groups(self) -> dict[str, list[str]]:
        """Group features by prefix type for organised display."""
        groups: dict[str, list[str]] = {}
        for col in self.feature_cols:
            parts = col.split("_")
            group = "_".join(parts[:2]) if len(parts) >= 3 else col
            groups.setdefault(group, []).append(col)
        return groups

    def top_correlated_features(self, n: int = 10) -> pd.DataFrame:
        """Return the top-N features most correlated with the label.

        Returns
        -------
        DataFrame with columns: feature, correlation, abs_correlation, group.
        """
        corrs = self.feature_label_correlations()
        top = corrs.head(n)
        groups = self.feature_groups()
        col_to_group = {col: g for g, cols in groups.items() for col in cols}
        return pd.DataFrame({
            "feature": top.index,
            "correlation": top.values,
            "abs_correlation": top.abs().values,
            "group": [col_to_group.get(c, "other") for c in top.index],
        }).reset_index(drop=True)
