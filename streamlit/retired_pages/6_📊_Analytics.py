"""Analytics page — feature correlations, importance, and temporal patterns."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bitbat.analytics.explainer import PredictionExplainer
from bitbat.analytics.feature_analysis import FeatureAnalyzer
from bitbat.gui.presets import DEFAULT_PRESET, get_preset
from bitbat.gui.widgets import db_query

st.set_page_config(page_title="Analytics — BitBat", page_icon="📊", layout="wide")

st.title("📊 Analytics")
st.markdown("Understand what drives BitBat's predictions.")

_DB = ROOT / "data" / "autonomous.db"

# ------------------------------------------------------------------
# Load config to find dataset path
# ------------------------------------------------------------------
try:
    import yaml
    _user_cfg_path = ROOT / "config" / "user_config.yaml"
    if _user_cfg_path.exists():
        _cfg = yaml.safe_load(_user_cfg_path.read_text()) or {}
        freq = _cfg.get("freq", "1h")
        horizon = _cfg.get("horizon", "4h")
    else:
        freq, horizon = "1h", "4h"
except Exception:
    freq, horizon = "1h", "4h"

data_dir = ROOT / "data"
dataset_path = data_dir / "features" / f"{freq}_{horizon}" / "dataset.parquet"

if not dataset_path.exists():
    st.warning(
        f"Feature dataset not found at `{dataset_path}`.  \n"
        "Build features using the **Advanced Pipeline** page first."
    )
    st.stop()

# Load analyzer
try:
    analyzer = FeatureAnalyzer(dataset_path)
    df = analyzer.load()
    feature_cols = analyzer.feature_cols
except Exception as exc:
    st.error(f"Failed to load dataset: {exc}")
    st.stop()

if not feature_cols:
    st.warning("No feature columns found in dataset.")
    st.stop()

st.caption(f"Dataset: {len(df):,} rows | {len(feature_cols)} features | {freq} / {horizon}")

# ------------------------------------------------------------------
# Feature overview
# ------------------------------------------------------------------
st.header("Feature Overview")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total Features", len(feature_cols))
with c2:
    groups = analyzer.feature_groups()
    st.metric("Feature Groups", len(groups))
with c3:
    price_feats = sum(1 for c in feature_cols if "ret" in c or "vol" in c or "macd" in c or "rsi" in c)
    sent_feats = sum(1 for c in feature_cols if "sent" in c)
    st.metric("Sentiment Features", sent_feats)

with st.expander("📋 Feature Groups"):
    for group, cols in groups.items():
        st.markdown(f"**{group}** ({len(cols)} features): " + ", ".join(cols[:5]) + ("..." if len(cols) > 5 else ""))

# ------------------------------------------------------------------
# Correlation with label
# ------------------------------------------------------------------
st.header("What Drives Predictions?")
st.markdown("Features most correlated with actual price direction:")

try:
    top_n = st.slider("Number of top features to show", 5, 30, 15, step=5)
    top_df = analyzer.top_correlated_features(n=top_n)

    if not top_df.empty:
        fig, ax = plt.subplots(figsize=(10, max(4, top_n // 2)))
        colors = ["#10B981" if v >= 0 else "#EF4444" for v in top_df["correlation"]]
        bars = ax.barh(top_df["feature"], top_df["correlation"], color=colors)
        ax.axvline(0, color="white", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Spearman Correlation with Direction (up=+1, flat=0, down=−1)")
        ax.set_title("Feature Importance — Correlation with Label")
        ax.tick_params(axis="y", labelsize=9)
        ax.set_facecolor("#1a1f2e")
        fig.patch.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

        # Plain-language table
        st.subheader("Top Features Summary")
        display_df = top_df[["feature", "correlation", "group"]].copy()
        display_df["strength"] = display_df["correlation"].abs().apply(
            lambda x: "Strong" if x >= 0.15 else ("Moderate" if x >= 0.07 else "Weak")
        )
        display_df["direction"] = display_df["correlation"].apply(
            lambda x: "↑ Predicts UP" if x > 0.02 else ("↓ Predicts DOWN" if x < -0.02 else "Neutral")
        )
        display_df["correlation"] = display_df["correlation"].round(3)
        st.dataframe(display_df, width="stretch", hide_index=True)
    else:
        st.info("Label correlations not available for this dataset.")
except Exception as exc:
    st.error(f"Could not compute correlations: {exc}")

# ------------------------------------------------------------------
# Correlation heatmap
# ------------------------------------------------------------------
st.header("Feature Correlation Heatmap")
st.markdown("How similar are the features to each other? Highly correlated features may be redundant.")

try:
    corr_method = st.selectbox("Correlation method", ["spearman", "pearson"], index=0)
    max_heatmap_features = 25

    mat = analyzer.correlation_matrix(method=corr_method)
    feat_cols_only = [c for c in mat.columns if c != "label_num"]

    if len(feat_cols_only) > max_heatmap_features:
        # Show top-correlated features by label correlation
        top_feats = analyzer.feature_label_correlations().head(max_heatmap_features).index.tolist()
        mat_sub = mat.loc[top_feats, top_feats]
        st.caption(f"Showing top {max_heatmap_features} features by label correlation (of {len(feat_cols_only)} total)")
    else:
        mat_sub = mat.loc[feat_cols_only, feat_cols_only]

    n = len(mat_sub)
    fig_size = max(8, n // 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))
    im = ax.imshow(mat_sub.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Correlation")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(mat_sub.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(mat_sub.index, fontsize=7)
    ax.set_title(f"{corr_method.title()} Correlation Matrix ({n} features)")
    ax.set_facecolor("#1a1f2e")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
except Exception as exc:
    st.error(f"Could not render correlation heatmap: {exc}")

# ------------------------------------------------------------------
# Price vs sentiment features
# ------------------------------------------------------------------
st.header("Price vs Sentiment Features")
st.markdown("How much do news sentiment features contribute compared to price features?")

try:
    price_corrs = []
    sent_corrs = []
    label_corrs = analyzer.feature_label_correlations()

    for feat, corr_val in label_corrs.items():
        if "sent" in feat:
            sent_corrs.append(abs(corr_val))
        else:
            price_corrs.append(abs(corr_val))

    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.metric(
            "Avg Price Feature Correlation",
            f"{np.mean(price_corrs):.3f}" if price_corrs else "N/A",
            help="Average absolute correlation between price features and direction",
        )
    with comp_col2:
        st.metric(
            "Avg Sentiment Feature Correlation",
            f"{np.mean(sent_corrs):.3f}" if sent_corrs else "N/A",
            help="Average absolute correlation between news sentiment features and direction",
        )

    if price_corrs and sent_corrs:
        if np.mean(price_corrs) > np.mean(sent_corrs) * 1.5:
            st.info("📈 Price patterns are driving most predictions. News sentiment has less influence.")
        elif np.mean(sent_corrs) > np.mean(price_corrs) * 1.5:
            st.info("📰 News sentiment is particularly influential for this dataset.")
        else:
            st.info("⚖️ Price patterns and news sentiment have roughly equal influence.")
except Exception as exc:
    st.caption(f"Could not compare feature groups: {exc}")

# ------------------------------------------------------------------
# Feature statistics
# ------------------------------------------------------------------
with st.expander("📊 Feature Statistics"):
    try:
        summary = analyzer.feature_summary()
        st.dataframe(summary, width="stretch")
    except Exception:
        st.info("Feature statistics not available.")

# ------------------------------------------------------------------
# Model-level feature importance (from trained model)
# ------------------------------------------------------------------
st.header("Why Does the Model Decide?")
st.markdown("This shows which features the trained model relies on most:")

model_path = ROOT / "models" / f"{freq}_{horizon}" / "xgb.json"

if not model_path.exists():
    st.info("No trained model found. Train a model in the **Advanced Pipeline** page.")
else:
    try:
        explainer = PredictionExplainer(model_path)
        importance = explainer.feature_importance_from_model()

        if not importance.empty:
            top_n_model = min(20, len(importance))
            imp_df = importance.head(top_n_model).reset_index()
            imp_df.columns = ["feature", "gain"]

            fig_imp, ax_imp = plt.subplots(figsize=(10, max(4, top_n_model // 2)))
            ax_imp.barh(imp_df["feature"], imp_df["gain"], color="#10B981")
            ax_imp.set_xlabel("Gain (higher = more important)", color="white")
            ax_imp.set_title("Model Feature Importance (Gain)", color="white")
            ax_imp.set_facecolor("#1a1f2e")
            fig_imp.patch.set_facecolor("#0e1117")
            ax_imp.tick_params(colors="white", labelsize=9)
            for spine in ax_imp.spines.values():
                spine.set_edgecolor("#444")
            plt.tight_layout()
            st.pyplot(fig_imp, clear_figure=True)
        else:
            st.info("Feature importance not available from this model.")

        # SHAP mean values on a sample of the dataset
        st.subheader("SHAP Values (Prediction Contributions)")
        st.markdown(
            "SHAP values show how much each feature *pushes* a prediction towards UP or DOWN on average:"
        )
        sample_size = min(500, len(df))
        X_sample = df[feature_cols].dropna().sample(sample_size, random_state=42)
        mean_shap = explainer.batch_mean_shap(X_sample)

        if not mean_shap.empty:
            top_shap = mean_shap.head(15)
            fig_shap, ax_shap = plt.subplots(figsize=(10, max(4, len(top_shap) // 2)))
            ax_shap.barh(top_shap["feature"], top_shap["mean_abs_shap"], color="#3B82F6")
            ax_shap.set_xlabel("Mean |SHAP value| — impact on prediction", color="white")
            ax_shap.set_title("Top Features by Mean Absolute SHAP Value", color="white")
            ax_shap.set_facecolor("#1a1f2e")
            fig_shap.patch.set_facecolor("#0e1117")
            ax_shap.tick_params(colors="white", labelsize=9)
            for spine in ax_shap.spines.values():
                spine.set_edgecolor("#444")
            plt.tight_layout()
            st.pyplot(fig_shap, clear_figure=True)

        # Single prediction explanation
        st.subheader("Explain a Specific Prediction")
        st.markdown("Choose a row from the dataset to see why BitBat would make that prediction:")
        row_idx = st.slider("Row index", 0, len(df) - 1, 0)
        sample_row = df[feature_cols].dropna().iloc[row_idx : row_idx + 1]
        if not sample_row.empty:
            explanation = explainer.explain_row(sample_row)
            st.markdown(explanation["plain_english"])

            contrib_df = (
                explanation["contributions"]
                .head(10)
                .reset_index()
            )
            contrib_df.columns = ["Feature", "Contribution"]
            contrib_df["Direction"] = contrib_df["Contribution"].apply(
                lambda v: "↑ Pushes UP" if v > 0 else "↓ Pushes DOWN"
            )
            contrib_df["Contribution"] = contrib_df["Contribution"].round(4)
            st.dataframe(contrib_df, width="stretch", hide_index=True)

    except Exception as exc:
        st.error(f"Could not generate explanations: {exc}")
