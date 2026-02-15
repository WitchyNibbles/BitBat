"""Enhanced Backtest page — what-if scenario analysis and preset comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bitbat.analytics.backtest_report import BacktestReport, compare_scenarios
from bitbat.backtest.engine import run as backtest_run
from bitbat.gui.presets import AGGRESSIVE, BALANCED, CONSERVATIVE

st.set_page_config(page_title="Backtest — BitBat", page_icon="🎯", layout="wide")

st.title("🎯 Backtest")
st.markdown(
    "Test how different strategies would have performed on historical feature data. "
    "Requires a trained model and feature dataset."
)

# ------------------------------------------------------------------
# Load config
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

dataset_path = ROOT / "data" / "features" / f"{freq}_{horizon}" / "dataset.parquet"
model_path = ROOT / "models" / f"{freq}_{horizon}" / "xgb.json"

# ------------------------------------------------------------------
# Guard: require both dataset and model
# ------------------------------------------------------------------
if not dataset_path.exists():
    st.warning(
        f"Feature dataset not found at `{dataset_path}`.  \n"
        "Build features using the **Advanced Pipeline** page first."
    )
    st.stop()

if not model_path.exists():
    st.info(
        "No trained model found.  \n"
        "Train a model in the **Advanced Pipeline** page, then return here."
    )
    st.stop()

# ------------------------------------------------------------------
# Load dataset + model predictions
# ------------------------------------------------------------------
try:
    df = pd.read_parquet(dataset_path)
    if "timestamp_utc" in df.columns:
        df = df.set_index("timestamp_utc")
    df.index = pd.to_datetime(df.index)

    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    if not feature_cols:
        st.error("No feature columns found in dataset.")
        st.stop()

    X = df[feature_cols].dropna().astype(float)

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
    # multi:softprob output: columns are [p_down, p_flat, p_up] (alphabetical)
    probas = booster.predict(dmatrix)
    proba_down = pd.Series(probas[:, 0], index=X.index, name="p_down")
    proba_up = pd.Series(probas[:, 2], index=X.index, name="p_up")

    # Reconstruct synthetic close price from 1-hour lagged returns
    ret_col = "feat_ret_1" if "feat_ret_1" in feature_cols else feature_cols[0]
    close = (1 + X[ret_col].fillna(0)).cumprod() * 100.0
    close.name = "close"

except Exception as exc:
    st.error(f"Failed to load data or generate predictions: {exc}")
    st.stop()

st.caption(
    f"Dataset: {len(X):,} bars | {len(feature_cols)} features | "
    f"freq={freq} / horizon={horizon}"
)

# ------------------------------------------------------------------
# What-if: single threshold explorer
# ------------------------------------------------------------------
st.header("What-If Analysis")
st.markdown(
    "Adjust the **entry threshold** to see how the strategy would have performed. "
    "A higher threshold means the model must be more confident before entering a trade."
)

col_slider, col_short = st.columns([3, 1])
with col_slider:
    threshold = st.slider(
        "Entry Confidence Threshold",
        min_value=0.50,
        max_value=0.90,
        value=0.65,
        step=0.05,
        format="%.0f%%",
        help="Model must predict P(up) ≥ threshold to enter a long position.",
    )
with col_short:
    allow_short = st.checkbox(
        "Allow Short Positions",
        value=False,
        help="Also enter short when P(down) ≥ threshold.",
    )

try:
    trades, equity = backtest_run(
        close,
        proba_up,
        proba_down,
        enter=threshold,
        allow_short=allow_short,
    )
    report = BacktestReport(
        equity,
        trades,
        preset_name=f"Threshold {threshold:.0%}",
        enter_threshold=threshold,
        allow_short=allow_short,
    )

    # Metrics row
    m = report.metrics()
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    with mc1:
        total_ret = m["total_return"] * 100
        st.metric("Total Return", f"{total_ret:+.1f}%")
    with mc2:
        st.metric("Sharpe Ratio", f"{m['sharpe']:.2f}")
    with mc3:
        dd = abs(m["max_drawdown"]) * 100
        st.metric("Max Drawdown", f"-{dd:.1f}%")
    with mc4:
        st.metric("Win Rate", f"{m['hit_rate']*100:.1f}%")
    with mc5:
        st.metric("Trades", f"{int(m['n_trades']):,}")

    # Equity curve
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(equity.index, equity.values, color="#10B981", linewidth=1.5, label="Strategy")
    ax.axhline(1.0, color="#EF4444", linewidth=1, linestyle="--", alpha=0.5, label="Buy & Hold (flat)")
    ax.fill_between(equity.index, equity.values, 1.0, alpha=0.12, color="#10B981")
    ax.set_ylabel("Equity (start = 1.0)", color="white")
    ax.set_title(f"Equity Curve — Threshold {threshold:.0%}", color="white")
    ax.set_facecolor("#1a1f2e")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(facecolor="#1a1f2e", labelcolor="white")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # Plain-language summary
    with st.expander("📝 Plain-Language Summary"):
        st.markdown(report.plain_summary())

except Exception as exc:
    st.error(f"Could not run backtest: {exc}")

# ------------------------------------------------------------------
# Preset comparison
# ------------------------------------------------------------------
st.header("Preset Comparison")
st.markdown(
    "How do **Conservative**, **Balanced**, and **Aggressive** presets compare "
    "on this dataset?"
)

PRESETS = [
    (CONSERVATIVE, "Conservative 🛡️"),
    (BALANCED, "Balanced ⚖️"),
    (AGGRESSIVE, "Aggressive 🚀"),
]

preset_reports: list[BacktestReport] = []
preset_equities: list[tuple[str, pd.Series]] = []

for preset, label in PRESETS:
    try:
        t, eq = backtest_run(
            close,
            proba_up,
            proba_down,
            enter=preset.enter_threshold,
            allow_short=False,
        )
        r = BacktestReport(
            eq,
            t,
            preset_name=label,
            enter_threshold=preset.enter_threshold,
        )
        preset_reports.append(r)
        preset_equities.append((label, eq))
    except Exception:
        pass

if preset_reports:
    # Comparison table
    comparison_df = compare_scenarios(preset_reports)
    st.dataframe(comparison_df, width="stretch", hide_index=True)

    # Overlaid equity curves
    preset_colors = {
        "Conservative 🛡️": "#3B82F6",
        "Balanced ⚖️": "#10B981",
        "Aggressive 🚀": "#EF4444",
    }

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    for label, eq in preset_equities:
        ax2.plot(
            eq.index,
            eq.values,
            color=preset_colors.get(label, "#888"),
            linewidth=1.5,
            label=label,
        )
    ax2.axhline(1.0, color="white", linewidth=0.8, linestyle="--", alpha=0.3)
    ax2.set_ylabel("Equity (start = 1.0)", color="white")
    ax2.set_title("Preset Equity Curves — Side by Side", color="white")
    ax2.set_facecolor("#1a1f2e")
    fig2.patch.set_facecolor("#0e1117")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444")
    ax2.legend(facecolor="#1a1f2e", labelcolor="white")
    plt.tight_layout()
    st.pyplot(fig2, clear_figure=True)

    # Individual plain-language cards
    st.subheader("Strategy Verdicts")
    card_cols = st.columns(len(preset_reports))
    for col, rep in zip(card_cols, preset_reports):
        with col:
            st.markdown(rep.plain_summary())

# ------------------------------------------------------------------
# Drawdown chart
# ------------------------------------------------------------------
st.header("Drawdown Analysis")
st.markdown("How deep did each strategy fall from its peak?")

if preset_equities:
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    for label, eq in preset_equities:
        dd = eq / eq.cummax() - 1
        ax3.fill_between(
            dd.index,
            dd.values * 100,
            0,
            alpha=0.35,
            label=label,
            color=preset_colors.get(label, "#888"),
        )
    ax3.set_ylabel("Drawdown %", color="white")
    ax3.set_title("Drawdown from Peak — All Presets", color="white")
    ax3.set_facecolor("#1a1f2e")
    fig3.patch.set_facecolor("#0e1117")
    ax3.tick_params(colors="white")
    for spine in ax3.spines.values():
        spine.set_edgecolor("#444")
    ax3.legend(facecolor="#1a1f2e", labelcolor="white")
    plt.tight_layout()
    st.pyplot(fig3, clear_figure=True)
