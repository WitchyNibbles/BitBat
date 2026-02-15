"""Advanced Pipeline page — full technical interface for power users."""

from __future__ import annotations

import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="Pipeline — BitBat", page_icon="🔬", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from bitbat import __version__
from bitbat.backtest.engine import run as backtest_run
from bitbat.backtest.metrics import summary as backtest_summary
from bitbat.cli import batch_realize as cli_batch_realize
from bitbat.cli import batch_run as cli_batch_run
from bitbat.cli import monitor_refresh as cli_monitor_refresh
from bitbat.config.loader import get_runtime_config_path, set_runtime_config
from bitbat.contracts import (
    ensure_feature_contract,
    ensure_predictions_contract,
    ensure_prices_contract,
)
from bitbat.dataset.build import DatasetMeta, build_xy
from bitbat.dataset.splits import walk_forward
from bitbat.ingest import news_gdelt as news_module
from bitbat.ingest import prices as prices_module
from bitbat.model.evaluate import classification_metrics
from bitbat.model.train import fit_xgb


@st.cache_data(show_spinner=False)
def _parquet_row_count(path: str) -> tuple[int | None, str | None]:
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - display in UI
        return None, str(exc)
    return len(frame), None


def _status_row(label: str, path: Path, kind: str) -> dict[str, Any]:
    exists = path.exists()
    rows: int | str = "-"
    error = ""
    if exists and kind == "parquet":
        count, err = _parquet_row_count(str(path))
        if count is not None:
            rows = count
        if err:
            error = err
    return {
        "Item": label,
        "Path": str(path),
        "Exists": exists,
        "Rows": rows,
        "Error": error,
    }


def _load_config() -> tuple[dict[str, Any], Path | None, str | None]:
    try:
        config = set_runtime_config(None)
        return config, get_runtime_config_path(), None
    except Exception as exc:  # pragma: no cover - display in UI
        return {}, None, str(exc)


def _init_global_overrides(config: dict[str, Any]) -> None:
    freq_default = str(config.get("freq", "1h"))
    horizon_default = str(config.get("horizon", "4h"))
    tau_default = float(config.get("tau", 0.0015))
    tau_default = min(max(tau_default, 0.0), 0.02)

    if "override_freq" not in st.session_state:
        st.session_state["override_freq"] = freq_default
    if "override_horizon" not in st.session_state:
        st.session_state["override_horizon"] = horizon_default
    if "override_tau" not in st.session_state:
        st.session_state["override_tau"] = tau_default


def _apply_global_overrides(config: dict[str, Any]) -> dict[str, Any]:
    overridden = dict(config)
    freq_value = st.session_state.get("override_freq")
    horizon_value = st.session_state.get("override_horizon")
    tau_value = st.session_state.get("override_tau")

    freq = str(freq_value).strip() if freq_value is not None else ""
    horizon = str(horizon_value).strip() if horizon_value is not None else ""
    if not freq:
        freq = str(config.get("freq", "1h"))
    if not horizon:
        horizon = str(config.get("horizon", "4h"))
    if tau_value is None:
        tau_value = config.get("tau", 0.0015)

    overridden["freq"] = freq
    overridden["horizon"] = horizon
    overridden["tau"] = float(tau_value)
    return overridden


def _resolve_date_range(
    start_date: date,
    end_date: date,
) -> tuple[datetime, datetime, list[str], list[str]]:
    errors: list[str] = []
    infos: list[str] = []

    if end_date < start_date:
        errors.append("End date must be on or after the start date.")

    start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=UTC)
    end_dt = datetime.combine(end_date, datetime.min.time(), tzinfo=UTC) + timedelta(days=1)
    now = datetime.now(UTC)

    if start_dt > now:
        errors.append("Start date is in the future. Update it before ingesting data.")

    if end_dt > now:
        end_dt = now
        infos.append("End date is in the future; capping to current UTC time.")

    if start_dt >= end_dt:
        errors.append("Date range must span at least one hour.")

    return start_dt, end_dt, infos, errors


def _render_preview(label: str, frame: pd.DataFrame | None, filename: str) -> None:
    st.subheader(f"{label} preview (last 10 rows)")
    if frame is None or frame.empty:
        st.info("No preview available yet.")
        return

    st.dataframe(frame, width="stretch")
    csv_payload = frame.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download {label.lower()} preview (CSV)",
        data=csv_payload,
        file_name=filename,
        mime="text/csv",
        width="stretch",
    )


@st.cache_data(show_spinner=False)
def _load_timestamp_bounds(path: str) -> tuple[datetime | None, datetime | None, str | None]:
    try:
        frame = pd.read_parquet(path, columns=["timestamp_utc"])
    except Exception as exc:  # pragma: no cover - display in UI
        return None, None, str(exc)
    if frame.empty:
        return None, None, "No timestamps found."
    timestamps = pd.to_datetime(frame["timestamp_utc"], errors="coerce").dropna()
    if timestamps.empty:
        return None, None, "No valid timestamps found."
    return timestamps.min().to_pydatetime(), timestamps.max().to_pydatetime(), None


@st.cache_data(show_spinner=False)
def _load_feature_dataset(
    path: str,
    *,
    require_label: bool,
    require_features_full: bool,
) -> pd.DataFrame:
    dataset = pd.read_parquet(path)
    if require_features_full and not any(
        col.startswith("feat_sent_") for col in dataset.columns
    ):
        require_features_full = False
    dataset = ensure_feature_contract(
        dataset,
        require_label=require_label,
        require_forward_return=require_label,
        require_features_full=require_features_full,
    )
    return dataset.sort_values("timestamp_utc").set_index("timestamp_utc")


@st.cache_data(show_spinner=False)
def _load_meta(path: str) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):  # pragma: no cover - UI display
        return None


@st.cache_data(show_spinner=False)
def _load_predictions(path: str, cache_bust: str) -> pd.DataFrame:
    _ = cache_bust
    frame = pd.read_parquet(path)
    frame = ensure_predictions_contract(frame)
    return frame.sort_values("timestamp_utc")


@st.cache_data(show_spinner=False)
def _load_prices(path: str, cache_bust: str) -> pd.DataFrame:
    _ = cache_bust
    frame = pd.read_parquet(path)
    frame = ensure_prices_contract(frame)
    return frame.sort_values("timestamp_utc")


@st.cache_data(show_spinner=False)
def _load_live_metrics(path: str, cache_bust: str) -> dict[str, Any] | None:
    _ = cache_bust
    target = Path(path)
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):  # pragma: no cover - UI display
        return None


def _align_predictions_prices(
    preds: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    if preds.empty or prices.empty:
        return pd.DataFrame()
    preds_sorted = preds.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    prices_indexed = prices.set_index("timestamp_utc").sort_index()
    aligned = preds_sorted.set_index("timestamp_utc")
    aligned["close"] = prices_indexed["close"].reindex(aligned.index).ffill()
    aligned = aligned.dropna(subset=["close"])
    return aligned


def _derive_predicted_label(
    preds: pd.DataFrame,
    threshold: float,
) -> pd.Series:
    threshold = float(np.clip(threshold, 0.0, 1.0))
    p_up = preds["p_up"].astype(float)
    p_down = preds["p_down"].astype(float)
    direction = np.where(p_up >= p_down, "up", "down")
    confidence = np.maximum(p_up, p_down)
    predicted = np.where(confidence >= threshold, direction, "flat")
    return pd.Series(predicted, index=preds.index, dtype="string")


def _render_confusion_matrix(confusion: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(confusion.values, cmap="Blues")
    ax.set_xticks(range(len(confusion.columns)))
    ax.set_yticks(range(len(confusion.index)))
    ax.set_xticklabels(confusion.columns)
    ax.set_yticklabels(confusion.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax)

    for (row, col), value in np.ndenumerate(confusion.values):
        ax.text(col, row, str(int(value)), ha="center", va="center", color="black")

    st.pyplot(fig, clear_figure=True)


@st.cache_data(show_spinner=False)
def _build_features_cached(
    prices_path: str,
    news_path: str | None,
    freq: str,
    horizon: str,
    tau: float,
    start: str,
    end: str,
    enable_sentiment: bool,
    output_root: str,
    seed: int,
    version: str,
    cache_bust: str,
) -> tuple[pd.DataFrame, pd.Series, DatasetMeta]:
    _ = cache_bust
    return build_xy(
        prices_path,
        news_path,
        freq=freq,
        horizon=horizon,
        tau=tau,
        start=start,
        end=end,
        enable_sentiment=enable_sentiment,
        output_root=output_root,
        seed=seed,
        version=version,
    )


def _build_cv_windows(
    index: pd.Index,
    start: str,
    end: str,
    embargo_bars: int,
) -> list[tuple[str, str, str, str]]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    idx = pd.Index(sorted(pd.to_datetime(list(index))))
    idx = idx[(idx >= start_ts) & (idx <= end_ts)]
    if len(idx) < 2:
        return []

    embargo_delta = pd.Timedelta(hours=embargo_bars)
    split_idx = max(1, int(len(idx) * 0.7))
    if split_idx >= len(idx):
        split_idx = len(idx) - 1

    for pos in range(split_idx, len(idx)):
        test_start = idx[pos]
        cutoff = test_start - embargo_delta
        train_candidates = idx[idx < cutoff]
        if train_candidates.empty:
            continue
        train_start = idx[0]
        train_end = train_candidates[-1]
        test_end = idx[-1]
        return [
            (
                train_start.isoformat(),
                train_end.isoformat(),
                test_start.isoformat(),
                test_end.isoformat(),
            )
        ]

    return []


@st.cache_data(show_spinner=False)
def _run_cv_cached(
    dataset_path: str,
    start: str,
    end: str,
    threshold: float,
    seed: int,
    freq: str,
    horizon: str,
    require_features_full: bool,
    cache_bust: str,
) -> dict[str, Any]:
    _ = cache_bust
    dataset = _load_feature_dataset(
        dataset_path,
        require_label=True,
        require_features_full=require_features_full,
    )
    feature_cols = [col for col in dataset.columns if col.startswith("feat_")]
    X = dataset[feature_cols]
    y = dataset["label"]

    windows = _build_cv_windows(X.index, start, end, embargo_bars=1)
    if not windows:
        return {"folds": [], "average_balanced_accuracy": 0.0, "average_mcc": 0.0}

    folds = walk_forward(X.index, windows=windows, embargo_bars=1)

    summary: list[dict[str, Any]] = []
    for fold in folds:
        if fold.train.empty or fold.test.empty:
            continue

        X_train = X.loc[fold.train].copy()
        X_test = X.loc[fold.test]
        y_train = y.loc[fold.train]
        y_test = y.loc[fold.test]

        X_train.attrs["freq"] = freq
        X_train.attrs["horizon"] = horizon

        booster, _ = fit_xgb(X_train, y_train, seed=seed)
        dtest = xgb.DMatrix(X_test, feature_names=list(X_test.columns))
        proba = booster.predict(dtest)
        metrics = classification_metrics(
            y_test,
            proba,
            threshold=threshold,
            class_labels=list(y.unique()),
        )
        summary.append(metrics)

    if not summary:
        return {"folds": [], "average_balanced_accuracy": 0.0, "average_mcc": 0.0}

    avg_balanced = float(np.mean([metric["balanced_accuracy"] for metric in summary]))
    avg_mcc = float(np.mean([metric["mcc"] for metric in summary]))
    return {
        "folds": summary,
        "average_balanced_accuracy": avg_balanced,
        "average_mcc": avg_mcc,
    }


@st.cache_data(show_spinner=False)
def _run_train_cached(
    dataset_path: str,
    threshold: float,
    class_weights: bool,
    seed: int,
    freq: str,
    horizon: str,
    require_features_full: bool,
    cache_bust: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    _ = cache_bust
    dataset = _load_feature_dataset(
        dataset_path,
        require_label=True,
        require_features_full=require_features_full,
    )
    feature_cols = [col for col in dataset.columns if col.startswith("feat_")]
    X = dataset[feature_cols]
    y = dataset["label"]

    X.attrs["freq"] = freq
    X.attrs["horizon"] = horizon

    booster, importance = fit_xgb(
        X,
        y,
        class_weights=class_weights,
        seed=seed,
    )
    dtrain = xgb.DMatrix(X, feature_names=list(X.columns))
    proba = booster.predict(dtrain)
    metrics = classification_metrics(
        y,
        proba,
        threshold=threshold,
        class_labels=list(y.unique()),
    )
    return metrics, importance


def _format_bounds(start: datetime | None, end: datetime | None) -> str:
    start_text = _format_timestamp(start)
    end_text = _format_timestamp(end)
    if start_text is None or end_text is None:
        return "Unavailable"
    return f"{start_text} → {end_text} (UTC)"


def _format_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d %H:%M")


def _build_feature_preview(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    preview = X.tail(200).copy()
    preview["label"] = y.loc[preview.index]
    return preview


def _plot_pr_curves(metrics: dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    pr_curves = metrics.get("pr_curves", {})
    if not pr_curves:
        ax.text(0.5, 0.5, "No PR curves available", ha="center", va="center")
        return fig

    for label, curve in pr_curves.items():
        ax.plot(curve["recall"], curve["precision"], label=str(label))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


def _dashboard(config: dict[str, Any]) -> None:
    st.header("Dashboard")

    freq = str(config.get("freq", "1h"))
    horizon = str(config.get("horizon", "4h"))
    data_dir = Path(config.get("data_dir", "data")).expanduser()

    st.caption(f"Data dir: {data_dir} | freq: {freq} | horizon: {horizon}")

    data_rows = [
        _status_row(
            "Prices (raw)",
            data_dir / "raw" / "prices" / f"btcusd_yf_{freq}.parquet",
            "parquet",
        ),
        _status_row(
            "News (raw)",
            data_dir / "raw" / "news" / "gdelt_1h" / "gdelt_crypto_1h.parquet",
            "parquet",
        ),
        _status_row(
            "Features dataset",
            data_dir / "features" / f"{freq}_{horizon}" / "dataset.parquet",
            "parquet",
        ),
    ]

    model_rows = [
        _status_row(
            "Model artifact",
            Path("models") / f"{freq}_{horizon}" / "xgb.json",
            "file",
        )
    ]

    prediction_rows = [
        _status_row(
            "Predictions",
            data_dir / "predictions" / f"{freq}_{horizon}.parquet",
            "parquet",
        )
    ]

    st.subheader("Data status")
    st.dataframe(pd.DataFrame(data_rows), width="stretch")

    st.subheader("Model status")
    st.dataframe(pd.DataFrame(model_rows), width="stretch")

    st.subheader("Predictions status")
    st.dataframe(pd.DataFrame(prediction_rows), width="stretch")


def _ingest_data(config: dict[str, Any]) -> None:
    st.header("Ingest Data")

    data_dir = Path(config.get("data_dir", "data")).expanduser()
    freq = str(config.get("freq", "1h"))

    st.caption(f"Data dir: {data_dir} | using freq: {freq}")

    today = date.today()
    default_start = today - timedelta(days=30)

    date_col, end_col = st.columns(2)
    with date_col:
        start_date = st.date_input(
            "Start date (UTC)",
            value=default_start,
            key="ingest_start_date",
        )
    with end_col:
        end_date = st.date_input(
            "End date (UTC)",
            value=today,
            key="ingest_end_date",
        )

    start_dt, end_dt, infos, errors = _resolve_date_range(start_date, end_date)
    for message in infos:
        st.info(message)
    for message in errors:
        st.error(message)

    st.caption(
        "Effective range (UTC): "
        f"{start_dt.strftime('%Y-%m-%d %H:%M')} → {end_dt.strftime('%Y-%m-%d %H:%M')}"
    )

    prices_root = data_dir / "raw" / "prices"
    news_root = data_dir / "raw" / "news" / "gdelt_1h"
    throttle_seconds = float(config.get("news_throttle_seconds", 10.0))
    retry_limit = int(config.get("news_retry_limit", 30))

    if "prices_preview" not in st.session_state:
        st.session_state["prices_preview"] = None
    if "news_preview" not in st.session_state:
        st.session_state["news_preview"] = None
    if "prices_preview_name" not in st.session_state:
        st.session_state["prices_preview_name"] = "prices_preview.csv"
    if "news_preview_name" not in st.session_state:
        st.session_state["news_preview_name"] = "news_preview.csv"

    price_col, news_col = st.columns(2)
    with price_col:
        st.subheader("Prices (Yahoo Finance)")
        symbol = st.text_input(
            "Symbol",
            value="BTC-USD",
            key="price_symbol",
        )
        interval = st.text_input(
            "Interval",
            value=freq,
            key="price_interval",
        )
        price_target = prices_module._target_path(symbol, interval, prices_root)
        st.caption(f"Output: {price_target}")
        pull_prices = st.button(
            "Pull prices",
            width="stretch",
            disabled=bool(errors),
        )
        if pull_prices:
            progress = st.progress(0)
            with st.spinner("Fetching price data from Yahoo Finance..."):
                try:
                    progress.progress(10)
                    frame = prices_module.fetch_yf(
                        symbol,
                        interval,
                        start_dt,
                        end=end_dt,
                        output_root=prices_root,
                    )
                    progress.progress(100)
                except Exception as exc:  # pragma: no cover - UI display
                    progress.empty()
                    st.error(f"Price ingest failed: {exc}")
                else:
                    st.success(f"Pulled {len(frame)} rows into {price_target}")
                    st.session_state["prices_preview"] = frame.tail(10).copy()
                    st.session_state["prices_preview_name"] = (
                        f"{price_target.stem}_preview.csv"
                    )

    with news_col:
        st.subheader("News (GDELT)")
        st.caption("Use for historical; realtime separate.")
        news_target = news_module._target_path(news_root)
        st.caption(f"Output: {news_target}")
        st.caption(f"Throttle: {throttle_seconds:.1f}s | Retries: {retry_limit}")
        pull_news = st.button(
            "Pull news",
            width="stretch",
            disabled=bool(errors),
        )
        if pull_news:
            progress = st.progress(0)
            with st.spinner("Fetching GDELT news data..."):
                try:
                    progress.progress(10)
                    frame = news_module.fetch(
                        start_dt,
                        end_dt,
                        output_root=news_root,
                        throttle_seconds=throttle_seconds,
                        retry_limit=retry_limit,
                    )
                    progress.progress(100)
                except Exception as exc:  # pragma: no cover - UI display
                    progress.empty()
                    st.error(f"News ingest failed: {exc}")
                else:
                    st.success(f"Pulled {len(frame)} rows into {news_target}")
                    st.session_state["news_preview"] = frame.tail(10).copy()
                    st.session_state["news_preview_name"] = (
                        f"{news_target.stem}_preview.csv"
                    )

    st.divider()
    preview_col, preview_news_col = st.columns(2)
    with preview_col:
        _render_preview(
            "Prices",
            st.session_state.get("prices_preview"),
            st.session_state.get("prices_preview_name", "prices_preview.csv"),
        )
    with preview_news_col:
        _render_preview(
            "News",
            st.session_state.get("news_preview"),
            st.session_state.get("news_preview_name", "news_preview.csv"),
        )


def _build_features_page(config: dict[str, Any]) -> None:
    st.header("Features")

    data_dir = Path(config.get("data_dir", "data")).expanduser()
    freq = str(config.get("freq", "1h"))
    horizon = str(config.get("horizon", "4h"))
    seed = int(config.get("seed", 42))
    default_tau = float(config.get("tau", 0.0015))

    prices_path = data_dir / "raw" / "prices" / f"btcusd_yf_{freq}.parquet"
    news_path = data_dir / "raw" / "news" / "gdelt_1h" / "gdelt_crypto_1h.parquet"
    dataset_path = data_dir / "features" / f"{freq}_{horizon}" / "dataset.parquet"
    meta_path = data_dir / "features" / f"{freq}_{horizon}" / "meta.json"

    st.caption(f"Data dir: {data_dir} | freq: {freq} | horizon: {horizon}")
    st.caption(f"Prices: {prices_path}")
    st.caption(f"News: {news_path}")

    start_bound, end_bound, bound_error = _load_timestamp_bounds(str(prices_path))
    if bound_error:
        st.warning(f"Unable to read price timestamps: {bound_error}")

    today = date.today()
    default_start = (start_bound.date() if start_bound else today - timedelta(days=30))
    default_end = (end_bound.date() if end_bound else today)

    date_col, end_col = st.columns(2)
    with date_col:
        start_date = st.date_input(
            "Start date (UTC)",
            value=default_start,
            key="features_start_date",
        )
    with end_col:
        end_date = st.date_input(
            "End date (UTC)",
            value=default_end,
            key="features_end_date",
        )

    tau = st.slider(
        "Label threshold (tau)",
        min_value=0.0,
        max_value=0.02,
        value=default_tau,
        step=0.0005,
        format="%.4f",
    )

    enable_sentiment_default = bool(config.get("enable_sentiment", True))
    enable_sentiment = st.checkbox(
        "Enable sentiment",
        value=enable_sentiment_default,
        key="features_enable_sentiment",
        help="Disable for price-only.",
    )

    errors: list[str] = []
    if not prices_path.exists():
        errors.append("Prices parquet not found; run ingest first.")
    if enable_sentiment and not news_path.exists():
        errors.append("News parquet not found; run ingest first.")
    if end_date < start_date:
        errors.append("End date must be on or after the start date.")

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())
    st.caption(
        "Effective range (UTC): "
        f"{start_dt.strftime('%Y-%m-%d %H:%M')} → {end_dt.strftime('%Y-%m-%d %H:%M')}"
    )

    for message in errors:
        st.error(message)

    news_arg = str(news_path) if enable_sentiment else None

    force_rebuild = st.checkbox("Force rebuild (ignore cache)", value=False)
    cache_bust = datetime.now(UTC).isoformat() if force_rebuild else "stable"

    build_col, load_col = st.columns(2)
    with build_col:
        build_features = st.button(
            "Build features",
            width="stretch",
            disabled=bool(errors),
        )
    with load_col:
        load_existing = st.button(
            "Load existing dataset",
            width="stretch",
            disabled=not dataset_path.exists(),
        )

    if build_features and not errors:
        with st.spinner("Building feature matrix..."):
            try:
                X, y, meta = _build_features_cached(
                    str(prices_path),
                    news_arg,
                    freq,
                    horizon,
                    tau,
                    start_dt.isoformat(),
                    end_dt.isoformat(),
                    enable_sentiment,
                    str(data_dir),
                    seed,
                    __version__,
                    cache_bust,
                )
            except Exception as exc:  # pragma: no cover - UI display
                st.error(f"Feature build failed: {exc}")
            else:
                st.success(f"Built {len(X)} rows into {dataset_path}")
                preview = _build_feature_preview(X, y)
                st.session_state["features_preview"] = preview
                st.session_state["features_summary"] = {
                    "rows": len(X),
                    "columns": X.shape[1],
                    "start": X.index.min(),
                    "end": X.index.max(),
                    "label_counts": y.value_counts().to_dict(),
                }
                st.session_state["features_meta"] = meta.__dict__

    if load_existing and dataset_path.exists():
        with st.spinner("Loading feature dataset..."):
            try:
                dataset = _load_feature_dataset(
                    str(dataset_path),
                    require_label=True,
                    require_features_full=enable_sentiment,
                )
            except Exception as exc:  # pragma: no cover - UI display
                st.error(f"Failed to load dataset: {exc}")
            else:
                feature_cols = [col for col in dataset.columns if col.startswith("feat_")]
                X = dataset[feature_cols]
                y = dataset["label"]
                preview = _build_feature_preview(X, y)
                st.session_state["features_preview"] = preview
                st.session_state["features_summary"] = {
                    "rows": len(X),
                    "columns": X.shape[1],
                    "start": X.index.min(),
                    "end": X.index.max(),
                    "label_counts": y.value_counts().to_dict(),
                }
                st.session_state["features_meta"] = _load_meta(str(meta_path))

    preview = st.session_state.get("features_preview")
    summary = st.session_state.get("features_summary")
    if preview is not None and summary is not None:
        st.subheader("Feature matrix preview")
        st.caption(
            f"Rows: {summary['rows']} | Columns: {summary['columns']} | "
            f"Range: {_format_bounds(summary['start'], summary['end'])}"
        )
        st.dataframe(preview.tail(10).reset_index(), width="stretch")

        feature_choices = [col for col in preview.columns if col.startswith("feat_")]
        default_features = feature_choices[:3]
        selected = st.multiselect(
            "Chart features",
            options=feature_choices,
            default=default_features,
        )
        if selected:
            st.line_chart(preview[selected], width="stretch")

        label_counts = summary.get("label_counts", {})
        if label_counts:
            label_frame = (
                pd.DataFrame(
                    {"label": list(label_counts.keys()), "count": list(label_counts.values())}
                )
                .set_index("label")
                .sort_index()
            )
            st.subheader("Label distribution")
            st.bar_chart(label_frame, width="stretch")

        meta = st.session_state.get("features_meta")
        if isinstance(meta, dict) and meta:
            st.subheader("Dataset metadata")
            st.json(meta, expanded=False)


def _render_metrics(metrics: dict[str, Any]) -> None:
    st.metric("Balanced accuracy", f"{metrics.get('balanced_accuracy', 0.0):.3f}")
    st.metric("MCC", f"{metrics.get('mcc', 0.0):.3f}")

    per_class = metrics.get("per_class", {})
    if per_class:
        st.subheader("Per-class metrics")
        rows = [
            {"label": label, **values}
            for label, values in per_class.items()
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch")

    st.subheader("Precision-recall curves")
    fig = _plot_pr_curves(metrics)
    st.pyplot(fig, clear_figure=True)

    fig_path = Path("metrics") / "confusion_matrix.png"
    if fig_path.exists():
        st.subheader("Confusion matrix")
        st.image(str(fig_path), use_column_width=True)


def _train_model_page(config: dict[str, Any]) -> None:
    st.header("Model")

    data_dir = Path(config.get("data_dir", "data")).expanduser()
    freq = str(config.get("freq", "1h"))
    horizon = str(config.get("horizon", "4h"))
    seed = int(config.get("seed", 42))
    threshold_default = float(config.get("enter_threshold", 0.6))
    enable_sentiment = bool(
        st.session_state.get(
            "features_enable_sentiment",
            config.get("enable_sentiment", True),
        )
    )

    dataset_path = data_dir / "features" / f"{freq}_{horizon}" / "dataset.parquet"
    st.caption(f"Dataset: {dataset_path}")

    if not dataset_path.exists():
        st.error("Feature dataset not found. Build features first.")
        return

    dataset = _load_feature_dataset(
        str(dataset_path),
        require_label=True,
        require_features_full=enable_sentiment,
    )
    if enable_sentiment and not any(
        col.startswith("feat_sent_") for col in dataset.columns
    ):
        enable_sentiment = False
    if dataset.empty:
        st.error("Feature dataset is empty. Rebuild features with a wider date range.")
        return
    st.caption(
        f"Rows: {len(dataset)} | Range: {_format_bounds(dataset.index.min(), dataset.index.max())}"
    )

    st.subheader("Cross-validation")
    cv_start = dataset.index.min().date()
    cv_end = dataset.index.max().date()
    cv_col, cv_end_col = st.columns(2)
    with cv_col:
        cv_start_date = st.date_input(
            "CV start date (UTC)",
            value=cv_start,
            key="cv_start_date",
        )
    with cv_end_col:
        cv_end_date = st.date_input(
            "CV end date (UTC)",
            value=cv_end,
            key="cv_end_date",
        )

    cv_threshold = st.slider(
        "CV confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=threshold_default,
        step=0.05,
    )
    cv_force = st.checkbox("Force CV rerun (ignore cache)", value=False)
    cv_cache_bust = datetime.now(UTC).isoformat() if cv_force else "stable"

    cv_errors: list[str] = []
    if cv_end_date < cv_start_date:
        cv_errors.append("CV end date must be on or after the start date.")
    for message in cv_errors:
        st.error(message)

    run_cv = st.button("Run CV", width="stretch", disabled=bool(cv_errors))
    if run_cv and not cv_errors:
        with st.spinner("Running cross-validation..."):
            try:
                cv_results = _run_cv_cached(
                    str(dataset_path),
                    datetime.combine(cv_start_date, datetime.min.time()).isoformat(),
                    datetime.combine(cv_end_date, datetime.max.time()).isoformat(),
                    cv_threshold,
                    seed,
                    freq,
                    horizon,
                    enable_sentiment,
                    cv_cache_bust,
                )
            except Exception as exc:  # pragma: no cover - UI display
                st.error(f"CV failed: {exc}")
            else:
                st.session_state["cv_results"] = cv_results

    cv_results = st.session_state.get("cv_results")
    if isinstance(cv_results, dict) and cv_results.get("folds"):
        st.subheader("CV summary")
        st.metric(
            "Average balanced accuracy",
            f"{cv_results.get('average_balanced_accuracy', 0.0):.3f}",
        )
        st.metric("Average MCC", f"{cv_results.get('average_mcc', 0.0):.3f}")

        fold_rows = []
        for idx, fold in enumerate(cv_results["folds"], start=1):
            fold_rows.append(
                {
                    "fold": idx,
                    "balanced_accuracy": fold.get("balanced_accuracy", 0.0),
                    "mcc": fold.get("mcc", 0.0),
                }
            )
        st.dataframe(pd.DataFrame(fold_rows), width="stretch")
        st.subheader("Latest fold details")
        _render_metrics(cv_results["folds"][-1])

    st.divider()
    st.subheader("Train model")
    train_threshold = st.slider(
        "Training confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=threshold_default,
        step=0.05,
    )
    class_weights = st.checkbox("Use class weights", value=True)
    train_force = st.checkbox("Force retrain (ignore cache)", value=False)
    train_cache_bust = datetime.now(UTC).isoformat() if train_force else "stable"

    train_model = st.button("Train model", width="stretch")
    if train_model:
        with st.spinner("Training model..."):
            try:
                metrics, importance = _run_train_cached(
                    str(dataset_path),
                    train_threshold,
                    class_weights,
                    seed,
                    freq,
                    horizon,
                    enable_sentiment,
                    train_cache_bust,
                )
            except Exception as exc:  # pragma: no cover - UI display
                st.error(f"Training failed: {exc}")
            else:
                st.session_state["train_metrics"] = metrics
                st.session_state["train_importance"] = importance
                st.success("Model trained and saved.")

    train_metrics = st.session_state.get("train_metrics")
    if isinstance(train_metrics, dict) and train_metrics:
        st.info("Metrics shown below are in-sample (training data).")
        _render_metrics(train_metrics)

    importance = st.session_state.get("train_importance")
    if isinstance(importance, dict) and importance:
        st.subheader("Feature importance (gain)")
        sorted_importance = (
            pd.DataFrame(
                [{"feature": key, "gain": value} for key, value in importance.items()]
            )
            .sort_values("gain", ascending=False)
            .head(20)
            .set_index("feature")
        )
        st.bar_chart(sorted_importance, width="stretch")


def _predictions_page(config: dict[str, Any]) -> None:
    st.header("Predictions")

    data_dir = Path(config.get("data_dir", "data")).expanduser()
    freq = str(config.get("freq", "1h"))
    horizon = str(config.get("horizon", "4h"))
    threshold = float(config.get("enter_threshold", 0.6))
    tau = float(config.get("tau", 0.0015))

    predictions_path = data_dir / "predictions" / f"{freq}_{horizon}.parquet"

    st.caption(f"Predictions: {predictions_path}")
    st.caption(f"Using confidence threshold: {threshold:.2f} | tau: {tau:.4f}")

    if "predictions_cache_bust" not in st.session_state:
        st.session_state["predictions_cache_bust"] = "stable"

    action_col, realize_col, refresh_col = st.columns(3)
    with action_col:
        run_batch = st.button("Batch Run", width="stretch")
    with realize_col:
        run_realize = st.button(
            "Realize",
            width="stretch",
            disabled=not predictions_path.exists(),
        )
    with refresh_col:
        refresh = st.button("Refresh", width="stretch")

    if refresh:
        st.session_state["predictions_cache_bust"] = datetime.now(UTC).isoformat()

    if run_batch:
        with st.spinner("Running batch inference..."):
            try:
                cli_batch_run.callback(freq=freq, horizon=horizon, model_version=None)
            except Exception as exc:  # pragma: no cover - UI display
                st.error(f"Batch run failed: {exc}")
            else:
                st.success("Batch prediction stored.")
                st.session_state["predictions_cache_bust"] = datetime.now(UTC).isoformat()

    if run_realize:
        with st.spinner("Realizing predictions..."):
            try:
                cli_batch_realize.callback(freq=freq, horizon=horizon, tau=tau)
            except Exception as exc:  # pragma: no cover - UI display
                st.error(f"Realize failed: {exc}")
            else:
                st.success("Predictions realized.")
                st.session_state["predictions_cache_bust"] = datetime.now(UTC).isoformat()

    if not predictions_path.exists():
        st.info("No predictions found yet. Run a batch prediction to create one.")
        return

    try:
        preds = _load_predictions(
            str(predictions_path),
            st.session_state["predictions_cache_bust"],
        )
    except Exception as exc:  # pragma: no cover - UI display
        st.error(f"Failed to load predictions: {exc}")
        return

    if preds.empty:
        st.info("Predictions file is empty.")
        return

    preds = preds.sort_values("timestamp_utc").reset_index(drop=True)
    predicted_label = _derive_predicted_label(preds, threshold)

    latest = preds.iloc[-1]
    avg_p_up = float(preds["p_up"].mean())
    avg_p_down = float(preds["p_down"].mean())

    realized_label = preds["realized_label"].replace("nan", pd.NA)
    realized_mask = realized_label.notna()
    hit_rate = 0.0
    realized_count = int(realized_mask.sum())
    if realized_count:
        hits = (predicted_label[realized_mask] == realized_label[realized_mask]).astype(int)
        hit_rate = float(hits.mean())

    metrics_cols = st.columns(3)
    with metrics_cols[0]:
        st.metric(
            "Latest prediction",
            predicted_label.iloc[-1].upper(),
            delta=f"p_up {latest['p_up']:.2%} | p_down {latest['p_down']:.2%}",
        )
        latest_ts = _format_timestamp(latest.get("timestamp_utc"))
        st.caption(
            f"Timestamp: {latest_ts} UTC" if latest_ts else "Timestamp: Unavailable"
        )
    with metrics_cols[1]:
        st.metric(
            "Average probabilities",
            f"{avg_p_up:.2%} up",
            delta=f"{avg_p_down:.2%} down",
        )
    with metrics_cols[2]:
        st.metric(
            "Realized hit rate",
            f"{hit_rate:.2%}",
            delta=f"{realized_count} realized",
        )

    st.subheader("Probability time series")
    ts_indexed = preds.set_index("timestamp_utc")
    st.line_chart(ts_indexed[["p_up"]], width="stretch")

    st.subheader("Hit rate (rolling)")
    if realized_count:
        window = st.slider("Rolling window (bars)", 5, 200, 50, step=5)
        hit_series = (predicted_label[realized_mask] == realized_label[realized_mask]).astype(
            int
        )
        hit_series.index = ts_indexed.index[realized_mask]
        rolling_hit = hit_series.rolling(window=window, min_periods=1).mean()
        st.line_chart(rolling_hit, width="stretch")
    else:
        st.info("No realized labels available yet for hit rate.")

    st.subheader("Confusion matrix")
    if realized_count:
        labels = ["down", "flat", "up"]
        confusion = pd.crosstab(
            realized_label[realized_mask],
            predicted_label[realized_mask],
            dropna=True,
        )
        confusion = confusion.reindex(index=labels, columns=labels, fill_value=0)
        _render_confusion_matrix(confusion)
        st.dataframe(confusion, width="stretch")
    else:
        st.info("No realized labels available yet for confusion matrix.")

    st.subheader("Prediction table")
    display_frame = preds.copy()
    display_frame["predicted_label"] = predicted_label
    display_frame = display_frame[
        [
            "timestamp_utc",
            "p_up",
            "p_down",
            "predicted_label",
            "realized_label",
            "realized_r",
            "model_version",
            "freq",
            "horizon",
        ]
    ]
    st.dataframe(display_frame, width="stretch")

    st.subheader("Export")
    csv_payload = preds.to_csv(index=False).encode("utf-8")
    json_payload = preds.to_json(orient="records", date_format="iso").encode("utf-8")
    export_col, export_json_col = st.columns(2)
    with export_col:
        st.download_button(
            "Download CSV",
            data=csv_payload,
            file_name=f"predictions_{freq}_{horizon}.csv",
            mime="text/csv",
            width="stretch",
        )
    with export_json_col:
        st.download_button(
            "Download JSON",
            data=json_payload,
            file_name=f"predictions_{freq}_{horizon}.json",
            mime="application/json",
            width="stretch",
        )


def _backtest_page(config: dict[str, Any]) -> None:
    st.header("Backtest")

    data_dir = Path(config.get("data_dir", "data")).expanduser()
    freq = str(config.get("freq", "1h"))
    horizon = str(config.get("horizon", "4h"))
    default_enter = float(config.get("enter_threshold", 0.6))
    default_allow_short = bool(config.get("allow_short", False))
    default_cost = float(config.get("cost_bps", 4.0))

    predictions_path = data_dir / "predictions" / f"{freq}_{horizon}.parquet"
    prices_path = data_dir / "raw" / "prices" / f"btcusd_yf_{freq}.parquet"

    st.caption(f"Predictions: {predictions_path}")
    st.caption(f"Prices: {prices_path}")

    if "backtest_cache_bust" not in st.session_state:
        st.session_state["backtest_cache_bust"] = "stable"

    controls_col, refresh_col = st.columns([3, 1])
    with controls_col:
        enter_threshold = st.slider(
            "Entry threshold",
            min_value=0.0,
            max_value=1.0,
            value=default_enter,
            step=0.05,
        )
        cost_bps = st.slider(
            "Round-trip cost (bps)",
            min_value=0.0,
            max_value=50.0,
            value=default_cost,
            step=0.5,
        )
        allow_short = st.checkbox("Allow short positions", value=default_allow_short)
    with refresh_col:
        refresh = st.button("Refresh data", width="stretch")

    if refresh:
        st.session_state["backtest_cache_bust"] = datetime.now(UTC).isoformat()

    run_backtest = st.button("Run backtest", width="stretch")

    if run_backtest:
        if not predictions_path.exists():
            st.error("Predictions file not found. Run batch predictions first.")
            return
        if not prices_path.exists():
            st.error("Prices file not found. Ingest prices first.")
            return

        with st.spinner("Running backtest..."):
            try:
                preds = _load_predictions(
                    str(predictions_path),
                    st.session_state["backtest_cache_bust"],
                )
                prices = _load_prices(
                    str(prices_path),
                    st.session_state["backtest_cache_bust"],
                )
            except Exception as exc:  # pragma: no cover - UI display
                st.error(f"Failed to load data: {exc}")
                return

            aligned = _align_predictions_prices(preds, prices)
            if aligned.empty:
                st.error("Unable to align predictions with prices.")
                return

            trades, equity = backtest_run(
                aligned["close"],
                aligned["p_up"],
                aligned["p_down"],
                enter=enter_threshold,
                allow_short=allow_short,
                cost_bps=cost_bps,
            )
            metrics = backtest_summary(equity, trades)
            st.session_state["backtest_results"] = {
                "metrics": metrics,
                "equity": equity,
                "trades": trades,
                "aligned": aligned,
            }

    results = st.session_state.get("backtest_results")
    if isinstance(results, dict) and results.get("equity") is not None:
        metrics = results["metrics"]
        equity = results["equity"]
        trades = results["trades"]
        aligned = results["aligned"]

        st.subheader("Key metrics")
        metric_cols = st.columns(5)
        with metric_cols[0]:
            st.metric("Sharpe", f"{metrics.get('sharpe', 0.0):.3f}")
        with metric_cols[1]:
            st.metric("Max drawdown", f"{metrics.get('max_drawdown', 0.0):.3f}")
        with metric_cols[2]:
            st.metric("Hit rate", f"{metrics.get('hit_rate', 0.0):.2%}")
        with metric_cols[3]:
            st.metric("Avg return", f"{metrics.get('avg_return', 0.0):.3%}")
        with metric_cols[4]:
            st.metric("Turnover", f"{metrics.get('turnover', 0.0):.1f}")

        st.caption(f"Aligned rows: {len(aligned)}")

        st.subheader("Equity curve")
        st.line_chart(equity, width="stretch")

        st.subheader("Trades")
        trades_view = trades.copy()
        trades_view["equity"] = equity
        trades_view = trades_view.reset_index().rename(columns={"index": "timestamp_utc"})
        st.dataframe(trades_view, width="stretch")


def _monitor_page(config: dict[str, Any]) -> None:
    st.header("Monitor")

    data_dir = Path(config.get("data_dir", "data")).expanduser()
    freq = str(config.get("freq", "1h"))
    horizon = str(config.get("horizon", "4h"))
    default_cost = float(config.get("cost_bps", 4.0))

    predictions_path = data_dir / "predictions" / f"{freq}_{horizon}.parquet"
    prices_path = data_dir / "raw" / "prices" / f"btcusd_yf_{freq}.parquet"
    live_metrics_path = Path("metrics") / f"live_{freq}_{horizon}.json"

    st.caption(f"Predictions: {predictions_path}")
    st.caption(f"Prices: {prices_path}")
    st.caption(f"Live metrics: {live_metrics_path}")

    if "monitor_cache_bust" not in st.session_state:
        st.session_state["monitor_cache_bust"] = "stable"

    controls_col, refresh_col = st.columns([3, 1])
    with controls_col:
        cost_bps = st.slider(
            "Monitoring cost (bps)",
            min_value=0.0,
            max_value=50.0,
            value=default_cost,
            step=0.5,
        )
    with refresh_col:
        refresh = st.button("Monitor refresh", width="stretch")

    if refresh:
        if not predictions_path.exists():
            st.error("Predictions file not found. Run batch predictions first.")
        else:
            with st.spinner("Refreshing live metrics..."):
                try:
                    cli_monitor_refresh.callback(
                        freq=freq,
                        horizon=horizon,
                        cost_bps=cost_bps,
                    )
                except Exception as exc:  # pragma: no cover - UI display
                    st.error(f"Monitor refresh failed: {exc}")
                else:
                    st.success("Live metrics refreshed.")
                    st.session_state["monitor_cache_bust"] = datetime.now(UTC).isoformat()

    live_metrics = _load_live_metrics(
        str(live_metrics_path),
        st.session_state["monitor_cache_bust"],
    )
    if live_metrics:
        st.subheader("Live metrics")
        metrics_cols = st.columns(5)
        with metrics_cols[0]:
            st.metric("Predictions", live_metrics.get("count", 0))
        with metrics_cols[1]:
            st.metric("Avg p_up", f"{live_metrics.get('avg_p_up', 0.0):.2%}")
        with metrics_cols[2]:
            st.metric("Avg p_down", f"{live_metrics.get('avg_p_down', 0.0):.2%}")
        with metrics_cols[3]:
            st.metric("Realized count", live_metrics.get("realized_count", 0))
        with metrics_cols[4]:
            st.metric("Hit rate", f"{live_metrics.get('hit_rate', 0.0):.2%}")
        if live_metrics.get("updated_at"):
            st.caption(f"Updated at: {live_metrics['updated_at']}")
    else:
        st.info("Run monitor refresh to generate live metrics.")

    if not predictions_path.exists():
        st.info("No predictions available for monitoring.")
        return
    if not prices_path.exists():
        st.info("No prices available for monitoring.")
        return

    try:
        preds = _load_predictions(
            str(predictions_path),
            st.session_state["monitor_cache_bust"],
        )
        prices = _load_prices(
            str(prices_path),
            st.session_state["monitor_cache_bust"],
        )
    except Exception as exc:  # pragma: no cover - UI display
        st.error(f"Failed to load monitoring data: {exc}")
        return

    aligned = _align_predictions_prices(preds, prices)
    if aligned.empty:
        st.info("No aligned predictions and prices to chart yet.")
        return

    st.subheader("Price trend")
    st.line_chart(aligned["close"], width="stretch")

    realized = aligned.dropna(subset=["realized_r"]).copy()
    if realized.empty:
        st.info("No realized returns available for performance charts.")
        return

    realized = realized.sort_index()
    cost = cost_bps / 10000.0
    realized["net_return"] = realized["realized_r"] - cost
    realized["equity"] = (1 + realized["net_return"]).cumprod()

    st.subheader("Cumulative net performance")
    st.line_chart(realized["equity"], width="stretch")

    window = st.slider(
        "Rolling hit rate window (bars)",
        min_value=5,
        max_value=200,
        value=50,
        step=5,
        key="monitor_hit_window",
    )
    realized["hit"] = (realized["realized_r"] > cost).astype(int)
    rolling_hit = realized["hit"].rolling(window=window, min_periods=1).mean()
    st.subheader("Rolling hit rate")
    st.line_chart(rolling_hit, width="stretch")

    st.subheader("Recent realized returns")
    table = (
        realized.reset_index()[
            [
                "timestamp_utc",
                "close",
                "p_up",
                "p_down",
                "realized_r",
                "realized_label",
            ]
        ]
        .sort_values("timestamp_utc")
        .tail(200)
    )
    st.dataframe(table, width="stretch")


PAGES = [
    "Dashboard",
    "Ingest Data",
    "Features",
    "Model",
    "Predictions",
    "Monitor",
    "Backtest",
]


def main() -> None:
    config, config_path, error = _load_config()

    with st.sidebar:
        st.header("Navigation")
        selection = st.radio("Page", PAGES)
        st.caption(
            f"Config: {config_path if config_path else 'Not loaded'}"
        )
        if error:
            st.error(error)
        else:
            _init_global_overrides(config)
            st.divider()
            st.subheader("Global overrides")

            freq_current = str(st.session_state["override_freq"])
            freq_options = [freq_current]
            for option in ["15m", "30m", "1h", "2h", "4h", "1d"]:
                if option not in freq_options:
                    freq_options.append(option)
            st.selectbox(
                "Frequency (freq)",
                options=freq_options,
                index=freq_options.index(freq_current),
                key="override_freq",
                help="Used for data paths and batch operations.",
            )
            st.text_input(
                "Horizon",
                value=str(st.session_state["override_horizon"]),
                key="override_horizon",
                help="Prediction horizon (for example: 4h, 1d).",
            )
            tau_current = float(st.session_state.get("override_tau", 0.0015))
            tau_current = min(max(tau_current, 0.0), 0.02)
            if tau_current != st.session_state.get("override_tau"):
                st.session_state["override_tau"] = tau_current
            st.slider(
                "Label threshold (tau)",
                min_value=0.0,
                max_value=0.02,
                value=tau_current,
                step=0.0005,
                format="%.4f",
                key="override_tau",
                help="Used for feature labels and realization.",
            )
            st.caption("Overrides replace config defaults across all pages.")

    if error:
        st.stop()

    effective_config = _apply_global_overrides(config)

    if selection == "Dashboard":
        _dashboard(effective_config)
    elif selection == "Ingest Data":
        _ingest_data(effective_config)
    elif selection == "Features":
        _build_features_page(effective_config)
    elif selection == "Model":
        _train_model_page(effective_config)
    elif selection == "Predictions":
        _predictions_page(effective_config)
    elif selection == "Monitor":
        _monitor_page(effective_config)
    elif selection == "Backtest":
        _backtest_page(effective_config)


main()
