"""Streamlit app for the Alpha BTC prediction pipeline."""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from alpha.config.loader import get_runtime_config_path, set_runtime_config
from alpha.ingest import news_gdelt as news_module
from alpha.ingest import prices as prices_module


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


def _resolve_date_range(
    start_date: date,
    end_date: date,
) -> tuple[datetime, datetime, list[str], list[str]]:
    errors: list[str] = []
    infos: list[str] = []

    if end_date < start_date:
        errors.append("End date must be on or after the start date.")

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)
    now = datetime.utcnow()

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

    st.dataframe(frame, use_container_width=True)
    csv_payload = frame.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download {label.lower()} preview (CSV)",
        data=csv_payload,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


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
    st.dataframe(pd.DataFrame(data_rows), use_container_width=True)

    st.subheader("Model status")
    st.dataframe(pd.DataFrame(model_rows), use_container_width=True)

    st.subheader("Predictions status")
    st.dataframe(pd.DataFrame(prediction_rows), use_container_width=True)


def _ingest_data(config: dict[str, Any]) -> None:
    st.header("Ingest Data")

    data_dir = Path(config.get("data_dir", "data")).expanduser()
    freq = str(config.get("freq", "1h"))

    st.caption(f"Data dir: {data_dir} | default freq: {freq}")

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
    throttle_seconds = float(config.get("news_throttle_seconds", 0.0))
    retry_limit = int(config.get("news_retry_limit", 3))

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
            use_container_width=True,
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
        news_target = news_module._target_path(news_root)
        st.caption(f"Output: {news_target}")
        st.caption(f"Throttle: {throttle_seconds:.1f}s | Retries: {retry_limit}")
        pull_news = st.button(
            "Pull news",
            use_container_width=True,
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


PAGES = [
    "Dashboard",
    "Ingest Data",
    "Build Features",
    "Train Model",
    "Predictions",
    "Backtest",
]


def main() -> None:
    st.set_page_config(page_title="Alpha BTC Predictor", layout="wide")

    config, config_path, error = _load_config()

    with st.sidebar:
        st.header("Navigation")
        selection = st.radio("Page", PAGES)
        st.caption(
            f"Config: {config_path if config_path else 'Not loaded'}"
        )
        if error:
            st.error(error)

    if error:
        st.stop()

    if selection == "Dashboard":
        _dashboard(config)
    elif selection == "Ingest Data":
        _ingest_data(config)
    elif selection == "Build Features":
        st.header("Build Features")
        st.info("Generate the feature dataset via the CLI before training the model.")
    elif selection == "Train Model":
        st.header("Train Model")
        st.info("Train the XGBoost model from the feature dataset using the CLI.")
    elif selection == "Predictions":
        st.header("Predictions")
        st.info("Generate and view predictions once a trained model is available.")
    elif selection == "Backtest":
        st.header("Backtest")
        st.info("Run backtests against stored predictions using the CLI.")


if __name__ == "__main__":
    main()
