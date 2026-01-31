"""Streamlit app for the Alpha BTC prediction pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from alpha.config.loader import get_runtime_config_path, set_runtime_config


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
        st.header("Ingest Data")
        st.info("Use the CLI to pull prices and news data before building features.")
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
