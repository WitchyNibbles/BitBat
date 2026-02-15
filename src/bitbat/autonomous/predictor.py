"""Live prediction generator for the autonomous monitoring pipeline.

Loads the latest ingested price data, generates features, runs model
inference, and stores the prediction in the autonomous database so the
dashboard and validation feedback loop can use it.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import xgboost as xgb

from bitbat.autonomous.db import AutonomousDB
from bitbat.config.loader import get_runtime_config, load_config
from bitbat.dataset.build import _generate_price_features
from bitbat.model.infer import predict_bar
from bitbat.model.persist import load as load_model

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _load_ingested_prices(data_dir: Path, freq: str) -> pd.DataFrame:
    """Load price bars from the date-partitioned ingestion directory.

    The ingestion service writes to ``data/raw/prices/date=YYYY-MM-DD/``.
    We also check for the legacy flat file used by the CLI.
    """
    prices_root = data_dir / "raw" / "prices"

    frames: list[pd.DataFrame] = []

    # 1. Date-partitioned files written by PriceIngestionService
    if prices_root.exists():
        for parquet_file in sorted(prices_root.glob("**/*.parquet")):
            try:
                df = pd.read_parquet(parquet_file)
                if "timestamp_utc" in df.columns and "close" in df.columns:
                    frames.append(df)
            except Exception as exc:
                logger.warning("Skipping unreadable price file %s: %s", parquet_file, exc)

    # 2. Legacy flat file from CLI ingest (data/raw/prices/btcusd_yf_1h.parquet)
    legacy_path = prices_root / f"btcusd_yf_{freq}.parquet"
    if legacy_path.exists():
        try:
            df = pd.read_parquet(legacy_path)
            if "timestamp_utc" in df.columns and "close" in df.columns:
                frames.append(df)
        except Exception as exc:
            logger.warning("Skipping legacy price file %s: %s", legacy_path, exc)

    if not frames:
        raise RuntimeError(f"No price data found under {prices_root}")

    merged = pd.concat(frames, ignore_index=True)
    merged["timestamp_utc"] = pd.to_datetime(
        merged["timestamp_utc"], utc=True, errors="coerce"
    ).dt.tz_localize(None)
    merged = (
        merged.sort_values("timestamp_utc")
        .drop_duplicates(subset=["timestamp_utc"], keep="last")
        .reset_index(drop=True)
    )
    return merged.set_index("timestamp_utc").sort_index()


class LivePredictor:
    """Generate a prediction from the latest ingested data and store it in the DB."""

    def __init__(
        self,
        db: AutonomousDB,
        freq: str = "1h",
        horizon: str = "4h",
    ) -> None:
        self.db = db
        self.freq = freq
        self.horizon = horizon

        config = get_runtime_config() or load_config()
        self.data_dir = Path(str(config.get("data_dir", "data"))).expanduser()
        self.model_dir = Path("models")

    def _model_path(self) -> Path:
        return self.model_dir / f"{self.freq}_{self.horizon}" / "xgb.json"

    def _load_model(self) -> xgb.Booster:
        model_path = self._model_path()
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
        return load_model(model_path)

    def _active_model_version(self) -> str:
        with self.db.session() as session:
            active = self.db.get_active_model(session, self.freq, self.horizon)
        if active is not None:
            return active.version
        # Fall back to a default version tag when no model is registered in DB
        from bitbat import __version__

        return __version__

    def predict_latest(self) -> dict[str, Any] | None:
        """Generate a prediction for the most recent bar and store it.

        Returns the prediction details dict, or None if prediction could not
        be made (e.g. insufficient data, missing model).
        """
        # Load model
        try:
            booster = self._load_model()
        except FileNotFoundError:
            logger.error("No model artifact found at %s — skipping prediction", self._model_path())
            return None

        # Load ingested prices
        try:
            prices = _load_ingested_prices(self.data_dir, self.freq)
        except RuntimeError as exc:
            logger.error("Cannot generate prediction: %s", exc)
            return None

        if len(prices) < 30:
            logger.warning(
                "Only %d price bars available — need at least 30 for features", len(prices)
            )
            return None

        # Generate features
        try:
            features = _generate_price_features(prices)
            features = features.dropna()
            rename_mapping = {
                col: col if col.startswith("feat_") else f"feat_{col}" for col in features.columns
            }
            features = features.rename(columns=rename_mapping)
        except Exception as exc:
            logger.error("Feature generation failed: %s", exc)
            return None

        if features.empty:
            logger.warning("Feature generation produced no usable rows")
            return None

        # Align features to model's expected columns
        expected_features = list(booster.feature_names or [])
        if not expected_features:
            logger.error("Model artifact missing feature names — cannot align features")
            return None

        available = set(features.columns)
        missing = sorted(set(expected_features) - available)
        if missing:
            logger.error("Features missing columns expected by model: %s", missing)
            return None

        aligned = features[expected_features]
        latest_ts = aligned.index.max()

        # Check for duplicate prediction — don't re-predict the same bar
        with self.db.session() as session:
            existing = self.db.get_unrealized_predictions(
                session=session,
                freq=self.freq,
                horizon=self.horizon,
            )
        for pred in existing:
            pred_ts = pd.Timestamp(pred.timestamp_utc)
            if pred_ts.tzinfo is not None:
                pred_ts = pred_ts.tz_localize(None)
            if pred_ts == pd.Timestamp(latest_ts):
                logger.info("Prediction for %s already exists — skipping", latest_ts)
                return None

        # Run inference
        feature_row = aligned.loc[latest_ts]
        prediction = predict_bar(booster, feature_row, timestamp=latest_ts)

        p_up = float(prediction["p_up"])
        p_down = float(prediction["p_down"])

        direction = "up" if p_up >= p_down else "down"

        # Determine if flat (neither probability is dominant)
        p_flat = max(0.0, 1.0 - p_up - p_down)
        if p_flat > p_up and p_flat > p_down:
            direction = "flat"

        model_version = self._active_model_version()

        timestamp_utc = pd.Timestamp(latest_ts)
        if hasattr(timestamp_utc, "tz_localize"):
            timestamp_utc = timestamp_utc.tz_localize(None)
        timestamp_py = timestamp_utc.to_pydatetime()

        # Store in autonomous.db
        with self.db.session() as session:
            self.db.store_prediction(
                session=session,
                timestamp_utc=timestamp_py,
                predicted_direction=direction,
                p_up=p_up,
                p_down=p_down,
                model_version=model_version,
                freq=self.freq,
                horizon=self.horizon,
            )

        logger.info(
            "Prediction stored: ts=%s direction=%s p_up=%.4f p_down=%.4f model=%s",
            timestamp_py,
            direction,
            p_up,
            p_down,
            model_version,
        )

        return {
            "timestamp_utc": timestamp_py,
            "predicted_direction": direction,
            "p_up": p_up,
            "p_down": p_down,
            "model_version": model_version,
        }
