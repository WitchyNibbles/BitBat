"""Prediction validator for autonomous outcome realization."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from bitbat.autonomous.db import AutonomousDB
from bitbat.autonomous.models import PredictionOutcome
from bitbat.config.loader import get_runtime_config, load_config
from bitbat.io.fs import read_parquet

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _normalize_timestamp(value: datetime | pd.Timestamp) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.to_pydatetime()


class PredictionValidator:
    """Validate predicted directions by comparing against realized prices."""

    def __init__(
        self,
        db: AutonomousDB,
        freq: str = "1h",
        horizon: str = "4h",
        tau: float | None = None,
    ) -> None:
        self.db = db
        self.freq = freq
        self.horizon = horizon

        config = get_runtime_config() or load_config()
        configured_tau = float(config.get("tau", 0.01))
        self.tau = configured_tau if tau is None else float(tau)
        self.horizon_delta = self._parse_horizon(horizon)

        logger.info(
            "Initialized validator with freq=%s horizon=%s tau=%.6f",
            self.freq,
            self.horizon,
            self.tau,
        )

    def _parse_horizon(self, horizon: str) -> timedelta:
        """Parse horizon text like `4h` or `2d` into a timedelta."""
        horizon_text = horizon.strip().lower()
        if len(horizon_text) < 2:
            raise ValueError(f"Invalid horizon format: {horizon}")

        number_text = horizon_text[:-1]
        unit = horizon_text[-1]
        if not number_text.isdigit():
            raise ValueError(f"Invalid horizon format: {horizon}")

        quantity = int(number_text)
        if quantity <= 0:
            raise ValueError(f"Horizon must be > 0: {horizon}")

        if unit == "h":
            return timedelta(hours=quantity)
        if unit == "d":
            return timedelta(days=quantity)
        if unit == "m":
            return timedelta(minutes=quantity)
        raise ValueError(f"Invalid horizon format: {horizon}")

    def find_predictions_to_validate(self) -> list[PredictionOutcome]:
        """Return unrealized predictions that have passed the horizon cutoff."""
        cutoff_time = _utcnow() - self.horizon_delta

        with self.db.session() as session:
            predictions = self.db.get_unrealized_predictions(
                session=session,
                freq=self.freq,
                horizon=self.horizon,
                cutoff_time=cutoff_time,
            )

        logger.info("Found %d predictions ready to validate", len(predictions))
        return predictions

    def fetch_price_data(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> dict[datetime, float]:
        """Load close prices from parquet files for a requested time window."""
        config = get_runtime_config() or load_config()
        data_dir = Path(str(config.get("data_dir", "data"))).expanduser()
        prices_root = data_dir / "raw" / "prices"

        if not prices_root.exists():
            logger.error("Prices directory not found: %s", prices_root)
            return {}

        datasets = sorted(prices_root.glob(f"*_{self.freq}.parquet"))
        if not datasets:
            datasets = sorted(prices_root.glob("*.parquet"))
        if not datasets:
            logger.warning("No price datasets found in %s", prices_root)
            return {}

        frames: list[pd.DataFrame] = []
        for dataset in datasets:
            try:
                frame = read_parquet(dataset)
            except Exception as exc:
                logger.warning("Skipping unreadable price dataset %s: %s", dataset, exc)
                continue

            if "timestamp_utc" not in frame.columns or "close" not in frame.columns:
                logger.warning("Skipping dataset missing columns: %s", dataset)
                continue

            normalized = frame[["timestamp_utc", "close"]].copy()
            normalized["timestamp_utc"] = pd.to_datetime(
                normalized["timestamp_utc"],
                utc=True,
                errors="coerce",
            ).dt.tz_localize(None)
            normalized["close"] = pd.to_numeric(normalized["close"], errors="coerce")
            normalized = normalized.dropna(subset=["timestamp_utc", "close"])
            if normalized.empty:
                continue
            frames.append(normalized)

        if not frames:
            logger.error("No usable price rows found in %s", prices_root)
            return {}

        merged = pd.concat(frames, ignore_index=True)
        merged = (
            merged.sort_values("timestamp_utc")
            .drop_duplicates(subset=["timestamp_utc"], keep="last")
            .reset_index(drop=True)
        )

        start_norm = _normalize_timestamp(start_time)
        end_norm = _normalize_timestamp(end_time)
        filtered = merged[
            (merged["timestamp_utc"] >= start_norm) & (merged["timestamp_utc"] <= end_norm)
        ].copy()

        result: dict[datetime, float] = {
            ts.to_pydatetime(): float(price)
            for ts, price in zip(filtered["timestamp_utc"], filtered["close"], strict=False)
        }

        logger.info(
            "Fetched %d prices for window %s to %s",
            len(result),
            start_norm,
            end_norm,
        )
        return result

    def get_price_at_timestamp(
        self,
        price_data: dict[datetime, float],
        timestamp: datetime,
        tolerance_minutes: int = 60,
    ) -> float | None:
        """Return exact or nearest price within a tolerance window."""
        if not price_data:
            logger.warning("Price data is empty")
            return None

        ts = _normalize_timestamp(timestamp)
        if ts in price_data:
            return price_data[ts]

        tolerance = timedelta(minutes=tolerance_minutes)
        min_time = ts - tolerance
        max_time = ts + tolerance

        nearby = {
            candidate_ts: price
            for candidate_ts, price in price_data.items()
            if min_time <= candidate_ts <= max_time
        }
        if not nearby:
            logger.warning(
                "No price found within %d minutes of %s",
                tolerance_minutes,
                ts,
            )
            return None

        closest_ts = min(nearby.keys(), key=lambda candidate: abs((candidate - ts).total_seconds()))
        gap_minutes = abs((closest_ts - ts).total_seconds()) / 60.0
        if gap_minutes > 5:
            logger.warning(
                "Using price %.1f minutes away (wanted %s, got %s)",
                gap_minutes,
                ts,
                closest_ts,
            )
        return nearby[closest_ts]

    def calculate_return(self, start_price: float, end_price: float) -> float:
        """Compute return as `(end - start) / start`."""
        if start_price == 0:
            raise ValueError("Cannot calculate return with start_price=0.")
        return (end_price - start_price) / start_price

    def classify_direction(self, actual_return: float) -> str:
        """Map numeric return to `up`, `down`, or `flat` using `tau`."""
        if actual_return > self.tau:
            return "up"
        if actual_return < -self.tau:
            return "down"
        return "flat"

    def validate_prediction(
        self,
        prediction: PredictionOutcome,
        price_data: dict[datetime, float],
    ) -> dict[str, Any] | None:
        """Validate one prediction and return computed fields, or None on failure."""
        try:
            prediction_time = _normalize_timestamp(prediction.timestamp_utc)

            start_price = self.get_price_at_timestamp(
                price_data=price_data,
                timestamp=prediction_time,
                tolerance_minutes=60,
            )
            if start_price is None:
                logger.error(
                    "Cannot validate prediction %s: no start price for %s",
                    prediction.id,
                    prediction_time,
                )
                return None

            target_time = prediction_time + self.horizon_delta
            end_price = self.get_price_at_timestamp(
                price_data=price_data,
                timestamp=target_time,
                tolerance_minutes=60,
            )
            if end_price is None:
                logger.error(
                    "Cannot validate prediction %s: no end price for %s",
                    prediction.id,
                    target_time,
                )
                return None

            actual_return = self.calculate_return(start_price, end_price)
            actual_direction = self.classify_direction(actual_return)
            correct = prediction.predicted_direction == actual_direction

            if abs(actual_return) > 0.5:
                logger.warning(
                    "Anomalous return for prediction %s: %.2f%% (start=%.4f end=%.4f)",
                    prediction.id,
                    actual_return * 100.0,
                    start_price,
                    end_price,
                )

            logger.debug(
                "Validated prediction %s predicted=%s actual=%s correct=%s return=%.6f",
                prediction.id,
                prediction.predicted_direction,
                actual_direction,
                correct,
                actual_return,
            )
            return {
                "prediction_id": int(prediction.id),
                "actual_return": float(actual_return),
                "actual_direction": actual_direction,
                "correct": bool(correct),
                "start_price": float(start_price),
                "end_price": float(end_price),
            }
        except Exception as exc:
            logger.error("Error validating prediction %s: %s", prediction.id, exc)
            return None

    def validate_batch(self, predictions: list[PredictionOutcome]) -> dict[str, Any]:
        """Validate a batch and update realized values in the database."""
        if not predictions:
            logger.info("No predictions to validate")
            return {
                "validated_count": 0,
                "correct_count": 0,
                "hit_rate": 0.0,
                "errors": [],
            }

        min_time = min(_normalize_timestamp(pred.timestamp_utc) for pred in predictions)
        max_time = max(
            _normalize_timestamp(pred.timestamp_utc) + self.horizon_delta
            for pred in predictions
        )
        window_start = min_time - timedelta(hours=1)
        window_end = max_time + timedelta(hours=1)

        logger.info("Fetching prices for batch window %s to %s", window_start, window_end)
        price_data = self.fetch_price_data(window_start, window_end)
        if not price_data:
            logger.error("No price data available for validation")
            return {
                "validated_count": 0,
                "correct_count": 0,
                "hit_rate": 0.0,
                "errors": ["No price data available"],
            }

        validated_count = 0
        correct_count = 0
        errors: list[str] = []

        for prediction in predictions:
            result = self.validate_prediction(prediction, price_data)
            if result is None:
                errors.append(f"Failed to validate prediction {prediction.id}")
                continue

            try:
                with self.db.session() as session:
                    self.db.realize_prediction(
                        session=session,
                        prediction_id=int(result["prediction_id"]),
                        actual_return=float(result["actual_return"]),
                        actual_direction=str(result["actual_direction"]),
                    )
            except Exception as exc:
                logger.error("Error updating prediction %s: %s", prediction.id, exc)
                errors.append(f"Database error for prediction {prediction.id}: {exc}")
                continue

            validated_count += 1
            if bool(result["correct"]):
                correct_count += 1

        hit_rate = (correct_count / validated_count) if validated_count else 0.0
        logger.info(
            "Batch validation complete validated=%d correct=%d hit_rate=%.2f%%",
            validated_count,
            correct_count,
            hit_rate * 100.0,
        )
        return {
            "validated_count": validated_count,
            "correct_count": correct_count,
            "hit_rate": hit_rate,
            "errors": errors,
        }

    def validate_all(self) -> dict[str, Any]:
        """Run full validation pass for all ready predictions."""
        logger.info("Starting validation run")
        predictions = self.find_predictions_to_validate()
        if not predictions:
            logger.info("No predictions ready to validate")
            return {
                "validated_count": 0,
                "correct_count": 0,
                "hit_rate": 0.0,
                "errors": [],
            }

        results = self.validate_batch(predictions)
        logger.info("Validation run complete: %s", results)
        return results
