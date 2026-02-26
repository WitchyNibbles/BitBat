"""Continuous retraining loop for the regression model."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from bitbat.autonomous.db import AutonomousDB

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class ContinuousTrainer:
    """Retrains on a rolling window whenever enough new realized predictions accumulate."""

    def __init__(
        self,
        db: AutonomousDB,
        freq: str,
        horizon: str,
        config: dict[str, Any],
    ) -> None:
        self.db = db
        self.freq = freq
        self.horizon = horizon

        ct_cfg = config.get("continuous_training", {})
        self.retrain_interval = int(ct_cfg.get("retrain_interval_seconds", 1800))
        self.min_new_samples = int(ct_cfg.get("min_new_samples", 6))
        self.rolling_window_bars = int(ct_cfg.get("rolling_window_bars", 17280))
        default_train_bars = int(ct_cfg.get("train_window_bars", max(self.rolling_window_bars // 2, 1)))
        self.train_window_bars = max(default_train_bars, 1)
        self.backtest_window_bars = int(
            ct_cfg.get("backtest_window_bars", max(self.rolling_window_bars - self.train_window_bars, 1))
        )
        self._last_retrain_time: datetime | None = None
        self._last_realized_count: int = 0

        self.data_dir = Path(str(config.get("data_dir", "data"))).expanduser()
        self.enable_garch = bool(config.get("enable_garch", False))
        self.seed = int(config.get("seed", 42))

    def should_retrain(self) -> bool:
        """Check if enough time and new samples have accumulated."""
        now = _utcnow()
        if self._last_retrain_time is not None:
            elapsed = (now - self._last_retrain_time).total_seconds()
            if elapsed < self.retrain_interval:
                return False

        with self.db.session() as session:
            realized = self.db.get_recent_predictions(
                session=session,
                freq=self.freq,
                horizon=self.horizon,
                days=7,
                realized_only=True,
            )
        current_count = len(realized)
        new_samples = current_count - self._last_realized_count
        if new_samples < self.min_new_samples:
            logger.info(
                "Not enough new samples for retraining: %d/%d",
                new_samples,
                self.min_new_samples,
            )
            return False

        return True

    def retrain(self) -> dict[str, Any]:
        """Execute a retraining cycle and deploy if improved."""
        t0 = time.monotonic()
        logger.info("Starting continuous retraining cycle")

        # Record event
        with self.db.session() as session:
            old_model = self.db.get_active_model(session, self.freq, self.horizon)
            old_version = old_model.version if old_model else "unknown"
            event = self.db.create_retraining_event(
                session=session,
                trigger_reason="continuous",
                trigger_metrics=None,
                old_model_version=old_version,
            )
            event_id = event.id

        try:
            result = self._do_retrain(old_version)
            duration = time.monotonic() - t0

            if result["deployed"]:
                with self.db.session() as session:
                    self.db.complete_retraining_event(
                        session=session,
                        event_id=event_id,
                        new_model_version=result["new_version"],
                        cv_improvement=result.get("rmse_improvement", 0.0),
                        training_duration_seconds=duration,
                    )
            else:
                with self.db.session() as session:
                    self.db.complete_retraining_event(
                        session=session,
                        event_id=event_id,
                        new_model_version=old_version,
                        cv_improvement=0.0,
                        training_duration_seconds=duration,
                    )

            self._last_retrain_time = _utcnow()
            with self.db.session() as session:
                realized = self.db.get_recent_predictions(
                    session=session,
                    freq=self.freq,
                    horizon=self.horizon,
                    days=7,
                    realized_only=True,
                )
            self._last_realized_count = len(realized)

            result["status"] = "completed"
            result["duration_seconds"] = round(duration, 1)
            return result

        except Exception as exc:
            duration = time.monotonic() - t0
            logger.error("Retraining failed: %s", exc, exc_info=True)
            with self.db.session() as session:
                self.db.fail_retraining_event(
                    session=session,
                    event_id=event_id,
                    error_message=str(exc),
                )
            return {"status": "failed", "error": str(exc), "duration_seconds": round(duration, 1)}

    def _do_retrain(self, old_version: str) -> dict[str, Any]:
        from bitbat.dataset.build import _generate_price_features
        from bitbat.labeling.returns import forward_return
        from bitbat.model.evaluate import regression_metrics
        from bitbat.model.infer import _ensure_model
        from bitbat.model.train import fit_xgb

        # Load prices
        prices = self._load_prices()
        if len(prices) < 100:
            raise ValueError(f"Insufficient price data: {len(prices)} bars")

        # Trim to rolling window
        if len(prices) > self.rolling_window_bars:
            prices = prices.iloc[-self.rolling_window_bars :]

        # Generate features
        features = _generate_price_features(prices, enable_garch=self.enable_garch, freq=self.freq)
        features = features.dropna()
        rename_mapping = {
            col: col if col.startswith("feat_") else f"feat_{col}" for col in features.columns
        }
        features = features.rename(columns=rename_mapping)

        # Compute targets
        y_returns = forward_return(prices["close"].to_frame(), self.horizon)
        y_returns = y_returns.loc[features.index]
        valid = y_returns.notna()
        features = features.loc[valid]
        y_returns = y_returns.loc[valid].astype("float64")

        if len(features) < 50:
            raise ValueError(f"Too few valid samples after feature generation: {len(features)}")

        required_samples = self.train_window_bars + self.backtest_window_bars
        if len(features) < required_samples:
            raise ValueError(
                "Not enough samples for configured windows: "
                f"{len(features)} < {required_samples}"
            )

        scoped_features = features.iloc[-required_samples:]
        scoped_returns = y_returns.iloc[-required_samples:]
        X_train = scoped_features.iloc[: self.train_window_bars]
        y_train = scoped_returns.iloc[: self.train_window_bars]
        X_holdout = scoped_features.iloc[self.train_window_bars :]
        y_holdout = scoped_returns.iloc[self.train_window_bars :]

        window_metadata = {
            "train_window_bars": self.train_window_bars,
            "backtest_window_bars": self.backtest_window_bars,
            "train_start": X_train.index.min().isoformat(),
            "train_end": X_train.index.max().isoformat(),
            "backtest_start": X_holdout.index.min().isoformat(),
            "backtest_end": X_holdout.index.max().isoformat(),
        }

        # Train new model
        X_train.attrs["freq"] = self.freq
        X_train.attrs["horizon"] = self.horizon
        new_booster, importance = fit_xgb(X_train, y_train, seed=self.seed)

        # Evaluate new model
        import xgboost as xgb

        dtest = xgb.DMatrix(X_holdout, feature_names=list(X_holdout.columns))
        new_preds = new_booster.predict(dtest)
        new_metrics = regression_metrics(y_holdout, new_preds)

        # Compare to current model
        model_path = Path("models") / f"{self.freq}_{self.horizon}" / "xgb.json"
        deployed = False
        rmse_improvement = 0.0

        if model_path.exists():
            old_booster = _ensure_model(model_path)
            old_preds = old_booster.predict(dtest)
            old_metrics = regression_metrics(y_holdout, old_preds)

            rmse_improvement = old_metrics["rmse"] - new_metrics["rmse"]
            if rmse_improvement > 0:
                deployed = True
                logger.info(
                    "New model improves RMSE by %.6f (%.6f -> %.6f)",
                    rmse_improvement,
                    old_metrics["rmse"],
                    new_metrics["rmse"],
                )
            else:
                logger.info(
                    "New model does not improve (old RMSE=%.6f, new RMSE=%.6f)",
                    old_metrics["rmse"],
                    new_metrics["rmse"],
                )
        else:
            deployed = True

        new_version = f"continuous-{int(time.time())}"

        if deployed:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            new_booster.save_model(str(model_path))

            with self.db.session() as session:
                self.db.deactivate_old_models(session, self.freq, self.horizon)
                self.db.store_model_version(
                    session=session,
                    version=new_version,
                    freq=self.freq,
                    horizon=self.horizon,
                    training_start=features.index.min().to_pydatetime(),
                    training_end=features.index.max().to_pydatetime(),
                    training_samples=len(X_train),
                    cv_score=new_metrics["rmse"],
                    features=list(X_train.columns),
                    hyperparameters=None,
                    training_metadata={
                        "trigger": "continuous",
                        "holdout_metrics": new_metrics,
                        "window_metadata": window_metadata,
                    },
                    is_active=True,
                )

        return {
            "deployed": deployed,
            "new_version": new_version if deployed else old_version,
            "new_rmse": new_metrics["rmse"],
            "new_mae": new_metrics["mae"],
            "new_directional_accuracy": new_metrics["directional_accuracy"],
            "rmse_improvement": rmse_improvement,
            "training_samples": len(X_train),
            "holdout_samples": len(X_holdout),
            "window_metadata": window_metadata,
        }

    def _load_prices(self) -> pd.DataFrame:
        prices_root = self.data_dir / "raw" / "prices"
        frames: list[pd.DataFrame] = []
        if prices_root.exists():
            for pf in sorted(prices_root.glob("**/*.parquet")):
                try:
                    df = pd.read_parquet(pf)
                    if "timestamp_utc" in df.columns and "close" in df.columns:
                        frames.append(df)
                except Exception:  # noqa: S110
                    pass
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
