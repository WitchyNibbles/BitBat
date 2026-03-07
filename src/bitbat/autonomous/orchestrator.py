"""One-click training orchestrator: ingest -> features -> train -> deploy."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_LOOKBACK_DAYS = 730  # 2 years of historical data for training


def _progress(
    callback: Callable[[str, float], None] | None,
    message: str,
    fraction: float,
) -> None:
    if callback is not None:
        callback(message, fraction)
    logger.info("[%.0f%%] %s", fraction * 100, message)


def one_click_train(  # noqa: C901
    preset_name: str = "balanced",
    progress_callback: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    """Execute the full pipeline in one call.

    Steps: load config -> ingest prices -> ingest news -> build features ->
    train model -> register in DB -> generate first prediction.

    Parameters
    ----------
    preset_name:
        One of "conservative", "balanced", "aggressive".
    progress_callback:
        Optional ``(message, fraction)`` callback for UI progress updates.

    Returns
    -------
    Dict with ``status``, ``model_version``, ``training_samples``,
    ``duration_seconds``, and ``error`` (on failure).
    """
    t0 = time.monotonic()

    # ------------------------------------------------------------------
    # Step 1: Load config and preset
    # ------------------------------------------------------------------
    _progress(progress_callback, "Loading configuration...", 0.02)

    try:
        from bitbat.common.presets import get_preset
        from bitbat.config.loader import get_runtime_config, load_config

        preset = get_preset(preset_name)
        freq = preset.freq
        horizon = preset.horizon

        config = get_runtime_config() or load_config()
        data_dir = Path(str(config.get("data_dir", "data"))).expanduser()
        enable_sentiment = bool(config.get("enable_sentiment", True))
        enable_garch = bool(config.get("enable_garch", False))
        enable_macro = bool(config.get("enable_macro", False))
        enable_onchain = bool(config.get("enable_onchain", False))
        db_url = config.get("autonomous", {}).get("database_url", "sqlite:///data/autonomous.db")
    except Exception as exc:
        return _fail("config", exc, t0)

    # ------------------------------------------------------------------
    # Step 2: Ingest prices
    # ------------------------------------------------------------------
    _progress(progress_callback, "Downloading price data...", 0.05)

    try:
        from bitbat.ingest import prices as prices_module

        start_dt = datetime.now(UTC) - timedelta(days=_LOOKBACK_DAYS)
        prices_module.fetch_yf("BTC-USD", freq, start_dt)
    except Exception as exc:
        return _fail("ingest_prices", exc, t0)

    prices_path = data_dir / "raw" / "prices" / f"btcusd_yf_{freq}.parquet"
    if not prices_path.exists():
        return _fail("ingest_prices", FileNotFoundError(f"Prices not found: {prices_path}"), t0)

    _progress(progress_callback, "Price data downloaded.", 0.25)

    # ------------------------------------------------------------------
    # Step 3: Ingest news (optional)
    # ------------------------------------------------------------------
    news_path: Path | None = None
    if enable_sentiment:
        _progress(progress_callback, "Downloading news data...", 0.26)
        try:
            from bitbat.ingest import news_cryptocompare as news_cc

            news_cc.fetch(
                from_dt=start_dt,
                to_dt=datetime.now(UTC),
                throttle_seconds=0.5,
            )
        except Exception as exc:
            logger.warning("News ingest failed (%s) — continuing without sentiment", exc)
            enable_sentiment = False

        news_subdir = f"cryptocompare_{freq}"
        candidate = data_dir / "raw" / "news" / news_subdir / "cryptocompare_btc_1h.parquet"
        if candidate.exists():
            news_path = candidate
        else:
            enable_sentiment = False

    _progress(progress_callback, "Data ingestion complete.", 0.40)

    # ------------------------------------------------------------------
    # Step 4: Build features
    # ------------------------------------------------------------------
    _progress(progress_callback, "Building features and labels...", 0.41)

    try:
        from bitbat.dataset.build import build_xy

        prices_df = pd.read_parquet(prices_path)
        ts_col = "timestamp_utc" if "timestamp_utc" in prices_df.columns else prices_df.columns[0]
        first_date = pd.to_datetime(prices_df[ts_col]).min()
        last_date = pd.to_datetime(prices_df[ts_col]).max()

        macro_parquet: Path | None = None
        onchain_parquet: Path | None = None
        if enable_macro:
            mp = data_dir / "raw" / "macro" / "fred.parquet"
            if mp.exists():
                macro_parquet = mp
        if enable_onchain:
            op = data_dir / "raw" / "onchain" / "blockchain_info.parquet"
            if op.exists():
                onchain_parquet = op

        X, y, meta = build_xy(
            prices_parquet=prices_path,
            news_parquet=news_path,
            freq=freq,
            horizon=horizon,
            start=str(first_date.date()),
            end=str(last_date.date()),
            enable_sentiment=enable_sentiment,
            enable_garch=enable_garch,
            macro_parquet=macro_parquet,
            onchain_parquet=onchain_parquet,
        )
    except Exception as exc:
        return _fail("build_features", exc, t0)

    if X.empty:
        return _fail("build_features", ValueError("Feature matrix is empty"), t0)

    _progress(progress_callback, f"Features built: {len(X)} samples.", 0.60)

    # ------------------------------------------------------------------
    # Step 5: Train model
    # ------------------------------------------------------------------
    _progress(progress_callback, "Training XGBoost model...", 0.61)

    try:
        from bitbat.model.train import fit_xgb

        X.attrs["freq"] = freq
        X.attrs["horizon"] = horizon
        booster, importance = fit_xgb(X, y)
    except Exception as exc:
        return _fail("train_model", exc, t0)

    _progress(progress_callback, "Model trained and saved.", 0.85)

    # ------------------------------------------------------------------
    # Step 6: Register model in DB
    # ------------------------------------------------------------------
    _progress(progress_callback, "Registering model...", 0.86)

    try:
        from bitbat import __version__
        from bitbat.autonomous.db import AutonomousDB
        from bitbat.autonomous.models import init_database

        init_database(db_url)
        db = AutonomousDB(db_url)

        version_tag = f"{__version__}-{int(time.time())}"

        with db.session() as session:
            # Deactivate previous models for this freq/horizon
            existing = db.get_active_model(session, freq, horizon)
            if existing is not None:
                existing.is_active = False

            db.store_model_version(
                session=session,
                version=version_tag,
                freq=freq,
                horizon=horizon,
                training_start=first_date.to_pydatetime(),
                training_end=last_date.to_pydatetime(),
                training_samples=len(X),
                cv_score=None,
                features=list(X.columns),
                hyperparameters=None,
                training_metadata=None,
                is_active=True,
            )
    except Exception as exc:
        return _fail("register_model", exc, t0)

    _progress(progress_callback, "Model registered.", 0.95)

    # ------------------------------------------------------------------
    # Step 7: Generate first prediction
    # ------------------------------------------------------------------
    _progress(progress_callback, "Generating first prediction...", 0.96)

    try:
        from bitbat.autonomous.predictor import LivePredictor

        predictor = LivePredictor(db, freq=freq, horizon=horizon)
        predictor.predict_latest()
    except Exception as exc:
        # Non-fatal — model is ready even if first prediction fails
        logger.warning("First prediction failed (non-fatal): %s", exc)

    duration = round(time.monotonic() - t0, 1)
    _progress(progress_callback, "Done!", 1.0)

    return {
        "status": "success",
        "model_version": version_tag,
        "training_samples": len(X),
        "duration_seconds": duration,
        "freq": freq,
        "horizon": horizon,
    }


def _fail(step: str, exc: Exception, t0: float) -> dict[str, Any]:
    logger.error("One-click train failed at step '%s': %s", step, exc)
    return {
        "status": "failed",
        "step": step,
        "error": str(exc),
        "duration_seconds": round(time.monotonic() - t0, 1),
    }
