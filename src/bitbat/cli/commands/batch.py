"""Batch job CLI commands."""

from __future__ import annotations

import click
import numpy as np
import pandas as pd

from bitbat import __version__
from bitbat.cli._helpers import (
    _config,
    _ensure_path_exists,
    _load_news,
    _load_prices_indexed,
    _model_path,
    _predictions_path,
    _resolve_setting,
    _sentiment_enabled,
)
from bitbat.contracts import ensure_feature_contract, ensure_predictions_contract
from bitbat.dataset.build import generate_price_features  # noqa: F401
from bitbat.features.sentiment import aggregate as aggregate_sentiment  # noqa: F401
from bitbat.labeling.returns import forward_return
from bitbat.model.infer import predict_bar  # noqa: F401
from bitbat.model.persist import load as load_model  # noqa: F401


@click.group(help="Batch jobs.")
def batch() -> None:
    """Batch command namespace."""


@batch.command("run")
@click.option("--freq", default=None, help="Bar frequency.")
@click.option("--horizon", default=None, help="Prediction horizon.")
@click.option("--model-version", default=None, help="Override model version tag.")
def batch_run(
    freq: str | None,
    horizon: str | None,
    model_version: str | None,
) -> None:
    """Generate a single prediction and store it."""
    freq_val = _resolve_setting(freq, "freq")
    horizon_val = _resolve_setting(horizon, "horizon")

    prices = _load_prices_indexed(freq_val)
    enable_sentiment = _sentiment_enabled()

    enable_garch = bool(_config().get("enable_garch", False))
    price_features = generate_price_features(prices, enable_garch=enable_garch)
    if enable_sentiment:
        news = _load_news()
        bar_df = prices.reset_index()[["timestamp_utc"]]
        sentiment_features = aggregate_sentiment(
            news_df=news,
            bar_df=bar_df,
            freq=freq_val,
        )
        features = price_features.join(sentiment_features, how="left")
    else:
        features = price_features.copy()
    features = features.dropna()
    rename_mapping = {
        column: column if column.startswith("feat_") else f"feat_{column}"
        for column in features.columns
    }
    features = features.rename(columns=rename_mapping)
    if features.empty:
        raise click.ClickException("No features available for batch inference.")

    model_path = _model_path(freq_val, horizon_val)
    _ensure_path_exists(model_path, "Model artifact")
    booster = load_model(model_path)

    features_with_ts = features.copy()
    features_with_ts["timestamp_utc"] = features_with_ts.index
    features_validated = ensure_feature_contract(
        features_with_ts,
        require_label=False,
        require_forward_return=False,
        require_features_full=enable_sentiment,
    )

    expected_features = list(booster.feature_names or [])
    if not expected_features:
        raise click.ClickException(
            "Model artifact missing feature names; cannot align batch features."
        )
    missing = sorted(set(expected_features) - set(features_validated.columns))
    if missing:
        raise click.ClickException(f"Batch features missing expected columns: {missing}")

    aligned_features = features_validated[expected_features]
    latest_ts = aligned_features.index.max()
    feature_row = aligned_features.loc[latest_ts]
    current_price = float(prices["close"].iloc[-1])
    tau = float(_config().get("tau", 0.01) or 0.01)
    prediction = predict_bar(
        booster,
        feature_row,
        timestamp=latest_ts,
        current_price=current_price,
        tau=tau,
    )

    timestamp_value = prediction.get("timestamp", latest_ts)
    timestamp_utc = pd.to_datetime(timestamp_value, utc=True, errors="coerce")
    if pd.isna(timestamp_utc):
        raise click.ClickException("Inference produced an invalid timestamp.")
    if hasattr(timestamp_utc, "tz_convert"):
        timestamp_utc = timestamp_utc.tz_convert(None)
    timestamp_py = timestamp_utc.to_pydatetime()

    record = {
        "timestamp_utc": timestamp_py,
        "predicted_return": prediction.get("predicted_return"),
        "predicted_price": prediction.get("predicted_price"),
        "freq": freq_val,
        "horizon": horizon_val,
        "model_version": model_version or __version__,
        "realized_r": np.nan,
    }
    new_df = ensure_predictions_contract(pd.DataFrame([record]))

    predictions_path = _predictions_path(freq_val, horizon_val)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    if predictions_path.exists():
        existing = ensure_predictions_contract(pd.read_parquet(predictions_path))
        combined = (
            pd.concat([existing, new_df], axis=0, ignore_index=True)
            .sort_values("timestamp_utc")
            .drop_duplicates(subset=["timestamp_utc", "horizon", "model_version"], keep="last")
        )
    else:
        combined = new_df
    combined = ensure_predictions_contract(combined)
    combined.to_parquet(predictions_path, index=False)
    latest_record = combined.iloc[-1]
    click.echo(f"Stored prediction for {latest_record['timestamp_utc']} at {predictions_path}")

    # Also store in autonomous.db so the dashboard and validation loop can see it.
    try:
        from bitbat.autonomous.db import AutonomousDB
        from bitbat.autonomous.models import init_database

        autonomous_cfg = _config().get("autonomous", {})
        db_url = str(autonomous_cfg.get("database_url", "sqlite:///data/autonomous.db"))
        init_database(db_url)
        db = AutonomousDB(db_url)

        predicted_return = prediction.get("predicted_return")
        predicted_direction = prediction["predicted_direction"]

        with db.session() as session:
            db.store_prediction(
                session=session,
                timestamp_utc=timestamp_py,
                predicted_direction=predicted_direction,
                model_version=model_version or __version__,
                freq=freq_val,
                horizon=horizon_val,
                predicted_return=predicted_return,
                predicted_price=prediction.get("predicted_price"),
                p_up=float(prediction.get("p_up", 0.0)),
                p_down=float(prediction.get("p_down", 0.0)),
                p_flat=float(prediction.get("p_flat", 0.0)),
            )
        click.echo(f"Also stored in autonomous DB ({db_url})")
    except Exception as exc:
        click.echo(f"Warning: could not store in autonomous DB: {exc}", err=True)


@batch.command("realize")
@click.option("--freq", default=None, help="Bar frequency.")
@click.option("--horizon", default=None, help="Prediction horizon.")
def batch_realize(
    freq: str | None,
    horizon: str | None,
) -> None:
    """Attach realized returns to stored predictions."""
    freq_val = _resolve_setting(freq, "freq")
    horizon_val = _resolve_setting(horizon, "horizon")

    predictions_path = _predictions_path(freq_val, horizon_val)
    _ensure_path_exists(predictions_path, "Predictions parquet")
    preds = ensure_predictions_contract(pd.read_parquet(predictions_path))
    if preds.empty:
        raise click.ClickException("No predictions available to realise.")

    preds["timestamp_utc"] = pd.to_datetime(
        preds["timestamp_utc"], utc=True, errors="coerce"
    ).dt.tz_localize(None)

    pending_mask = preds["realized_r"].isna()
    if not pending_mask.any():
        click.echo("All predictions already realised.")
        return

    prices = _load_prices_indexed(freq_val)
    returns = forward_return(prices[["close"]], horizon_val)
    reindexed = pd.to_datetime(returns.index)
    if getattr(reindexed, "tz", None) is not None:
        reindexed = reindexed.tz_localize(None)
    returns.index = reindexed

    preds.loc[pending_mask, "realized_r"] = preds.loc[pending_mask, "timestamp_utc"].map(returns)
    updated_mask = preds["realized_r"].notna()

    preds = ensure_predictions_contract(preds)
    preds.to_parquet(predictions_path, index=False)
    click.echo(f"Updated {updated_mask.sum()} predictions with realised returns.")
