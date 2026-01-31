"""Alpha command line interface."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import xgboost as xgb

from alpha import __version__
from alpha.backtest.engine import run as run_strategy
from alpha.backtest.metrics import summary as summarize_backtest
from alpha.config.loader import get_runtime_config, set_runtime_config
from alpha.contracts import ensure_feature_contract, ensure_predictions_contract
from alpha.dataset.build import _generate_price_features, build_xy
from alpha.dataset.splits import walk_forward
from alpha.features.sentiment import aggregate as aggregate_sentiment
from alpha.ingest import news_gdelt as news_module
from alpha.ingest import prices as prices_module
from alpha.labeling.returns import forward_return
from alpha.labeling.targets import classify
from alpha.model.evaluate import classification_metrics
from alpha.model.infer import predict_bar
from alpha.model.persist import load as load_model
from alpha.model.train import fit_xgb
from alpha.timealign.calendar import ensure_utc


def _config() -> dict[str, Any]:
    return get_runtime_config()


def _data_path(*parts: str | Path) -> Path:
    base = Path(_config()["data_dir"]).expanduser()
    return base.joinpath(*parts)


def _resolve_setting(value: Any | None, key: str) -> str:
    result = value if value not in (None, "") else _config().get(key)
    if result is None:
        raise KeyError(f"Configuration missing required setting '{key}'")
    return str(result)


def _parse_datetime(raw: str, label: str) -> datetime:
    try:
        return datetime.fromisoformat(raw)
    except ValueError as exc:
        raise click.BadParameter(f"Invalid {label} datetime: {raw}") from exc


def _ensure_path_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise click.ClickException(f"{description} not found: {path}")


def _feature_dataset_path(freq: str, horizon: str) -> Path:
    return _data_path("features", f"{freq}_{horizon}", "dataset.parquet")


def _load_feature_dataset(freq: str, horizon: str, *, require_label: bool) -> pd.DataFrame:
    dataset_path = _feature_dataset_path(freq, horizon)
    _ensure_path_exists(dataset_path, "Feature dataset")
    dataset = pd.read_parquet(dataset_path)
    dataset = ensure_feature_contract(
        dataset,
        require_label=require_label,
        require_forward_return=require_label,
    )
    return dataset.sort_values("timestamp_utc").set_index("timestamp_utc")


def _load_prices_indexed(freq: str) -> pd.DataFrame:
    prices_path = _data_path("raw", "prices", f"btcusd_yf_{freq}.parquet")
    _ensure_path_exists(prices_path, "Prices parquet")
    return (
        ensure_utc(pd.read_parquet(prices_path), "timestamp_utc")
        .set_index("timestamp_utc")
        .sort_index()
    )


def _load_news() -> pd.DataFrame:
    news_path = _data_path("raw", "news", "gdelt_1h", "gdelt_crypto_1h.parquet")
    _ensure_path_exists(news_path, "News parquet")
    return ensure_utc(pd.read_parquet(news_path), "published_utc").sort_values("published_utc")


def _predictions_path(freq: str, horizon: str) -> Path:
    return _data_path("predictions", f"{freq}_{horizon}.parquet")


def _model_path(freq: str, horizon: str) -> Path:
    return Path("models") / f"{freq}_{horizon}" / "xgb.json"


@click.group(name="alpha", invoke_without_command=True)
@click.option(
    "--config",
    type=click.Path(exists=False, dir_okay=False, file_okay=True, path_type=Path),
    default=None,
    help="Path to configuration file (overrides ALPHA_CONFIG).",
)
@click.option("--version", is_flag=True, help="Show version and exit.")
@click.pass_context
def _cli(ctx: click.Context, config: Path | None, version: bool) -> None:
    set_runtime_config(config)
    if version:
        click.echo(__version__)
        ctx.exit()
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


@_cli.group(help="Price data operations.")
def prices() -> None:
    """Price command namespace."""


@_cli.group(help="News ingestion.")
def news() -> None:
    """News command namespace."""


@_cli.group(help="Feature generation.")
def features() -> None:
    """Features command namespace."""


@_cli.group(help="Model lifecycle commands.")
def model() -> None:
    """Model command namespace."""


@_cli.group(help="Backtest utilities.")
def backtest() -> None:
    """Backtest command namespace."""


@_cli.group(help="Batch jobs.")
def batch() -> None:
    """Batch command namespace."""


@_cli.group(help="Monitoring commands.")
def monitor() -> None:
    """Monitor command namespace."""


@prices.command("pull")
@click.option("--symbol", required=True, help="Ticker symbol to download.")
@click.option("--start", required=True, help="Start date (YYYY-MM-DD).")
@click.option("--interval", default=None, help="Data interval (defaults to config freq).")
@click.option(
    "--output",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Output directory for prices parquet.",
)
def prices_pull(
    symbol: str,
    start: str,
    interval: str | None,
    output: Path | None,
) -> None:
    """Pull price data from Yahoo Finance."""
    freq = interval or _resolve_setting(None, "freq")
    start_dt = _parse_datetime(start, "--start")
    out_root = output.expanduser() if output else _data_path("raw", "prices")

    frame = prices_module.fetch_yf(symbol, freq, start_dt, output_root=out_root)
    target_path = prices_module._target_path(symbol, freq, out_root)
    click.echo(f"Pulled {len(frame)} rows for {symbol} {freq} into {target_path}")


@news.command("pull")
@click.option("--from", "from_dt", required=True, help="Start datetime (ISO8601).")
@click.option("--to", "to_dt", required=True, help="End datetime (ISO8601).")
@click.option(
    "--output",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Output directory for news parquet.",
)
def news_pull(from_dt: str, to_dt: str, output: Path | None) -> None:
    """Fetch news from GDELT."""
    start = _parse_datetime(from_dt, "--from")
    end = _parse_datetime(to_dt, "--to")
    out_root = output.expanduser() if output else _data_path("raw", "news", "gdelt_1h")

    throttle_seconds = float(_config().get("news_throttle_seconds", 0.0))
    retry_limit = int(_config().get("news_retry_limit", 3))
    frame = news_module.fetch(
        start,
        end,
        output_root=out_root,
        throttle_seconds=throttle_seconds,
        retry_limit=retry_limit,
    )
    target_path = news_module._target_path(out_root)
    click.echo(f"Pulled {len(frame)} news rows into {target_path}")


@features.command("build")
@click.option("--start", default=None, help="Start datetime for feature build.")
@click.option("--end", default=None, help="End datetime for feature build.")
@click.option(
    "--tau",
    type=float,
    default=None,
    help="Label threshold override (defaults to config).",
)
def features_build(
    start: str | None,
    end: str | None,
    tau: float | None,
) -> None:
    """Build feature matrix and labels."""
    freq = _resolve_setting(None, "freq")
    horizon = _resolve_setting(None, "horizon")
    tau_val = tau if tau is not None else float(_config()["tau"])

    prices_path = _data_path("raw", "prices", f"btcusd_yf_{freq}.parquet")
    news_path = _data_path("raw", "news", "gdelt_1h", "gdelt_crypto_1h.parquet")
    _ensure_path_exists(prices_path, "Prices parquet")
    _ensure_path_exists(news_path, "News parquet")

    if start is None or end is None:
        sample = pd.read_parquet(prices_path, columns=["timestamp_utc"])
        timestamps = pd.to_datetime(sample["timestamp_utc"])
        default_start = timestamps.min().isoformat()
        default_end = timestamps.max().isoformat()
    else:
        default_start = start
        default_end = end

    X, y, _meta = build_xy(
        prices_path,
        news_path,
        freq=freq,
        horizon=horizon,
        tau=tau_val,
        start=default_start,
        end=default_end,
        seed=int(_config().get("seed", 0)),
        version=__version__,
    )
    click.echo(f"Built feature matrix with {len(X)} rows.")


@model.command("cv")
@click.option("--start", required=True, help="Training window start (ISO8601).")
@click.option("--end", required=True, help="Training window end (ISO8601).")
@click.option("--freq", default=None, help="Bar frequency.")
@click.option("--horizon", default=None, help="Prediction horizon.")
@click.option("--tau", type=float, default=None, help="Label threshold (defaults to config).")
@click.option(
    "--windows",
    type=str,
    nargs=4,
    multiple=True,
    help="Custom walk-forward windows (train_start train_end test_start test_end).",
)
def model_cv(
    start: str,
    end: str,
    freq: str | None,
    horizon: str | None,
    tau: float | None,
    windows: Iterable[tuple[str, str, str, str]],
) -> None:
    """Run walk-forward cross validation."""
    freq_val = _resolve_setting(freq, "freq")
    horizon_val = _resolve_setting(horizon, "horizon")
    tau_val = tau if tau is not None else float(_config()["tau"])

    window_spec: list[tuple[str, str, str, str]] = []
    for a, b, c, d in windows:
        window_spec.append((a, b, c, d))
    if not window_spec:
        window_spec.append((start, end, start, end))

    dataset = _load_feature_dataset(freq_val, horizon_val, require_label=True)
    feature_cols = [col for col in dataset.columns if col.startswith("feat_")]
    X = dataset[feature_cols]
    y = dataset["label"]

    folds = walk_forward(X.index, windows=window_spec, embargo_bars=1)

    summary: list[dict[str, Any]] = []
    for idx, fold in enumerate(folds):
        if fold.train.empty or fold.test.empty:
            continue

        X_train = X.loc[fold.train].copy()
        X_test = X.loc[fold.test]
        y_train = y.loc[fold.train]
        y_test = y.loc[fold.test]

        X_train.attrs["freq"] = freq_val
        X_train.attrs["horizon"] = horizon_val

        booster, _ = fit_xgb(X_train, y_train, seed=int(_config()["seed"]))
        dtest = xgb.DMatrix(X_test, feature_names=list(X_test.columns))
        proba = booster.predict(dtest)
        metrics = classification_metrics(
            y_test,
            proba,
            threshold=tau_val,
            class_labels=list(y.unique()),
        )
        summary.append(metrics)
        click.echo(
            "Fold "
            f"{idx + 1}: balanced_accuracy={metrics['balanced_accuracy']:.3f}, "
            f"mcc={metrics['mcc']:.3f}"
        )

    if summary:
        avg_balanced = float(np.mean([metric["balanced_accuracy"] for metric in summary]))
        avg_mcc = float(np.mean([metric["mcc"] for metric in summary]))
        aggregate = {
            "folds": summary,
            "average_balanced_accuracy": avg_balanced,
            "average_mcc": avg_mcc,
        }
        metrics_dir = Path("metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "cv_summary.json").write_text(
            json.dumps(aggregate, indent=2),
            encoding="utf-8",
        )
        click.echo(f"Aggregate: balanced_accuracy={avg_balanced:.3f}, mcc={avg_mcc:.3f}")


@model.command("train")
@click.option("--freq", default=None, help="Bar frequency.")
@click.option("--horizon", default=None, help="Prediction horizon.")
@click.option(
    "--class-weights/--no-class-weights",
    default=True,
    show_default=True,
    help="Enable class weighting during training.",
)
def model_train(
    freq: str | None,
    horizon: str | None,
    class_weights: bool,
) -> None:
    """Train the XGBoost model."""
    freq_val = _resolve_setting(freq, "freq")
    horizon_val = _resolve_setting(horizon, "horizon")

    dataset = _load_feature_dataset(freq_val, horizon_val, require_label=True)
    feature_cols = [col for col in dataset.columns if col.startswith("feat_")]
    X = dataset[feature_cols]
    y = dataset["label"]
    X.attrs["freq"] = freq_val
    X.attrs["horizon"] = horizon_val

    booster, _ = fit_xgb(X, y, class_weights=class_weights, seed=int(_config()["seed"]))
    model_path = _model_path(freq_val, horizon_val)
    click.echo(f"Trained model saved to {model_path}")


@model.command("infer")
@click.option(
    "--features",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Parquet file of feature rows.",
)
@click.option(
    "--output",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    default=None,
    help="Where to store predictions parquet.",
)
@click.option(
    "--model",
    "model_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to trained model (defaults to models/freq_horizon/xgb.json).",
)
@click.option("--freq", default=None, help="Bar frequency for defaults.")
@click.option("--horizon", default=None, help="Prediction horizon for defaults.")
def model_infer(
    features: Path,
    output: Path | None,
    model_path: Path | None,
    freq: str | None,
    horizon: str | None,
) -> None:
    """Run inference on a feature parquet."""
    freq_val = _resolve_setting(freq, "freq")
    horizon_val = _resolve_setting(horizon, "horizon")
    resolved_model_path = model_path if model_path else _model_path(freq_val, horizon_val)
    _ensure_path_exists(resolved_model_path, "Model artifact")
    _ensure_path_exists(features, "Feature parquet")

    feature_frame = pd.read_parquet(features)
    if feature_frame.empty:
        raise click.ClickException("No feature rows found.")

    feature_frame = ensure_feature_contract(
        feature_frame,
        require_label=False,
        require_forward_return=False,
    ).sort_values("timestamp_utc").set_index("timestamp_utc")
    feature_cols = [col for col in feature_frame.columns if col.startswith("feat_")]

    booster = load_model(resolved_model_path)
    records: list[dict[str, Any]] = []
    for ts, row in feature_frame[feature_cols].iterrows():
        result = predict_bar(booster, row, timestamp=ts)
        records.append(
            {
                "timestamp_utc": result.get("timestamp", ts),
                "p_up": result.get("p_up"),
                "p_down": result.get("p_down"),
                "freq": freq_val,
                "horizon": horizon_val,
                "model_version": __version__,
                "realized_r": float("nan"),
                "realized_label": pd.NA,
            }
        )

    predictions = ensure_predictions_contract(pd.DataFrame(records))
    if output is None:
        result = predictions.copy()
        result["timestamp_utc"] = result["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        click.echo(result.to_json(orient="records"))
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(output, index=False)
        click.echo(f"Wrote {len(predictions)} predictions to {output}")


@backtest.command("run")
@click.option("--freq", default=None, help="Bar frequency.")
@click.option("--horizon", default=None, help="Prediction horizon.")
@click.option(
    "--enter-threshold",
    "enter_threshold",
    type=float,
    default=None,
    help="Entry probability threshold.",
)
@click.option(
    "--allow-short",
    "allow_short_flag",
    is_flag=True,
    default=False,
    help="Enable short positions.",
)
@click.option(
    "--no-allow-short",
    "no_allow_short_flag",
    is_flag=True,
    default=False,
    help="Disable short positions.",
)
@click.option(
    "--cost-bps",
    "--cost_bps",
    type=float,
    default=None,
    help="Round-trip cost in basis points.",
)
def backtest_run(
    freq: str | None,
    horizon: str | None,
    enter_threshold: float | None,
    allow_short_flag: bool,
    no_allow_short_flag: bool,
    cost_bps: float | None,
) -> None:
    """Run backtest using stored predictions."""
    freq_val = _resolve_setting(freq, "freq")
    horizon_val = _resolve_setting(horizon, "horizon")
    enter = enter_threshold if enter_threshold is not None else float(_config()["enter_threshold"])
    cost = cost_bps if cost_bps is not None else float(_config()["cost_bps"])

    if allow_short_flag and no_allow_short_flag:
        raise click.BadParameter("Specify only one of --allow-short or --no-allow-short.")

    predictions_path = _predictions_path(freq_val, horizon_val)
    _ensure_path_exists(predictions_path, "Predictions parquet")
    preds = ensure_predictions_contract(pd.read_parquet(predictions_path))
    if preds.empty:
        raise click.ClickException("No predictions available for backtest.")

    preds["timestamp_utc"] = pd.to_datetime(
        preds["timestamp_utc"], utc=True, errors="coerce"
    ).dt.tz_localize(None)
    preds = preds.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    prices = _load_prices_indexed(freq_val)
    close = prices["close"].reindex(preds["timestamp_utc"]).ffill()

    proba_up = preds.set_index("timestamp_utc")["p_up"]
    proba_down = preds.set_index("timestamp_utc")["p_down"]

    if allow_short_flag:
        allow_short_val = True
    elif no_allow_short_flag:
        allow_short_val = False
    else:
        allow_short_val = bool(_config().get("allow_short", False))

    trades, equity = run_strategy(
        close,
        proba_up,
        proba_down,
        enter=enter,
        allow_short=allow_short_val,
        cost_bps=cost,
    )
    metrics = summarize_backtest(equity, trades)
    click.echo(
        f"Backtest complete: sharpe={metrics['sharpe']:.3f}, "
        f"max_drawdown={metrics['max_drawdown']:.3f}"
    )


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
    news = _load_news()

    price_features = _generate_price_features(prices)
    bar_df = prices.reset_index()[["timestamp_utc"]]
    sentiment_features = aggregate_sentiment(news_df=news, bar_df=bar_df, freq=freq_val)
    features = price_features.join(sentiment_features, how="left").dropna()
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
    prediction = predict_bar(booster, feature_row, timestamp=latest_ts)

    timestamp_value = prediction.get("timestamp", latest_ts)
    timestamp_utc = pd.to_datetime(timestamp_value, utc=True, errors="coerce")
    if pd.isna(timestamp_utc):
        raise click.ClickException("Inference produced an invalid timestamp.")
    if hasattr(timestamp_utc, "tz_convert"):
        timestamp_utc = timestamp_utc.tz_convert(None)
    timestamp_py = timestamp_utc.to_pydatetime()

    record = {
        "timestamp_utc": timestamp_py,
        "p_up": float(prediction["p_up"]),
        "p_down": float(prediction["p_down"]),
        "freq": freq_val,
        "horizon": horizon_val,
        "model_version": model_version or __version__,
        "realized_r": np.nan,
        "realized_label": pd.NA,
    }
    new_df = ensure_predictions_contract(pd.DataFrame([record]))

    predictions_path = _predictions_path(freq_val, horizon_val)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    if predictions_path.exists():
        existing = ensure_predictions_contract(pd.read_parquet(predictions_path))
        combined = (
            pd.concat([existing, new_df], axis=0, ignore_index=True)
            .sort_values("timestamp_utc")
            .drop_duplicates(
                subset=["timestamp_utc", "horizon", "model_version"], keep="last"
            )
        )
    else:
        combined = new_df
    combined = ensure_predictions_contract(combined)
    combined.to_parquet(predictions_path, index=False)
    latest_record = combined.iloc[-1]
    click.echo(f"Stored prediction for {latest_record['timestamp_utc']} at {predictions_path}")


@batch.command("realize")
@click.option("--freq", default=None, help="Bar frequency.")
@click.option("--horizon", default=None, help="Prediction horizon.")
@click.option("--tau", type=float, default=None, help="Label threshold (defaults to config).")
def batch_realize(
    freq: str | None,
    horizon: str | None,
    tau: float | None,
) -> None:
    """Attach realized returns and labels to stored predictions."""
    freq_val = _resolve_setting(freq, "freq")
    horizon_val = _resolve_setting(horizon, "horizon")
    tau_val = tau if tau is not None else float(_config()["tau"])

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
    labels = classify(preds.loc[updated_mask, "realized_r"], tau_val)
    preds.loc[labels.index, "realized_label"] = labels.astype("string")

    preds = ensure_predictions_contract(preds)
    preds.to_parquet(predictions_path, index=False)
    click.echo(f"Updated {updated_mask.sum()} predictions with realised returns.")


@monitor.command("refresh")
@click.option("--freq", default=None, help="Bar frequency.")
@click.option("--horizon", default=None, help="Prediction horizon.")
@click.option(
    "--cost-bps",
    "--cost_bps",
    type=float,
    default=None,
    help="Trading costs assumed for live metrics.",
)
def monitor_refresh(
    freq: str | None,
    horizon: str | None,
    cost_bps: float | None,
) -> None:
    """Refresh monitoring metrics from prediction store."""
    freq_val = _resolve_setting(freq, "freq")
    horizon_val = _resolve_setting(horizon, "horizon")
    cost = cost_bps if cost_bps is not None else float(_config()["cost_bps"])

    predictions_path = _predictions_path(freq_val, horizon_val)
    _ensure_path_exists(predictions_path, "Predictions parquet")
    preds = ensure_predictions_contract(pd.read_parquet(predictions_path))
    if preds.empty:
        raise click.ClickException("No live predictions to monitor.")

    preds["timestamp_utc"] = pd.to_datetime(
        preds["timestamp_utc"], utc=True, errors="coerce"
    )
    preds = preds.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")

    realised = preds["realized_r"].astype(float)
    realized_mask = realised.notna()
    hit_rate = 0.0
    if realized_mask.any():
        hit_rate = float((realised[realized_mask] > cost / 10000).mean())
    live_metrics = {
        "count": int(len(preds)),
        "avg_p_up": float(preds["p_up"].mean()),
        "avg_p_down": float(preds["p_down"].mean()),
        "realized_count": int(realized_mask.sum()),
        "hit_rate": hit_rate,
        "updated_at": datetime.now(UTC).isoformat(),
    }

    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_path = metrics_dir / f"live_{freq_val}_{horizon_val}.json"
    output_path.write_text(json.dumps(live_metrics, indent=2), encoding="utf-8")
    click.echo(f"Wrote monitoring snapshot to {output_path}")


def main() -> None:
    """Entry point used by tests and scripts."""
    _cli.main(standalone_mode=False)


if __name__ == "__main__":
    _cli()
