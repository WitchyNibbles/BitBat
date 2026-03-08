"""BitBat command line interface."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn

import click
import numpy as np
import pandas as pd
import xgboost as xgb

from bitbat import __version__
from bitbat.autonomous.schema_compat import SchemaCompatibilityError, format_missing_columns
from bitbat.backtest.engine import run as run_strategy
from bitbat.backtest.metrics import summary as summarize_backtest
from bitbat.config.loader import (
    get_runtime_config,
    get_runtime_config_path,
    get_runtime_config_source,
    set_runtime_config,
)
from bitbat.contracts import ensure_feature_contract, ensure_predictions_contract
from bitbat.dataset.build import build_xy, generate_price_features
from bitbat.dataset.splits import generate_rolling_windows, walk_forward
from bitbat.features.sentiment import aggregate as aggregate_sentiment
from bitbat.ingest import prices as prices_module
from bitbat.labeling.returns import forward_return
from bitbat.model.evaluate import (
    build_candidate_report,
    compute_multiple_testing_safeguards,
    regression_metrics,
    select_champion_report,
)
from bitbat.model.infer import predict_bar
from bitbat.model.optimize import HyperparamOptimizer
from bitbat.model.persist import (
    default_model_artifact_path,
    save_baseline_artifact,
)
from bitbat.model.persist import (
    load as load_model,
)
from bitbat.model.train import fit_random_forest, fit_xgb
from bitbat.timealign.calendar import ensure_utc

if TYPE_CHECKING:
    from bitbat.autonomous.db import MonitorDatabaseError


def _config() -> dict[str, Any]:
    return get_runtime_config()


def _sentiment_enabled() -> bool:
    return bool(_config().get("enable_sentiment", True))


def _resolve_news_source(source: str | None = None) -> str:
    configured = (
        source if source not in (None, "") else _config().get("news_source", "cryptocompare")
    )
    resolved = str(configured).strip().lower()
    if resolved not in {"gdelt", "cryptocompare"}:
        raise click.ClickException(
            f"Unsupported news_source '{resolved}'. Expected one of: gdelt, cryptocompare."
        )
    return resolved


def _news_backend(source: str) -> Any:
    if source == "gdelt":
        from bitbat.ingest import news_gdelt as backend
    else:
        from bitbat.ingest import news_cryptocompare as backend
    return backend


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


def _load_feature_dataset(
    freq: str,
    horizon: str,
    *,
    require_label: bool,
    require_forward_return: bool | None = None,
) -> pd.DataFrame:
    dataset_path = _feature_dataset_path(freq, horizon)
    _ensure_path_exists(dataset_path, "Feature dataset")
    dataset = pd.read_parquet(dataset_path)
    _require_fwd = require_forward_return if require_forward_return is not None else require_label
    dataset = ensure_feature_contract(
        dataset,
        require_label=require_label,
        require_forward_return=_require_fwd,
        require_features_full=_sentiment_enabled(),
    )
    return dataset.sort_values("timestamp_utc").set_index("timestamp_utc")


def _load_prices_indexed(freq: str) -> pd.DataFrame:
    from bitbat.io.prices import load_prices_for_cli

    data_dir = Path(_config()["data_dir"]).expanduser()
    try:
        return load_prices_for_cli(freq, data_dir=data_dir)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc


def _load_news() -> pd.DataFrame:
    source = _resolve_news_source()
    backend = _news_backend(source)
    news_root = _data_path("raw", "news", f"{source}_1h")
    news_path = backend._target_path(news_root)
    _ensure_path_exists(news_path, "News parquet")
    return ensure_utc(pd.read_parquet(news_path), "published_utc").sort_values("published_utc")


def _predictions_path(freq: str, horizon: str) -> Path:
    return _data_path("predictions", f"{freq}_{horizon}.parquet")


def _model_path(freq: str, horizon: str) -> Path:
    return default_model_artifact_path(freq, horizon, family="xgb")


def _resolve_model_families(selection: str | None) -> list[Literal["xgb", "random_forest"]]:
    configured = _config().get("model", {})
    default_family = str(configured.get("baseline_family", "xgb")).strip().lower()
    requested = str(selection or default_family).strip().lower()

    if requested == "both":
        return ["xgb", "random_forest"]
    if requested not in {"xgb", "random_forest"}:
        raise click.ClickException(
            f"Unsupported model family '{requested}'. Expected xgb, random_forest, or both."
        )
    return [requested]


def _predict_baseline(
    family: str,
    model: Any,
    features: pd.DataFrame,
) -> np.ndarray:
    if family == "xgb":
        dtest = xgb.DMatrix(features, feature_names=list(features.columns))
        return np.asarray(model.predict(dtest), dtype="float64")
    return np.asarray(model.predict(features.astype(float)), dtype="float64")


def _raise_monitor_schema_error(exc: SchemaCompatibilityError, db_url: str) -> NoReturn:
    missing = format_missing_columns(exc.report) or "unknown"
    raise click.ClickException(
        "\n".join([
            "Autonomous DB schema is incompatible for monitor commands.",
            f"Missing columns: {missing}",
            (
                "Run: poetry run python scripts/init_autonomous_db.py "
                f'--database-url "{db_url}" --audit'
            ),
            (
                "Then: poetry run python scripts/init_autonomous_db.py "
                f'--database-url "{db_url}" --upgrade'
            ),
        ])
    ) from exc


def _raise_monitor_runtime_db_error(exc: MonitorDatabaseError) -> NoReturn:
    raise click.ClickException(
        "\n".join([
            "Autonomous monitor runtime database failure.",
            f"Step: {exc.step}",
            f"Error class: {exc.error_class}",
            f"Detail: {exc.detail}",
            f"Remediation: {exc.remediation}",
        ])
    ) from exc


def _monitor_config_source_label(source: str) -> str:
    labels = {
        "explicit": "--config",
        "env": "BITBAT_CONFIG",
        "default": "default-config",
    }
    return labels.get(source, source)


def _emit_monitor_startup_context(freq: str, horizon: str) -> None:
    source = get_runtime_config_source()
    config_path = get_runtime_config_path()
    click.echo(
        "Monitor startup config: "
        f"source={_monitor_config_source_label(source)}, path={config_path}"
    )
    click.echo(f"Resolved runtime pair: freq={freq}, horizon={horizon}")


def _raise_monitor_model_preflight_error(exc: FileNotFoundError) -> NoReturn:
    raise click.ClickException(
        "\n".join([
            "Autonomous monitor startup blocked: model artifact missing.",
            f"Detail: {exc}",
            "Remediation:",
            "  1. Use --config or BITBAT_CONFIG to select the intended freq/horizon pair.",
            "  2. Train/copy the expected model artifact: models/<freq>_<horizon>/xgb.json.",
        ])
    ) from exc


@click.group(name="bitbat", invoke_without_command=True)
@click.option(
    "--config",
    type=click.Path(exists=False, dir_okay=False, file_okay=True, path_type=Path),
    default=None,
    help="Path to configuration file (overrides BITBAT_CONFIG).",
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


@_cli.group(help="Prediction validation commands.")
def validate() -> None:
    """Validation command namespace."""


@_cli.group(help="Data ingestion commands.")
def ingest() -> None:
    """Ingestion command namespace."""


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
    "--source",
    type=click.Choice(["cryptocompare", "gdelt"], case_sensitive=False),
    default=None,
    help="News source override (defaults to config news_source).",
)
@click.option(
    "--output",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Output directory for news parquet.",
)
def news_pull(from_dt: str, to_dt: str, source: str | None, output: Path | None) -> None:
    """Fetch historical news for feature training."""
    start = _parse_datetime(from_dt, "--from")
    end = _parse_datetime(to_dt, "--to")
    source_name = _resolve_news_source(source)
    backend = _news_backend(source_name)
    out_root = output.expanduser() if output else _data_path("raw", "news", f"{source_name}_1h")

    throttle_seconds = float(_config().get("news_throttle_seconds", 10.0))
    retry_limit = int(_config().get("news_retry_limit", 30))
    frame = backend.fetch(
        start,
        end,
        output_root=out_root,
        throttle_seconds=throttle_seconds,
        retry_limit=retry_limit,
    )
    target_path = backend._target_path(out_root)
    click.echo(f"Pulled {len(frame)} {source_name} news rows into {target_path}")


@features.command("build")
@click.option("--start", default=None, help="Start datetime for feature build.")
@click.option("--end", default=None, help="End datetime for feature build.")
@click.option(
    "--label-mode",
    type=click.Choice(["return_direction", "triple_barrier"], case_sensitive=False),
    default=None,
    help="Target labeling mode for dataset generation.",
)
@click.option(
    "--take-profit",
    type=float,
    default=None,
    help="Take-profit threshold for `--label-mode triple_barrier`.",
)
@click.option(
    "--stop-loss",
    type=float,
    default=None,
    help="Stop-loss threshold for `--label-mode triple_barrier`.",
)
def features_build(
    start: str | None,
    end: str | None,
    label_mode: str | None,
    take_profit: float | None,
    stop_loss: float | None,
) -> None:
    """Build feature matrix and labels."""
    freq = _resolve_setting(None, "freq")
    horizon = _resolve_setting(None, "horizon")
    enable_sentiment = _sentiment_enabled()
    configured_mode = str(_config().get("label_mode", "return_direction"))
    resolved_label_mode = (label_mode or configured_mode).strip().lower()
    if resolved_label_mode not in {"return_direction", "triple_barrier"}:
        raise click.ClickException(
            f"Unsupported label mode '{resolved_label_mode}'. "
            "Expected 'return_direction' or 'triple_barrier'."
        )
    if resolved_label_mode != "triple_barrier" and (
        take_profit is not None or stop_loss is not None
    ):
        raise click.ClickException(
            "--take-profit/--stop-loss can only be used with --label-mode triple_barrier."
        )

    prices_path = _data_path("raw", "prices", f"btcusd_yf_{freq}.parquet")
    _ensure_path_exists(prices_path, "Prices parquet")
    news_path = None
    if enable_sentiment:
        source = _resolve_news_source()
        backend = _news_backend(source)
        news_path = backend._target_path(_data_path("raw", "news", f"{source}_1h"))
        _ensure_path_exists(news_path, "News parquet")

    if start is None or end is None:
        sample = pd.read_parquet(prices_path, columns=["timestamp_utc"])
        timestamps = pd.to_datetime(sample["timestamp_utc"])
        default_start = timestamps.min().isoformat()
        default_end = timestamps.max().isoformat()
    else:
        default_start = start
        default_end = end

    enable_garch = bool(_config().get("enable_garch", False))
    enable_macro = bool(_config().get("enable_macro", False))
    enable_onchain = bool(_config().get("enable_onchain", False))

    macro_path = None
    if enable_macro:
        macro_candidate = _data_path("raw", "macro", "fred.parquet")
        if macro_candidate.exists():
            macro_path = macro_candidate

    onchain_path = None
    if enable_onchain:
        onchain_candidate = _data_path("raw", "onchain", "blockchain_info.parquet")
        if onchain_candidate.exists():
            onchain_path = onchain_candidate

    tau_config = _config().get("tau")
    tau_value = float(tau_config) if tau_config is not None else None
    barrier_tp: float | None = None
    barrier_sl: float | None = None
    if resolved_label_mode == "triple_barrier":
        default_tp = float(_config().get("triple_barrier_take_profit", tau_value or 0.01))
        barrier_tp = float(take_profit) if take_profit is not None else default_tp
        default_sl = float(_config().get("triple_barrier_stop_loss", barrier_tp))
        barrier_sl = float(stop_loss) if stop_loss is not None else default_sl

    X, y, _meta = build_xy(
        prices_path,
        news_path,
        freq=freq,
        horizon=horizon,
        start=default_start,
        end=default_end,
        tau=tau_value,
        enable_sentiment=enable_sentiment,
        enable_garch=enable_garch,
        macro_parquet=macro_path,
        onchain_parquet=onchain_path,
        label_mode=resolved_label_mode,
        barrier_take_profit=barrier_tp,
        barrier_stop_loss=barrier_sl,
        output_root=Path(_config()["data_dir"]).expanduser(),
        seed=int(_config().get("seed", 0)),
        version=__version__,
    )
    click.echo(f"Built feature matrix with {len(X)} rows.")


@model.command("cv")
@click.option("--start", required=True, help="Training window start (ISO8601).")
@click.option("--end", required=True, help="Training window end (ISO8601).")
@click.option("--freq", default=None, help="Bar frequency.")
@click.option("--horizon", default=None, help="Prediction horizon.")
@click.option(
    "--family",
    type=click.Choice(["xgb", "random_forest", "both"], case_sensitive=False),
    default=None,
    help="Baseline model family selection (default: config model.baseline_family or xgb).",
)
@click.option(
    "--train-window",
    default=None,
    help="Rolling train window duration (e.g. 365D). Uses config model.cv.train_window by default.",
)
@click.option(
    "--backtest-window",
    default=None,
    help=(
        "Rolling backtest window duration (e.g. 90D). "
        "Uses config model.cv.backtest_window by default."
    ),
)
@click.option(
    "--window-step",
    default=None,
    help="Window step duration (e.g. 30D). Defaults to backtest window when unset.",
)
@click.option(
    "--windows",
    type=str,
    nargs=4,
    multiple=True,
    help="Custom walk-forward windows (train_start train_end test_start test_end).",
)
@click.option(
    "--embargo-bars",
    type=int,
    default=None,
    help="Bars embargoed before each test window (default: model.cv.embargo_bars or 1).",
)
@click.option(
    "--purge-bars",
    type=int,
    default=None,
    help=(
        "Bars purged before each test window to avoid horizon overlap "
        "(default: model.cv.purge_bars or 0)."
    ),
)
@click.option(
    "--label-horizon",
    type=str,
    default=None,
    help="Optional label horizon (for example 4h) used to infer purge bars when purge is unset.",
)
def model_cv(  # noqa: C901
    start: str,
    end: str,
    freq: str | None,
    horizon: str | None,
    family: str | None,
    train_window: str | None,
    backtest_window: str | None,
    window_step: str | None,
    windows: Iterable[tuple[str, str, str, str]],
    embargo_bars: int | None,
    purge_bars: int | None,
    label_horizon: str | None,
) -> None:
    """Run walk-forward cross validation."""
    freq_val = _resolve_setting(freq, "freq")
    horizon_val = _resolve_setting(horizon, "horizon")
    selected_families = _resolve_model_families(family)

    dataset = _load_feature_dataset(
        freq_val,
        horizon_val,
        require_label=False,
        require_forward_return=True,
    )
    feature_cols = [col for col in dataset.columns if col.startswith("feat_")]
    X = dataset[feature_cols]
    y = dataset["r_forward"]

    model_cfg = _config().get("model", {})
    cv_cfg = model_cfg.get("cv", {}) if isinstance(model_cfg, dict) else {}
    cfg_train_window = cv_cfg.get("train_window")
    cfg_backtest_window = cv_cfg.get("backtest_window")
    cfg_window_step = cv_cfg.get("window_step")
    cfg_embargo_bars = cv_cfg.get("embargo_bars")
    cfg_purge_bars = cv_cfg.get("purge_bars")
    cfg_label_horizon = cv_cfg.get("label_horizon")

    resolved_train_window = train_window or cfg_train_window
    resolved_backtest_window = backtest_window or cfg_backtest_window
    resolved_window_step = window_step or cfg_window_step
    resolved_embargo_bars = (
        embargo_bars
        if embargo_bars is not None
        else int(cfg_embargo_bars)
        if cfg_embargo_bars not in (None, "")
        else 1
    )
    resolved_purge_bars = (
        purge_bars
        if purge_bars is not None
        else int(cfg_purge_bars)
        if cfg_purge_bars not in (None, "")
        else 0
    )
    resolved_label_horizon = label_horizon if label_horizon not in (None, "") else cfg_label_horizon

    window_spec: list[tuple[str, str, str, str]] = list(windows)
    if not window_spec:
        if resolved_train_window and resolved_backtest_window:
            window_spec = generate_rolling_windows(
                X.index,
                train_window=str(resolved_train_window),
                backtest_window=str(resolved_backtest_window),
                step=(
                    str(resolved_window_step) if resolved_window_step not in ("", None) else None
                ),
                start=start,
                end=end,
            )
            if not window_spec:
                raise click.ClickException(
                    "No rolling windows generated. Adjust --start/--end or window durations."
                )
        else:
            window_spec = [(start, end, start, end)]

    folds = walk_forward(
        X.index,
        windows=window_spec,
        embargo_bars=int(resolved_embargo_bars),
        purge_bars=int(resolved_purge_bars),
        label_horizon=(
            str(resolved_label_horizon) if resolved_label_horizon not in (None, "") else None
        ),
    )

    summary_by_family: dict[str, list[dict[str, Any]]] = {
        model_family: [] for model_family in selected_families
    }
    cv_cost_bps = float(_config().get("cost_bps", 4.0))
    cv_fee_bps = float(_config().get("fee_bps", cv_cost_bps))
    cv_slippage_bps = float(_config().get("slippage_bps", 0.0))
    cv_allow_short = bool(_config().get("allow_short", False))
    for idx, fold in enumerate(folds):
        if fold.train.empty or fold.test.empty:
            continue

        X_train = X.loc[fold.train].copy()
        X_test = X.loc[fold.test]
        y_train = y.loc[fold.train]
        y_test = y.loc[fold.test]

        X_train.attrs["freq"] = freq_val
        X_train.attrs["horizon"] = horizon_val

        for model_family in selected_families:
            if model_family == "xgb":
                model, _ = fit_xgb(X_train, y_train, seed=int(_config()["seed"]), persist=False)
            else:
                model, _ = fit_random_forest(
                    X_train,
                    y_train,
                    seed=int(_config()["seed"]),
                    persist=False,
                )

            predicted = _predict_baseline(model_family, model, X_test)
            regression = regression_metrics(y_test, predicted)
            predicted_series = pd.Series(predicted, index=X_test.index, dtype="float64")
            realized_returns = y_test.astype("float64")
            synthetic_close = pd.Series(
                100.0 * np.cumprod(1.0 + realized_returns.to_numpy()),
                index=realized_returns.index,
                dtype="float64",
            )
            trades, equity = run_strategy(
                synthetic_close,
                predicted_series,
                allow_short=cv_allow_short,
                cost_bps=cv_cost_bps,
                fee_bps=cv_fee_bps,
                slippage_bps=cv_slippage_bps,
            )
            risk = summarize_backtest(equity, trades)

            metrics = {
                **regression,
                "net_sharpe": float(risk.get("net_sharpe", risk.get("sharpe", 0.0))),
                "gross_sharpe": float(risk.get("gross_sharpe", 0.0)),
                "max_drawdown": float(risk.get("max_drawdown", 0.0)),
                "net_return": float(risk.get("net_return", 0.0)),
                "gross_return": float(risk.get("gross_return", 0.0)),
                "total_costs": float(risk.get("total_costs", 0.0)),
                "total_fee_costs": float(risk.get("total_fee_costs", 0.0)),
                "total_slippage_costs": float(risk.get("total_slippage_costs", 0.0)),
            }
            summary_by_family[model_family].append(metrics)
            click.echo(
                "Fold "
                f"{idx + 1} [{model_family}]: rmse={metrics['rmse']:.6f}, "
                f"mae={metrics['mae']:.6f}"
            )

    family_metrics: dict[str, dict[str, Any]] = {}
    for model_family, folds_summary in summary_by_family.items():
        if not folds_summary:
            continue
        avg_rmse = float(np.mean([metric["rmse"] for metric in folds_summary]))
        avg_mae = float(np.mean([metric["mae"] for metric in folds_summary]))
        family_metrics[model_family] = {
            "folds": folds_summary,
            "average_rmse": avg_rmse,
            "average_mae": avg_mae,
        }
        click.echo(f"Aggregate [{model_family}]: rmse={avg_rmse:.6f}, mae={avg_mae:.6f}")

    if family_metrics:
        primary_family = selected_families[0]
        primary_metrics = family_metrics.get(
            primary_family,
            {"folds": [], "average_rmse": 0.0, "average_mae": 0.0},
        )
        optimization_cfg = model_cfg.get("optimization", {}) if isinstance(model_cfg, dict) else {}
        promotion_cfg = model_cfg.get("promotion_gate", {}) if isinstance(model_cfg, dict) else {}
        safeguard_trial_count = int(
            optimization_cfg.get("trials", max(len(primary_metrics["folds"]), 1))
        )
        safeguard_min_deflated_sharpe = float(optimization_cfg.get("min_deflated_sharpe", 0.0))
        safeguard_max_overfit_probability = float(
            optimization_cfg.get("max_overfit_probability", 0.50)
        )
        promotion_min_consecutive = int(promotion_cfg.get("min_consecutive_outperformance", 2))
        promotion_drawdown_floor = float(promotion_cfg.get("max_drawdown_floor", -0.35))

        candidate_reports: dict[str, dict[str, Any]] = {}
        for model_family, folds_summary in summary_by_family.items():
            if not folds_summary:
                continue
            safeguards = compute_multiple_testing_safeguards(
                [
                    {"outer_score": float(fold_metric.get("rmse", 0.0))}
                    for fold_metric in folds_summary
                ],
                trial_count=safeguard_trial_count,
                min_deflated_sharpe=safeguard_min_deflated_sharpe,
                max_overfit_probability=safeguard_max_overfit_probability,
            )
            candidate_reports[model_family] = build_candidate_report(
                candidate_id=model_family,
                family=model_family,
                fold_metrics=folds_summary,
                safeguards=safeguards,
            )
        champion_decision = select_champion_report(
            candidate_reports=candidate_reports,
            incumbent_id=primary_family if primary_family in candidate_reports else None,
            max_drawdown_floor=promotion_drawdown_floor,
            min_consecutive_outperformance=promotion_min_consecutive,
        )
        if champion_decision.get("winner"):
            click.echo(
                "Champion "
                f"[{champion_decision['winner']}]: "
                f"promote={champion_decision['promote_candidate']} "
                f"({champion_decision.get('reason', 'rule-applied')})"
            )

        primary_report = candidate_reports.get(primary_family, {})
        primary_directional = (
            primary_report.get("metrics", {}).get("directional", {})
            if isinstance(primary_report, dict)
            else {}
        )
        aggregate = {
            "primary_family": primary_family,
            "selected_families": selected_families,
            "family_metrics": family_metrics,
            "folds": primary_metrics["folds"],
            "average_rmse": primary_metrics["average_rmse"],
            "average_mae": primary_metrics["average_mae"],
            "mean_directional_accuracy": float(
                primary_directional.get("mean_directional_accuracy", 0.0)
            ),
            "candidate_reports": candidate_reports,
            "champion_decision": champion_decision,
        }
        metrics_dir = Path("metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "cv_summary.json").write_text(
            json.dumps(aggregate, indent=2),
            encoding="utf-8",
        )


@model.command("optimize")
@click.option("--start", required=True, help="Optimization window start (ISO8601).")
@click.option("--end", required=True, help="Optimization window end (ISO8601).")
@click.option("--freq", default=None, help="Bar frequency.")
@click.option("--horizon", default=None, help="Prediction horizon.")
@click.option(
    "--trials",
    type=int,
    default=None,
    help="Optuna trial count (default: model.optimization.trials or 20).",
)
@click.option(
    "--timeout",
    type=int,
    default=None,
    help="Optional optimization timeout in seconds (default: model.optimization.timeout_seconds).",
)
@click.option(
    "--train-window",
    default=None,
    help="Rolling train window duration (e.g. 365D).",
)
@click.option(
    "--backtest-window",
    default=None,
    help="Rolling backtest window duration (e.g. 90D).",
)
@click.option(
    "--window-step",
    default=None,
    help="Window step duration (e.g. 30D). Defaults to backtest window when unset.",
)
@click.option(
    "--windows",
    type=str,
    nargs=4,
    multiple=True,
    help="Custom nested windows (train_start train_end test_start test_end).",
)
@click.option(
    "--embargo-bars",
    type=int,
    default=None,
    help="Bars embargoed before each test window.",
)
@click.option(
    "--purge-bars",
    type=int,
    default=None,
    help="Bars purged before each test window to avoid overlap leakage.",
)
@click.option(
    "--label-horizon",
    type=str,
    default=None,
    help="Optional label horizon used to infer purge bars when purge is unset.",
)
def model_optimize(
    start: str,
    end: str,
    freq: str | None,
    horizon: str | None,
    trials: int | None,
    timeout: int | None,
    train_window: str | None,
    backtest_window: str | None,
    window_step: str | None,
    windows: Iterable[tuple[str, str, str, str]],
    embargo_bars: int | None,
    purge_bars: int | None,
    label_horizon: str | None,
) -> None:
    """Run nested walk-forward hyperparameter optimization and persist provenance."""
    freq_val = _resolve_setting(freq, "freq")
    horizon_val = _resolve_setting(horizon, "horizon")
    dataset = _load_feature_dataset(
        freq_val,
        horizon_val,
        require_label=False,
        require_forward_return=True,
    )
    feature_cols = [col for col in dataset.columns if col.startswith("feat_")]
    if not feature_cols:
        raise click.ClickException(
            "No feature columns found in dataset (expected columns prefixed with 'feat_')."
        )
    X = dataset[feature_cols]
    y = dataset["r_forward"]

    model_cfg = _config().get("model", {})
    optimization_cfg = model_cfg.get("optimization", {}) if isinstance(model_cfg, dict) else {}
    cv_cfg = model_cfg.get("cv", {}) if isinstance(model_cfg, dict) else {}

    resolved_trials = int(trials if trials is not None else optimization_cfg.get("trials", 20))
    timeout_raw = timeout if timeout is not None else optimization_cfg.get("timeout_seconds")
    resolved_timeout = int(timeout_raw) if timeout_raw not in (None, "", 0, "0") else None
    resolved_train_window = (
        train_window or optimization_cfg.get("train_window") or cv_cfg.get("train_window")
    )
    resolved_backtest_window = (
        backtest_window or optimization_cfg.get("backtest_window") or cv_cfg.get("backtest_window")
    )
    resolved_window_step = (
        window_step or optimization_cfg.get("window_step") or cv_cfg.get("window_step")
    )
    resolved_embargo_bars = (
        embargo_bars
        if embargo_bars is not None
        else int(optimization_cfg.get("embargo_bars", cv_cfg.get("embargo_bars", 1)))
    )
    resolved_purge_bars = (
        purge_bars
        if purge_bars is not None
        else int(optimization_cfg.get("purge_bars", cv_cfg.get("purge_bars", 0)))
    )
    resolved_label_horizon = (
        label_horizon
        if label_horizon not in (None, "")
        else optimization_cfg.get("label_horizon") or cv_cfg.get("label_horizon")
    )

    window_spec: list[tuple[str, str, str, str]] = list(windows)
    if not window_spec:
        if resolved_train_window and resolved_backtest_window:
            window_spec = generate_rolling_windows(
                X.index,
                train_window=str(resolved_train_window),
                backtest_window=str(resolved_backtest_window),
                step=(
                    str(resolved_window_step) if resolved_window_step not in ("", None) else None
                ),
                start=start,
                end=end,
            )
            if not window_spec:
                raise click.ClickException(
                    "No optimization windows generated. Adjust --start/--end or window durations."
                )
        else:
            window_spec = [(start, end, start, end)]

    folds = walk_forward(
        X.index,
        windows=window_spec,
        embargo_bars=int(resolved_embargo_bars),
        purge_bars=int(resolved_purge_bars),
        label_horizon=(
            str(resolved_label_horizon) if resolved_label_horizon not in (None, "") else None
        ),
    )
    if not folds:
        raise click.ClickException("No optimization folds produced from requested windows.")

    optimizer = HyperparamOptimizer(X, y, folds, seed=int(_config().get("seed", 0)))
    result = optimizer.optimize(n_trials=resolved_trials, timeout=resolved_timeout)
    summary = result.summary()
    payload = {
        **summary,
        "freq": freq_val,
        "horizon": horizon_val,
        "feature_count": len(feature_cols),
        "fold_count": len(folds),
        "config": {
            "n_trials": resolved_trials,
            "timeout_seconds": resolved_timeout,
            "embargo_bars": int(resolved_embargo_bars),
            "purge_bars": int(resolved_purge_bars),
            "label_horizon": (
                str(resolved_label_horizon) if resolved_label_horizon not in (None, "") else None
            ),
        },
    }

    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_path = metrics_dir / "optimization_summary.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    click.echo(
        "Optimization complete: "
        f"best_rmse={float(summary.get('best_score', 0.0)):.6f}, "
        f"trials={resolved_trials}, outer_folds={len(summary.get('outer_folds', []))}. "
        f"Saved {output_path}"
    )


@model.command("train")
@click.option("--freq", default=None, help="Bar frequency.")
@click.option("--horizon", default=None, help="Prediction horizon.")
@click.option(
    "--family",
    type=click.Choice(["xgb", "random_forest", "both"], case_sensitive=False),
    default=None,
    help="Baseline model family selection (default: config model.baseline_family or xgb).",
)
def model_train(
    freq: str | None,
    horizon: str | None,
    family: str | None,
) -> None:
    """Train baseline model families."""
    freq_val = _resolve_setting(freq, "freq")
    horizon_val = _resolve_setting(horizon, "horizon")
    selected_families = _resolve_model_families(family)

    dataset = _load_feature_dataset(
        freq_val,
        horizon_val,
        require_label=False,
        require_forward_return=True,
    )
    feature_cols = [col for col in dataset.columns if col.startswith("feat_")]
    X = dataset[feature_cols]
    y = dataset["r_forward"]
    X.attrs["freq"] = freq_val
    X.attrs["horizon"] = horizon_val

    trained_paths: list[tuple[str, Path]] = []
    for model_family in selected_families:
        if model_family == "xgb":
            model, _ = fit_xgb(X, y, seed=int(_config()["seed"]), persist=False)
        else:
            model, _ = fit_random_forest(X, y, seed=int(_config()["seed"]), persist=False)
        model_path = save_baseline_artifact(
            model,
            family=model_family,
            freq=freq_val,
            horizon=horizon_val,
            metadata={"source": "cli:model-train"},
        )
        trained_paths.append((model_family, model_path))

    if len(trained_paths) == 1 and trained_paths[0][0] == "xgb":
        click.echo(f"Trained model saved to {trained_paths[0][1]}")
    else:
        for model_family, model_path in trained_paths:
            click.echo(f"Trained {model_family} model saved to {model_path}")


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

    feature_frame = (
        ensure_feature_contract(
            feature_frame,
            require_label=False,
            require_forward_return=False,
            require_features_full=_sentiment_enabled(),
        )
        .sort_values("timestamp_utc")
        .set_index("timestamp_utc")
    )
    feature_cols = [col for col in feature_frame.columns if col.startswith("feat_")]

    booster = load_model(resolved_model_path)
    records: list[dict[str, Any]] = []
    for ts, row in feature_frame[feature_cols].iterrows():
        result = predict_bar(booster, row, timestamp=ts)
        records.append({
            "timestamp_utc": result.get("timestamp", ts),
            "predicted_return": result.get("predicted_return"),
            "predicted_price": result.get("predicted_price"),
            "freq": freq_val,
            "horizon": horizon_val,
            "model_version": __version__,
            "realized_r": float("nan"),
        })

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
@click.option(
    "--fee-bps",
    "--fee_bps",
    type=float,
    default=None,
    help="Transaction fee component in basis points.",
)
@click.option(
    "--slippage-bps",
    "--slippage_bps",
    type=float,
    default=None,
    help="Slippage component in basis points.",
)
def backtest_run(
    freq: str | None,
    horizon: str | None,
    allow_short_flag: bool,
    no_allow_short_flag: bool,
    cost_bps: float | None,
    fee_bps: float | None,
    slippage_bps: float | None,
) -> None:
    """Run backtest using stored predictions."""
    freq_val = _resolve_setting(freq, "freq")
    horizon_val = _resolve_setting(horizon, "horizon")
    cfg = _config()
    cost = cost_bps if cost_bps is not None else float(cfg["cost_bps"])
    resolved_fee_bps = fee_bps if fee_bps is not None else float(cfg.get("fee_bps", cost))
    resolved_slippage_bps = (
        slippage_bps if slippage_bps is not None else float(cfg.get("slippage_bps", 0.0))
    )

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

    predicted_returns = preds.set_index("timestamp_utc")["predicted_return"]

    if allow_short_flag:
        allow_short_val = True
    elif no_allow_short_flag:
        allow_short_val = False
    else:
        allow_short_val = bool(_config().get("allow_short", False))

    trades, equity = run_strategy(
        close,
        predicted_returns,
        allow_short=allow_short_val,
        cost_bps=cost,
        fee_bps=resolved_fee_bps,
        slippage_bps=resolved_slippage_bps,
    )
    metrics = summarize_backtest(equity, trades)
    click.echo(
        "Backtest complete: "
        f"net_sharpe={metrics['net_sharpe']:.3f}, "
        f"gross_sharpe={metrics['gross_sharpe']:.3f}, "
        f"max_drawdown={metrics['max_drawdown']:.3f}, "
        f"costs={metrics['total_costs']:.6f} "
        f"(fee={metrics['total_fee_costs']:.6f}, "
        f"slippage={metrics['total_slippage_costs']:.6f})"
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
        "predicted_return": float(prediction["predicted_return"]),
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

        predicted_return = float(prediction["predicted_return"])
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

    preds["timestamp_utc"] = pd.to_datetime(preds["timestamp_utc"], utc=True, errors="coerce")
    preds = preds.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")

    realised = preds["realized_r"].astype(float)
    realized_mask = realised.notna()
    hit_rate = 0.0
    if realized_mask.any():
        hit_rate = float((realised[realized_mask] > cost / 10000).mean())
    live_metrics = {
        "count": int(len(preds)),
        "avg_predicted_return": float(preds["predicted_return"].mean()),
        "realized_count": int(realized_mask.sum()),
        "hit_rate": hit_rate,
        "updated_at": datetime.now(UTC).isoformat(),
    }

    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_path = metrics_dir / f"live_{freq_val}_{horizon_val}.json"
    output_path.write_text(json.dumps(live_metrics, indent=2), encoding="utf-8")
    click.echo(f"Wrote monitoring snapshot to {output_path}")


@monitor.command("run-once")
@click.option("--freq", default=None, help="Bar frequency (defaults to config).")
@click.option("--horizon", default=None, help="Prediction horizon (defaults to config).")
def monitor_run_once(freq: str | None, horizon: str | None) -> None:
    """Run one autonomous monitoring iteration."""
    from bitbat.autonomous.agent import MonitoringAgent
    from bitbat.autonomous.db import AutonomousDB, MonitorDatabaseError
    from bitbat.autonomous.models import init_database

    freq_val = _resolve_setting(freq, "freq")
    horizon_val = _resolve_setting(horizon, "horizon")
    db_url = str(
        _config().get("autonomous", {}).get("database_url", "sqlite:///data/autonomous.db")
    )
    _emit_monitor_startup_context(freq_val, horizon_val)

    try:
        init_database(db_url)
        db = AutonomousDB(db_url)
        agent = MonitoringAgent(db, freq=freq_val, horizon=horizon_val)
        result = agent.run_once()
    except FileNotFoundError as exc:
        _raise_monitor_model_preflight_error(exc)
    except SchemaCompatibilityError as exc:
        _raise_monitor_schema_error(exc, db_url)
    except MonitorDatabaseError as exc:
        _raise_monitor_runtime_db_error(exc)

    click.echo("Monitoring run completed")
    click.echo(f"  Validations: {result['validations']}")
    click.echo(f"  Drift detected: {result['drift_detected']}")
    click.echo(f"  Retraining triggered: {result['retraining_triggered']}")
    prediction_payload = result.get("prediction")
    prediction_state = result.get("prediction_state")
    if prediction_state is None and isinstance(prediction_payload, dict):
        prediction_state = (
            "generated" if prediction_payload.get("status") == "generated" else "none"
        )  # noqa: E501
    prediction_state = str(prediction_state or "unknown")

    prediction_reason = result.get("prediction_reason")
    if prediction_reason is None and isinstance(prediction_payload, dict):
        prediction_reason = prediction_payload.get("reason")
    prediction_reason = str(prediction_reason or "unknown")

    realization_state = str(result.get("realization_state") or "unknown")
    pending_validations = int(result.get("pending_validations", 0))
    prediction_message = str(result.get("prediction_message") or "")
    cycle_diagnostic = str(result.get("cycle_diagnostic") or "").strip()
    if not cycle_diagnostic:
        if prediction_state == "generated":
            cycle_diagnostic = "prediction_generated"
        elif prediction_message:
            cycle_diagnostic = f"{prediction_reason}: {prediction_message}"
        else:
            cycle_diagnostic = prediction_reason

    click.echo(f"  Prediction state: {prediction_state}")
    click.echo(f"  Prediction reason: {prediction_reason}")
    click.echo(f"  Realization state: {realization_state}")
    click.echo(f"  Pending validations: {pending_validations}")
    click.echo(f"  Cycle diagnostic: {cycle_diagnostic}")

    metrics = result.get("metrics", {})
    diagnostics = metrics.get("window_diagnostics") if isinstance(metrics, dict) else None
    if isinstance(diagnostics, dict):
        click.echo(f"  Regime: {diagnostics.get('regime', 'unknown')}")
        click.echo(f"  Drift score: {float(diagnostics.get('drift_score', 0.0)):.6f}")


@monitor.command("start")
@click.option("--freq", default=None, help="Bar frequency (defaults to config).")
@click.option("--horizon", default=None, help="Prediction horizon (defaults to config).")
@click.option("--interval", type=int, default=None, help="Seconds between monitoring cycles.")
def monitor_start(freq: str | None, horizon: str | None, interval: int | None) -> None:
    """Start continuous autonomous monitoring."""
    from bitbat.autonomous.agent import MonitoringAgent
    from bitbat.autonomous.db import AutonomousDB, MonitorDatabaseError
    from bitbat.autonomous.models import init_database

    freq_val = _resolve_setting(freq, "freq")
    horizon_val = _resolve_setting(horizon, "horizon")
    db_url = str(
        _config().get("autonomous", {}).get("database_url", "sqlite:///data/autonomous.db")
    )
    interval_seconds = (
        interval
        if interval is not None
        else int(_config().get("autonomous", {}).get("validation_interval", 300))
    )
    _emit_monitor_startup_context(freq_val, horizon_val)

    try:
        init_database(db_url)
        db = AutonomousDB(db_url)
        agent = MonitoringAgent(db, freq=freq_val, horizon=horizon_val)
    except FileNotFoundError as exc:
        _raise_monitor_model_preflight_error(exc)
    except SchemaCompatibilityError as exc:
        _raise_monitor_schema_error(exc, db_url)
    except MonitorDatabaseError as exc:
        _raise_monitor_runtime_db_error(exc)
    click.echo(
        "Starting monitoring loop "
        f"(freq={freq_val}, horizon={horizon_val}, interval={interval_seconds}s)"
    )
    agent.run_forever(interval_seconds=interval_seconds)


@monitor.command("status")
@click.option("--freq", default=None, help="Bar frequency (defaults to config).")
@click.option("--horizon", default=None, help="Prediction horizon (defaults to config).")
def monitor_status(freq: str | None, horizon: str | None) -> None:
    """Show latest autonomous monitoring status."""
    from bitbat.autonomous.db import AutonomousDB
    from bitbat.autonomous.models import PerformanceSnapshot, RetrainingEvent, init_database

    freq_val = _resolve_setting(freq, "freq")
    horizon_val = _resolve_setting(horizon, "horizon")
    db_url = str(
        _config().get("autonomous", {}).get("database_url", "sqlite:///data/autonomous.db")
    )

    try:
        init_database(db_url)
        db = AutonomousDB(db_url)
    except SchemaCompatibilityError as exc:
        _raise_monitor_schema_error(exc, db_url)

    with db.session() as session:
        lifecycle_counts = db.get_prediction_counts(
            session=session,
            freq=freq_val,
            horizon=horizon_val,
        )
        latest_snapshot = (
            session.query(PerformanceSnapshot)
            .filter(
                PerformanceSnapshot.freq == freq_val,
                PerformanceSnapshot.horizon == horizon_val,
            )
            .order_by(PerformanceSnapshot.snapshot_time.desc())
            .first()
        )
        last_retraining = (
            session.query(RetrainingEvent).order_by(RetrainingEvent.started_at.desc()).first()
        )

    click.echo(f"Monitoring status for {freq_val}/{horizon_val}")
    click.echo(f"  Total predictions: {int(lifecycle_counts['total_predictions'])}")
    click.echo(f"  Unrealized predictions: {int(lifecycle_counts['unrealized_predictions'])}")
    click.echo(f"  Realized predictions: {int(lifecycle_counts['realized_predictions'])}")
    click.echo(f"  Pending validations: {int(lifecycle_counts['unrealized_predictions'])}")
    if latest_snapshot is None:
        click.echo("  Latest snapshot: none")
    else:
        click.echo(f"  Latest snapshot: {latest_snapshot.snapshot_time}")
        click.echo(f"  Hit rate: {latest_snapshot.hit_rate or 0.0:.2%}")
        click.echo(f"  Sharpe: {latest_snapshot.sharpe_ratio or 0.0:.3f}")
        click.echo(f"  Realized predictions: {latest_snapshot.realized_predictions}")

    if last_retraining is None:
        click.echo("  Last retraining: none")
    else:
        click.echo(
            "  Last retraining: "
            f"id={last_retraining.id}, status={last_retraining.status}, "
            f"started_at={last_retraining.started_at}"
        )


@monitor.command("snapshots")
@click.option("--freq", default=None, help="Bar frequency (defaults to config).")
@click.option("--horizon", default=None, help="Prediction horizon (defaults to config).")
@click.option("--last", "last_count", type=int, default=10, help="Number of snapshots to show.")
def monitor_snapshots(freq: str | None, horizon: str | None, last_count: int) -> None:
    """Print recent performance snapshots."""
    from bitbat.autonomous.db import AutonomousDB
    from bitbat.autonomous.models import PerformanceSnapshot, init_database

    freq_val = _resolve_setting(freq, "freq")
    horizon_val = _resolve_setting(horizon, "horizon")
    db_url = str(
        _config().get("autonomous", {}).get("database_url", "sqlite:///data/autonomous.db")
    )

    try:
        init_database(db_url)
        db = AutonomousDB(db_url)
    except SchemaCompatibilityError as exc:
        _raise_monitor_schema_error(exc, db_url)

    with db.session() as session:
        snapshots = (
            session.query(PerformanceSnapshot)
            .filter(
                PerformanceSnapshot.freq == freq_val,
                PerformanceSnapshot.horizon == horizon_val,
            )
            .order_by(PerformanceSnapshot.snapshot_time.desc())
            .limit(max(last_count, 1))
            .all()
        )

    if not snapshots:
        click.echo("No snapshots found.")
        return

    click.echo(f"Recent snapshots ({len(snapshots)}):")
    for snapshot in snapshots:
        click.echo(
            f"  {snapshot.snapshot_time} | "
            f"hit_rate={(snapshot.hit_rate or 0.0):.2%} | "
            f"sharpe={(snapshot.sharpe_ratio or 0.0):.3f} | "
            f"realized={snapshot.realized_predictions}"
        )


@validate.command("run")
@click.option("--freq", default=None, help="Bar frequency (defaults to config).")
@click.option("--horizon", default=None, help="Prediction horizon (defaults to config).")
def validate_run(freq: str | None, horizon: str | None) -> None:
    """Validate pending predictions against realized outcomes."""
    from bitbat.autonomous.db import AutonomousDB
    from bitbat.autonomous.models import init_database
    from bitbat.autonomous.validator import PredictionValidator

    freq_val = _resolve_setting(freq, "freq")
    horizon_val = _resolve_setting(horizon, "horizon")
    db_url = str(
        _config().get("autonomous", {}).get("database_url", "sqlite:///data/autonomous.db")
    )

    click.echo(f"Starting validation: freq={freq_val}, horizon={horizon_val}")

    init_database(db_url)
    db = AutonomousDB(db_url)
    validator = PredictionValidator(db=db, freq=freq_val, horizon=horizon_val)
    results = validator.validate_all()

    click.echo("")
    click.echo("Validation complete")
    click.echo(f"  Validated: {results['validated_count']} predictions")
    click.echo(f"  Correct: {results['correct_count']}")
    click.echo(f"  Hit rate: {results['hit_rate']:.2%}")

    errors = list(results.get("errors", []))
    if errors:
        click.echo("")
        click.echo(f"Errors ({len(errors)}):")
        for error in errors[:5]:
            click.echo(f"  {error}")
        if len(errors) > 5:
            click.echo(f"  ... and {len(errors) - 5} more")


@ingest.command("prices-once")
@click.option("--symbol", default="BTC-USD", show_default=True, help="Yahoo Finance ticker.")
@click.option(
    "--interval",
    default="1h",
    show_default=True,
    help="Bar interval (e.g. '1h', '1d').",
)
def ingest_prices_once(symbol: str, interval: str) -> None:
    """Fetch the latest prices once and store them."""
    from bitbat.autonomous.price_ingestion import PriceIngestionService

    service = PriceIngestionService(symbol=symbol, interval=interval)
    count = service.fetch_with_retry()
    click.echo(f"Ingested {count} price bars")


@ingest.command("news-once")
def ingest_news_once() -> None:
    """Fetch the latest news from all sources once and store them."""
    from bitbat.autonomous.news_ingestion import NewsIngestionService

    service = NewsIngestionService()
    count = service.fetch_all_sources()
    click.echo(f"Ingested {count} news articles")


@ingest.command("macro-once")
def ingest_macro_once() -> None:
    """Fetch the latest FRED macro data once and store it."""
    from bitbat.autonomous.macro_ingestion import MacroIngestionService

    data_dir = Path(_config()["data_dir"]).expanduser()
    service = MacroIngestionService(data_dir=data_dir)
    count = service.fetch_with_retry()
    click.echo(f"Ingested {count} macro rows")


@ingest.command("onchain-once")
def ingest_onchain_once() -> None:
    """Fetch the latest on-chain data once and store it."""
    from bitbat.autonomous.onchain_ingestion import OnchainIngestionService

    data_dir = Path(_config()["data_dir"]).expanduser()
    service = OnchainIngestionService(data_dir=data_dir)
    count = service.fetch_with_retry()
    click.echo(f"Ingested {count} on-chain rows")


@ingest.command("status")
def ingest_status() -> None:
    """Show ingestion data and rate-limit status."""
    from bitbat.autonomous.news_ingestion import NewsIngestionService
    from bitbat.autonomous.price_ingestion import PriceIngestionService

    price_service = PriceIngestionService()
    last_price_ts = price_service._get_last_timestamp()

    click.echo("Ingestion Status\n")
    click.echo("Prices:")
    click.echo(f"  Last update : {last_price_ts or 'never'}")
    click.echo(f"  Data dir    : {price_service.prices_dir}")

    news_service = NewsIngestionService()
    rate_status = news_service.newsapi_limiter.get_status()

    click.echo("\nNews APIs:")
    click.echo(f"  NewsAPI key  : {'set' if news_service.newsapi_key else 'not set'}")
    click.echo(
        f"  NewsAPI usage: "
        f"{rate_status['requests_made']}/{rate_status['limit']} "
        f"({rate_status['requests_remaining']} remaining)"
    )
    reset = rate_status.get("time_until_reset")
    click.echo(f"  Reset in     : {reset or 'N/A'}")
    click.echo(f"  Reddit keys  : {'set' if news_service.reddit_client_id else 'not set'}")
    click.echo(f"  News dir     : {news_service.news_dir}")


def main() -> None:
    """Entry point used by tests and scripts."""
    _cli.main(standalone_mode=False)


if __name__ == "__main__":
    _cli()
