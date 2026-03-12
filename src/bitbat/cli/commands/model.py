"""Model command group for the BitBat CLI."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import xgboost as xgb  # noqa: F401

from bitbat import __version__
from bitbat.cli._helpers import (
    _config,
    _load_feature_dataset,
    _model_path,
    _resolve_model_families,
    _resolve_setting,
)
from bitbat.dataset.build import build_xy, generate_price_features  # noqa: F401
from bitbat.dataset.splits import generate_rolling_windows, walk_forward  # noqa: F401
from bitbat.model.evaluate import (
    build_candidate_report,
    compute_multiple_testing_safeguards,  # noqa: F401
    regression_metrics,  # noqa: F401
    select_champion_report,
)
from bitbat.model.optimize import HyperparamOptimizer  # noqa: F401
from bitbat.model.persist import load as load_model  # noqa: F401
from bitbat.model.persist import save_baseline_artifact
from bitbat.model.train import fit_random_forest, fit_xgb  # noqa: F401

# ---------------------------------------------------------------------------
# Private helpers for model_cv (C901 complexity reduction)
# ---------------------------------------------------------------------------


def _resolve_cv_embargo_purge(
    embargo_bars: int | None,
    purge_bars: int | None,
    label_horizon: str | None,
    cfg: dict,
) -> tuple[int, int]:
    """Resolve embargo_bars and purge_bars from CLI options + config defaults."""
    cv_cfg = cfg.get("cv", {}) if isinstance(cfg, dict) else {}
    cfg_embargo_bars = cv_cfg.get("embargo_bars")
    cfg_purge_bars = cv_cfg.get("purge_bars")

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
    return int(resolved_embargo_bars), int(resolved_purge_bars)


def _resolve_cv_window_spec(
    train_window: str | None,
    backtest_window: str | None,
    window_step: str | None,
    windows: Iterable[tuple[str, str, str, str]],
    cfg: dict,
    start: str,
    end: str,
    index: Any,
) -> list[tuple[str, str, str, str]]:
    """Resolve rolling window parameters from CLI options + config defaults."""
    cv_cfg = cfg.get("cv", {}) if isinstance(cfg, dict) else {}
    resolved_train_window = train_window or cv_cfg.get("train_window")
    resolved_backtest_window = backtest_window or cv_cfg.get("backtest_window")
    resolved_window_step = window_step or cv_cfg.get("window_step")

    window_spec: list[tuple[str, str, str, str]] = list(windows)
    if not window_spec:
        if resolved_train_window and resolved_backtest_window:
            window_spec = generate_rolling_windows(
                index,
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
    return window_spec


def _run_cv_folds(
    folds: list,
    selected_families: list,
    ds: pd.DataFrame,
    target: str,
    freq: str,
    horizon: str,
    embargo_bars: int,
    purge_bars: int,
) -> dict[str, list[dict[str, Any]]]:
    """Iterate folds, train each model family, collect per-fold metrics."""
    from bitbat.backtest.engine import run as run_strategy
    from bitbat.backtest.metrics import summary as summarize_backtest
    from bitbat.cli._helpers import _predict_baseline

    feature_cols = [col for col in ds.columns if col.startswith("feat_")]
    X = ds[feature_cols]
    y = ds[target]

    cv_cost_bps = float(_config().get("cost_bps", 4.0))
    cv_fee_bps = float(_config().get("fee_bps", cv_cost_bps))
    cv_slippage_bps = float(_config().get("slippage_bps", 0.0))
    cv_allow_short = bool(_config().get("allow_short", False))

    summary_by_family: dict[str, list[dict[str, Any]]] = {
        model_family: [] for model_family in selected_families
    }

    for idx, fold in enumerate(folds):
        if fold.train.empty or fold.test.empty:
            continue

        X_train = X.loc[fold.train].copy()
        X_test = X.loc[fold.test]
        y_train = y.loc[fold.train]
        y_test = y.loc[fold.test]

        X_train.attrs["freq"] = freq
        X_train.attrs["horizon"] = horizon

        for model_family in selected_families:
            if model_family == "xgb":
                trained_model, _ = fit_xgb(
                    X_train, y_train, seed=int(_config()["seed"]), persist=False
                )
            else:
                trained_model, _ = fit_random_forest(
                    X_train,
                    y_train,
                    seed=int(_config()["seed"]),
                    persist=False,
                )

            predicted = _predict_baseline(model_family, trained_model, X_test)
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

    return summary_by_family


def _build_family_metrics(summary_by_family: dict) -> dict[str, dict[str, Any]]:
    """Compute avg_rmse, avg_mae per model family."""
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
    return family_metrics


def _run_champion_selection(
    family_metrics: dict,
    summary_by_family: dict,
    selected_families: list,
    freq: str,
    horizon: str,
    ds: pd.DataFrame,
    target: str,
) -> None:
    """Build candidate reports, select champion, write cv_summary.json."""
    primary_family = selected_families[0]
    primary_metrics = family_metrics.get(
        primary_family,
        {"folds": [], "average_rmse": 0.0, "average_mae": 0.0},
    )

    model_cfg = _config().get("model", {})
    optimization_cfg = model_cfg.get("optimization", {}) if isinstance(model_cfg, dict) else {}
    promotion_cfg = model_cfg.get("promotion_gate", {}) if isinstance(model_cfg, dict) else {}
    safeguard_trial_count = int(
        optimization_cfg.get("trials", max(len(primary_metrics["folds"]), 1))
    )
    safeguard_min_deflated_sharpe = float(optimization_cfg.get("min_deflated_sharpe", 0.0))
    safeguard_max_overfit_probability = float(optimization_cfg.get("max_overfit_probability", 0.50))
    promotion_min_consecutive = int(promotion_cfg.get("min_consecutive_outperformance", 2))
    promotion_drawdown_floor = float(promotion_cfg.get("max_drawdown_floor", -0.35))

    candidate_reports: dict[str, dict[str, Any]] = {}
    for model_family, folds_summary in summary_by_family.items():
        if not folds_summary:
            continue
        safeguards = compute_multiple_testing_safeguards(
            [{"outer_score": float(fold_metric.get("rmse", 0.0))} for fold_metric in folds_summary],
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


# ---------------------------------------------------------------------------
# Model command group
# ---------------------------------------------------------------------------


@click.group(help="Model lifecycle commands.")
def model() -> None:
    """Model command namespace."""


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
def model_cv(
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

    model_cfg = _config().get("model", {})
    cv_cfg = model_cfg.get("cv", {}) if isinstance(model_cfg, dict) else {}
    cfg_label_horizon = cv_cfg.get("label_horizon")
    resolved_label_horizon = label_horizon if label_horizon not in (None, "") else cfg_label_horizon

    resolved_embargo_bars, resolved_purge_bars = _resolve_cv_embargo_purge(
        embargo_bars, purge_bars, resolved_label_horizon, model_cfg
    )

    feature_cols = [col for col in dataset.columns if col.startswith("feat_")]
    X = dataset[feature_cols]

    window_spec = _resolve_cv_window_spec(
        train_window,
        backtest_window,
        window_step,
        windows,
        model_cfg,
        start,
        end,
        X.index,
    )

    folds = walk_forward(
        X.index,
        windows=window_spec,
        embargo_bars=resolved_embargo_bars,
        purge_bars=resolved_purge_bars,
        label_horizon=(
            str(resolved_label_horizon) if resolved_label_horizon not in (None, "") else None
        ),
    )

    summary_by_family = _run_cv_folds(
        folds,
        selected_families,
        dataset,
        "r_forward",
        freq_val,
        horizon_val,
        resolved_embargo_bars,
        resolved_purge_bars,
    )

    family_metrics = _build_family_metrics(summary_by_family)

    if family_metrics:
        _run_champion_selection(
            family_metrics,
            summary_by_family,
            selected_families,
            freq_val,
            horizon_val,
            dataset,
            "r_forward",
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
        require_label=True,
        require_forward_return=True,
    )
    feature_cols = [col for col in dataset.columns if col.startswith("feat_")]
    X = dataset[feature_cols]
    X.attrs["freq"] = freq_val
    X.attrs["horizon"] = horizon_val

    trained_paths: list[tuple[str, Path]] = []
    for model_family in selected_families:
        if model_family == "xgb":
            # XGBoost expects direction labels (up/down/flat); fit_xgb encodes via DIRECTION_CLASSES
            y_xgb = dataset["label"].astype(str)
            trained_model, _ = fit_xgb(X, y_xgb, seed=int(_config()["seed"]), persist=False)
        else:
            y_rf = dataset["r_forward"]
            trained_model, _ = fit_random_forest(
                X, y_rf, seed=int(_config()["seed"]), persist=False
            )
        model_path = save_baseline_artifact(
            trained_model,
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
    from bitbat.cli._helpers import _ensure_path_exists, _sentiment_enabled
    from bitbat.contracts import ensure_feature_contract, ensure_predictions_contract
    from bitbat.model.infer import predict_bar

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
        out = predictions.copy()
        out["timestamp_utc"] = out["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        click.echo(out.to_json(orient="records"))
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(output, index=False)
        click.echo(f"Wrote {len(predictions)} predictions to {output}")
