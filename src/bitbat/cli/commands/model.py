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
from bitbat.config.loader import resolve_metrics_dir
from bitbat.dataset.build import build_xy, generate_price_features  # noqa: F401
from bitbat.dataset.splits import generate_rolling_windows, walk_forward  # noqa: F401
from bitbat.model.evaluate import (
    build_candidate_report,
    classification_probability_metrics,
    compute_multiple_testing_safeguards,  # noqa: F401
    evaluate_replay_promotion_gate,
    regression_metrics,  # noqa: F401
    select_champion_report,
)
from bitbat.model.manifest import write_candidate_manifests
from bitbat.model.optimize import HyperparamOptimizer  # noqa: F401
from bitbat.model.persist import (
    BaselineFamily,
    load_metadata,
    normalize_artifact_role,
    normalize_label_mode,
    save_baseline_artifact,
)
from bitbat.model.persist import load as load_model  # noqa: F401
from bitbat.model.train import fit_random_forest, fit_xgb  # noqa: F401
from bitbat_v2.config import BitBatV2Config
from bitbat_v2.evaluation import (
    load_candles_from_parquet,
    simulate_legacy_model_replay,
    unsupported_runtime_replay_summary,
)

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


def _run_cv_folds(  # noqa: C901
    folds: list,
    selected_families: list,
    ds: pd.DataFrame,
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
    return_target = ds["r_forward"]
    label_target = ds["label"].astype(str) if "label" in ds.columns else None

    cv_cost_bps = float(_config().get("cost_bps", 4.0))
    cv_fee_bps = float(_config().get("fee_bps", cv_cost_bps))
    cv_slippage_bps = float(_config().get("slippage_bps", 0.0))
    cv_allow_short = bool(_config().get("allow_short", False))

    summary_by_family: dict[str, list[dict[str, Any]]] = {
        model_family: [] for model_family in selected_families
    }

    def _class_labels(y_values: pd.Series) -> tuple[str, ...]:
        observed = {str(value) for value in y_values.dropna().astype(str).unique()}
        if observed.issubset({"take_profit", "stop_loss", "timeout"}):
            return ("take_profit", "stop_loss", "timeout")
        if observed.issubset({"act", "pass"}):
            return ("pass", "act")
        return ("up", "down", "flat")

    for idx, fold in enumerate(folds):
        if fold.train.empty or fold.test.empty:
            continue

        X_train = X.loc[fold.train].copy()
        X_test = X.loc[fold.test]
        y_return_train = return_target.loc[fold.train]
        y_return_test = return_target.loc[fold.test]
        y_label_train = label_target.loc[fold.train] if label_target is not None else None
        y_label_test = label_target.loc[fold.test] if label_target is not None else None

        X_train.attrs["freq"] = freq
        X_train.attrs["horizon"] = horizon

        for model_family in selected_families:
            class_metrics: dict[str, float] | None = None
            if model_family == "xgb":
                if y_label_train is None or y_label_test is None:
                    raise click.ClickException(
                        "XGBoost CV requires label targets in the feature dataset."
                    )
                trained_model, _ = fit_xgb(
                    X_train,
                    y_label_train,
                    seed=int(_config()["seed"]),
                    persist=False,
                )
                raw_predictions = _predict_baseline(
                    model_family,
                    trained_model,
                    X_test,
                    return_probabilities=True,
                )
                if raw_predictions.ndim == 2:
                    class_labels = _class_labels(y_label_train)
                    try:
                        class_metrics = classification_probability_metrics(
                            y_label_test,
                            raw_predictions,
                            class_labels=class_labels,
                        )
                    except TypeError:
                        class_metrics = classification_probability_metrics(
                            y_label_test,
                            raw_predictions,
                        )
                    predicted = _predict_baseline(model_family, trained_model, X_test)
                else:
                    predicted = np.asarray(raw_predictions, dtype="float64")
            else:
                trained_model, _ = fit_random_forest(
                    X_train,
                    y_return_train,
                    seed=int(_config()["seed"]),
                    persist=False,
                )
                predicted = _predict_baseline(model_family, trained_model, X_test)
            regression = regression_metrics(y_return_test, predicted)
            predicted_series = pd.Series(predicted, index=X_test.index, dtype="float64")
            realized_returns = y_return_test.astype("float64")
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
                "objective_mode": "regression",
            }
            if class_metrics is not None:
                metrics["pr_auc"] = float(class_metrics["pr_auc"])
                metrics["logloss"] = float(class_metrics["mlogloss"])
                metrics["directional_accuracy"] = float(class_metrics["directional_accuracy"])
                metrics["objective_mode"] = "classification"
            summary_by_family[model_family].append(metrics)
            if metrics["objective_mode"] == "classification":
                click.echo(
                    "Fold "
                    f"{idx + 1} [{model_family}]: pr_auc={metrics['pr_auc']:.4f}, "
                    f"logloss={metrics['logloss']:.6f}, rmse={metrics['rmse']:.6f}"
                )
            else:
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
        pr_aucs = [float(metric["pr_auc"]) for metric in folds_summary if "pr_auc" in metric]
        loglosses = [float(metric["logloss"]) for metric in folds_summary if "logloss" in metric]
        objective_mode = (
            "classification"
            if any(metric.get("objective_mode") == "classification" for metric in folds_summary)
            else "regression"
        )
        family_metrics[model_family] = {
            "folds": folds_summary,
            "average_rmse": avg_rmse,
            "average_mae": avg_mae,
            "average_pr_auc": float(np.mean(pr_aucs)) if pr_aucs else None,
            "average_logloss": float(np.mean(loglosses)) if loglosses else None,
            "objective_mode": objective_mode,
        }
        if objective_mode == "classification" and pr_aucs and loglosses:
            click.echo(
                f"Aggregate [{model_family}]: pr_auc={float(np.mean(pr_aucs)):.4f}, "
                f"logloss={float(np.mean(loglosses)):.6f}, rmse={avg_rmse:.6f}"
            )
        else:
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
        "average_pr_auc": primary_metrics.get("average_pr_auc"),
        "average_logloss": primary_metrics.get("average_logloss"),
        "objective_mode": primary_metrics.get("objective_mode", "regression"),
        "mean_directional_accuracy": float(
            primary_directional.get("mean_directional_accuracy", 0.0)
        ),
        "candidate_reports": candidate_reports,
        "champion_decision": champion_decision,
    }
    metrics_dir = resolve_metrics_dir()
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "cv_summary.json").write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )


def _replay_gate_thresholds() -> dict[str, float | int]:
    model_cfg = _config().get("model", {})
    promotion_cfg = model_cfg.get("promotion_gate", {}) if isinstance(model_cfg, dict) else {}
    return {
        "replay_window_bars": int(promotion_cfg.get("replay_window_bars", 288)),
        "min_trade_count": int(promotion_cfg.get("min_replay_trades", 2)),
        "max_hold_rate": float(promotion_cfg.get("max_hold_rate", 0.98)),
        "max_calibration_brier": float(promotion_cfg.get("max_calibration_brier", 0.80)),
        "min_mean_expected_value_return": float(
            promotion_cfg.get("min_mean_expected_value_return", 0.0)
        ),
        "min_net_pnl_pct": float(promotion_cfg.get("min_net_pnl_pct", 0.0)),
    }


def _cost_assumptions() -> dict[str, float]:
    return {
        "cost_bps": float(_config().get("cost_bps", 4.0)),
        "fee_bps": float(_config().get("fee_bps", _config().get("cost_bps", 4.0))),
        "slippage_bps": float(_config().get("slippage_bps", 0.0)),
    }


def _artifact_entry(
    path: Path,
    *,
    artifact_role: str,
) -> dict[str, Any]:
    metadata = load_metadata(path)
    return {
        "artifact_role": normalize_artifact_role(artifact_role),
        "path": str(path),
        "metadata": metadata,
    }


def _train_path_aware_auxiliary_artifacts(
    *,
    X: pd.DataFrame,
    dataset: pd.DataFrame,
    freq: str,
    horizon: str,
) -> list[tuple[str, Path]]:
    auxiliary_paths: list[tuple[str, Path]] = []
    side_label = dataset.get("side_label")
    if side_label is None:
        side_label = pd.Series("flat", index=dataset.index, dtype="string")
        side_label.loc[dataset["r_forward"].astype(float) > 0.0] = "up"
        side_label.loc[dataset["r_forward"].astype(float) < 0.0] = "down"
    meta_label = dataset.get("meta_label")
    if meta_label is None:
        meta_label = pd.Series(
            np.where(dataset["label"].astype(str) == "timeout", "pass", "act"),
            index=dataset.index,
            dtype="string",
        )

    side_model, _ = fit_xgb(
        X,
        side_label.astype(str),
        label_mode="direction",
        seed=int(_config()["seed"]),
        persist=False,
    )
    side_path = save_baseline_artifact(
        side_model,
        family="xgb",
        freq=freq,
        horizon=horizon,
        label_mode="direction",
        artifact_role="side",
        metadata={
            "source": "cli:model-train",
            "model_role": "side",
            "source_label_mode": "triple_barrier",
            "class_labels": ["up", "down", "flat"],
        },
    )
    auxiliary_paths.append(("side", side_path))

    action_model, _ = fit_xgb(
        X,
        meta_label.astype(str),
        label_mode="meta_label",
        seed=int(_config()["seed"]),
        persist=False,
    )
    action_path = save_baseline_artifact(
        action_model,
        family="xgb",
        freq=freq,
        horizon=horizon,
        label_mode="meta_label",
        artifact_role="action",
        metadata={
            "source": "cli:model-train",
            "model_role": "action",
            "source_label_mode": "triple_barrier",
            "class_labels": ["pass", "act"],
        },
    )
    auxiliary_paths.append(("action", action_path))
    return auxiliary_paths


def _feature_family_name(column: str) -> str:
    if column.startswith("feat_sent_"):
        return "sentiment"
    if "garch" in column:
        return "volatility"
    if "macro" in column:
        return "macro"
    if "onchain" in column:
        return "onchain"
    return "price"


def _build_ablation_scenarios(feature_cols: list[str]) -> list[dict[str, Any]]:
    family_map: dict[str, list[str]] = {}
    for column in feature_cols:
        family = _feature_family_name(column)
        family_map.setdefault(family, []).append(column)

    scenarios: list[dict[str, Any]] = [
        {
            "scenario_id": "all_features",
            "included_families": sorted(family_map),
            "excluded_families": [],
            "columns": list(feature_cols),
        }
    ]
    price_columns = family_map.get("price", [])
    if price_columns:
        scenarios.append({
            "scenario_id": "price_only",
            "included_families": ["price"],
            "excluded_families": sorted([name for name in family_map if name != "price"]),
            "columns": list(price_columns),
        })
    for family in sorted(name for name in family_map if name != "price"):
        remaining = [column for column in feature_cols if _feature_family_name(column) != family]
        if not remaining:
            continue
        scenarios.append({
            "scenario_id": f"drop_{family}",
            "included_families": sorted(name for name in family_map if name != family),
            "excluded_families": [family],
            "columns": remaining,
        })
    return scenarios


def _runtime_replay_summary_for_artifact(
    *,
    family: str,
    label_mode: str,
    freq: str,
    horizon: str,
) -> dict[str, Any]:
    model_dir = Path(_config().get("models_dir", "models")) / f"{freq}_{horizon}"
    has_meta_policy = (model_dir / "xgb.side.json").exists() and (
        model_dir / "xgb.action.meta_label.json"
    ).exists()
    if family != "xgb":
        return unsupported_runtime_replay_summary(
            signal_source="legacy_ml",
            model_name=f"{family}_{freq}_{horizon}",
            compatibility_reason="runtime_incompatible_family",
        ).to_dict()
    if label_mode != "direction" and not has_meta_policy:
        return unsupported_runtime_replay_summary(
            signal_source="legacy_ml",
            model_name=f"{family}_{freq}_{horizon}",
            compatibility_reason=f"runtime_incompatible_label_mode:{label_mode}",
        ).to_dict()

    thresholds = _replay_gate_thresholds()
    prices_path = Path(_config()["data_dir"]) / "raw" / "prices" / f"btcusd_yf_{freq}.parquet"
    if not prices_path.exists():
        return unsupported_runtime_replay_summary(
            signal_source="legacy_ml",
            model_name=f"{family}_{freq}_{horizon}",
            compatibility_reason="replay_prices_missing",
        ).to_dict()

    replay_config = BitBatV2Config(
        database_url="sqlite:///:memory:",
        demo_mode=False,
        signal_source="legacy_ml",
        legacy_signal_freq=freq,
        legacy_signal_horizon=horizon,
        fee_bps=float(_config().get("fee_bps", _config().get("cost_bps", 4.0))),
        slippage_bps=float(_config().get("slippage_bps", 0.0)),
    )
    candles = load_candles_from_parquet(prices_path, replay_config)
    replay_window_bars = int(thresholds["replay_window_bars"])
    if replay_window_bars > 0:
        candles = candles[-replay_window_bars:]
    if not candles:
        return unsupported_runtime_replay_summary(
            signal_source="legacy_ml",
            model_name=f"{family}_{freq}_{horizon}",
            compatibility_reason="replay_window_empty",
        ).to_dict()

    return simulate_legacy_model_replay(
        candles,
        replay_config,
        tau=float(_config().get("tau", 0.01)),
    ).to_dict()


def _update_cv_summary_with_replay_gate(
    *,
    trained_paths: list[tuple[BaselineFamily, Path]],
    auxiliary_artifacts: dict[str, list[tuple[str, Path]]] | None = None,
    dataset_meta: dict[str, Any] | None = None,
    label_mode: str,
    freq: str,
    horizon: str,
) -> None:
    summary_path = resolve_metrics_dir() / "cv_summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        primary_family = trained_paths[0][0] if trained_paths else "xgb"
        payload = {
            "candidate_reports": {
                family: {"candidate_id": family, "family": family}
                for family, _model_path in trained_paths
            },
            "champion_decision": {
                "winner": primary_family,
                "promote_candidate": False,
                "promotion_gate": {"pass": False, "reasons": ["cv-summary-missing"]},
                "reason": "cv-summary-missing",
            },
        }
    candidate_reports = payload.get("candidate_reports", {})
    if not isinstance(candidate_reports, dict):
        return

    thresholds = _replay_gate_thresholds()
    auxiliary_artifacts = auxiliary_artifacts or {}
    replay_payload: dict[str, Any] = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "winner": payload.get("champion_decision", {}).get("winner"),
        "families": {},
    }
    artifact_registry: dict[str, dict[str, Any]] = {}
    for family, model_path in trained_paths:
        report = candidate_reports.get(family)
        if not isinstance(report, dict):
            continue
        artifact_metadata = load_metadata(model_path)
        replay_summary = _runtime_replay_summary_for_artifact(
            family=family,
            label_mode=label_mode,
            freq=freq,
            horizon=horizon,
        )
        replay_gate = evaluate_replay_promotion_gate(
            replay_summary,
            min_trade_count=int(thresholds["min_trade_count"]),
            max_hold_rate=float(thresholds["max_hold_rate"]),
            max_calibration_brier=float(thresholds["max_calibration_brier"]),
            min_mean_expected_value_return=float(thresholds["min_mean_expected_value_return"]),
            min_net_pnl_pct=float(thresholds["min_net_pnl_pct"]),
        )
        report["artifact_path"] = str(model_path)
        report["artifact_metadata"] = artifact_metadata
        report["replay_summary"] = replay_summary
        report["replay_gate"] = replay_gate
        family_auxiliary = auxiliary_artifacts.get(family, [])
        report["auxiliary_artifacts"] = [
            _artifact_entry(aux_path, artifact_role=role) for role, aux_path in family_auxiliary
        ]
        replay_payload["families"][family] = {
            "artifact_path": str(model_path),
            "artifact_metadata": artifact_metadata,
            "replay_summary": replay_summary,
            "replay_gate": replay_gate,
            "auxiliary_artifacts": report["auxiliary_artifacts"],
        }
        artifact_registry[family] = {
            "primary": _artifact_entry(model_path, artifact_role="primary"),
            "auxiliary": report["auxiliary_artifacts"],
        }

    champion_decision = payload.get("champion_decision", {})
    if isinstance(champion_decision, dict):
        winner = champion_decision.get("winner")
        winner_report = candidate_reports.get(winner) if isinstance(winner, str) else None
        if isinstance(winner_report, dict):
            replay_gate = winner_report.get("replay_gate", {})
            winner_replay_summary = winner_report.get("replay_summary")
            champion_decision["replay_gate"] = replay_gate
            champion_decision["artifact_path"] = winner_report.get("artifact_path")
            champion_decision["artifact_metadata"] = winner_report.get("artifact_metadata")
            promotion_gate = champion_decision.get("promotion_gate", {})
            if not isinstance(promotion_gate, dict):
                promotion_gate = {}
            existing_reasons = list(promotion_gate.get("reasons", []))
            replay_reasons = replay_gate.get("reasons", []) if isinstance(replay_gate, dict) else []
            merged_reasons = existing_reasons + [
                reason for reason in replay_reasons if reason not in existing_reasons
            ]
            promotion_gate["replay_gate"] = replay_gate
            promotion_gate["replay_summary"] = winner_replay_summary
            promotion_gate["pass"] = bool(promotion_gate.get("pass", True)) and bool(
                isinstance(replay_gate, dict) and replay_gate.get("pass", False)
            )
            promotion_gate["reasons"] = merged_reasons
            champion_decision["promotion_gate"] = promotion_gate
            if not bool(isinstance(replay_gate, dict) and replay_gate.get("pass", False)):
                champion_decision["promote_candidate"] = False
                if isinstance(replay_gate, dict) and replay_gate.get("runtime_compatible") is False:
                    champion_decision["reason"] = "winner-runtime-incompatible"
                else:
                    champion_decision["reason"] = "replay-gate-failed"

    payload["candidate_reports"] = candidate_reports
    payload["champion_decision"] = champion_decision
    payload["runtime_replay"] = replay_payload
    payload["candidate_manifests"] = write_candidate_manifests(
        freq=freq,
        horizon=horizon,
        dataset_meta=dataset_meta or {},
        candidate_reports=candidate_reports,
        artifact_registry=artifact_registry,
        champion_decision=champion_decision if isinstance(champion_decision, dict) else {},
        cost_assumptions=_cost_assumptions(),
    )
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (resolve_metrics_dir() / "runtime_replay_summary.json").write_text(
        json.dumps(replay_payload, indent=2),
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
        require_label="xgb" in selected_families,
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
        require_label=True,
        require_forward_return=True,
    )
    feature_cols = [col for col in dataset.columns if col.startswith("feat_")]
    if not feature_cols:
        raise click.ClickException(
            "No feature columns found in dataset (expected columns prefixed with 'feat_')."
        )
    X = dataset[feature_cols]
    y = dataset["label"].astype(str)

    model_cfg = _config().get("model", {})
    optimization_cfg = model_cfg.get("optimization", {}) if isinstance(model_cfg, dict) else {}
    cv_cfg = model_cfg.get("cv", {}) if isinstance(model_cfg, dict) else {}

    resolved_trials = int(trials if trials is not None else optimization_cfg.get("trials", 20))
    timeout_raw = timeout if timeout is not None else optimization_cfg.get("timeout_seconds")
    resolved_timeout = None if timeout_raw in (None, "", 0, "0") else int(str(timeout_raw))
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

    metrics_dir = resolve_metrics_dir()
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_path = metrics_dir / "optimization_summary.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    objective_mode = str(summary.get("objective_mode", "regression"))
    if objective_mode == "classification":
        click.echo(
            "Optimization complete: "
            f"best_pr_auc={float(summary.get('best_pr_auc', 0.0)):.6f}, "
            f"score={float(summary.get('best_score', 0.0)):.6f}, "
            f"trials={resolved_trials}, outer_folds={len(summary.get('outer_folds', []))}. "
            f"Saved {output_path}"
        )
    else:
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
    dataset_meta_path = Path(_config()["data_dir"]) / "features" / f"{freq_val}_{horizon_val}"
    dataset_meta_path = dataset_meta_path / "meta.json"
    dataset_meta: dict[str, Any] = {}
    if dataset_meta_path.exists():
        dataset_meta = json.loads(dataset_meta_path.read_text(encoding="utf-8"))
    label_mode = normalize_label_mode(str(dataset_meta.get("label_mode", "direction")))
    feature_cols = [col for col in dataset.columns if col.startswith("feat_")]
    X = dataset[feature_cols]
    X.attrs["freq"] = freq_val
    X.attrs["horizon"] = horizon_val

    class_labels = (
        ["up", "down", "flat"]
        if label_mode == "direction"
        else ["take_profit", "stop_loss", "timeout"]
    )
    trained_paths: list[tuple[BaselineFamily, Path]] = []
    auxiliary_artifacts: dict[str, list[tuple[str, Path]]] = {}
    for model_family in selected_families:
        if model_family == "xgb":
            y_xgb = dataset["label"].astype(str)
            trained_model, _ = fit_xgb(
                X,
                y_xgb,
                label_mode=label_mode,
                seed=int(_config()["seed"]),
                persist=False,
            )
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
            label_mode=label_mode,
            metadata={
                "source": "cli:model-train",
                "model_role": "primary",
                "class_labels": class_labels,
            },
        )
        trained_paths.append((model_family, model_path))
        if model_family == "xgb" and label_mode == "triple_barrier":
            auxiliary_artifacts[model_family] = _train_path_aware_auxiliary_artifacts(
                X=X,
                dataset=dataset,
                freq=freq_val,
                horizon=horizon_val,
            )

    _update_cv_summary_with_replay_gate(
        trained_paths=trained_paths,
        auxiliary_artifacts=auxiliary_artifacts,
        dataset_meta=dataset_meta,
        label_mode=label_mode,
        freq=freq_val,
        horizon=horizon_val,
    )

    if len(trained_paths) == 1 and trained_paths[0][0] == "xgb":
        click.echo(f"Trained model saved to {trained_paths[0][1]}")
    else:
        for model_family, model_path in trained_paths:
            click.echo(f"Trained {model_family} model saved to {model_path}")
            for role, auxiliary_path in auxiliary_artifacts.get(model_family, []):
                click.echo(f"Trained {model_family} {role} artifact saved to {auxiliary_path}")


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

    metadata = load_metadata(resolved_model_path)
    label_mode = normalize_label_mode(str(metadata.get("label_mode", "direction")))
    if label_mode != "direction":
        raise click.ClickException(
            "model infer currently supports only direction artifacts. "
            f"Artifact {resolved_model_path} declares label_mode='{label_mode}'."
        )

    booster = load_model(resolved_model_path, expected_label_mode="direction")
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


@model.command("ablate-features")
@click.option("--freq", default=None, help="Bar frequency.")
@click.option("--horizon", default=None, help="Prediction horizon.")
@click.option("--start", default=None, help="Optional train/eval start (ISO8601).")
@click.option("--end", default=None, help="Optional train/eval end (ISO8601).")
def model_ablate_features(
    freq: str | None,
    horizon: str | None,
    start: str | None,
    end: str | None,
) -> None:
    """Run deterministic bar-stack feature-family ablations on the directional benchmark path."""
    freq_val = _resolve_setting(freq, "freq")
    horizon_val = _resolve_setting(horizon, "horizon")
    dataset = _load_feature_dataset(
        freq_val,
        horizon_val,
        require_label=True,
        require_forward_return=True,
    )
    dataset_meta_path = (
        Path(_config()["data_dir"]) / "features" / f"{freq_val}_{horizon_val}" / "meta.json"
    )
    dataset_meta: dict[str, Any] = {}
    if dataset_meta_path.exists():
        dataset_meta = json.loads(dataset_meta_path.read_text(encoding="utf-8"))
    label_mode = normalize_label_mode(str(dataset_meta.get("label_mode", "direction")))
    if label_mode != "direction":
        raise click.ClickException(
            "model ablate-features currently supports only direction datasets so scenarios "
            "stay runtime-comparable."
        )

    feature_cols = [col for col in dataset.columns if col.startswith("feat_")]
    if not feature_cols:
        raise click.ClickException("No feature columns found for feature ablation.")

    start_value = start or str(dataset.index.min())
    end_value = end or str(dataset.index.max())
    model_cfg = _config().get("model", {})
    resolved_embargo_bars, resolved_purge_bars = _resolve_cv_embargo_purge(
        None, None, None, model_cfg
    )
    window_spec = _resolve_cv_window_spec(
        None,
        None,
        None,
        [],
        model_cfg,
        start_value,
        end_value,
        dataset.index,
    )
    folds = walk_forward(
        dataset.index,
        windows=window_spec,
        embargo_bars=resolved_embargo_bars,
        purge_bars=resolved_purge_bars,
    )
    if not folds:
        raise click.ClickException("No ablation folds produced from the current dataset/window.")

    scenarios = _build_ablation_scenarios(feature_cols)
    reports: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_dataset = dataset[["label", "r_forward", *scenario["columns"]]].copy()
        summary_by_family = _run_cv_folds(
            folds,
            ["xgb"],
            scenario_dataset,
            freq_val,
            horizon_val,
            resolved_embargo_bars,
            resolved_purge_bars,
        )
        folds_summary = summary_by_family.get("xgb", [])
        report = build_candidate_report(
            candidate_id=scenario["scenario_id"],
            family="xgb",
            fold_metrics=folds_summary,
        )
        reports.append({
            "scenario_id": scenario["scenario_id"],
            "included_families": scenario["included_families"],
            "excluded_families": scenario["excluded_families"],
            "feature_count": len(scenario["columns"]),
            "metrics": report["metrics"],
            "windows": report["windows"],
        })

    reports.sort(
        key=lambda item: (
            float(item["metrics"]["risk"]["mean_net_return"]),
            float(item["metrics"]["risk"]["mean_net_sharpe"]),
            float(item["metrics"]["directional"]["mean_directional_accuracy"]),
        ),
        reverse=True,
    )
    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "freq": freq_val,
        "horizon": horizon_val,
        "label_mode": label_mode,
        "reports": reports,
        "recommended_scenario": reports[0]["scenario_id"] if reports else None,
    }
    metrics_dir = resolve_metrics_dir()
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_path = metrics_dir / "feature_ablation.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    click.echo(
        "Feature ablation complete: "
        f"recommended={payload['recommended_scenario']}, "
        f"scenarios={len(reports)}. Saved {output_path}"
    )
