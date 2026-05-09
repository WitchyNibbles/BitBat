"""Walk-forward candidate selection for preset-specific model profiles."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from bitbat.dataset.splits import walk_forward
from bitbat.model.evaluate import (
    build_candidate_report,
    classification_probability_metrics,
    compute_multiple_testing_safeguards,
    regression_metrics,
    select_champion_report,
)
from bitbat.model.mode_profiles import ModeModelProfile
from bitbat.model.train import fit_random_forest, fit_xgb


def _expanding_windows(index: pd.Index, *, fold_count: int = 3) -> list[tuple[str, str, str, str]]:
    if len(index) < (fold_count + 1) * 8:
        return []

    segments = [segment for segment in np.array_split(index, fold_count + 1) if len(segment) > 0]
    if len(segments) < 2:
        return []

    windows: list[tuple[str, str, str, str]] = []
    for position in range(1, len(segments)):
        train = pd.Index(np.concatenate([segment.to_numpy() for segment in segments[:position]]))
        test = pd.Index(segments[position])
        if train.empty or test.empty:
            continue
        windows.append((
            pd.Timestamp(train.min()).isoformat(sep=" "),
            pd.Timestamp(train.max()).isoformat(sep=" "),
            pd.Timestamp(test.min()).isoformat(sep=" "),
            pd.Timestamp(test.max()).isoformat(sep=" "),
        ))
    return windows


def _predict_xgb(
    model: xgb.Booster,
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    dtest = xgb.DMatrix(features, feature_names=list(features.columns))
    probabilities = np.asarray(model.predict(dtest), dtype="float64")
    if probabilities.ndim == 1:
        return probabilities, probabilities
    positive_idx = 0
    negative_idx = 1 if probabilities.shape[1] > 1 else 0
    score = probabilities[:, positive_idx] - probabilities[:, negative_idx]
    return probabilities, np.asarray(score, dtype="float64")


def _build_summary_by_family(
    dataset: pd.DataFrame,
    *,
    profile: ModeModelProfile,
    candidate_families: list[str],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    feature_cols = [column for column in dataset.columns if column.startswith("feat_")]
    X = dataset[feature_cols].sort_index()
    y_returns = dataset["r_forward"].astype("float64").sort_index()
    y_labels = dataset["label"].astype(str).sort_index()
    windows = _expanding_windows(X.index)
    if not windows:
        return {}

    folds = walk_forward(X.index, windows=windows, label_horizon=profile.horizon)
    summary_by_family: dict[str, list[dict[str, Any]]] = {
        family: [] for family in candidate_families
    }
    class_labels = list(profile.class_labels)

    for fold_index, fold in enumerate(folds, start=1):
        if fold.train.empty or fold.test.empty:
            continue

        X_train = X.loc[fold.train].copy()
        X_test = X.loc[fold.test].copy()
        X_train.attrs["freq"] = profile.freq
        X_train.attrs["horizon"] = profile.horizon
        y_return_train = y_returns.loc[fold.train]
        y_return_test = y_returns.loc[fold.test]
        y_label_train = y_labels.loc[fold.train]
        y_label_test = y_labels.loc[fold.test]

        for family in candidate_families:
            classification_metrics: dict[str, Any] | None = None
            if family == "xgb":
                trained_model, _ = fit_xgb(
                    X_train,
                    y_label_train,
                    seed=seed,
                    persist=False,
                    class_labels=class_labels,
                )
                probabilities, predicted = _predict_xgb(trained_model, X_test)
                classification_metrics = classification_probability_metrics(
                    y_label_test,
                    probabilities,
                    class_labels=tuple(class_labels),
                )
            else:
                trained_model, _ = fit_random_forest(
                    X_train,
                    y_return_train,
                    seed=seed,
                    persist=False,
                )
                predicted = np.asarray(trained_model.predict(X_test.astype(float)), dtype="float64")

            regression = regression_metrics(y_return_test, predicted)
            summary: dict[str, Any] = {
                **regression,
                "net_sharpe": float(np.mean(predicted)) if len(predicted) else 0.0,
                "gross_sharpe": float(np.mean(np.abs(predicted))) if len(predicted) else 0.0,
                "max_drawdown": float(min(np.min(predicted), 0.0)) if len(predicted) else 0.0,
                "net_return": float(np.mean(predicted)) if len(predicted) else 0.0,
                "gross_return": float(np.mean(np.abs(predicted))) if len(predicted) else 0.0,
                "total_costs": 0.0,
                "total_fee_costs": 0.0,
                "total_slippage_costs": 0.0,
                "window_id": f"fold-{fold_index}",
            }
            if classification_metrics is not None:
                summary["pr_auc"] = float(classification_metrics["pr_auc"])
                summary["logloss"] = float(classification_metrics["mlogloss"])
                summary["directional_accuracy"] = float(
                    classification_metrics["directional_accuracy"]
                )
            summary_by_family[family].append(summary)

    return {family: folds for family, folds in summary_by_family.items() if folds}


def summarize_candidate_selection(
    summary_by_family: dict[str, list[dict[str, Any]]],
    *,
    candidate_families: list[str],
    incumbent_family: str,
    trial_count: int = 3,
    min_deflated_sharpe: float = 0.0,
    max_overfit_probability: float = 0.50,
    min_consecutive_outperformance: int = 2,
    max_drawdown_floor: float = -0.35,
) -> dict[str, Any]:
    """Build deterministic candidate reports and choose the winning family."""
    if not summary_by_family:
        return {
            "selected_family": incumbent_family,
            "candidate_reports": {},
            "champion_decision": {
                "winner": incumbent_family,
                "promote_candidate": False,
                "reason": "no-candidates",
            },
        }

    candidate_reports: dict[str, dict[str, Any]] = {}
    for family in candidate_families:
        folds = summary_by_family.get(family, [])
        if not folds:
            continue
        safeguards = compute_multiple_testing_safeguards(
            [{"outer_score": float(fold.get("rmse", 0.0))} for fold in folds],
            trial_count=max(trial_count, len(folds)),
            min_deflated_sharpe=min_deflated_sharpe,
            max_overfit_probability=max_overfit_probability,
        )
        candidate_reports[family] = build_candidate_report(
            candidate_id=family,
            family=family,
            fold_metrics=folds,
            safeguards=safeguards,
        )

    champion_decision = select_champion_report(
        candidate_reports=candidate_reports,
        incumbent_id=incumbent_family,
        max_drawdown_floor=max_drawdown_floor,
        min_consecutive_outperformance=min_consecutive_outperformance,
    )
    selected_family = (
        str(champion_decision.get("winner"))
        if champion_decision.get("promote_candidate")
        else incumbent_family
    )
    return {
        "selected_family": selected_family,
        "candidate_reports": candidate_reports,
        "champion_decision": champion_decision,
    }


def select_mode_candidate(
    dataset: pd.DataFrame,
    *,
    profile: ModeModelProfile,
    seed: int = 42,
) -> dict[str, Any]:
    """Run preset-specific walk-forward selection and return winner metadata."""
    summary_by_family = _build_summary_by_family(
        dataset,
        profile=profile,
        candidate_families=list(profile.candidate_families),
        seed=seed,
    )
    selection = summarize_candidate_selection(
        summary_by_family,
        candidate_families=list(profile.candidate_families),
        incumbent_family=profile.family,
        trial_count=max(len(next(iter(summary_by_family.values()), [])), 1),
    )
    selection["summary_by_family"] = summary_by_family
    return selection
