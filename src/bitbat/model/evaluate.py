"""Model evaluation routines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def window_diagnostics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    window_id: str | None = None,
    family: str | None = None,
) -> dict[str, Any]:
    """Compute deterministic regime/drift diagnostics for one evaluation window."""
    y_t = np.asarray(y_true, dtype="float64")
    y_p = np.asarray(y_pred, dtype="float64")
    if y_t.shape != y_p.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    error = y_p - y_t
    mae = float(np.mean(np.abs(error))) if y_t.size else 0.0
    rmse = float(np.sqrt(np.mean(error**2))) if y_t.size else 0.0
    bias = float(np.mean(error)) if y_t.size else 0.0

    realized_volatility = float(np.std(y_t)) if y_t.size else 0.0
    predicted_volatility = float(np.std(y_p)) if y_p.size else 0.0
    volatility_ratio = (
        float(predicted_volatility / realized_volatility)
        if realized_volatility > 0
        else 0.0
    )

    directional_accuracy = (
        float(np.mean(np.sign(y_t) == np.sign(y_p)))
        if y_t.size
        else 0.0
    )
    directional_stability = (
        float(np.mean(np.sign(y_p[1:]) == np.sign(y_p[:-1])))
        if y_p.size > 1
        else 1.0
    )

    if realized_volatility >= 0.01:
        regime = "high_volatility"
    elif realized_volatility >= 0.003:
        regime = "medium_volatility"
    else:
        regime = "low_volatility"

    diagnostics = {
        "window_id": window_id,
        "family": family,
        "regime": regime,
        "drift_score": float(abs(bias) + mae),
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "realized_volatility": realized_volatility,
        "predicted_volatility": predicted_volatility,
        "volatility_ratio": volatility_ratio,
        "directional_accuracy": directional_accuracy,
        "directional_stability": directional_stability,
        "n_samples": int(y_t.size),
    }
    return diagnostics


def write_window_diagnostics(
    diagnostics: dict[str, Any] | list[dict[str, Any]],
    *,
    output_path: str | Path = Path("metrics") / "window_diagnostics.json",
) -> Path:
    """Persist diagnostics payload for downstream retraining analysis."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return target


def regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Compute regression metrics from true and predicted returns.

    Writes `metrics/regression_metrics.json` and
    `metrics/prediction_scatter.png` as side effects.

    Args:
        y_true: Ground-truth forward returns for each sample.
        y_pred: Predicted forward returns.

    Returns:
        Mapping of MAE, RMSE, R-squared, directional accuracy, and correlation.
    """
    y_t = np.asarray(y_true, dtype="float64")
    y_p = np.asarray(y_pred, dtype="float64")

    if y_t.shape != y_p.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    residuals = y_t - y_p
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Directional accuracy: sign match
    sign_match = np.sign(y_t) == np.sign(y_p)
    directional_accuracy = float(np.mean(sign_match))

    # Correlation
    correlation = (
        float(np.corrcoef(y_t, y_p)[0, 1])
        if y_t.std() > 0 and y_p.std() > 0
        else 0.0
    )

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2),
        "directional_accuracy": directional_accuracy,
        "correlation": correlation,
        "n_samples": int(len(y_t)),
    }

    # Scatter plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_t, y_p, alpha=0.3, s=10)
    lims = [min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual Return")
    ax.set_ylabel("Predicted Return")
    ax.set_title(f"Predicted vs Actual (R²={r2:.3f})")

    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "regression_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fig_path = metrics_dir / "prediction_scatter.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    return metrics


def _fold_mean(folds: list[dict[str, Any]], key: str) -> float:
    values = [float(fold[key]) for fold in folds if key in fold]
    if not values:
        return 0.0
    return float(np.mean(values))


def _fold_sum(folds: list[dict[str, Any]], key: str) -> float:
    values = [float(fold[key]) for fold in folds if key in fold]
    if not values:
        return 0.0
    return float(np.sum(values))


def compute_multiple_testing_safeguards(
    outer_folds: list[dict[str, Any]],
    *,
    trial_count: int,
    min_deflated_sharpe: float = 0.0,
    max_overfit_probability: float = 0.50,
) -> dict[str, Any]:
    """Compute deterministic multiple-testing safeguards from outer-fold outcomes.

    Uses reciprocal RMSE as a stability-adjusted performance signal, then applies
    a conservative trial-count/dispersion penalty to produce a deflated Sharpe-style
    score and an overfit probability proxy.
    """
    outer_scores = [
        float(fold["outer_score"])
        for fold in outer_folds
        if "outer_score" in fold and float(fold["outer_score"]) > 0.0
    ]
    if not outer_scores:
        return {
            "trial_count": int(max(trial_count, 0)),
            "window_count": 0,
            "raw_sharpe": 0.0,
            "deflated_sharpe": 0.0,
            "overfit_probability": 1.0,
            "thresholds": {
                "min_deflated_sharpe": float(min_deflated_sharpe),
                "max_overfit_probability": float(max_overfit_probability),
            },
            "pass": False,
            "reasons": ["no-outer-fold-scores"],
        }

    score_vector = np.asarray(outer_scores, dtype="float64")
    performance_signal = 1.0 / np.clip(score_vector, 1e-12, None)
    mean_signal = float(np.mean(performance_signal))
    std_signal = float(np.std(performance_signal, ddof=0))
    window_count = int(performance_signal.size)

    if std_signal <= 1e-12:
        raw_sharpe = float(mean_signal * np.sqrt(window_count))
    else:
        raw_sharpe = float((mean_signal / std_signal) * np.sqrt(window_count))

    coefficient_of_variation = (
        float(std_signal / abs(mean_signal)) if abs(mean_signal) > 1e-12 else 1.0
    )
    trial_penalty = float(np.sqrt(2.0 * np.log(max(int(trial_count), 1))))
    adjusted_penalty = float(trial_penalty * (1.0 + coefficient_of_variation))
    deflated_sharpe = float(raw_sharpe - adjusted_penalty)

    clipped_deflated = float(np.clip(deflated_sharpe, -60.0, 60.0))
    overfit_probability = float(1.0 / (1.0 + np.exp(clipped_deflated)))

    reasons: list[str] = []
    if deflated_sharpe < float(min_deflated_sharpe):
        reasons.append("deflated_sharpe_below_threshold")
    if overfit_probability > float(max_overfit_probability):
        reasons.append("overfit_probability_above_threshold")

    return {
        "trial_count": int(max(trial_count, 0)),
        "window_count": window_count,
        "raw_sharpe": round(raw_sharpe, 6),
        "deflated_sharpe": round(deflated_sharpe, 6),
        "overfit_probability": round(overfit_probability, 6),
        "thresholds": {
            "min_deflated_sharpe": float(min_deflated_sharpe),
            "max_overfit_probability": float(max_overfit_probability),
        },
        "pass": len(reasons) == 0,
        "reasons": reasons,
    }


def evaluate_promotion_gate(
    *,
    candidate_report: dict[str, Any],
    incumbent_report: dict[str, Any],
    min_consecutive_outperformance: int = 2,
    max_drawdown_floor: float = -0.25,
) -> dict[str, Any]:
    """Evaluate promotion safety versus incumbent across consecutive windows.

    A candidate passes only when it beats the incumbent over enough consecutive
    out-of-sample windows and stays within configured drawdown guardrails.
    """
    candidate_windows = list(candidate_report.get("windows", []))
    incumbent_windows = list(incumbent_report.get("windows", []))
    aligned_windows = min(len(candidate_windows), len(incumbent_windows))

    if aligned_windows == 0:
        return {
            "pass": False,
            "min_consecutive_outperformance": int(max(min_consecutive_outperformance, 1)),
            "max_consecutive_outperformance": 0,
            "drawdown_floor": float(max_drawdown_floor),
            "drawdown_ok": False,
            "window_count": 0,
            "window_verdicts": [],
            "reasons": ["no-window-overlap"],
        }

    required_streak = int(max(min_consecutive_outperformance, 1))
    streak = 0
    max_streak = 0
    drawdown_ok = True
    window_verdicts: list[dict[str, Any]] = []

    for idx in range(aligned_windows):
        candidate_window = candidate_windows[idx]
        incumbent_window = incumbent_windows[idx]

        candidate_return = float(candidate_window.get("net_return", 0.0))
        incumbent_return = float(incumbent_window.get("net_return", 0.0))
        beats_incumbent = candidate_return > incumbent_return

        streak = streak + 1 if beats_incumbent else 0
        max_streak = max(max_streak, streak)

        candidate_drawdown = float(candidate_window.get("max_drawdown", 0.0))
        window_drawdown_ok = candidate_drawdown >= float(max_drawdown_floor)
        drawdown_ok = bool(drawdown_ok and window_drawdown_ok)

        window_verdicts.append({
            "window_id": (
                str(candidate_window.get("window_id"))
                if candidate_window.get("window_id") is not None
                else f"fold-{idx + 1}"
            ),
            "candidate_net_return": round(candidate_return, 6),
            "incumbent_net_return": round(incumbent_return, 6),
            "candidate_beats_incumbent": bool(beats_incumbent),
            "candidate_max_drawdown": round(candidate_drawdown, 6),
            "drawdown_ok": bool(window_drawdown_ok),
        })

    reasons: list[str] = []
    if max_streak < required_streak:
        reasons.append("insufficient_consecutive_outperformance")
    if not drawdown_ok:
        reasons.append("drawdown_guardrail_violated")

    return {
        "pass": len(reasons) == 0,
        "min_consecutive_outperformance": required_streak,
        "max_consecutive_outperformance": int(max_streak),
        "drawdown_floor": float(max_drawdown_floor),
        "drawdown_ok": bool(drawdown_ok),
        "window_count": int(aligned_windows),
        "window_verdicts": window_verdicts,
        "reasons": reasons,
    }


def build_candidate_report(
    *,
    candidate_id: str,
    family: str,
    fold_metrics: list[dict[str, Any]],
    safeguards: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic multi-metric candidate report for model selection."""
    folds = list(fold_metrics)
    report = {
        "candidate_id": candidate_id,
        "family": family,
        "n_folds": len(folds),
        "windows": [
            {
                "window_id": fold.get("window_id", f"fold-{idx + 1}"),
                "net_return": round(float(fold.get("net_return", 0.0)), 6),
                "max_drawdown": round(float(fold.get("max_drawdown", 0.0)), 6),
                "directional_accuracy": round(
                    float(fold.get("directional_accuracy", 0.0)),
                    6,
                ),
                "rmse": round(float(fold.get("rmse", 0.0)), 6),
            }
            for idx, fold in enumerate(folds)
        ],
        "metrics": {
            "regression": {
                "mean_rmse": round(_fold_mean(folds, "rmse"), 6),
                "mean_mae": round(_fold_mean(folds, "mae"), 6),
            },
            "directional": {
                "mean_directional_accuracy": round(
                    _fold_mean(folds, "directional_accuracy"), 6
                ),
                "mean_correlation": round(_fold_mean(folds, "correlation"), 6),
            },
            "risk": {
                "mean_net_sharpe": round(_fold_mean(folds, "net_sharpe"), 6),
                "mean_gross_sharpe": round(_fold_mean(folds, "gross_sharpe"), 6),
                "mean_net_return": round(_fold_mean(folds, "net_return"), 6),
                "mean_gross_return": round(_fold_mean(folds, "gross_return"), 6),
                "mean_max_drawdown": round(_fold_mean(folds, "max_drawdown"), 6),
                "total_costs": round(_fold_sum(folds, "total_costs"), 6),
                "total_fee_costs": round(_fold_sum(folds, "total_fee_costs"), 6),
                "total_slippage_costs": round(_fold_sum(folds, "total_slippage_costs"), 6),
            },
        },
    }
    if safeguards is not None:
        report["safeguards"] = safeguards
    return report


def select_champion_report(
    *,
    candidate_reports: dict[str, dict[str, Any]],
    incumbent_id: str | None = None,
    min_directional_accuracy: float = 0.50,
    max_drawdown_floor: float = -0.35,
    min_consecutive_outperformance: int = 2,
) -> dict[str, Any]:
    """Select champion from candidate reports with explicit deterministic rules."""
    if not candidate_reports:
        return {
            "winner": None,
            "incumbent": incumbent_id,
            "promote_candidate": False,
            "rule": {
                "min_directional_accuracy": min_directional_accuracy,
                "max_drawdown_floor": max_drawdown_floor,
                "min_consecutive_outperformance": int(max(min_consecutive_outperformance, 1)),
            },
            "promotion_gate": {
                "pass": False,
                "reasons": ["no-candidates"],
            },
            "reason": "no-candidates",
        }

    ranked: list[tuple[tuple[float, float, float, float, float], str, dict[str, Any]]] = []
    for candidate_id in sorted(candidate_reports):
        report = candidate_reports[candidate_id]
        metrics = report.get("metrics", {})
        directional = metrics.get("directional", {})
        risk = metrics.get("risk", {})
        regression = metrics.get("regression", {})
        safeguards = report.get("safeguards", {})

        directional_accuracy = float(directional.get("mean_directional_accuracy", 0.0))
        net_sharpe = float(risk.get("mean_net_sharpe", 0.0))
        net_return = float(risk.get("mean_net_return", 0.0))
        max_drawdown = float(risk.get("mean_max_drawdown", 0.0))
        mean_rmse = float(regression.get("mean_rmse", 0.0))
        safeguards_pass = (
            bool(safeguards.get("pass"))
            if isinstance(safeguards, dict) and "pass" in safeguards
            else True
        )

        eligible = int(
            directional_accuracy >= min_directional_accuracy
            and max_drawdown >= max_drawdown_floor
            and safeguards_pass
        )
        # Ordered tuple keeps ranking deterministic and transparent.
        score = (
            float(eligible),
            net_sharpe,
            net_return,
            -mean_rmse,
            max_drawdown,
        )
        ranked.append((score, candidate_id, report))

    ranked.sort(reverse=True)
    _, winner_id, winner_report = ranked[0]
    winner_metrics = winner_report.get("metrics", {})
    winner_directional = winner_metrics.get("directional", {})
    winner_risk = winner_metrics.get("risk", {})
    winner_safeguards = winner_report.get("safeguards", {})
    winner_safeguards_pass = (
        bool(winner_safeguards.get("pass"))
        if isinstance(winner_safeguards, dict) and "pass" in winner_safeguards
        else True
    )

    winner_eligible = (
        float(winner_directional.get("mean_directional_accuracy", 0.0))
        >= min_directional_accuracy
        and float(winner_risk.get("mean_max_drawdown", 0.0)) >= max_drawdown_floor
        and winner_safeguards_pass
    )
    promote_candidate = bool(winner_eligible)
    reason = "winner-meets-thresholds" if winner_safeguards_pass else "winner-failed-safeguards"
    promotion_gate: dict[str, Any] = {
        "pass": bool(promote_candidate),
        "min_consecutive_outperformance": int(max(min_consecutive_outperformance, 1)),
        "max_consecutive_outperformance": 0,
        "drawdown_floor": float(max_drawdown_floor),
        "drawdown_ok": bool(float(winner_risk.get("mean_max_drawdown", 0.0)) >= max_drawdown_floor),
        "window_count": int(len(winner_report.get("windows", []))),
        "window_verdicts": [],
        "reasons": [],
    }
    if incumbent_id and winner_id == incumbent_id:
        promote_candidate = False
        reason = "incumbent-retained-by-rule"
        promotion_gate["pass"] = False
        promotion_gate["reasons"] = ["incumbent-retained"]

    if incumbent_id and incumbent_id in candidate_reports and winner_id != incumbent_id:
        incumbent = candidate_reports[incumbent_id]
        incumbent_metrics = incumbent.get("metrics", {})
        incumbent_directional = incumbent_metrics.get("directional", {})
        incumbent_risk = incumbent_metrics.get("risk", {})
        winner_directional_acc = float(winner_directional.get("mean_directional_accuracy", 0.0))
        winner_net_sharpe = float(winner_risk.get("mean_net_sharpe", 0.0))
        winner_net_return = float(winner_risk.get("mean_net_return", 0.0))
        incumbent_directional_acc = float(
            incumbent_directional.get("mean_directional_accuracy", 0.0)
        )
        incumbent_net_sharpe = float(incumbent_risk.get("mean_net_sharpe", 0.0))
        incumbent_net_return = float(incumbent_risk.get("mean_net_return", 0.0))

        promote_candidate = bool(
            winner_eligible
            and winner_directional_acc >= incumbent_directional_acc
            and winner_net_sharpe >= incumbent_net_sharpe
            and winner_net_return >= incumbent_net_return
        )
        reason = (
            "candidate-beats-incumbent" if promote_candidate else "incumbent-retained-by-rule"
        )
        promotion_gate = evaluate_promotion_gate(
            candidate_report=winner_report,
            incumbent_report=incumbent,
            min_consecutive_outperformance=int(max(min_consecutive_outperformance, 1)),
            max_drawdown_floor=float(max_drawdown_floor),
        )
        if not promotion_gate.get("pass", False):
            promote_candidate = False
            reason = "promotion-gate-failed"
        if not winner_safeguards_pass:
            reason = "winner-failed-safeguards"
            promote_candidate = False

    return {
        "winner": winner_id,
        "incumbent": incumbent_id,
        "promote_candidate": promote_candidate,
        "rule": {
            "min_directional_accuracy": min_directional_accuracy,
            "max_drawdown_floor": max_drawdown_floor,
            "min_consecutive_outperformance": int(max(min_consecutive_outperformance, 1)),
        },
        "winner_metrics": winner_metrics,
        "safeguards": winner_safeguards,
        "promotion_gate": promotion_gate,
        "reason": reason,
    }
