"""Performance metric helpers for autonomous monitoring."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

import numpy as np

from bitbat.autonomous.models import PredictionOutcome


class PerformanceMetrics:
    """Calculate monitoring metrics from realized prediction outcomes."""

    def __init__(self, predictions: Sequence[PredictionOutcome]) -> None:
        self.predictions = list(predictions)
        self.realized = [pred for pred in self.predictions if pred.actual_return is not None]

    def _returns(self) -> np.ndarray:
        return np.array([float(pred.actual_return) for pred in self.realized], dtype="float64")

    def _correct_flags(self) -> list[bool]:
        flags: list[bool] = []
        for pred in self.realized:
            if pred.correct is None:
                continue
            flags.append(bool(pred.correct))
        return flags

    def total_predictions(self) -> int:
        """Return count of all predictions in window."""
        return len(self.predictions)

    def realized_predictions(self) -> int:
        """Return count of predictions with realized outcomes."""
        return len(self.realized)

    def hit_rate(self) -> float:
        """Return directional hit rate."""
        flags = self._correct_flags()
        if not flags:
            return 0.0
        return float(np.mean(np.array(flags, dtype="float64")))

    def sharpe_ratio(self) -> float:
        """Return simple Sharpe ratio estimate."""
        returns = self._returns()
        if returns.size < 2:
            return 0.0
        std = float(np.std(returns, ddof=1))
        if std == 0.0:
            return 0.0
        return float(np.mean(returns) / std * sqrt(returns.size))

    def average_return(self) -> float:
        """Return average realized return."""
        returns = self._returns()
        if returns.size == 0:
            return 0.0
        return float(np.mean(returns))

    def _longest_streak(self, target: bool) -> int:
        flags = self._correct_flags()
        best = 0
        current = 0
        for flag in flags:
            if flag is target:
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best

    def win_streak(self) -> int:
        """Return longest streak of correct predictions."""
        return self._longest_streak(True)

    def lose_streak(self) -> int:
        """Return longest streak of incorrect predictions."""
        return self._longest_streak(False)

    def current_streak(self) -> tuple[int, str]:
        """Return current streak length and kind (`win`, `loss`, or `none`)."""
        flags = self._correct_flags()
        if not flags:
            return (0, "none")

        last = flags[-1]
        count = 0
        for flag in reversed(flags):
            if flag is last:
                count += 1
            else:
                break
        return (count, "win" if last else "loss")

    def calibration_score(self) -> dict[str, float]:
        """Return confidence calibration summary."""
        rows: list[tuple[float, bool]] = []
        for pred in self.realized:
            if pred.correct is None:
                continue
            # In regression mode p_up/p_down may be None; skip calibration.
            if pred.p_up is None and pred.p_down is None:
                continue
            p_flat = (
                float(pred.p_flat)
                if pred.p_flat is not None
                else max(0.0, 1.0 - float(pred.p_up or 0.0) - float(pred.p_down or 0.0))
            )
            confidence = max(float(pred.p_up or 0.0), float(pred.p_down or 0.0), p_flat)
            rows.append((confidence, bool(pred.correct)))

        if not rows:
            return {
                "high_confidence_count": 0.0,
                "high_confidence_accuracy": 0.0,
                "mean_confidence": 0.0,
                "calibration_error": 0.0,
            }

        high_conf = [(conf, corr) for conf, corr in rows if conf >= 0.6]
        if not high_conf:
            return {
                "high_confidence_count": 0.0,
                "high_confidence_accuracy": 0.0,
                "mean_confidence": 0.0,
                "calibration_error": 0.0,
            }

        mean_conf = float(np.mean([conf for conf, _ in high_conf]))
        accuracy = float(np.mean([1.0 if corr else 0.0 for _, corr in high_conf]))
        return {
            "high_confidence_count": float(len(high_conf)),
            "high_confidence_accuracy": accuracy,
            "mean_confidence": mean_conf,
            "calibration_error": abs(mean_conf - accuracy),
        }

    def _predicted_returns(self) -> np.ndarray:
        """Return predicted_return values for realized predictions that have it."""
        return np.array(
            [
                float(pred.predicted_return)
                for pred in self.realized
                if pred.predicted_return is not None
            ],
            dtype="float64",
        )

    def _actual_returns_for_predicted(self) -> np.ndarray:
        """Return actual_return values paired with predictions that have predicted_return."""
        return np.array(
            [
                float(pred.actual_return)
                for pred in self.realized
                if pred.predicted_return is not None
            ],
            dtype="float64",
        )

    def mae(self) -> float:
        """Mean absolute error between predicted_return and actual_return."""
        predicted = self._predicted_returns()
        actual = self._actual_returns_for_predicted()
        if predicted.size == 0:
            return 0.0
        return float(np.mean(np.abs(predicted - actual)))

    def rmse(self) -> float:
        """Root mean squared error between predicted_return and actual_return."""
        predicted = self._predicted_returns()
        actual = self._actual_returns_for_predicted()
        if predicted.size == 0:
            return 0.0
        return float(np.sqrt(np.mean((predicted - actual) ** 2)))

    def directional_accuracy(self) -> float:
        """Fraction of realized predictions whose direction matched the outcome.

        Regression-era rows carry ``predicted_return`` and can still be scored
        by return sign. Classification rows only store direction labels and
        class probabilities, so we fall back to comparing the realized and
        predicted directions directly.
        """
        predicted = self._predicted_returns()
        actual = self._actual_returns_for_predicted()
        if predicted.size > 0:
            return float(np.mean(np.sign(predicted) == np.sign(actual)))

        matches: list[bool] = []
        for pred in self.realized:
            if pred.actual_direction is None:
                continue
            matches.append(pred.predicted_direction == pred.actual_direction)

        if not matches:
            return 0.0
        return float(np.mean(np.array(matches, dtype="float64")))

    def max_drawdown(self) -> float:
        """Return maximum drawdown on cumulative return path."""
        returns = self._returns()
        if returns.size == 0:
            return 0.0

        equity = np.cumprod(1.0 + returns)
        peaks = np.maximum.accumulate(equity)
        drawdowns = (equity - peaks) / peaks
        return float(abs(np.min(drawdowns)))

    def to_dict(self) -> dict[str, float | int | str]:
        """Return all metrics in one dictionary."""
        streak_count, streak_type = self.current_streak()
        calibration = self.calibration_score()

        return {
            "total_predictions": self.total_predictions(),
            "realized_predictions": self.realized_predictions(),
            "hit_rate": self.hit_rate(),
            "sharpe_ratio": self.sharpe_ratio(),
            "average_return": self.average_return(),
            "max_drawdown": self.max_drawdown(),
            "win_streak": self.win_streak(),
            "lose_streak": self.lose_streak(),
            "current_streak_count": streak_count,
            "current_streak_type": streak_type,
            "high_confidence_count": calibration["high_confidence_count"],
            "high_confidence_accuracy": calibration["high_confidence_accuracy"],
            "calibration_mean_confidence": calibration["mean_confidence"],
            "calibration_error": calibration["calibration_error"],
            "calibration_score": calibration["high_confidence_accuracy"],
            "mae": self.mae(),
            "rmse": self.rmse(),
            "directional_accuracy": self.directional_accuracy(),
        }
