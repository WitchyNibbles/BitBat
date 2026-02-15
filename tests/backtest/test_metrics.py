from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:  # pragma: no cover - dependency guard
    import matplotlib.pyplot as plt  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("matplotlib not installed", allow_module_level=True)

from bitbat.backtest.metrics import summary


def test_summary_outputs_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.date_range(datetime(2024, 1, 1), periods=10, freq="1h")
    equity = pd.Series(np.linspace(1, 1.1, len(idx)), index=idx)
    trades = pd.DataFrame({"position": np.random.choice([0, 1], size=len(idx))}, index=idx)

    monkeypatch.chdir(tmp_path)
    metrics = summary(equity, trades)

    assert "sharpe" in metrics

    metrics_dir = Path("metrics")
    assert (metrics_dir / "backtest_metrics.json").exists()
    assert (metrics_dir / "equity_curve.png").exists()
    assert (metrics_dir / "trades.csv").exists()


def test_summary_contains_cost_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.date_range(datetime(2024, 1, 1), periods=20, freq="1h")
    equity = pd.Series(np.linspace(1, 1.05, len(idx)), index=idx)
    trades = pd.DataFrame(
        {
            "position": np.random.choice([0, 1], size=len(idx)),
            "costs": np.full(len(idx), 0.001),
            "gross_pnl": np.random.normal(0.001, 0.01, size=len(idx)),
            "pnl": np.random.normal(0.0005, 0.01, size=len(idx)),
        },
        index=idx,
    )

    monkeypatch.chdir(tmp_path)
    metrics = summary(equity, trades)

    assert "net_sharpe" in metrics
    assert "gross_sharpe" in metrics
    assert "total_costs" in metrics
    assert "net_return" in metrics
    assert metrics["sharpe"] == metrics["net_sharpe"]
    assert metrics["total_costs"] > 0
