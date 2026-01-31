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

from alpha.backtest.metrics import summary


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
