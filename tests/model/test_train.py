from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:  # pragma: no cover - dependency guard
    import xgboost  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("xgboost not installed", allow_module_level=True)

from bitbat.model.train import fit_xgb


def test_fit_xgb_trains_and_saves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, 6)), columns=[f"f{i}" for i in range(6)])
    X.attrs["freq"] = "1h"
    X.attrs["horizon"] = "2h"
    labels = np.array(["down", "flat", "up"])
    y = pd.Series(labels[rng.integers(0, 3, size=50)])

    monkeypatch.chdir(tmp_path)
    booster, importance = fit_xgb(X, y, seed=0)

    assert booster is not None
    assert set(importance.keys()) == set(X.columns)
    model_path = Path("models") / "1h_2h" / "xgb.json"
    assert model_path.exists()
