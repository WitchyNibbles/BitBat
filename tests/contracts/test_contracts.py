from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from bitbat.contracts import (
    ContractError,
    ensure_feature_contract,
    ensure_predictions_contract,
)


def test_feature_contract_requires_feat_prefix() -> None:
    frame = pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=3, freq="1h"),
        "value": [1.0, 2.0, 3.0],
    })

    with pytest.raises(ContractError):
        ensure_feature_contract(frame, require_label=False, require_forward_return=False)


def test_feature_contract_happy_path() -> None:
    frame = pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=2, freq="1h"),
        "feat_a": [0.1, 0.2],
        "label": ["up", "down"],
        "r_forward": [0.05, -0.02],
    })

    validated = ensure_feature_contract(
        frame,
        require_label=True,
        require_forward_return=True,
        require_features_full=False,
    )
    assert list(validated.columns) == ["timestamp_utc", "feat_a", "label", "r_forward"]


def test_feature_contract_full_requires_sentiment() -> None:
    frame = pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=1, freq="1h"),
        "feat_price": [0.1],
    })

    with pytest.raises(ContractError):
        ensure_feature_contract(
            frame,
            require_label=False,
            require_forward_return=False,
            require_features_full=True,
        )


def test_predictions_contract_normalises_types() -> None:
    frame = pd.DataFrame({
        "timestamp_utc": [datetime(2024, 1, 1, 12, 0, 0)],
        "p_up": ["0.7"],
        "p_down": [0.2],
        "horizon": ["4h"],
        "freq": ["1h"],
        "model_version": ["0.1"],
        "realized_r": [None],
        "realized_label": [None],
    })

    validated = ensure_predictions_contract(frame)
    assert list(validated.columns) == [
        "timestamp_utc",
        "p_up",
        "p_down",
        "horizon",
        "freq",
        "model_version",
        "realized_r",
        "realized_label",
    ]
    assert validated["timestamp_utc"].dt.tz is None
