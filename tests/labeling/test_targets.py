from __future__ import annotations

import pandas as pd
import pandas.testing as pd_testing
import pytest

from bitbat.labeling.targets import classify


def test_classify_threshold_edges() -> None:
    series = pd.Series([0.02, 0.01, 0.0, -0.01, -0.02])
    labels = classify(series, tau=0.01)
    expected = pd.Series(["up", "flat", "flat", "flat", "down"], index=series.index, name="target")
    pd_testing.assert_series_equal(labels, expected)


def test_classify_symmetry() -> None:
    values = pd.Series([0.015, -0.015, 0.002, -0.002])
    labels = classify(values, tau=0.01)
    expected = pd.Series(["up", "down", "flat", "flat"], index=values.index, name="target")
    pd_testing.assert_series_equal(labels, expected)


def test_classify_negative_tau_rejected() -> None:
    with pytest.raises(ValueError):
        classify(pd.Series([0.0]), tau=-0.1)
