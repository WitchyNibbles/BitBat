from __future__ import annotations

import pandas as pd
import pandas.testing as pd_testing
import pytest

try:  # pragma: no cover - dependency guard
    import vaderSentiment  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("vaderSentiment not installed", allow_module_level=True)

from alpha.features.sentiment import score_vader


def test_score_vader_respects_sentiment_direction() -> None:
    texts = pd.Series(
        [
            "This is an excellent profit and a fantastic success",
            "This is a terrible loss and an awful disaster",
            "Neutral statement without strong sentiment",
            "",
        ]
    )
    scores = score_vader(texts)
    assert scores.iloc[0] > 0.5
    assert scores.iloc[1] < 0
    assert scores.iloc[0] > scores.iloc[2] > scores.iloc[1]
    assert pd.isna(scores.iloc[3])


def test_score_vader_deterministic() -> None:
    texts = pd.Series(["bullish sentiment", "bearish outlook"])
    first = score_vader(texts)
    second = score_vader(texts)
    pd_testing.assert_series_equal(first, second)
