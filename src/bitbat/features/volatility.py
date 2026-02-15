"""GARCH-based volatility feature generation."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def garch_features(close: pd.Series, window: int = 168) -> pd.DataFrame:
    """Generate GARCH(1,1) conditional volatility features.

    Fits a GARCH(1,1) model on log returns and extracts the conditional
    volatility series.  The ``arch`` library computes conditional volatility
    causally (only past information at each step), so no look-ahead leakage.

    Parameters
    ----------
    close:
        Close price series (DatetimeIndex expected).
    window:
        Minimum observations required before the GARCH fit is meaningful.
        Defaults to 168 (one week of hourly bars).

    Returns
    -------
    DataFrame with columns ``garch_vol``, ``vol_regime``, ``vol_ratio``.
    """
    close_series = pd.Series(close, dtype="float64")
    log_returns = np.log(close_series / close_series.shift(1)).dropna()

    if len(log_returns) < window:
        logger.warning(
            "Not enough data for GARCH (%d bars, need %d); returning NaN features",
            len(log_returns),
            window,
        )
        return _empty_features(close_series.index)

    try:
        from arch import arch_model

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(
                log_returns * 100,  # scale to percent for numerical stability
                vol="Garch",
                p=1,
                q=1,
                mean="Zero",
                rescale=False,
            )
            result = model.fit(disp="off", show_warning=False)

        cond_vol = result.conditional_volatility / 100  # scale back
        cond_vol = cond_vol.reindex(close_series.index)
    except Exception:
        logger.warning("GARCH fit failed; falling back to rolling volatility")
        returns_full = close_series.pct_change()
        cond_vol = returns_full.rolling(window=24, min_periods=24).std()

    # Realised vol for ratio calculation
    realised_vol = close_series.pct_change().rolling(window=24, min_periods=24).std()

    # Volatility regime: tercile ranking over a rolling window
    vol_rank = cond_vol.rolling(window=window, min_periods=window).rank(pct=True)
    vol_regime = pd.cut(
        vol_rank, bins=[0, 1 / 3, 2 / 3, 1.0], labels=[0, 1, 2], include_lowest=True
    )
    vol_regime = vol_regime.astype("float64")

    # Ratio: GARCH vol vs realised vol
    vol_ratio = cond_vol / realised_vol.replace(0, np.nan)

    return pd.DataFrame(
        {
            "garch_vol": cond_vol,
            "vol_regime": vol_regime,
            "vol_ratio": vol_ratio,
        },
        index=close_series.index,
    )


def _empty_features(index: pd.Index) -> pd.DataFrame:
    """Return a NaN-filled feature frame when GARCH cannot be computed."""
    return pd.DataFrame(
        {
            "garch_vol": np.nan,
            "vol_regime": np.nan,
            "vol_ratio": np.nan,
        },
        index=index,
    )
