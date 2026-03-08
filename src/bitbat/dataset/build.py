"""Dataset assembly entrypoints."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bitbat.contracts import ensure_feature_contract
from bitbat.features.price import (
    atr,
    lagged_returns,
    macd,
    obv,
    obv_fold_aware,
    rolling_std,
    rolling_z,
    rsi,
)
from bitbat.features.sentiment import aggregate as aggregate_sentiment
from bitbat.labeling.targets import direction_from_prices
from bitbat.labeling.triple_barrier import triple_barrier
from bitbat.timealign.asof import align_features_asof
from bitbat.timealign.calendar import ensure_utc

logger = logging.getLogger(__name__)


@dataclass
class DatasetMeta:
    columns: list[str]
    freq: str
    horizon: str
    label_mode: str
    start: str
    end: str
    rows: int
    target_mean: float
    target_std: float
    target_min: float
    target_max: float
    seed: int | None
    version: str


def _load_parquet(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    return pd.read_parquet(target)


def generate_price_features(
    prices: pd.DataFrame,
    *,
    enable_garch: bool = False,
    freq: str | None = None,
    fold_boundaries: list[int] | None = None,
) -> pd.DataFrame:
    close = prices["close"]
    features = lagged_returns(close, freq=freq)
    std_result = rolling_std(close, freq=freq)
    features[std_result.name] = std_result
    z_result = rolling_z(close, freq=freq)
    features[z_result.name] = z_result
    features = features.join(atr(prices, freq=freq), how="left")
    features = features.join(macd(close), how="left")
    if fold_boundaries:
        features["obv"] = obv_fold_aware(close, prices["volume"], fold_boundaries)
    else:
        features["obv"] = obv(close, prices["volume"])

    rsi_result = rsi(close, freq=freq)
    features[rsi_result.name] = rsi_result

    if enable_garch:
        try:
            from bitbat.features.volatility import garch_features

            vol_feats = garch_features(close, freq=freq)
            features = features.join(vol_feats, how="left")
        except ImportError:
            logger.warning("arch library not installed; skipping GARCH features")
        except Exception:
            logger.warning("GARCH feature generation failed; skipping", exc_info=True)

    return features


def join_auxiliary_features(
    features: pd.DataFrame,
    *,
    macro_parquet: str | Path | None = None,
    onchain_parquet: str | Path | None = None,
    freq: str = "1h",
) -> pd.DataFrame:
    """Join macro and on-chain features into the main feature frame."""
    target_index = features.index

    if macro_parquet is not None:
        try:
            from bitbat.features.macro import generate_macro_features

            macro_raw = _load_parquet(macro_parquet)
            macro_feats = generate_macro_features(macro_raw, freq=freq)
            macro_feats = align_features_asof(
                target_index,
                macro_feats,
                source_name="macro",
            )
            features = features.join(macro_feats, how="left")
            logger.info("Joined %d macro feature columns", len(macro_feats.columns))
        except Exception:
            logger.warning("Macro feature generation failed; skipping", exc_info=True)

    if onchain_parquet is not None:
        try:
            from bitbat.features.onchain import generate_onchain_features

            onchain_raw = _load_parquet(onchain_parquet)
            onchain_feats = generate_onchain_features(onchain_raw, freq=freq)
            onchain_feats = align_features_asof(
                target_index,
                onchain_feats,
                source_name="onchain",
            )
            features = features.join(onchain_feats, how="left")
            logger.info("Joined %d on-chain feature columns", len(onchain_feats.columns))
        except Exception:
            logger.warning("On-chain feature generation failed; skipping", exc_info=True)

    return features


def build_xy(
    prices_parquet: str | Path,
    news_parquet: str | Path | None,
    freq: str,
    horizon: str,
    start: str,
    end: str,
    *,
    tau: float | None = None,
    enable_sentiment: bool = True,
    enable_garch: bool = False,
    macro_parquet: str | Path | None = None,
    onchain_parquet: str | Path | None = None,
    label_mode: str = "return_direction",
    barrier_take_profit: float | None = None,
    barrier_stop_loss: float | None = None,
    output_root: str | Path | None = None,
    seed: int | None = None,
    version: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, DatasetMeta]:
    """Build the primary dataset (features + regression targets).

    This is the main dataset builder in the pipeline. It assembles price
    features (and optional sentiment, GARCH, macro, and on-chain features),
    computes targets from one canonical label path (default return+direction,
    optional triple-barrier), enforces the feature contract, and writes the
    resulting dataset + metadata to disk.
    """
    prices = _load_parquet(prices_parquet)

    prices = ensure_utc(prices, "timestamp_utc").set_index("timestamp_utc").sort_index()

    price_features = generate_price_features(prices, enable_garch=enable_garch, freq=freq)

    if enable_sentiment:
        if news_parquet is None:
            raise ValueError("news_parquet is required when enable_sentiment=True")
        news = _load_parquet(news_parquet)
        news = ensure_utc(news, "published_utc").sort_values("published_utc")
        sentiment_features = aggregate_sentiment(
            news_df=news,
            bar_df=prices.reset_index()[["timestamp_utc"]],
            freq=freq,
        )
        sentiment_features = align_features_asof(
            price_features.index,
            sentiment_features,
            source_name="sentiment",
        )
        features = price_features.join(sentiment_features, how="left")
    else:
        features = price_features.copy()

    # Join auxiliary feature sources (macro, on-chain)
    features = join_auxiliary_features(
        features,
        macro_parquet=macro_parquet,
        onchain_parquet=onchain_parquet,
        freq=freq,
    )

    features = features.dropna()

    rename_mapping = {
        column: column if column.startswith("feat_") else f"feat_{column}"
        for column in features.columns
    }
    features = features.rename(columns=rename_mapping)

    resolved_label_mode = str(label_mode).strip().lower()
    if resolved_label_mode in {"return_direction", "direction"}:
        label_frame = direction_from_prices(
            prices["close"].to_frame(),
            horizon=horizon,
            tau=0.0 if tau is None else tau,
            return_name="r_forward",
            label_name="label",
        )
        contract_label_mode = "direction"
    elif resolved_label_mode == "triple_barrier":
        tp = barrier_take_profit if barrier_take_profit is not None else (tau or 0.01)
        sl = barrier_stop_loss if barrier_stop_loss is not None else tp
        label_frame = triple_barrier(
            prices["close"].to_frame(),
            horizon=horizon,
            take_profit=tp,
            stop_loss=sl,
            return_name="r_forward",
            label_name="label",
        )
        contract_label_mode = "triple_barrier"
    else:
        raise ValueError(
            f"Unsupported label_mode '{label_mode}'. "
            "Use 'return_direction' or 'triple_barrier'."
        )

    label_frame = label_frame.loc[features.index]

    valid_mask = label_frame["r_forward"].notna() & label_frame["label"].notna()
    features = features.loc[valid_mask]
    y_returns = label_frame.loc[valid_mask, "r_forward"].astype("float64")
    y_direction = label_frame.loc[valid_mask, "label"].astype("string")

    first_idx = features.index.min()
    idx_start = max(pd.Timestamp(start), first_idx)
    idx_end = pd.Timestamp(end)
    features = features.loc[idx_start:idx_end]
    y_returns = y_returns.loc[idx_start:idx_end]
    y_direction = y_direction.loc[idx_start:idx_end]

    dataset = features.copy()
    dataset["timestamp_utc"] = dataset.index
    dataset["label"] = y_direction
    dataset["r_forward"] = y_returns.astype("float64")
    dataset = dataset.reset_index(drop=True)
    dataset = ensure_feature_contract(
        dataset,
        require_label=True,
        require_forward_return=True,
        require_features_full=enable_sentiment,
        label_mode=contract_label_mode,
    )
    dataset = dataset.sort_values("timestamp_utc")

    feature_cols = [col for col in dataset.columns if col.startswith("feat_")]
    indexed_dataset = dataset.set_index("timestamp_utc")
    X = indexed_dataset[feature_cols]
    y = indexed_dataset["r_forward"].astype("float64")

    meta = DatasetMeta(
        columns=list(X.columns),
        freq=freq,
        horizon=horizon,
        label_mode=resolved_label_mode,
        start=start,
        end=end,
        rows=len(X),
        target_mean=float(y.mean()),
        target_std=float(y.std()),
        target_min=float(y.min()),
        target_max=float(y.max()),
        seed=seed,
        version=version or "unknown",
    )

    output_base = Path(output_root) if output_root is not None else Path("data")
    output_dir = output_base / "features" / f"{freq}_{horizon}"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_dir / "dataset.parquet", index=False)
    (output_dir / "meta.json").write_text(json.dumps(meta.__dict__), encoding="utf-8")

    return X, y, meta


# Backward-compatibility aliases — keep old private names working for any
# untracked callers while the public names are the canonical API.
_generate_price_features = generate_price_features
_join_auxiliary_features = join_auxiliary_features
