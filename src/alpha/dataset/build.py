"""Dataset assembly entrypoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alpha.contracts import ensure_feature_contract
from alpha.features.price import (
    atr,
    lagged_returns,
    macd,
    obv,
    rolling_std,
    rolling_z,
)
from alpha.features.sentiment import aggregate as aggregate_sentiment
from alpha.labeling.returns import forward_return
from alpha.labeling.targets import classify
from alpha.timealign.calendar import ensure_utc


@dataclass
class DatasetMeta:
    columns: list[str]
    freq: str
    horizon: str
    tau: float
    start: str
    end: str
    rows: int
    positives: int
    negatives: int
    flats: int
    seed: int | None
    version: str


def _load_parquet(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    return pd.read_parquet(target)


def _generate_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    close = prices["close"]
    features = lagged_returns(close)
    features["rolling_std_24"] = rolling_std(close)
    features["rolling_z_24"] = rolling_z(close, 24)
    features = features.join(atr(prices, 14), how="left")
    features = features.join(macd(close), how="left")
    features["obv"] = obv(close, prices["volume"])
    return features


def build_xy(
    prices_parquet: str | Path,
    news_parquet: str | Path | None,
    freq: str,
    horizon: str,
    tau: float,
    start: str,
    end: str,
    *,
    enable_sentiment: bool = True,
    output_root: str | Path | None = None,
    seed: int | None = None,
    version: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, DatasetMeta]:
    """Build the primary dataset (features + labels) used for model training.

    This is the main dataset builder in the pipeline. It assembles price
    features (and optional sentiment features), aligns labels, enforces the
    feature contract, and writes the resulting dataset + metadata to disk.
    """
    prices = _load_parquet(prices_parquet)

    prices = ensure_utc(prices, "timestamp_utc").set_index("timestamp_utc").sort_index()

    price_features = _generate_price_features(prices)

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
        features = price_features.join(sentiment_features, how="left")
    else:
        features = price_features.copy()
    features = features.dropna()

    rename_mapping = {
        column: column if column.startswith("feat_") else f"feat_{column}"
        for column in features.columns
    }
    features = features.rename(columns=rename_mapping)

    y_returns = forward_return(prices["close"].to_frame(), horizon)
    y_returns = y_returns.loc[features.index]
    labels = classify(y_returns, tau)

    valid_mask = labels.notna()
    features = features.loc[valid_mask]
    labels = labels.loc[valid_mask]
    y_returns = y_returns.loc[valid_mask]

    first_idx = features.index.min()
    idx_start = max(pd.Timestamp(start), first_idx)
    idx_end = pd.Timestamp(end)
    features = features.loc[idx_start:idx_end]
    labels = labels.loc[idx_start:idx_end]
    y_returns = y_returns.loc[idx_start:idx_end]

    dataset = features.copy()
    dataset["timestamp_utc"] = dataset.index
    dataset["label"] = labels.astype("string")
    dataset["r_forward"] = y_returns.astype("float64")
    dataset = dataset.reset_index(drop=True)
    dataset = ensure_feature_contract(
        dataset,
        require_label=True,
        require_forward_return=True,
        require_features_full=enable_sentiment,
    )
    dataset = dataset.sort_values("timestamp_utc")

    feature_cols = [col for col in dataset.columns if col.startswith("feat_")]
    indexed_dataset = dataset.set_index("timestamp_utc")
    X = indexed_dataset[feature_cols]
    y = indexed_dataset["label"]

    meta = DatasetMeta(
        columns=list(X.columns),
        freq=freq,
        horizon=horizon,
        tau=tau,
        start=start,
        end=end,
        rows=len(X),
        positives=int((y == "up").sum()),
        negatives=int((y == "down").sum()),
        flats=int((y == "flat").sum()),
        seed=seed,
        version=version or "unknown",
    )

    output_base = Path(output_root) if output_root is not None else Path("data")
    output_dir = output_base / "features" / f"{freq}_{horizon}"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_dir / "dataset.parquet", index=False)
    (output_dir / "meta.json").write_text(json.dumps(meta.__dict__), encoding="utf-8")

    return X, y, meta
