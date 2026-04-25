"""Configuration for the clean-room BitBat v2 runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class BitBatV2Config:
    """Runtime configuration for the BitBat operator console."""

    database_url: str = "sqlite:///data/bitbat_v2.db"
    product_id: str = "BTC-USD"
    granularity_seconds: int = 300
    starting_cash_usd: float = 10_000.0
    order_size_btc: float = 0.01
    max_position_size_btc: float = 0.05
    signal_threshold: float = 0.0015
    sell_signal_threshold: float = 0.0012
    stale_after_seconds: int = 180
    trend_lookback_candles: int = 12
    short_trend_lookback_candles: int = 3
    max_range_ratio: float = 0.012
    min_body_strength: float = 0.35
    model_name: str = "ritual-momentum-v1"
    venue: str = "coinbase"
    demo_mode: bool = False
    operator_token: str | None = None
    autorun_enabled: bool = False
    autorun_interval_seconds: int = 15

    @classmethod
    def from_env(cls) -> BitBatV2Config:
        return cls(
            database_url=os.getenv("BITBAT_V2_DATABASE_URL", cls.database_url),
            product_id=os.getenv("BITBAT_V2_PRODUCT_ID", cls.product_id),
            granularity_seconds=int(
                os.getenv("BITBAT_V2_GRANULARITY_SECONDS", str(cls.granularity_seconds))
            ),
            starting_cash_usd=float(
                os.getenv("BITBAT_V2_STARTING_CASH_USD", str(cls.starting_cash_usd))
            ),
            order_size_btc=float(os.getenv("BITBAT_V2_ORDER_SIZE_BTC", str(cls.order_size_btc))),
            max_position_size_btc=float(
                os.getenv("BITBAT_V2_MAX_POSITION_BTC", str(cls.max_position_size_btc))
            ),
            signal_threshold=float(
                os.getenv("BITBAT_V2_SIGNAL_THRESHOLD", str(cls.signal_threshold))
            ),
            sell_signal_threshold=float(
                os.getenv(
                    "BITBAT_V2_SELL_SIGNAL_THRESHOLD",
                    str(cls.sell_signal_threshold),
                )
            ),
            stale_after_seconds=int(
                os.getenv("BITBAT_V2_STALE_AFTER_SECONDS", str(cls.stale_after_seconds))
            ),
            trend_lookback_candles=int(
                os.getenv(
                    "BITBAT_V2_TREND_LOOKBACK_CANDLES",
                    str(cls.trend_lookback_candles),
                )
            ),
            short_trend_lookback_candles=int(
                os.getenv(
                    "BITBAT_V2_SHORT_TREND_LOOKBACK_CANDLES",
                    str(cls.short_trend_lookback_candles),
                )
            ),
            max_range_ratio=float(
                os.getenv("BITBAT_V2_MAX_RANGE_RATIO", str(cls.max_range_ratio))
            ),
            min_body_strength=float(
                os.getenv("BITBAT_V2_MIN_BODY_STRENGTH", str(cls.min_body_strength))
            ),
            model_name=os.getenv("BITBAT_V2_MODEL_NAME", cls.model_name),
            venue=os.getenv("BITBAT_V2_VENUE", cls.venue),
            demo_mode=os.getenv(
                "BITBAT_V2_DEMO_MODE",
                str(cls.demo_mode).lower(),
            ).lower()
            == "true",
            operator_token=os.getenv("BITBAT_V2_OPERATOR_TOKEN"),
            autorun_enabled=os.getenv(
                "BITBAT_V2_AUTORUN_ENABLED",
                str(cls.autorun_enabled).lower(),
            ).lower()
            == "true",
            autorun_interval_seconds=int(
                os.getenv(
                    "BITBAT_V2_AUTORUN_INTERVAL_SECONDS",
                    str(cls.autorun_interval_seconds),
                )
            ),
        )
