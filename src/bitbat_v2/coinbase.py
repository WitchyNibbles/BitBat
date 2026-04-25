"""Coinbase public market data adapter for BitBat v2."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from .domain import Candle


class CoinbaseMarketDataClient:
    """Fetch public candle data from the Coinbase Exchange REST API.

    Docs used:
    https://docs.cdp.coinbase.com/exchange/reference/exchangerestapi_getproductcandles
    """

    def __init__(
        self,
        base_url: str = "https://api.exchange.coinbase.com",
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=10.0)

    def fetch_candles(
        self,
        product_id: str,
        granularity_seconds: int,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        response = self.client.get(
            f"{self.base_url}/products/{product_id}/candles",
            params={
                "granularity": granularity_seconds,
                "start": start.astimezone(UTC).isoformat(),
                "end": end.astimezone(UTC).isoformat(),
            },
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        rows = response.json()
        candles: list[Candle] = []
        for row in rows:
            if len(row) != 6:
                raise ValueError("Coinbase candle rows must contain 6 values")
            timestamp, low, high, open_, close, volume = row
            low_value = float(low)
            high_value = float(high)
            open_value = float(open_)
            close_value = float(close)
            volume_value = float(volume)
            if min(low_value, high_value, open_value, close_value) <= 0:
                raise ValueError("Coinbase candle prices must be positive")
            if low_value > high_value:
                raise ValueError("Coinbase candle low cannot exceed high")
            if not (low_value <= open_value <= high_value):
                raise ValueError("Coinbase candle open is outside the range")
            if not (low_value <= close_value <= high_value):
                raise ValueError("Coinbase candle close is outside the range")
            if volume_value < 0:
                raise ValueError("Coinbase candle volume cannot be negative")
            candles.append(
                Candle(
                    product_id=product_id,
                    granularity_seconds=granularity_seconds,
                    start=datetime.fromtimestamp(int(timestamp), tz=UTC),
                    open=open_value,
                    high=high_value,
                    low=low_value,
                    close=close_value,
                    volume=volume_value,
                )
            )
        candles.sort(key=lambda candle: candle.start)
        return candles
