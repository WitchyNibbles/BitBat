from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx

from bitbat_v2.coinbase import CoinbaseMarketDataClient


def test_fetch_candles_parses_coinbase_exchange_response() -> None:
    payload = [
        [1_745_572_800, 93_000.0, 95_000.0, 94_000.0, 94_500.0, 11.2],
        [1_745_573_100, 94_400.0, 96_000.0, 94_500.0, 95_700.0, 13.7],
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/products/BTC-USD/candles"
        assert request.url.params["granularity"] == "300"
        return httpx.Response(200, content=json.dumps(payload))

    transport = httpx.MockTransport(handler)
    client = CoinbaseMarketDataClient(
        base_url="https://api.exchange.coinbase.com",
        client=httpx.Client(transport=transport),
    )

    candles = client.fetch_candles(
        product_id="BTC-USD",
        granularity_seconds=300,
        start=datetime(2025, 4, 25, 10, 0, tzinfo=UTC),
        end=datetime(2025, 4, 25, 10, 10, tzinfo=UTC),
    )

    assert len(candles) == 2
    assert candles[0].start < candles[1].start
    assert candles[0].product_id == "BTC-USD"
    assert candles[0].close == 94_500.0
    assert candles[1].volume == 13.7


def test_fetch_candles_rejects_invalid_rows() -> None:
    payload = [
        [1_745_572_800, 96_000.0, 95_000.0, 94_000.0, 94_500.0, 11.2],
    ]

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps(payload))

    transport = httpx.MockTransport(handler)
    client = CoinbaseMarketDataClient(
        base_url="https://api.exchange.coinbase.com",
        client=httpx.Client(transport=transport),
    )

    try:
        client.fetch_candles(
            product_id="BTC-USD",
            granularity_seconds=300,
            start=datetime(2025, 4, 25, 10, 0, tzinfo=UTC),
            end=datetime(2025, 4, 25, 10, 10, tzinfo=UTC),
        )
    except ValueError as exc:
        assert "low cannot exceed high" in str(exc)
    else:
        raise AssertionError("Expected invalid Coinbase candle row to raise ValueError")
