"""Lightweight ASGI test client that avoids TestClient's blocking portal."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from fastapi import FastAPI


class SyncASGIClient:
    """Sync wrapper over httpx.AsyncClient + ASGITransport for pytest tests."""

    def __init__(self, app: FastAPI, base_url: str = "http://testserver") -> None:
        self._app = app
        self._base_url = base_url

    async def _request_async(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        transport = httpx.ASGITransport(app=self._app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url=self._base_url,
            follow_redirects=True,
        ) as client:
            return await client.request(method, url, **kwargs)

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        return asyncio.run(self._request_async(method, url, **kwargs))

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)
