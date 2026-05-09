from __future__ import annotations

import pytest

from bitbat.api.app import create_app
from bitbat.api.cors import ALLOWED_BROWSER_ORIGINS
from tests.api.client import SyncASGIClient


@pytest.mark.parametrize("origin", ALLOWED_BROWSER_ORIGINS)
def test_legacy_api_allows_configured_local_browser_origin_preflight(origin: str) -> None:
    client = SyncASGIClient(create_app())

    response = client.request(
        "OPTIONS",
        "/health",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_legacy_api_rejects_unknown_origin_preflight() -> None:
    client = SyncASGIClient(create_app())

    response = client.request(
        "OPTIONS",
        "/health",
        headers={
            "Origin": "http://evil.example",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers
