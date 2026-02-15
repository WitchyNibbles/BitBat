"""Tests for health endpoints (Phase 4, Session 1)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from bitbat.api.app import create_app


@pytest.fixture()
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_status_ok(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_has_version(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert "version" in data
        assert isinstance(data["version"], str)

    def test_has_uptime(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


class TestDetailedHealthEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health/detailed")
        assert resp.status_code == 200

    def test_has_services_list(self, client: TestClient) -> None:
        data = client.get("/health/detailed").json()
        assert "services" in data
        assert isinstance(data["services"], list)

    def test_services_have_name_and_status(self, client: TestClient) -> None:
        data = client.get("/health/detailed").json()
        for svc in data["services"]:
            assert "name" in svc
            assert "status" in svc
            assert svc["status"] in ("ok", "unavailable", "error")

    def test_three_services_checked(self, client: TestClient) -> None:
        data = client.get("/health/detailed").json()
        names = {s["name"] for s in data["services"]}
        assert names == {"database", "model", "dataset"}

    def test_overall_status_is_string(self, client: TestClient) -> None:
        data = client.get("/health/detailed").json()
        assert data["status"] in ("ok", "degraded", "error")
