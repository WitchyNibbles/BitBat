"""Tests for settings endpoints (APIC-01 default fallback, APIC-02 sub-hourly persistence)."""

from __future__ import annotations

import pytest

from bitbat.api.app import create_app
from tests.api.client import SyncASGIClient


@pytest.fixture()
def client(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> SyncASGIClient:
    """Create a test client with user config path pointed at a temp directory."""
    monkeypatch.setattr(
        "bitbat.api.routes.system._USER_CONFIG_PATH",
        tmp_path / "user_config.yaml",
    )
    app = create_app()
    return SyncASGIClient(app)


# ---------------------------------------------------------------------------
# APIC-01: Default fallback to default.yaml
# ---------------------------------------------------------------------------


class TestSettingsDefaultFallback:
    """GET /system/settings returns default.yaml values when no user config exists."""

    def test_get_settings_default_returns_yaml_defaults(
        self, client: SyncASGIClient
    ) -> None:
        """Default freq=5m, horizon=30m from default.yaml (not balanced preset 1h/4h)."""
        resp = client.get("/system/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert data["freq"] == "5m"
        assert data["horizon"] == "30m"
        assert "valid_freqs" in data
        assert "valid_horizons" in data
        assert isinstance(data["valid_freqs"], list)
        assert isinstance(data["valid_horizons"], list)

    def test_get_settings_default_includes_valid_options(
        self, client: SyncASGIClient
    ) -> None:
        """Response includes valid_freqs and valid_horizons with expected values."""
        resp = client.get("/system/settings")
        assert resp.status_code == 200
        data = resp.json()
        # Must contain at least the common sub-hourly and hourly frequencies
        for freq in ["5m", "15m", "30m", "1h", "4h"]:
            assert freq in data["valid_freqs"], f"{freq} missing from valid_freqs"
        # Horizons: at least 15m through 24h (1m too short for a horizon)
        for horizon in ["15m", "30m", "1h", "4h", "24h"]:
            assert horizon in data["valid_horizons"], f"{horizon} missing from valid_horizons"


# ---------------------------------------------------------------------------
# APIC-02: Sub-hourly persistence and validation
# ---------------------------------------------------------------------------


class TestSettingsSubHourlyPersistence:
    """PUT /system/settings accepts and persists sub-hourly freq/horizon values."""

    def test_put_settings_sub_hourly_persists(
        self, client: SyncASGIClient
    ) -> None:
        """PUT freq=15m, horizon=1h then GET returns them unchanged."""
        put_resp = client.request(
            "PUT",
            "/system/settings",
            json={"freq": "15m", "horizon": "1h"},
        )
        assert put_resp.status_code == 200

        get_resp = client.get("/system/settings")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["freq"] == "15m"
        assert data["horizon"] == "1h"

    def test_put_settings_all_sub_hourly_accepted(
        self, client: SyncASGIClient
    ) -> None:
        """All sub-hourly freq values (5m, 15m, 30m) accepted without error."""
        for freq in ["5m", "15m", "30m"]:
            resp = client.request(
                "PUT",
                "/system/settings",
                json={"freq": freq},
            )
            assert resp.status_code == 200, f"freq={freq} rejected with {resp.status_code}"

    def test_put_settings_invalid_freq_rejected(
        self, client: SyncASGIClient
    ) -> None:
        """PUT with unsupported freq value returns 422."""
        resp = client.request(
            "PUT",
            "/system/settings",
            json={"freq": "7m"},
        )
        assert resp.status_code == 422

    def test_put_settings_partial_update_merges(
        self, client: SyncASGIClient
    ) -> None:
        """PUT with only freq does not wipe horizon — partial merge."""
        # First set a known state
        client.request(
            "PUT",
            "/system/settings",
            json={"freq": "1h", "horizon": "4h"},
        )
        # Now update only freq
        client.request(
            "PUT",
            "/system/settings",
            json={"freq": "15m"},
        )
        # Verify horizon is still 4h
        get_resp = client.get("/system/settings")
        data = get_resp.json()
        assert data["freq"] == "15m"
        assert data["horizon"] == "4h"


# ---------------------------------------------------------------------------
# TEST-01 / TEST-02: Preset and sub-hourly settings round-trip
# ---------------------------------------------------------------------------


class TestSettingsPresetRoundTrip:
    """PUT preset or sub-hourly values, then GET and verify round-trip."""

    def test_put_preset_scalper_round_trip(
        self, client: SyncASGIClient
    ) -> None:
        """PUT preset=scalper resolves to 5m/30m/0.003/0.55 on GET."""
        put_resp = client.request(
            "PUT",
            "/system/settings",
            json={"preset": "scalper"},
        )
        assert put_resp.status_code == 200

        get_resp = client.get("/system/settings")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["freq"] == "5m"
        assert data["horizon"] == "30m"
        assert data["tau"] == 0.003
        assert data["enter_threshold"] == 0.55
        assert data["preset"] == "scalper"

    def test_put_preset_swing_round_trip(
        self, client: SyncASGIClient
    ) -> None:
        """PUT preset=swing resolves to 15m/1h/0.007/0.60 on GET."""
        put_resp = client.request(
            "PUT",
            "/system/settings",
            json={"preset": "swing"},
        )
        assert put_resp.status_code == 200

        get_resp = client.get("/system/settings")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["freq"] == "15m"
        assert data["horizon"] == "1h"
        assert data["tau"] == 0.007
        assert data["enter_threshold"] == 0.60
        assert data["preset"] == "swing"

    def test_sub_hourly_freq_horizon_round_trip(
        self, client: SyncASGIClient
    ) -> None:
        """PUT explicit freq=5m/horizon=30m persists without preset resolution."""
        put_resp = client.request(
            "PUT",
            "/system/settings",
            json={"freq": "5m", "horizon": "30m"},
        )
        assert put_resp.status_code == 200

        get_resp = client.get("/system/settings")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["freq"] == "5m"
        assert data["horizon"] == "30m"
