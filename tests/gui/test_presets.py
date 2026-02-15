"""Tests for the GUI presets module (SESSION 1)."""

from __future__ import annotations

from bitbat.gui.presets import (
    AGGRESSIVE,
    BALANCED,
    CONSERVATIVE,
    DEFAULT_PRESET,
    PRESETS,
    Preset,
    get_preset,
    list_presets,
)


class TestPresetDataclass:
    def test_conservative_is_preset(self) -> None:
        assert isinstance(CONSERVATIVE, Preset)

    def test_balanced_is_preset(self) -> None:
        assert isinstance(BALANCED, Preset)

    def test_aggressive_is_preset(self) -> None:
        assert isinstance(AGGRESSIVE, Preset)

    def test_presets_have_icons(self) -> None:
        for p in PRESETS.values():
            assert p.icon, f"{p.name} must have an icon"

    def test_presets_have_colors(self) -> None:
        for p in PRESETS.values():
            assert p.color.startswith("#"), f"{p.name} color must be a hex string"

    def test_to_dict_keys(self) -> None:
        d = BALANCED.to_dict()
        assert set(d) == {"freq", "horizon", "tau", "enter_threshold"}

    def test_to_dict_values_types(self) -> None:
        d = BALANCED.to_dict()
        assert isinstance(d["freq"], str)
        assert isinstance(d["horizon"], str)
        assert isinstance(d["tau"], float)
        assert isinstance(d["enter_threshold"], float)

    def test_to_display_keys(self) -> None:
        disp = BALANCED.to_display()
        assert "Update Frequency" in disp
        assert "Forecast Period" in disp
        assert "Movement Sensitivity" in disp
        assert "Confidence Required" in disp

    def test_to_display_values_strings(self) -> None:
        disp = BALANCED.to_display()
        for v in disp.values():
            assert isinstance(v, str)

    def test_conservative_has_higher_threshold(self) -> None:
        assert CONSERVATIVE.enter_threshold > BALANCED.enter_threshold > AGGRESSIVE.enter_threshold

    def test_conservative_has_longer_horizon(self) -> None:
        # Conservative = 24h, Balanced = 4h, Aggressive = 1h
        assert CONSERVATIVE.horizon == "24h"
        assert AGGRESSIVE.horizon == "1h"

    def test_freq_formats(self) -> None:
        p = Preset(
            name="test",
            description="",
            freq="4h",
            horizon="4h",
            tau=0.01,
            enter_threshold=0.65,
            color="#fff",
            icon="🧪",
        )
        assert "4 hours" in p._format_freq()

    def test_freq_daily_format(self) -> None:
        p = Preset(
            name="test",
            description="",
            freq="1d",
            horizon="4h",
            tau=0.01,
            enter_threshold=0.65,
            color="#fff",
            icon="🧪",
        )
        assert "Daily" in p._format_freq()

    def test_tau_sensitivity_labels(self) -> None:
        p_high = Preset(
            name="h",
            description="",
            freq="1h",
            horizon="4h",
            tau=0.025,
            enter_threshold=0.65,
            color="#fff",
            icon="x",
        )
        assert "High" in p_high._format_tau()

        p_low = Preset(
            name="l",
            description="",
            freq="1h",
            horizon="4h",
            tau=0.003,
            enter_threshold=0.65,
            color="#fff",
            icon="x",
        )
        assert "Low" in p_low._format_tau()


class TestPresetRegistry:
    def test_three_presets_defined(self) -> None:
        assert len(PRESETS) == 3

    def test_all_keys_present(self) -> None:
        assert "conservative" in PRESETS
        assert "balanced" in PRESETS
        assert "aggressive" in PRESETS

    def test_default_preset_is_balanced(self) -> None:
        assert DEFAULT_PRESET == "balanced"

    def test_get_preset_known(self) -> None:
        assert get_preset("balanced") is BALANCED
        assert get_preset("conservative") is CONSERVATIVE
        assert get_preset("aggressive") is AGGRESSIVE

    def test_get_preset_case_insensitive(self) -> None:
        assert get_preset("Balanced") is BALANCED
        assert get_preset("CONSERVATIVE") is CONSERVATIVE

    def test_get_preset_unknown_falls_back_to_balanced(self) -> None:
        result = get_preset("nonexistent")
        assert result is BALANCED

    def test_list_presets_returns_copy(self) -> None:
        p1 = list_presets()
        p2 = list_presets()
        assert p1 == p2
        assert p1 is not PRESETS  # must be a copy
