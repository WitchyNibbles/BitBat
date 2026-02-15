"""
User-friendly configuration presets for the BitBat GUI.

Translates simple preset names into technical parameters, and
provides plain-language display helpers for non-technical users.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Preset:
    """A named configuration preset."""

    name: str
    description: str
    freq: str
    horizon: str
    tau: float
    enter_threshold: float
    color: str
    icon: str

    def to_dict(self) -> dict[str, Any]:
        """Return the underlying technical config dict."""
        return {
            "freq": self.freq,
            "horizon": self.horizon,
            "tau": self.tau,
            "enter_threshold": self.enter_threshold,
        }

    def to_display(self) -> dict[str, str]:
        """Return a user-friendly label → value dict for UI rendering."""
        return {
            "Update Frequency": self._format_freq(),
            "Forecast Period": self._format_horizon(),
            "Movement Sensitivity": self._format_tau(),
            "Confidence Required": f"{self.enter_threshold:.0%}",
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_freq(self) -> str:
        mapping = {"1h": "Every hour", "4h": "Every 4 hours", "1d": "Daily"}
        return mapping.get(self.freq, self.freq)

    def _format_horizon(self) -> str:
        mapping = {"1h": "1 hour ahead", "4h": "4 hours ahead", "24h": "1 day ahead"}
        return mapping.get(self.horizon, self.horizon)

    def _format_tau(self) -> str:
        if self.tau >= 0.02:
            return "High — only very clear signals"
        elif self.tau >= 0.01:
            return "Medium — balanced"
        else:
            return "Low — more sensitive"


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

CONSERVATIVE = Preset(
    name="Conservative",
    description="Fewer predictions with higher accuracy focus. Best for risk-averse users.",
    freq="1h",
    horizon="24h",
    tau=0.02,
    enter_threshold=0.75,
    color="#3B82F6",  # blue
    icon="🛡️",
)

BALANCED = Preset(
    name="Balanced",
    description="Good mix of prediction frequency and accuracy. Recommended for most users.",
    freq="1h",
    horizon="4h",
    tau=0.01,
    enter_threshold=0.65,
    color="#10B981",  # green
    icon="⚖️",
)

AGGRESSIVE = Preset(
    name="Aggressive",
    description="More frequent predictions with higher risk. Best for active traders.",
    freq="1h",
    horizon="1h",
    tau=0.005,
    enter_threshold=0.55,
    color="#EF4444",  # red
    icon="🚀",
)

PRESETS: dict[str, Preset] = {
    "conservative": CONSERVATIVE,
    "balanced": BALANCED,
    "aggressive": AGGRESSIVE,
}

DEFAULT_PRESET = "balanced"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_preset(name: str) -> Preset:
    """Return the preset for *name*, falling back to BALANCED if not found."""
    return PRESETS.get(name.lower(), BALANCED)


def list_presets() -> dict[str, Preset]:
    """Return a copy of the full preset registry."""
    return PRESETS.copy()
