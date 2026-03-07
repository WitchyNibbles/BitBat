"""
User-friendly configuration presets for the BitBat pipeline.

Translates simple preset names into technical parameters, and
provides plain-language display helpers for non-technical users.

Canonical implementation — re-exported by bitbat.gui.presets for
backward compatibility with GUI-layer callers.
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
        mapping = {
            "5m": "Every 5 min",
            "15m": "Every 15 min",
            "30m": "Every 30 min",
            "1h": "Every hour",
            "4h": "Every 4 hours",
            "1d": "Daily",
        }
        return mapping.get(self.freq, self.freq)

    def _format_horizon(self) -> str:
        mapping = {
            "15m": "15 min ahead",
            "30m": "30 min ahead",
            "1h": "1 hour ahead",
            "4h": "4 hours ahead",
            "24h": "1 day ahead",
        }
        return mapping.get(self.horizon, self.horizon)

    def _format_tau(self) -> str:
        if self.tau >= 0.02:
            return "High — only very clear signals"
        if self.tau >= 0.01:
            return "Medium — balanced"
        return "Low — more sensitive"


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

SCALPER = Preset(
    name="Scalper",
    description=(
        "Rapid sub-hourly predictions for active scalp trading."
        " Highest frequency, shortest horizon."
    ),
    freq="5m",
    horizon="30m",
    tau=0.003,
    enter_threshold=0.55,
    color="#F59E0B",  # amber
    icon="\u26a1",
)

CONSERVATIVE = Preset(
    name="Conservative",
    description="Fewer predictions with higher accuracy focus. Best for risk-averse users.",
    freq="1h",
    horizon="24h",
    tau=0.02,
    enter_threshold=0.75,
    color="#3B82F6",  # blue
    icon="\U0001f6e1\ufe0f",
)

BALANCED = Preset(
    name="Balanced",
    description="Good mix of prediction frequency and accuracy. Recommended for most users.",
    freq="1h",
    horizon="4h",
    tau=0.01,
    enter_threshold=0.65,
    color="#10B981",  # green
    icon="\u2696\ufe0f",
)

AGGRESSIVE = Preset(
    name="Aggressive",
    description="More frequent predictions with higher risk. Best for active traders.",
    freq="1h",
    horizon="1h",
    tau=0.005,
    enter_threshold=0.55,
    color="#EF4444",  # red
    icon="\U0001f680",
)

SWING = Preset(
    name="Swing",
    description="Sub-hourly signals for swing positions. Balances speed with confirmation time.",
    freq="15m",
    horizon="1h",
    tau=0.007,
    enter_threshold=0.60,
    color="#8B5CF6",  # purple
    icon="\U0001f30a",
)

PRESETS: dict[str, Preset] = {
    "scalper": SCALPER,
    "conservative": CONSERVATIVE,
    "balanced": BALANCED,
    "aggressive": AGGRESSIVE,
    "swing": SWING,
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
