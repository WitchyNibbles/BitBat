"""User-friendly configuration presets for the BitBat GUI.

Canonical implementation lives in bitbat.common.presets.
This module re-exports for backward compatibility with GUI-layer callers.
"""

from bitbat.common.presets import (  # noqa: F401
    AGGRESSIVE,
    BALANCED,
    CONSERVATIVE,
    DEFAULT_PRESET,
    PRESETS,
    SCALPER,
    SWING,
    Preset,
    get_preset,
    list_presets,
)
