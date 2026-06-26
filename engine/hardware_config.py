"""
Hardware profile configurations constraining town selection.

Maps hardware profiles to memory-appropriate town selections. This ensures
LLM agents respect system constraints when generating scenarios.
"""
from typing import Literal

HardwareProfile = Literal[
    "high_performance",
    "standard",
    "low_vram_2gb",
    "low_vram_2gb_signals",
]

# Town geometry complexity (approx VRAM consumption in MB during simulation)
TOWN_VRAM_USAGE = {
    "Town01": 800,      # Smallest, dense pedestrian area
    "Town02": 900,      # Small, residential
    "Town03": 1200,     # Medium, signalized junctions
    "Town04": 1600,     # Large, highway segments
    "Town05": 1400,     # Large, complex urban
}

HARDWARE_PROFILES: dict[HardwareProfile, dict] = {
    "high_performance": {
        "description": "Unrestricted town selection (4GB+ VRAM)",
        "preferred_towns": ("Town01", "Town02", "Town03", "Town04", "Town05"),
        "vram_budget_mb": 4000,
    },
    "standard": {
        "description": "Mid-range systems (2-4GB VRAM)",
        "preferred_towns": ("Town01", "Town02", "Town03"),
        "vram_budget_mb": 2500,
    },
    "low_vram_2gb": {
        "description": "Constrained systems (2GB VRAM max)",
        "preferred_towns": ("Town01", "Town02"),  # smallest, least geometry-heavy maps
        "vram_budget_mb": 1200,
    },
    "low_vram_2gb_signals": {
        # Town01 is used here instead of Town03 because Town03 (1200MB) exceeds
        # the practical VRAM budget on Quadro M620 class hardware under load.
        # Town01 (800MB) has sufficient pedestrian spawn points and junction
        # proximity for signal-adjacent scenarios at this memory constraint.
        "description": "Signal-adjacent scenario testing (Town01, 2GB VRAM constrained)",
        "preferred_towns": ("Town01",),
        "vram_budget_mb": 1200,
    },
}


def get_preferred_towns(profile: HardwareProfile) -> tuple[str, ...]:
    """Return the list of towns appropriate for a given hardware profile."""
    if profile not in HARDWARE_PROFILES:
        raise ValueError(
            f"Unknown hardware profile: {profile}. "
            f"Valid options: {list(HARDWARE_PROFILES.keys())}"
        )
    return HARDWARE_PROFILES[profile]["preferred_towns"]


def get_profile_info(profile: HardwareProfile) -> dict:
    """Return full configuration for a hardware profile."""
    if profile not in HARDWARE_PROFILES:
        raise ValueError(f"Unknown hardware profile: {profile}")
    return HARDWARE_PROFILES[profile]