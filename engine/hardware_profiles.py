"""
engine/hardware_profiles.py

Resolution / quality / actor-density presets for running Prompt-to-Drive
on modest hardware. Added specifically because CARLA's default settings
assume a desktop-class discrete GPU with >=6GB VRAM; mobile workstation
GPUs (e.g., the Quadro M620, 2GB VRAM, Maxwell) routinely segfault under
CARLA's default Vulkan renderer at default resolution with many actors.

Use a profile via:
    python run_scenario.py --hardware-profile low_vram_2gb ...
or set P2D_HARDWARE_PROFILE=low_vram_2gb in .env.

This does NOT fix CARLA server-side crashes by itself -- you must ALSO
launch the CARLA server binary with matching flags. See
docs/TROUBLESHOOTING.md for the full checklist, including the
exact CarlaUE4 launch flags for a 2GB-VRAM Maxwell GPU.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    camera_width: int
    camera_height: int
    max_vehicles: int          # hard cap regardless of what the ScenarioSpec requests
    capture_every_s: float     # less frequent capture = less VRAM/encoding pressure
    fixed_delta_seconds: float # larger tick = lower simulation load
    preferred_towns: tuple     # smaller/simpler maps first
    recommended_carla_flags: str
    notes: str


PROFILES: dict[str, HardwareProfile] = {
    "default": HardwareProfile(
        name="default",
        camera_width=960,
        camera_height=540,
        max_vehicles=20,
        capture_every_s=2.0,
        fixed_delta_seconds=0.05,
        preferred_towns=("Town01", "Town02", "Town03", "Town04", "Town05"),
        recommended_carla_flags="",
        notes="Assumes a discrete desktop GPU with >=6GB VRAM.",
    ),
    "low_vram_2gb_signals": HardwareProfile(
        name="low_vram_2gb_signals",
        camera_width=480,
        camera_height=270,
        max_vehicles=4,
        capture_every_s=3.0,
        fixed_delta_seconds=0.1,
        # Town01 (800MB) is used instead of Town03 (1200MB) because Town03
        # exceeds practical VRAM budget on Quadro M620 class hardware under
        # load. Town01 has adequate pedestrian and junction-adjacent spawn
        # points for signal-related scenarios at this memory constraint.
        preferred_towns=("Town01",),
        recommended_carla_flags="-quality-level=Low -ResX=800 -ResY=600 -opengl -RenderOffScreen",
        notes="Signal-adjacent profile for 2GB VRAM; Town01 used instead of Town03 to fit 2GB budget.",
    ),
    "low_vram_2gb": HardwareProfile(
        name="low_vram_2gb",
        camera_width=480,
        camera_height=270,
        max_vehicles=6,
        capture_every_s=3.0,
        fixed_delta_seconds=0.1,         # 10 Hz instead of 20 Hz
        preferred_towns=("Town01", "Town02"),  # smallest, least geometry-heavy maps
        recommended_carla_flags=(
            "-quality-level=Low -ResX=800 -ResY=600 -opengl -RenderOffScreen"
        ),
        notes=(
            "Tuned for the Dell Precision 3530 / Quadro M620 (2GB VRAM, Maxwell). "
            "-opengl is important: this GPU's Vulkan driver support is the most "
            "common segfault cause under CARLA 0.9.14+. Low quality level disables "
            "most post-processing and shadow rendering, which is what actually "
            "consumes the limited VRAM. max_vehicles=6 keeps NPC physics/rendering "
            "load low enough to avoid frame-capture stalls causing camera-callback "
            "timeouts in carla_runner.py."
        ),
    ),
    "headless_ci": HardwareProfile(
        name="headless_ci",
        camera_width=320,
        camera_height=180,
        max_vehicles=4,
        capture_every_s=4.0,
        fixed_delta_seconds=0.1,
        preferred_towns=("Town01",),
        recommended_carla_flags="-quality-level=Low -ResX=640 -ResY=480 -opengl -RenderOffScreen -nosound",
        notes="Smallest footprint -- for CI smoke tests, not for real evaluation data.",
    ),
}


def get_profile(name: str | None) -> HardwareProfile:
    import os

    name = name or os.getenv("P2D_HARDWARE_PROFILE", "default")
    if name not in PROFILES:
        raise ValueError(f"Unknown hardware profile '{name}'. Options: {list(PROFILES)}")
    return PROFILES[name]


def get_preferred_towns(name: str) -> tuple[str, ...]:
    """Towns the Generator may choose under this profile (smallest maps first)."""
    return get_profile(name).preferred_towns


def get_profile_info(name: str) -> dict:
    """Human-readable summary of a profile, for CLI output."""
    prof = get_profile(name)
    return {"name": prof.name, "description": prof.notes, "preferred_towns": prof.preferred_towns}
