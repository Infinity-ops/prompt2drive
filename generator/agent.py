"""
generator/agent.py

Agent 1 — Generator.

Translates a natural-language driving scenario description into a
validated ScenarioSpec using a pluggable text backend (OpenAI or
Hugging Face -- see backends/llm_backend.py). Supports an optional
refinement_context produced by the Reflector agent on retry iterations.

Design principle: the backend never writes code or touches the
simulator. It only ever fills in a fixed, typed schema (ScenarioSpec).
This is the generation-side half of the hallucination-mitigation
strategy described in the paper; the engine-side half is the
deterministic CarlaRunner.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shelve
from pathlib import Path
from typing import Optional, Union

from backends.llm_backend import TextBackend, get_backend
from engine.scenario_spec import ScenarioSpec

logger = logging.getLogger("prompt_to_drive.generator")

CACHE_PATH = Path(".cache/generator_cache")
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """You are the Generator agent in Prompt-to-Drive, a closed-loop \
simulation synthesis system for autonomous driving validation.

Your job: convert a natural-language driving scenario description into a \
ScenarioSpec JSON object that a deterministic CARLA execution engine can run.

You MUST only use these exact fields and value ranges:
- weather: one of "clear", "rain", "fog", "night", "cloudy"
- town: one of "Town01", "Town02", "Town03", "Town04", "Town05"
  (Town03 and Town04 have signalized intersections; Town01 has dense
   pedestrian-friendly sidewalks; Town04 has highway segments)
- num_vehicles: integer 0-20
- pedestrian: boolean
- cyclist: boolean
- event_type: one of "crossing", "lane_change", "emergency_brake",
  "red_light_violation", "obstruction", "overtaking"
- duration_s: integer 8-30 (use >=15 for crossing/emergency_brake events)
- spawn_point_index: integer 0-200 (use 0-12 for junction-proximate scenes)
- notes: a one-sentence rationale for your choices

Examples:
prompt: "A pedestrian suddenly crosses near a traffic signal in rainy weather."
-> {"weather": "rain", "town": "Town03", "num_vehicles": 3, "pedestrian": true,
    "cyclist": false, "event_type": "crossing", "duration_s": 15,
    "spawn_point_index": 6, "notes": "Town03 junction, rain, pedestrian crossing."}

prompt: "Sudden lane change to avoid a parked vehicle on a sunny highway."
-> {"weather": "clear", "town": "Town04", "num_vehicles": 6, "pedestrian": false,
    "cyclist": false, "event_type": "lane_change", "duration_s": 12,
    "spawn_point_index": 20, "notes": "Town04 highway segment for lane change."}

prompt: "A cyclist runs a red light in foggy conditions at night."
-> {"weather": "fog", "town": "Town03", "num_vehicles": 4, "pedestrian": false,
    "cyclist": true, "event_type": "red_light_violation", "duration_s": 14,
    "spawn_point_index": 8, "notes": "Town03 junction with fog for visibility test."}

If a refinement context is provided (from a previous failed verification),
apply the required corrections precisely while preserving fields that were
already verified as correct. Return ONLY the JSON object, no prose."""

BASELINE_SYSTEM_PROMPT = """You are a CARLA simulation expert. Generate SAFE, WORKING Python code for \
CARLA scenarios directly from the user's natural language description.

CRITICAL SAFETY RULES:
1. ALWAYS use world.try_spawn_actor() instead of world.spawn_actor()
2. ALWAYS check if actor is not None before using it
3. LIMIT actors: maximum 3 vehicles, maximum 2 pedestrians
4. Do NOT include client connection, world retrieval, time.sleep, or destroy calls
5. Do NOT include if __name__ == '__main__' blocks
6. ALWAYS add spawned actors to actor_list for cleanup
7. The camera will automatically follow the first spawned vehicle; ensure at least
   one vehicle is spawned for a dynamic view.
8. Autopilot should ONLY be enabled if the user explicitly requests the vehicle to
   "drive", "move", "follow a path", or if it's part of a "traffic" scenario.

AVAILABLE VARIABLES: client, world, actor_list (list to append spawned actors to).

Return ONLY Python code, no markdown fences, no explanation.

Note: this is intentionally a free-form, unconstrained generation baseline (the
naive comparator in Experiment 1) -- it has no semantic verification of its own
output, no schema validation, and no closed-loop refinement. Its prompt
engineering is adapted from an earlier, hand-tuned practitioner prototype, since
a weak, unrealistic strawman baseline would be a less defensible comparator than
one reflecting genuine prior effort to make free-form generation work well."""


class Generator:
    """
    backend: a TextBackend instance, or a string ("openai" / "huggingface")
    which will be resolved via backends.llm_backend.get_backend(). Defaults
    to the P2D_TEXT_BACKEND environment variable, or "openai" if unset.
    """

    def __init__(self, backend: Optional[Union[TextBackend, str]] = None):
        if backend is None:
            backend = os.getenv("P2D_TEXT_BACKEND", "openai")
        self.backend: TextBackend = get_backend(backend) if isinstance(backend, str) else backend

    def generate(self, prompt: str, refinement_context: str = "") -> ScenarioSpec:
        """
        Convert a natural language prompt (+ optional refinement context)
        into a validated ScenarioSpec. Results are cached by backend name +
        input hash so repeated runs never re-spend API/inference budget,
        and so the same prompt run under two backends is cached separately.
        """
        cache_key = hashlib.md5(
            (self.backend.name + "||" + prompt + "||" + refinement_context).encode()
        ).hexdigest()

        with shelve.open(str(CACHE_PATH)) as cache:
            if cache_key in cache:
                logger.debug("Generator cache hit for %s [%s]", cache_key[:8], self.backend.name)
                return ScenarioSpec(**cache[cache_key])

            user_content = prompt if not refinement_context else f"{prompt}\n\n{refinement_context}"
            parsed = self.backend.generate_structured(SYSTEM_PROMPT, user_content, ScenarioSpec)

            cache[cache_key] = parsed.model_dump()
            return self._clamp_for_hardware(parsed)

    @staticmethod
    def _clamp_for_hardware(spec: ScenarioSpec) -> ScenarioSpec:
        """
        Optional post-generation safety clamp for resource-constrained
        hardware. Activated by setting P2D_LOWSPEC_MODE=1 -- restricts the
        town to Town01/Town02 (smaller maps, less geometry to render) and
        caps num_vehicles, so a weak GPU (e.g. 2GB VRAM mobile cards) is
        less likely to stall or crash mid-run. This does NOT change what
        the Generator *chose* semantically; it only bounds render load.
        """
        if os.getenv("P2D_LOWSPEC_MODE") != "1":
            return spec
        data = spec.model_dump()
        if data["town"] not in {"Town01", "Town02"}:
            data["town"] = "Town01"
        data["num_vehicles"] = min(data["num_vehicles"], 8)
        return ScenarioSpec(**data)


def generate_baseline_code(prompt: str, backend: Optional[Union[TextBackend, str]] = None) -> str:
    """
    Baseline comparator used in Experiment 1: direct, free-form generation
    with NO schema constraint, run under whichever backend is passed in.
    This is intentionally the "naive" approach the paper compares against.
    """
    if backend is None:
        backend = os.getenv("P2D_TEXT_BACKEND", "openai")
    backend_obj = get_backend(backend) if isinstance(backend, str) else backend
    return backend_obj.generate_text(BASELINE_SYSTEM_PROMPT, prompt)
