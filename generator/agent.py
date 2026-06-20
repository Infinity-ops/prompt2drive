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

BASELINE_SYSTEM_PROMPT = (
    "You are an expert CARLA Python API developer. Write a complete, runnable "
    "Python script using the carla module that implements the scenario described "
    "by the user. Assume `carla`, `client`, and `world` are already available. "
    "Return ONLY Python code."
)


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
            return parsed


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
