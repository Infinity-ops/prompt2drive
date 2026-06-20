"""
generator/agent.py

Agent 1 — Generator.

Translates a natural-language driving scenario description into a
validated ScenarioSpec using an LLM with structured output. Supports an
optional refinement_context produced by the Reflector agent on retry
iterations.

Design principle: the LLM never writes code or touches the simulator. It
only ever fills in a fixed, typed schema (ScenarioSpec). This is the
generation-side half of the hallucination-mitigation strategy described
in the paper; the engine-side half is the deterministic CarlaRunner.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shelve
from pathlib import Path
from typing import Optional

from openai import OpenAI
from pydantic import ValidationError

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


class Generator:
    def __init__(self, model: str = "gpt-4o-2024-08-06", client: Optional[OpenAI] = None):
        self.model = model
        self.client = client or OpenAI()

    def generate(self, prompt: str, refinement_context: str = "") -> ScenarioSpec:
        """
        Convert a natural language prompt (+ optional refinement context)
        into a validated ScenarioSpec. Results are cached by the combined
        input hash so repeated runs never re-spend API budget.
        """
        cache_key = hashlib.md5((prompt + "||" + refinement_context).encode()).hexdigest()

        with shelve.open(str(CACHE_PATH)) as cache:
            if cache_key in cache:
                logger.debug("Generator cache hit for %s", cache_key[:8])
                return ScenarioSpec(**cache[cache_key])

            user_content = prompt if not refinement_context else f"{prompt}\n\n{refinement_context}"

            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format=ScenarioSpec,
                temperature=0.2,
            )

            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise ValidationError.from_exception_data(
                    "ScenarioSpec", [{"type": "missing", "loc": (), "input": None}]
                )

            cache[cache_key] = parsed.model_dump()
            return parsed


def generate_baseline_code(prompt: str, client: Optional[OpenAI] = None,
                            model: str = "gpt-4o-2024-08-06") -> str:
    """
    Baseline comparator used in Experiment 1: direct, free-form LLM code
    generation with NO schema constraint. This is intentionally the "naive"
    approach the paper compares against -- do not improve its prompt beyond
    what a typical practitioner would write in a first attempt.
    """
    client = client or OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert CARLA Python API developer. Write a complete, "
                    "runnable Python script using the carla module that implements the "
                    "scenario described by the user. Assume `carla`, `client`, and `world` "
                    "are already available. Return ONLY Python code."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content
