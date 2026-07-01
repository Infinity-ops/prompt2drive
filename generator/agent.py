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
import logging
import shelve
from pathlib import Path
from typing import Optional

from openai import OpenAI
from pydantic import ValidationError

from backends.llm_backend import TextBackend
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
  (Town01 is preferred for pedestrian scenarios; Town03 and Town04 are
   preferred for signalized-intersection scenarios when available; Town04
   is preferred for highway scenarios)
- num_vehicles: integer 0-20
- pedestrian: boolean
- cyclist: boolean
- event_type: one of "crossing", "lane_change", "emergency_brake",
  "red_light_violation", "obstruction", "overtaking"
- duration_s: integer 8-30 (use >=15 for crossing/emergency_brake events)
- spawn_point_index: integer 0-200 (use 0-12 for junction-proximate scenes)
- notes: a one-sentence rationale for your choices

Examples (select town based on scenario features, not repetition):
prompt: "A pedestrian suddenly crosses near a traffic signal in rainy weather."
-> {"weather": "rain", "town": "Town01", "num_vehicles": 3, "pedestrian": true,
    "cyclist": false, "event_type": "crossing", "duration_s": 15,
    "spawn_point_index": 6, "notes": "Town01 pedestrian-friendly area, rain, crossing."}

prompt: "Sudden lane change to avoid a parked vehicle on a sunny highway."
-> {"weather": "clear", "town": "Town04", "num_vehicles": 6, "pedestrian": false,
    "cyclist": false, "event_type": "lane_change", "duration_s": 12,
    "spawn_point_index": 20, "notes": "Town04 highway segment for lane change."}

prompt: "A cyclist runs a red light in foggy conditions at a busy junction."
-> {"weather": "fog", "town": "Town03", "num_vehicles": 4, "pedestrian": false,
    "cyclist": true, "event_type": "red_light_violation", "duration_s": 14,
    "spawn_point_index": 8, "notes": "Town03 junction with signalized intersection."}

If a refinement context is provided (from a previous failed verification),
apply the required corrections precisely while preserving fields that were
already verified as correct. Return ONLY the JSON object, no prose."""


class Generator:
    def __init__(
        self,
        backend: TextBackend,
        preferred_towns: Optional[tuple[str, ...]] = None,
    ):
        self.backend = backend
        self.preferred_towns = preferred_towns

    def generate(self, prompt: str, refinement_context: str = "") -> ScenarioSpec:
        """
        Convert a natural-language prompt (+ optional refinement context)
        into a validated ScenarioSpec. Results are cached by the combined
        input hash so repeated runs never re-spend API budget.
        """
        hardware_key = ",".join(self.preferred_towns or ())
        cache_key = hashlib.md5(
            (prompt + "||" + refinement_context + "||" + hardware_key).encode()
        ).hexdigest()
        with shelve.open(str(CACHE_PATH)) as cache:
            if cache_key in cache:
                logger.debug("Generator cache hit for %s", cache_key[:8])
                cached_spec = ScenarioSpec.model_validate(cache[cache_key])

                # Do not allow a cache created under another hardware profile
                # to bypass the current town restriction.
                if (
                    self.preferred_towns
                    and cached_spec.town not in self.preferred_towns
                ):
                    logger.warning(
                        "Ignoring cached scenario for forbidden town '%s'; "
                        "allowed towns are %s.",
                        cached_spec.town,
                        self.preferred_towns,
                    )
                else:
                    return cached_spec

            user_content = (
                prompt
                if not refinement_context
                else f"{prompt}\n\n{refinement_context}"
            )

            system_prompt = SYSTEM_PROMPT

            if self.preferred_towns:
                towns_str = ", ".join(self.preferred_towns)
                system_prompt += (
                    f"\n\n*** HARDWARE CONSTRAINT: You must use only these towns: "
                    f"{towns_str}. "
                    "This restriction overrides any town preference stated above. "
                    "If the user requests a feature normally associated with an "
                    "unavailable town (for example, a traffic signal or highway), "
                    "preserve the requested driving event, weather, "
                    "pedestrian/cyclist behavior, and vehicle behavior using the "
                    "best available spawn point in an allowed town. "
                    "Do not select a forbidden town. "
                    "Do not claim in notes that an unavailable map feature was "
                    "guaranteed. ***"
                )

            # RC#7: generate_structured must run regardless of preferred_towns.
            try:
                parsed = self.backend.generate_structured(
                    system_prompt=system_prompt,
                    user_prompt=user_content,
                    schema=ScenarioSpec,
                )
            except ValidationError:
                logger.exception("Backend returned an invalid ScenarioSpec.")
                raise

            if (
                self.preferred_towns
                and parsed.town not in self.preferred_towns
            ):
                raise ValueError(
                    f"Generator selected forbidden town '{parsed.town}'. "
                    f"Allowed towns: {', '.join(self.preferred_towns)}"
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