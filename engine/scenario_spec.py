"""
ScenarioSpec: the constrained JSON intermediate representation at the core
of Prompt-to-Drive.

The Generator agent fills this schema from natural language; the
deterministic CARLA engine converts it into API calls. Keeping this schema
narrow and strongly typed is the single most important design decision in
the framework: it bounds the LLM's hallucination surface to a small set of
structured fields instead of arbitrary free-form code.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

Weather = Literal["clear", "rain", "fog", "night", "cloudy"]
Town = Literal["Town01", "Town02", "Town03", "Town04", "Town05"]
EventType = Literal[
    "crossing",
    "lane_change",
    "emergency_brake",
    "red_light_violation",
    "obstruction",
    "overtaking",
]


class ScenarioSpec(BaseModel):
    """Structured, schema-constrained representation of a driving scenario."""

    weather: Weather
    town: Town
    num_vehicles: int = Field(ge=0, le=20)
    pedestrian: bool
    event_type: EventType
    duration_s: int = Field(ge=8, le=30)
    spawn_point_index: int = Field(default=0, ge=0, le=200)
    cyclist: bool = False
    notes: Optional[str] = Field(
        default=None,
        description="Free-text rationale from the Generator. Not used by the engine.",
    )

    @field_validator("duration_s")
    @classmethod
    def _min_duration_for_event(cls, v: int, info):
        # crossing / emergency_brake need a minimum window to be observable
        event = info.data.get("event_type")
        if event in {"crossing", "emergency_brake"} and v < 12:
            return 12
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "weather": "rain",
                "town": "Town03",
                "num_vehicles": 4,
                "pedestrian": True,
                "event_type": "crossing",
                "duration_s": 15,
                "spawn_point_index": 6,
                "cyclist": False,
                "notes": "Pedestrian crossing near a signalized junction in rain.",
            }
        }
    }


class ScenarioRecord(BaseModel):
    """Full record of one scenario synthesis attempt, persisted to disk."""

    scenario_id: str
    prompt: str
    iteration: int
    spec: ScenarioSpec
    refinement_context: Optional[str] = None
