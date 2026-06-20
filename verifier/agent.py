"""
verifier/agent.py

Agent 2 — Verifier.

Implements hybrid semantic verification of a completed simulation run
against the original natural-language intent, combining:
  (a) a VLM channel that visually inspects rendered frames, and
  (b) a deterministic log channel that checks the CARLA state log.

See Section 3.3 of the paper for the full design rationale. The hybrid
score is what the Reflector consumes to decide whether (and how) to
trigger another generation iteration.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import shelve
from pathlib import Path
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel

from engine.scenario_spec import ScenarioSpec

logger = logging.getLogger("prompt_to_drive.verifier")

CACHE_PATH = Path(".cache/verifier_cache")
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_THRESHOLD = 0.72
SCORE_WEIGHTS = dict(vlm=0.5, log=0.3, exec=0.2)


class VerificationResult(BaseModel):
    weather_correct: bool
    pedestrian_present: bool
    vehicles_present: bool
    event_observable: bool
    intent_alignment_score: float
    missing_elements: list[str]
    excess_elements: list[str]
    confidence: float
    notes: str


class HybridVerificationResult(BaseModel):
    vlm_result: VerificationResult
    log_checks_passed: int
    log_checks_total: int
    execution_success: bool
    final_score: float
    overall_pass: bool


VERIFY_PROMPT_TEMPLATE = """You are a simulation verification expert for autonomous driving.

Original user intent: "{prompt}"

Executed scenario specification:
{spec_json}

Examine the attached simulation frames carefully (sampled at the start, \
midpoint, and end of the run). Return ONLY a JSON object with these exact \
fields, no markdown fences, no explanation outside the JSON:

{{
  "weather_correct": <bool: visible weather matches "{weather}">,
  "pedestrian_present": <bool: pedestrian visible if the spec requires one>,
  "vehicles_present": <bool: vehicle count is approximately correct>,
  "event_observable": <bool: "{event}" is clearly occurring or has occurred>,
  "intent_alignment_score": <float 0.0-1.0: overall semantic match to the original intent>,
  "missing_elements": [<strings: semantic elements from the prompt absent from the simulation>],
  "excess_elements": [<strings: elements present but not requested>],
  "confidence": <float 0.0-1.0: your confidence in this assessment>,
  "notes": "<one sentence diagnostic note>"
}}"""


class Verifier:
    def __init__(self, model: str = "gpt-4o", client: Optional[OpenAI] = None,
                 threshold: float = DEFAULT_THRESHOLD):
        self.model = model
        self.client = client or OpenAI()
        self.threshold = threshold

    # ------------------------------------------------------------------ #
    def verify(
        self,
        original_prompt: str,
        spec: ScenarioSpec,
        frame_paths: list[str],
        log_path: str,
        exec_success: bool,
    ) -> HybridVerificationResult:
        cache_key = hashlib.md5(
            (original_prompt + str(spec.model_dump()) + str(frame_paths)).encode()
        ).hexdigest()

        with shelve.open(str(CACHE_PATH)) as cache:
            if cache_key in cache:
                logger.debug("Verifier cache hit for %s", cache_key[:8])
                return HybridVerificationResult(**cache[cache_key])

            if not exec_success or not frame_paths:
                # No point asking the VLM to inspect a failed run.
                vlm_result = VerificationResult(
                    weather_correct=False,
                    pedestrian_present=False,
                    vehicles_present=False,
                    event_observable=False,
                    intent_alignment_score=0.0,
                    missing_elements=["execution_failed"],
                    excess_elements=[],
                    confidence=1.0,
                    notes="Simulation did not execute successfully; no frames to inspect.",
                )
                log_passed, log_total = 0, 4
            else:
                vlm_result = self._call_vlm(original_prompt, spec, frame_paths)
                log_passed, log_total = self._validate_log(log_path, spec)

            s_vlm = vlm_result.intent_alignment_score
            s_log = log_passed / log_total if log_total > 0 else 0.0
            s_exec = 1.0 if exec_success else 0.0
            final_score = round(
                SCORE_WEIGHTS["vlm"] * s_vlm + SCORE_WEIGHTS["log"] * s_log + SCORE_WEIGHTS["exec"] * s_exec,
                4,
            )

            result = HybridVerificationResult(
                vlm_result=vlm_result,
                log_checks_passed=log_passed,
                log_checks_total=log_total,
                execution_success=exec_success,
                final_score=final_score,
                overall_pass=final_score >= self.threshold,
            )

            cache[cache_key] = result.model_dump()
            return result

    # ------------------------------------------------------------------ #
    def _call_vlm(self, prompt: str, spec: ScenarioSpec, frame_paths: list[str]) -> VerificationResult:
        selected = self._select_frames(frame_paths)

        content = [
            {
                "type": "text",
                "text": VERIFY_PROMPT_TEMPLATE.format(
                    prompt=prompt,
                    spec_json=json.dumps(spec.model_dump(), indent=2),
                    weather=spec.weather,
                    event=spec.event_type,
                ),
            }
        ]
        content.extend(self._encode_frame(f) for f in selected if Path(f).exists())

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=600,
            temperature=0.0,  # deterministic verification
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("Verifier JSON parse failure, raw output: %s", raw[:500])
            data = dict(
                weather_correct=False, pedestrian_present=False, vehicles_present=False,
                event_observable=False, intent_alignment_score=0.0,
                missing_elements=["verifier_parse_error"], excess_elements=[],
                confidence=0.0, notes="VLM response could not be parsed as JSON.",
            )
        return VerificationResult(**data)

    @staticmethod
    def _select_frames(frame_paths: list[str]) -> list[str]:
        if len(frame_paths) <= 3:
            return frame_paths
        return [frame_paths[0], frame_paths[len(frame_paths) // 2], frame_paths[-1]]

    @staticmethod
    def _encode_frame(frame_path: str) -> dict:
        with open(frame_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}}

    @staticmethod
    def _validate_log(log_path: str, spec: ScenarioSpec) -> tuple[int, int]:
        try:
            with open(log_path) as f:
                log = json.load(f)
        except Exception:
            return 0, 4

        checks = [
            log.get("weather_api_called") == spec.weather,
            log.get("pedestrian_spawned", False) == spec.pedestrian,
            abs(log.get("vehicle_count", 0) - spec.num_vehicles) <= 2,
            log.get("execution_completed", False),
        ]
        return sum(checks), len(checks)
