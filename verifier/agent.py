"""
verifier/agent.py

Agent 2 — Verifier.

Implements predicate-based semantic verification of a completed simulation
run against the original natural-language intent.

Design history note (read this before changing the scoring logic): the
first version of this module computed a single opaque
`intent_alignment_score` float directly from the VLM ("how well does this
match, 0 to 1?") and weighted it into a hybrid score. A reviewer correctly
identified this as the weakest part of the design: an unexplained float
from a vision-language model is not an auditable verification result, and
"the model thinks it looks right" is not evidence for a safety case.

This version instead extracts a small set of NAMED, BOOLEAN predicates
from each frame set (weather_correct, pedestrian_present, vehicles_present,
event_observable), restricts each predicate to the scenarios where it is
APPLICABLE (e.g. pedestrian_present only counts if the spec actually
requested a pedestrian), and defines the semantic alignment score as the
fraction of applicable predicates that hold:

    predicate_agreement = (# applicable predicates that hold) / (# applicable predicates)

This number is fully explainable: for any score, you can list exactly
which predicates passed and which failed, in plain language, without
reference to model internals. See docs/SEMANTIC_VERIFICATION_METRIC.md
for the full formal definition used in the paper's Methods section.

The hybrid final score combines this predicate agreement with the
deterministic log channel and execution success, using the same
structure as before -- only the VLM channel's *definition* changed, not
its role in the hybrid combination.

Uncertainty: because VLM outputs are not deterministic even at
temperature 0 across different backends, `verify()` queries the vision
backend `n_samples` times (default 3) and reports both the per-sample
mean/std of predicate_agreement and a majority-vote predicate set. This
directly answers "why should I trust this score" with a quantified
agreement-across-samples number rather than a single unrepeated call.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shelve
import statistics
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, ValidationError

from backends.llm_backend import VisionBackend, get_vision_backend
from engine.scenario_spec import ScenarioSpec

logger = logging.getLogger("prompt_to_drive.verifier")

CACHE_PATH = Path(".cache/verifier_cache")
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_THRESHOLD = float(os.getenv("P2D_VERIFICATION_THRESHOLD", "0.72"))
DEFAULT_N_SAMPLES = int(os.getenv("P2D_VERIFIER_N_SAMPLES", "3"))

# Hybrid score combination weights.
SCORE_WEIGHTS = dict(vlm=0.5, log=0.3, exec=0.2)

# Predicate logic tracking rules
PREDICATE_APPLICABILITY = {
    "weather_correct": lambda spec: True,  # always applicable
    "pedestrian_present": lambda spec: spec.pedestrian,
    "vehicles_present": lambda spec: spec.num_vehicles > 0,
    "event_observable": lambda spec: True,  # always applicable
}


class VerificationResult(BaseModel):
    weather_correct: bool
    pedestrian_present: bool
    vehicles_present: bool
    event_observable: bool
    intent_alignment_score: float  # auxiliary holistic score, kept for tracking/comparison
    excess_elements: list[str]
    confidence: float
    notes: str

    @property
    def predicates(self) -> dict[str, bool]:
        return {
            "weather_correct": self.weather_correct,
            "pedestrian_present": self.pedestrian_present,
            "vehicles_present": self.vehicles_present,
            "event_observable": self.event_observable,
        }


class HybridVerificationResult(BaseModel):
    vlm_result: VerificationResult              # majority-vote predicates across samples
    predicate_agreement: float                  # PRIMARY formal semantic score (0.0-1.0)
    predicate_agreement_std: float               # std across n_samples -- uncertainty estimate
    applicable_predicates: list[str]
    failed_predicates: list[str]                 # applicable predicates that did NOT hold
    n_samples: int
    log_checks_passed: int
    log_checks_total: int
    execution_success: bool
    final_score: float
    overall_pass: bool
    backend_name: str = "unknown"

    @property
    def missing_elements(self) -> list[str]:
        return list(self.failed_predicates)


VERIFY_PROMPT_TEMPLATE = """You are a simulation verification expert for autonomous driving.

Original user intent: "{prompt}"

Executed scenario specification:
{spec_json}

Examine the 3 attached sequential simulation frames carefully (sampled at the start, midpoint, and end of the run). Return ONLY a JSON object with these exact fields, no markdown fences, no explanation outside the JSON:

{{
  "weather_correct": <bool: visible weather matches "{weather}">,
  "pedestrian_present": <bool: pedestrian visible if the spec requires one>,
  "vehicles_present": <bool: vehicle count is approximately correct>,
  "event_observable": <bool: "{event}" is clearly occurring or has occurred>,
  "intent_alignment_score": <float 0.0-1.0: your own holistic impression of overall match -- this is recorded for comparison against the formal predicate score, not used directly>,
  "excess_elements": [<strings: elements present but not requested>],
  "confidence": <float 0.0-1.0: your confidence in this specific assessment>,
  "notes": "<one sentence diagnostic note>"
}}"""


class Verifier:
    def __init__(
        self,
        backend: Optional[Union[VisionBackend, str]] = None,
        threshold: float = DEFAULT_THRESHOLD,
        n_samples: int = DEFAULT_N_SAMPLES,
    ):
        if backend is None:
            backend = os.getenv("P2D_VISION_BACKEND", "openai")
        self.backend: VisionBackend = get_vision_backend(backend) if isinstance(backend, str) else backend
        self.threshold = threshold
        self.n_samples = max(1, n_samples)

    def verify(
        self,
        original_prompt: str,
        spec: ScenarioSpec,
        frame_paths: list[str],
        log_path: str,
        exec_success: bool,
    ) -> HybridVerificationResult:
        # FIX 1: Pre-extract the matching frame sequences right at the beginning
        selected = self._select_frames(frame_paths)
        
        # FIX 2: Compute cache key on 'selected' (the actual 3 frames used) rather than raw frame paths
        cache_key = hashlib.md5(
            (self.backend.name + "||" + str(self.n_samples) + "||" + original_prompt
             + "||" + str(spec.model_dump()) + "||" + str(selected)).encode()
        ).hexdigest()

        applicable = [name for name, fn in PREDICATE_APPLICABILITY.items() if fn(spec)]

        try:
            with shelve.open(str(CACHE_PATH)) as db:
                if cache_key in db:
                    logger.info("Verifier cache hit for scenario descriptor.")
                    cached_data = db[cache_key]
                    log_passed, log_total = self._validate_log(log_path, spec) if exec_success else (0, 4)
                    cached_data.log_checks_passed = log_passed
                    cached_data.log_checks_total = log_total
                    cached_data.execution_success = exec_success
                    
                    s_vlm = cached_data.predicate_agreement
                    s_log = log_passed / log_total if log_total > 0 else 0.0
                    s_exec = 1.0 if exec_success else 0.0
                    cached_data.final_score = round(
                        SCORE_WEIGHTS["vlm"] * s_vlm + SCORE_WEIGHTS["log"] * s_log + SCORE_WEIGHTS["exec"] * s_exec, 4
                    )
                    cached_data.overall_pass = cached_data.final_score >= self.threshold
                    return cached_data
        except Exception as e:
            logger.debug("Verifier cache read skipped or file locked: %s", e)

        if not exec_success or not selected:
            samples = [self._failed_execution_result()]
        else:
            # FIX 3: Pass full frame paths instead of slicing down to a single element block (`[:1]`)
            samples = [self._call_vlm(original_prompt, spec, frame_paths) for _ in range(self.n_samples)]

        majority_result, agreement_scores = self._aggregate_samples(samples, applicable)
        agreement_mean = statistics.fmean(agreement_scores)
        agreement_std = statistics.pstdev(agreement_scores) if len(agreement_scores) > 1 else 0.0

        failed_predicates = [p for p in applicable if not majority_result.predicates[p]]
        log_passed, log_total = self._validate_log(log_path, spec) if exec_success else (0, 4)

        s_vlm = agreement_mean
        s_log = log_passed / log_total if log_total > 0 else 0.0
        s_exec = 1.0 if exec_success else 0.0
        final_score = round(
            SCORE_WEIGHTS["vlm"] * s_vlm + SCORE_WEIGHTS["log"] * s_log + SCORE_WEIGHTS["exec"] * s_exec,
            4,
        )

        result = HybridVerificationResult(
            vlm_result=majority_result,
            predicate_agreement=round(agreement_mean, 4),
            predicate_agreement_std=round(agreement_std, 4),
            applicable_predicates=applicable,
            failed_predicates=failed_predicates,
            n_samples=len(samples),
            log_checks_passed=log_passed,
            log_checks_total=log_total,
            execution_success=exec_success,
            final_score=final_score,
            overall_pass=final_score >= self.threshold,
            backend_name=self.backend.name,
        )

        try:
            with shelve.open(str(CACHE_PATH)) as db:
                db[cache_key] = result
        except Exception as e:
            logger.debug("Could not write to verifier state shelf: %s", e)

        return result

    def _aggregate_samples(
        self, samples: list[VerificationResult], applicable: list[str]
    ) -> tuple[VerificationResult, list[float]]:
        agreement_scores = []
        for s in samples:
            preds = s.predicates
            applicable_results = [preds[p] for p in applicable]
            agreement_scores.append(sum(applicable_results) / len(applicable_results) if applicable_results else 1.0)

        majority: dict[str, bool] = {}
        for p in PREDICATE_APPLICABILITY:
            votes = [s.predicates[p] for s in samples]
            majority[p] = sum(votes) > len(votes) / 2

        majority_result = VerificationResult(
            weather_correct=majority["weather_correct"],
            pedestrian_present=majority["pedestrian_present"],
            vehicles_present=majority["vehicles_present"],
            event_observable=majority["event_observable"],
            intent_alignment_score=statistics.fmean(s.intent_alignment_score for s in samples),
            excess_elements=sorted(set().union(*[set(s.excess_elements) for s in samples])) if samples else [],
            confidence=statistics.fmean(s.confidence for s in samples),
            notes=samples[0].notes if samples else "",
        )
        return majority_result, agreement_scores

    def _failed_execution_result(self) -> VerificationResult:
        return VerificationResult(
            weather_correct=False, pedestrian_present=False, vehicles_present=False,
            event_observable=False, intent_alignment_score=0.0, excess_elements=[],
            confidence=1.0, notes="Simulation did not execute successfully; no frames to inspect.",
        )

    def _call_vlm(self, prompt: str, spec: ScenarioSpec, frame_paths: list[str]) -> VerificationResult:
        selected = self._select_frames(frame_paths)
        prompt_text = VERIFY_PROMPT_TEMPLATE.format(
            prompt=prompt,
            spec_json=json.dumps(spec.model_dump(), indent=2),
            weather=spec.weather,
            event=spec.event_type,
        )

        try:
            # Filters the 3 temporal tracking frames through path check validation
            valid_frames = [str(Path(f).resolve()) for f in selected if Path(f).exists()]
            
            data = self.backend.verify_frames(prompt_text, valid_frames)
            data.pop("missing_elements", None)  # tolerate older structural formats smoothly
            
            # Defensive Structural Safeguard for model text truncation
            if "intent_alignment_score" not in data:
                data["intent_alignment_score"] = 1.0 if data.get("event_observable", False) else 0.0
            if "excess_elements" not in data:
                data["excess_elements"] = []
            if "confidence" not in data:
                data["confidence"] = 1.0
            if "notes" not in data:
                data["notes"] = "Generated via multi-image fallback structural path."

            return VerificationResult(**data)
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            logger.error("Verifier (%s) failed to produce valid output: %s", self.backend.name, exc)
            return VerificationResult(
                weather_correct=False, pedestrian_present=False, vehicles_present=False,
                event_observable=False, intent_alignment_score=0.0, excess_elements=[],
                confidence=0.0, notes=f"Backend '{self.backend.name}' response could not be parsed: {exc}",
            )

    @staticmethod
    def _select_frames(frame_paths: list[str]) -> list[str]:
        """Slices out exactly 3 frames tracking start, midpoint, and end states."""
        if len(frame_paths) <= 3:
            return frame_paths
        return [frame_paths[0], frame_paths[len(frame_paths) // 2], frame_paths[-1]]

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