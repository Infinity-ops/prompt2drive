"""
reflector/agent.py

Agent 3 — Reflector.

Performs structured root-cause analysis of a failed VerificationResult and
produces a targeted refinement_context string for the Generator's next
iteration. The design principle (Section 3.4 of the paper) is: every
failed predicate maps to exactly one specific, actionable instruction.
This keeps reflection auditable and avoids regressing fields that were
already verified correct.

As of the predicate-based verifier rewrite, `missing_elements` is a list
of FORMAL PREDICATE NAMES (e.g. "weather_correct", "event_observable") --
not free-text VLM output -- so corrections are looked up by predicate
name and parameterized with the scenario's actual spec values, rather than
matched against arbitrary strings.
"""
from __future__ import annotations

import logging

from engine.scenario_spec import ScenarioSpec
from verifier.agent import HybridVerificationResult

logger = logging.getLogger("prompt_to_drive.reflector")

# Town/spawn guidance keyed by event type -- used to parameterize corrections
# with the scenario's actual requested event rather than a generic message.
EVENT_GUIDANCE = {
    "crossing": "duration_s >=15s to capture the full crossing arc; ensure pedestrian=true",
    "lane_change": "a straight multi-lane segment: prefer Town04 (highway) or Town01 (main boulevard)",
    "emergency_brake": "an NPC vehicle spawned ahead of the ego vehicle; use num_vehicles>=3",
    "red_light_violation": "Town03 or Town04 (signalized intersections) and spawn_point_index 5-12",
    "obstruction": "a stationary NPC vehicle placed directly in the ego vehicle's lane",
    "overtaking": "a multi-lane segment with at least one slow-moving NPC vehicle ahead",
}

WEATHER_GUIDANCE = {
    "rain": "CARLA renders this with high precipitation and wet road reflections -- should be clearly visible",
    "fog": "increase duration_s to >=15s so foggy conditions are observable across multiple frames",
    "night": "increase duration_s to >=20s since dark scenes need more frames to be visually verifiable",
    "clear": "verify cloudiness/precipitation are both at minimum in the weather preset",
    "cloudy": "verify cloudiness is elevated but precipitation remains at zero",
}


def _correction_for_predicate(predicate: str, spec: ScenarioSpec) -> str:
    """Generates a correction instruction for a single failed predicate,
    parameterized with this scenario's actual requested values."""
    if predicate == "weather_correct":
        guidance = WEATHER_GUIDANCE.get(spec.weather, "re-check the weather preset against the prompt")
        return f"- weather_correct failed: requested weather='{spec.weather}'. {guidance}."
    if predicate == "pedestrian_present":
        return (
            "- pedestrian_present failed: pedestrian=true was requested but not visually confirmed. "
            "Prefer Town01 or Town03 (denser sidewalk geometry); use spawn_point_index 0-5 for sidewalk proximity."
        )
    if predicate == "vehicles_present":
        return (
            f"- vehicles_present failed: requested num_vehicles={spec.num_vehicles} but count was not "
            f"approximately confirmed. Check spawn_point availability in {spec.town}; consider increasing "
            f"num_vehicles slightly to compensate for spawn collisions."
        )
    if predicate == "event_observable":
        guidance = EVENT_GUIDANCE.get(spec.event_type, "re-examine event_type against the prompt's description")
        return f"- event_observable failed: requested event_type='{spec.event_type}'. Needs {guidance}."
    return f"- {predicate} failed: re-examine the corresponding ScenarioSpec field(s)."


def reflect(
    original_prompt: str,
    spec: ScenarioSpec,
    verification_result: HybridVerificationResult,
    iteration: int,
) -> str:
    """
    Build a refinement_context string to prepend to the next Generator
    call. This is the entire interface between the Verifier and the next
    Generator iteration -- keep it self-contained and explicit.

    `spec` is the ScenarioSpec that was actually executed (and verified)
    in the iteration that just failed -- corrections are parameterized
    with its actual values (e.g. "requested weather='rain'"), not generic
    text, since the predicate names alone don't carry that context.
    """
    missing = verification_result.missing_elements  # derived from failed_predicates, formal & deterministic
    excess = verification_result.vlm_result.excess_elements
    score = verification_result.predicate_agreement  # formal predicate-agreement score, not an opaque holistic score

    corrections = [_correction_for_predicate(p, spec) for p in missing] or [
        "- No applicable predicates failed; broadly re-examine all fields "
        "for alignment with the original prompt."
    ]

    excess_notes = ""
    if excess:
        excess_notes = (
            f"\nAvoid reintroducing unintended elements from the previous "
            f"attempt: {excess}. Constrain the scenario more tightly to "
            f"only what the prompt describes."
        )

    context = f"""[REFLECTION -- iteration {iteration}]
Previous predicate agreement: {score:.0%} (applicable predicates: {verification_result.applicable_predicates}; sample uncertainty std: {verification_result.predicate_agreement_std:.3f} over {verification_result.n_samples} samples)
Verifier notes: {verification_result.vlm_result.notes}
Failed predicates (the specific things that did NOT hold): {missing}

REQUIRED CORRECTIONS for this regeneration:
{chr(10).join(corrections)}{excess_notes}

Apply all corrections above. Preserve any field NOT implicated in the
failed-predicate list, since it was already verified as correct.""".strip()

    logger.info(
        "Reflector iteration %s: predicate_agreement=%.2f failed_predicates=%s", iteration, score, missing
    )
    return context
