"""
reflector/agent.py

Agent 3 — Reflector.

Performs structured root-cause analysis of a failed VerificationResult and
produces a targeted refinement_context string for the Generator's next
iteration. The design principle (Section 3.4 of the paper) is: every
correction maps exactly one missing element to one specific, actionable
instruction. This keeps reflection auditable and avoids regressing fields
that were already verified correct.
"""
from __future__ import annotations

import logging

from verifier.agent import HybridVerificationResult

logger = logging.getLogger("prompt_to_drive.reflector")

# Rule-based corrections for the most common missing-element categories.
# Extend this table as new failure modes are observed in your own dataset.
ELEMENT_CORRECTIONS: dict[str, str] = {
    "traffic_light": (
        "Ensure the scenario uses Town03 or Town04, both of which have "
        "signalized intersections. Set spawn_point_index in the 5-12 range "
        "so actors are placed near a junction."
    ),
    "junction": (
        "Set spawn_point_index in the 5-12 range to place actors near a "
        "signalized junction in Town03 or Town04."
    ),
    "rain": (
        "Set weather='rain'. CARLA renders this with high precipitation "
        "and wet road reflections, which should be clearly visible."
    ),
    "fog": (
        "Set weather='fog'. Increase duration_s to >=15 so the foggy "
        "conditions are observable across multiple frames."
    ),
    "night": (
        "Set weather='night'. Increase duration_s to >=20 since dark "
        "scenes need more frames to be visually verifiable."
    ),
    "pedestrian": (
        "Set pedestrian=true. Prefer Town01 or Town03 which have denser "
        "sidewalk geometry. Use spawn_point_index in the 0-5 range for "
        "sidewalk proximity."
    ),
    "cyclist": "Set cyclist=true and ensure event_type matches the described cyclist behavior.",
    "crossing": (
        "Set event_type='crossing' and pedestrian=true. duration_s must be "
        ">=15s to capture the full crossing arc. Include num_vehicles>=2 "
        "for realistic context."
    ),
    "lane_change": (
        "Set event_type='lane_change'. Requires a straight multi-lane "
        "segment: prefer Town04 (highway) or Town01 (main boulevard)."
    ),
    "emergency_brake": (
        "Set event_type='emergency_brake'. An NPC vehicle must be spawned "
        "ahead of the ego vehicle. Use num_vehicles>=3 for realistic density."
    ),
    "vehicles": "Increase num_vehicles -- the previous count was insufficient for the described density.",
    "weather": "Re-check the weather field against the prompt; the previous value did not visually match.",
}


def _lookup_correction(element: str) -> str:
    element_lower = element.lower()
    for key, correction in ELEMENT_CORRECTIONS.items():
        if key in element_lower:
            return f"- {correction}"
    return (
        f"- Element '{element}' was reported missing from the simulation. "
        f"Re-examine the relevant ScenarioSpec field(s) and ensure this "
        f"element is present and visually verifiable."
    )


def reflect(
    original_prompt: str,
    verification_result: HybridVerificationResult,
    iteration: int,
) -> str:
    """
    Build a refinement_context string to prepend to the next Generator
    call. This is the entire interface between the Verifier and the next
    Generator iteration -- keep it self-contained and explicit.
    """
    missing = verification_result.vlm_result.missing_elements
    excess = verification_result.vlm_result.excess_elements
    score = verification_result.final_score

    corrections = [_lookup_correction(e) for e in missing] or [
        "- No specific missing elements identified; broadly re-examine all "
        "fields for alignment with the original prompt."
    ]

    excess_notes = ""
    if excess:
        excess_notes = (
            f"\nAvoid reintroducing unintended elements from the previous "
            f"attempt: {excess}. Constrain the scenario more tightly to "
            f"only what the prompt describes."
        )

    affected_fields = sorted({e.split("_")[0] for e in missing}) if missing else ["all fields"]

    context = f"""[REFLECTION -- iteration {iteration}]
Previous semantic alignment score: {score:.2f} (pass threshold: {verification_result.overall_pass and "PASSED" or "0.72"})
Verifier notes: {verification_result.vlm_result.notes}
Missing elements identified: {missing}

REQUIRED CORRECTIONS for this regeneration:
{chr(10).join(corrections)}{excess_notes}

Apply all corrections above. Preserve any field NOT implicated in the
missing-element list, since it was already verified as correct.
Fields most likely needing adjustment: {", ".join(affected_fields)}.""".strip()

    logger.info(
        "Reflector iteration %s: score=%.2f missing=%s", iteration, score, missing
    )
    return context
