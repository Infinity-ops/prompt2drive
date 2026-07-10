"""
reflector/agent.py

Agent 3 — Reflector.
Performs structured root-cause analysis of a failed VerificationResult and
uses an LLM backend (via Cloud) to produce a targeted refinement_context 
string for the Generator's next iteration. Fully falls back to rule-based engineering.
"""
from __future__ import annotations

import logging
from typing import Optional, Any
from verifier.agent import HybridVerificationResult

logger = logging.getLogger("prompt_to_drive.reflector")

# Master Rule-Based Lookup Table (Serves as your primary backup strategy)
ELEMENT_CORRECTIONS: dict[str, str] = {
    "traffic_light": (
        "Ensure the scenario uses a town with signalized intersections "
        "(Town03 or Town04 are recommended). Set spawn_point_index in the 5-12 range "
        "so actors are placed near a junction."
    ),
    "junction": (
        "Set spawn_point_index in the 5-12 range to place actors near a "
        "signalized junction (consider Town03 or Town04, but select based on other "
        "scenario requirements)."
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
        "Set pedestrian=true. Select a town with dense pedestrian infrastructure "
        "(Town01 or Town03 are options). Use spawn_point_index in the 0-5 range for "
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


class Reflector:
    def __init__(self, backend: Optional[Any] = None):
        """
        Initializes the Reflector Agent with a cloud or local text backend instance.
        If no backend is provided, it defaults immediately to deterministic rules.
        """
        self.backend = backend

    def reflect(
        self,
        prompt: str,  # Fixed parameter name to align with pipeline caller
        verification_result: HybridVerificationResult,
        iteration: int,
        preferred_towns: tuple[str, ...] | None = None,
    ) -> str:
        """
        Builds a refinement context string using either cloud LLM inference or 
        rule-based backup logic depending on availability and system exceptions.
        """
        missing = verification_result.missing_elements
        excess = getattr(verification_result.vlm_result, "excess_elements", None)
        score = verification_result.final_score
        
        # Safe dot-notation object property lookup to prevent AttributeError
        notes = getattr(verification_result.vlm_result, "notes", "No additional verifier notes provided.")

        # --- Rule-Based Verification Corrections Assembly ---
        corrections = []
        if missing:
            for e in missing:
                matched = False
                element_lower = e.lower()
                for key, correction in ELEMENT_CORRECTIONS.items():
                    if key in element_lower:
                        corrections.append(f"- {correction}")
                        matched = True
                        break
                if not matched:
                    corrections.append(
                        f"- Element '{e}' was reported missing. Re-examine relevant ScenarioSpec fields "
                        f"to ensure this element is present and visually verifiable."
                    )
        else:
            corrections.append("- No specific missing elements identified; broadly re-examine all fields.")

        excess_notes = ""
        if excess:
            excess_notes = f"\nAvoid reintroducing unintended elements from previous attempt: {excess}."

        hardware_note = ""
        if preferred_towns:
            towns_str = ", ".join(preferred_towns)
            hardware_note = f"\nIMPORTANT: Hardware constraint active. Only use these maps: {towns_str}."

        # --- Pipeline Routing Action ---
        if not self.backend:
            logger.info("No Reflector backend assigned. Executing deterministic rule-based reflection.")
            return self._format_rule_context(iteration, score, missing, notes, corrections, excess_notes, hardware_note)

        logger.info("Invoking cloud LLM Reflector backend for iteration %d", iteration)

        system_prompt = (
            "You are an expert autonomous driving scenario engineer. Your job is to analyze a failed "
            "simulation run and output explicit instructions for the scenario generator to fix it. "
            "Keep instructions high-level, action-oriented, and conversational. Do not output code."
        )

        user_content = f"""
User Target Intent Prompt: "{prompt}"

Simulation Execution Failure Context:
- Current Alignment Score: {score:.2f} / 1.00
- Confirmed Missing Elements: {missing}
- Visual Verifier Notes: "{notes}"
{hardware_note}

Review the baseline rule corrections for these failures:
{chr(10).join(corrections)}
{excess_notes}

Synthesize these insights into a concise, human-like instruction list directing the generator on how to fix the fields.
        """.strip()

        try:
            # Executes cloud inference (adapts cleanly to both OpenAI wrapper format and local string generation scripts)
            if hasattr(self.backend, "generate"):
                response = self.backend.generate(prompt=user_content, system_prompt=system_prompt)
            else:
                # Fallback implementation if the api wrapper uses a basic call signature
                response = self.backend(f"{system_prompt}\n\n{user_content}")

            return f"""[LLM REFLECTION -- iteration {iteration}]
Previous score: {score:.2f}
{response.strip()}

Apply all adjustments above. Preserve fields not implicated in errors.""".strip()

        except Exception as e:
            logger.error("Cloud LLM reflection inference failed: %s. Activating Rule-Based fallback safety block.", e)
            return self._format_rule_context(iteration, score, missing, notes, corrections, excess_notes, hardware_note)

    def _format_rule_context(
        self, iteration: int, score: float, missing: list[str], notes: str, 
        corrections: list[str], excess_notes: str, hardware_note: str
    ) -> str:
        """Helper method to format the rule-based backup block identically to old script requirements."""
        affected_fields = sorted({e.split("_")[0] for e in missing}) if missing else ["all fields"]
        
        return f"""[RULE REFLECTION -- iteration {iteration}]
Previous semantic alignment score: {score:.2f}
Verifier notes: {notes}
Missing elements identified: {missing}

REQUIRED CORRECTIONS for this regeneration:
{chr(10).join(corrections)}{excess_notes}{hardware_note}

Apply all corrections above. Preserve any field NOT implicated in the missing-element list.
Fields most likely needing adjustment: {", ".join(affected_fields)}.""".strip()


# Main operational interface entry point mapped exactly to eval/run_full_evaluation.py
def reflect(
    prompt: str,  # Fixed wrapper positional name
    verification_result: HybridVerificationResult,
    iteration: int,
    preferred_towns: tuple[str, ...] | None = None,
    backend: Optional[Any] = None,
) -> str:
    agent = Reflector(backend=backend)
    return agent.reflect(prompt, verification_result, iteration, preferred_towns)