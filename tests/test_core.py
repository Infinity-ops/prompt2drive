"""
tests/test_core.py

Unit tests that require neither a running CARLA server nor an OpenAI API
key -- they exercise the schema validation, reflector correction logic,
and verifier scoring math in isolation. Run with:

    pytest tests/ -v
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from engine.scenario_spec import ScenarioSpec
from reflector.agent import _lookup_correction, reflect
from verifier.agent import HybridVerificationResult, VerificationResult


# --------------------------------------------------------------------- #
# ScenarioSpec schema
# --------------------------------------------------------------------- #
def test_scenario_spec_valid():
    spec = ScenarioSpec(
        weather="rain", town="Town03", num_vehicles=4, pedestrian=True,
        event_type="crossing", duration_s=15, spawn_point_index=6,
    )
    assert spec.weather == "rain"
    assert spec.cyclist is False  # default


def test_scenario_spec_rejects_invalid_weather():
    with pytest.raises(ValidationError):
        ScenarioSpec(
            weather="tornado", town="Town03", num_vehicles=4, pedestrian=True,
            event_type="crossing", duration_s=15,
        )


def test_scenario_spec_rejects_out_of_range_vehicles():
    with pytest.raises(ValidationError):
        ScenarioSpec(
            weather="clear", town="Town01", num_vehicles=999, pedestrian=False,
            event_type="overtaking", duration_s=10,
        )


def test_scenario_spec_enforces_min_duration_for_crossing():
    spec = ScenarioSpec(
        weather="clear", town="Town01", num_vehicles=2, pedestrian=True,
        event_type="crossing", duration_s=8,  # below the 12s floor for crossing
    )
    assert spec.duration_s == 12


# --------------------------------------------------------------------- #
# Reflector correction lookup
# --------------------------------------------------------------------- #
def test_lookup_correction_known_element():
    correction = _lookup_correction("traffic_light")
    assert "Town03" in correction or "Town04" in correction


def test_lookup_correction_unknown_element_has_fallback():
    correction = _lookup_correction("unicorn_sighting")
    assert "unicorn_sighting" in correction


def test_reflect_produces_actionable_context():
    vlm_result = VerificationResult(
        weather_correct=False, pedestrian_present=True, vehicles_present=True,
        event_observable=True, intent_alignment_score=0.55,
        missing_elements=["rain", "traffic_light"], excess_elements=[],
        confidence=0.9, notes="Weather appears clear, not rainy as requested.",
    )
    hybrid = HybridVerificationResult(
        vlm_result=vlm_result, log_checks_passed=2, log_checks_total=4,
        execution_success=True, final_score=0.55, overall_pass=False,
    )
    context = reflect("A pedestrian crosses in the rain near a signal.", hybrid, iteration=1)
    assert "REQUIRED CORRECTIONS" in context
    assert "rain" in context.lower()


# --------------------------------------------------------------------- #
# Verifier hybrid score math
# --------------------------------------------------------------------- #
def test_hybrid_score_weighting():
    # s_vlm=1.0, s_log=1.0, s_exec=1.0 -> final score should be 1.0
    vlm_result = VerificationResult(
        weather_correct=True, pedestrian_present=True, vehicles_present=True,
        event_observable=True, intent_alignment_score=1.0,
        missing_elements=[], excess_elements=[], confidence=1.0, notes="Perfect match.",
    )
    s_vlm, s_log, s_exec = 1.0, 1.0, 1.0
    expected = round(0.5 * s_vlm + 0.3 * s_log + 0.2 * s_exec, 4)
    assert expected == 1.0


def test_overall_pass_threshold():
    result = HybridVerificationResult(
        vlm_result=VerificationResult(
            weather_correct=True, pedestrian_present=True, vehicles_present=True,
            event_observable=True, intent_alignment_score=0.9,
            missing_elements=[], excess_elements=[], confidence=0.9, notes="Good.",
        ),
        log_checks_passed=4, log_checks_total=4, execution_success=True,
        final_score=0.8, overall_pass=True,
    )
    assert result.overall_pass is True
    assert result.final_score >= 0.72
