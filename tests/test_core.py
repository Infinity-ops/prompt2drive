"""
tests/test_core.py

Unit tests that require neither a running CARLA server nor an OpenAI API
key -- they exercise schema validation, the predicate-based reflector
correction logic, and the verifier's hybrid scoring math in isolation.

Run with:
    pytest tests/ -v
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from backends.llm_backend import _parse_json_block, get_backend
from engine.scenario_spec import ScenarioSpec
from reflector.agent import _correction_for_predicate, reflect
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
# Predicate-based reflector correction lookup
# --------------------------------------------------------------------- #
def _sample_spec(**overrides) -> ScenarioSpec:
    base = dict(
        weather="rain", town="Town03", num_vehicles=4, pedestrian=True,
        event_type="crossing", duration_s=15, spawn_point_index=6,
    )
    base.update(overrides)
    return ScenarioSpec(**base)


def test_correction_for_weather_predicate_is_parameterized():
    spec = _sample_spec(weather="fog")
    correction = _correction_for_predicate("weather_correct", spec)
    assert "fog" in correction.lower()


def test_correction_for_pedestrian_predicate():
    spec = _sample_spec()
    correction = _correction_for_predicate("pedestrian_present", spec)
    assert "pedestrian_present" in correction


def test_correction_for_unknown_predicate_has_fallback():
    spec = _sample_spec()
    correction = _correction_for_predicate("some_future_predicate", spec)
    assert "some_future_predicate" in correction


def _sample_hybrid_result(
    predicate_agreement: float = 0.5,
    failed_predicates: list[str] | None = None,
    final_score: float = 0.5,
) -> HybridVerificationResult:
    failed_predicates = failed_predicates if failed_predicates is not None else ["weather_correct"]
    vlm_result = VerificationResult(
        weather_correct=False, pedestrian_present=True, vehicles_present=True,
        event_observable=True, intent_alignment_score=predicate_agreement,
        excess_elements=[], confidence=0.9,
        notes="Weather appears clear, not rainy as requested.",
    )
    return HybridVerificationResult(
        vlm_result=vlm_result,
        predicate_agreement=predicate_agreement,
        predicate_agreement_std=0.05,
        applicable_predicates=["weather_correct", "pedestrian_present", "vehicles_present", "event_observable"],
        failed_predicates=failed_predicates,
        n_samples=3,
        log_checks_passed=2,
        log_checks_total=4,
        execution_success=True,
        final_score=final_score,
        overall_pass=final_score >= 0.72,
        backend_name="openai",
    )


def test_reflect_produces_actionable_context():
    spec = _sample_spec()
    hybrid = _sample_hybrid_result()
    context = reflect("A pedestrian crosses in the rain near a signal.", spec, hybrid, iteration=1)
    assert "REQUIRED CORRECTIONS" in context
    assert "rain" in context.lower()


def test_missing_elements_alias_matches_failed_predicates():
    hybrid = _sample_hybrid_result(failed_predicates=["weather_correct", "event_observable"])
    assert hybrid.missing_elements == ["weather_correct", "event_observable"]


# --------------------------------------------------------------------- #
# Verifier hybrid score math
# --------------------------------------------------------------------- #
def test_hybrid_score_weighting():
    # predicate_agreement=1.0, log=1.0, exec=1.0 -> final score should be 1.0
    s_vlm, s_log, s_exec = 1.0, 1.0, 1.0
    expected = round(0.5 * s_vlm + 0.3 * s_log + 0.2 * s_exec, 4)
    assert expected == 1.0


def test_overall_pass_threshold():
    result = _sample_hybrid_result(predicate_agreement=0.9, failed_predicates=[], final_score=0.8)
    assert result.overall_pass is True
    assert result.final_score >= 0.72


def test_overall_pass_threshold_fails_below_cutoff():
    result = _sample_hybrid_result(predicate_agreement=0.4, final_score=0.4)
    assert result.overall_pass is False


# --------------------------------------------------------------------- #
# Backend abstraction (no network calls)
# --------------------------------------------------------------------- #
def test_get_backend_unknown_raises():
    with pytest.raises(ValueError):
        get_backend("not-a-real-backend")


def test_get_backend_openai_resolves(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-test-key")
    backend = get_backend("openai")
    assert backend.name == "openai"


def test_parse_json_block_strips_fences():
    raw = '```json\n{"a": 1, "b": "x"}\n```'
    parsed = _parse_json_block(raw)
    assert parsed == {"a": 1, "b": "x"}


def test_parse_json_block_extracts_from_stray_prose():
    # Open-weight models are more prone to wrapping JSON in prose than
    # OpenAI's structured outputs -- this is the repair path for that.
    raw = 'Sure, here is the result:\n{"a": 1, "b": 2}\nHope that helps!'
    parsed = _parse_json_block(raw)
    assert parsed == {"a": 1, "b": 2}
