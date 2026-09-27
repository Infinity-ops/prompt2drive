import tempfile, os
from engine.scenario_spec import ScenarioSpec
import generator.agent as ga

class FakeBackend:
    name="fake"
    def generate_structured(self, system_prompt, user_prompt, schema):
        return ScenarioSpec(weather="rain", town="Town04", num_vehicles=3, pedestrian=True,
                            event_type="crossing", duration_s=15, spawn_point_index=20, notes="highway")

def test_forbidden_town_is_clamped(tmp_path, monkeypatch):
    monkeypatch.setattr(ga, "CACHE_PATH", tmp_path / "gen_cache")
    spec = ga.Generator(FakeBackend(), preferred_towns=("Town01","Town02")).generate("a pedestrian crosses in rain")
    assert spec.town == "Town01"
    assert (spec.weather, spec.pedestrian, spec.event_type, spec.num_vehicles) == ("rain", True, "crossing", 3)
    assert "Town04 -> Town01" in spec.notes

def test_no_profile_keeps_town(tmp_path, monkeypatch):
    monkeypatch.setattr(ga, "CACHE_PATH", tmp_path / "gen_cache")
    assert ga.Generator(FakeBackend()).generate("x").town == "Town04"
