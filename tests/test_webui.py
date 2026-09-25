"""
tests/test_webui.py

Covers the UI integration without CARLA, GPUs or API keys:
  * the pipeline's optional status_callback / live_frame_callback hook
    (event order, and that it cannot change the persisted results),
  * replay of a recorded run,
  * the web server end-to-end over a WebSocket in replay mode.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.carla_runner import ExecutionResult
from engine.scenario_spec import ScenarioSpec
from orchestrator.pipeline import PromptToDrivePipeline
from verifier.agent import HybridVerificationResult, VerificationResult

SPEC = ScenarioSpec(weather="rain", town="Town01", num_vehicles=2, pedestrian=True,
                    event_type="crossing", duration_s=15)


class FakeGenerator:
    def generate(self, prompt, refinement_context=""):
        return SPEC


class FakeRunner:
    def run(self, spec, scenario_id, output_dir, live_frame_callback=None, rng_seed=None):
        if live_frame_callback:
            live_frame_callback("frame-object")
        return ExecutionResult(scenario_id=scenario_id, success=True, frame_paths=["a.png"], log_path="")


class FakeVerifier:
    threshold = 0.75

    def __init__(self):
        self.calls = 0

    def verify(self, **kw):
        self.calls += 1
        ok = self.calls > 1  # fail first, pass second
        vlm = VerificationResult(weather_correct=ok, pedestrian_present=True, vehicles_present=True,
                                 event_observable=True, intent_alignment_score=0.8, excess_elements=[],
                                 confidence=0.9, notes="n")
        return HybridVerificationResult(
            vlm_result=vlm, predicate_agreement=1.0 if ok else 0.75, predicate_agreement_std=0.0,
            applicable_predicates=list(vlm.predicates), failed_predicates=[] if ok else ["weather_correct"],
            n_samples=3, log_checks_passed=4, log_checks_total=4, execution_success=True,
            final_score=0.9 if ok else 0.6, overall_pass=ok, backend_name="fake")


class FakeReflector:
    def reflect(self, **kw):
        return "Set weather to rain."


def _pipeline(tmp_path):
    return PromptToDrivePipeline(generator=FakeGenerator(), verifier=FakeVerifier(),
                                 reflector=FakeReflector(), runner=FakeRunner(),
                                 max_iterations=3, output_root=str(tmp_path))


def _strip_timing(d: dict) -> dict:
    d = json.loads(json.dumps(d))
    for k in list(d):
        if k.startswith("total_") or k == "peak_memory_mb":
            d.pop(k)
    for it in d["iterations"]:
        it.pop("latencies"); it.pop("peak_memory_mb")
    return d


def test_status_callback_order_and_frames(tmp_path):
    events, frames = [], []
    run = _pipeline(tmp_path).run("rainy crossing", "t1",
                                  status_callback=lambda s, p: events.append((s, p)),
                                  live_frame_callback=frames.append)
    stages = [s for s, _ in events]
    assert stages == [
        "run_start",
        "generator", "generator_done", "engine", "engine_done", "verifier", "verifier_done",
        "reflector", "reflector_done",
        "generator", "generator_done", "engine", "engine_done", "verifier", "verifier_done",
        "done",
    ]
    assert run.final_status == "passed"
    assert frames == ["frame-object", "frame-object"]
    vd = [p for s, p in events if s == "verifier_done"]
    assert vd[0]["missing_elements"] == ["weather_correct"] and vd[0]["iteration"] == 1
    assert vd[1]["passed"] is True and vd[1]["threshold"] == 0.75
    json.dumps([p for _, p in events])  # every payload must be JSON-serialisable


def test_callback_cannot_change_results(tmp_path):
    def boom(stage, payload):
        raise RuntimeError("UI crashed")

    a = _pipeline(tmp_path / "a").run("rainy crossing", "t2")
    b = _pipeline(tmp_path / "b").run("rainy crossing", "t2", status_callback=boom)
    ja = json.loads((tmp_path / "a" / "t2" / "pipeline_run.json").read_text())
    jb = json.loads((tmp_path / "b" / "t2" / "pipeline_run.json").read_text())
    assert a.final_status == b.final_status == "passed"
    assert _strip_timing(ja) == _strip_timing(jb)


# ---------------------------------------------------------------- replay
@pytest.fixture
def recorded(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    events = []
    _pipeline(tmp_path).run("A pedestrian crosses in rain.", "rec01",
                            status_callback=lambda s, p: events.append(s))
    frames = tmp_path / "rec01" / "iter_0" / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    PIL.new("RGB", (64, 36), (40, 60, 80)).save(frames / "frame_000.png")
    return tmp_path


def test_replay_streams_recorded_run(recorded):
    from webui import replay

    runs = replay.list_runs(recorded)
    assert [r.scenario_id for r in runs] == ["rec01"]
    assert replay.find_run(recorded, "  a pedestrian CROSSES in rain. ") is not None
    events, frames = [], []
    replay.replay_run(runs[0], lambda s, p: events.append((s, p)),
                      lambda jpg, it: frames.append(it), sleep=lambda s: None)
    stages = [s for s, _ in events]
    assert stages[0] == "run_start" and stages[-1] == "done"
    assert stages.count("verifier_done") == 2 and "reflector_done" in stages
    assert frames == [1]
    refl = next(p for s, p in events if s == "reflector_done")
    assert refl["refinement_context"] == "Set weather to rain."
    assert events[-1][1]["final_status"] == "passed"


def test_server_replay_over_websocket(recorded, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from webui import replay, server

    monkeypatch.setattr(replay, "SPEEDUP", 1e9)
    monkeypatch.setattr(replay, "MIN_STAGE_S", 0.0)
    cfg = server.UIConfig(mode="replay", runs_dir=str(recorded), token="s3cret")
    with TestClient(server.create_app(cfg)) as client:
        assert client.get("/api/config").status_code == 401
        conf = client.get("/api/config?token=s3cret").json()
        assert conf["mode"] == "replay" and conf["examples"][0]["scenario_id"] == "rec01"
        assert client.get("/").status_code == 200

        with client.websocket_connect("/ws?token=s3cret") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"type": "generate", "prompt": "A pedestrian crosses in rain."})
            stages = []
            while True:
                msg = ws.receive_json()
                if msg["type"] == "event":
                    stages.append(msg["stage"])
                    if msg["stage"] in ("done", "error"):
                        break
            assert stages[-1] == "done"

        with client.websocket_connect("/ws?token=s3cret") as ws:
            ws.receive_json()
            ws.send_json({"type": "generate", "prompt": "Something never recorded"})
            while (msg := ws.receive_json())["type"] != "event":
                pass
            assert msg["stage"] == "error"
