"""
webui/replay.py

Replay mode: re-streams a RECORDED Prompt-to-Drive run (pipeline_run.json +
saved frames) through the same event protocol the live pipeline uses.
Needs no CARLA server, no GPU and no API keys, which is what makes a public
demo possible. Nothing is generated or re-scored: every number shown comes
from the recorded file, and the UI labels the session as a replay.

Expected layout (exactly what the pipeline writes):

    <runs_dir>/<scenario_id>/pipeline_run.json
    <runs_dir>/<scenario_id>/iter_0/frames/frame_000.png ...
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from webui.frames import file_to_jpeg_b64

Emit = Callable[[str, dict], None]
FrameEmit = Callable[[str, int], None]

# Replays are paced from the recorded latencies, compressed so a 3-minute
# run plays back in well under a minute.
SPEEDUP = 6.0
MIN_STAGE_S, MAX_STAGE_S = 0.6, 4.0
MAX_FRAMES_PER_ITER = 12        # sparse Verifier frames (frames/*.png)
MAX_PREVIEW_PER_ITER = 150      # smooth UI preview (preview/*.jpg), when recorded
PREVIEW_FRAME_S = 0.12


@dataclass
class RecordedRun:
    scenario_id: str
    prompt: str
    final_status: str
    n_iterations: int
    path: Path

    def summary(self) -> dict:
        return dict(scenario_id=self.scenario_id, prompt=self.prompt,
                    final_status=self.final_status, iterations=self.n_iterations)


def list_runs(runs_dir: str | Path) -> list[RecordedRun]:
    root = Path(runs_dir)
    runs: list[RecordedRun] = []
    if not root.is_dir():
        return runs
    for f in sorted(root.glob("*/pipeline_run.json")):
        try:
            data = json.loads(f.read_text())
            runs.append(RecordedRun(
                scenario_id=data["scenario_id"], prompt=data["prompt"],
                final_status=data.get("final_status", "unknown"),
                n_iterations=len(data.get("iterations", [])), path=f.parent,
            ))
        except (OSError, ValueError, KeyError):
            continue  # skip malformed runs rather than failing the whole listing
    return runs


def find_run(runs_dir: str | Path, prompt: str) -> Optional[RecordedRun]:
    wanted = " ".join(prompt.split()).lower()
    for run in list_runs(runs_dir):
        if " ".join(run.prompt.split()).lower() == wanted:
            return run
    return None


def _pace(seconds: Optional[float], sleep: Callable[[float], None]) -> None:
    s = (seconds or 0.0) / SPEEDUP
    sleep(min(max(s, MIN_STAGE_S), MAX_STAGE_S))


def replay_run(run: RecordedRun, emit: Emit, emit_frame: FrameEmit,
               sleep: Callable[[float], None] = time.sleep,
               should_stop: Callable[[], bool] = lambda: False) -> None:
    data = json.loads((run.path / "pipeline_run.json").read_text())
    iters = data.get("iterations", [])
    emit("run_start", dict(scenario_id=data["scenario_id"], prompt=data["prompt"],
                           mode=data.get("mode", "full"), max_iterations=len(iters), replay=True))

    for idx, it in enumerate(iters):
        if should_stop():
            return
        n = idx + 1
        lat = it.get("latencies", {}) or {}
        spec = it.get("spec", {}) or {}

        emit("generator", dict(iteration=n, refinement_context=it.get("refinement_context")))
        _pace(lat.get("generator_s"), sleep)
        emit("generator_done", dict(iteration=n, spec=spec, latency_s=lat.get("generator_s")))

        emit("engine", dict(iteration=n, town=spec.get("town")))
        it_dir = run.path / f"iter_{it.get('iteration', idx)}"
        preview = sorted((it_dir / "preview").glob("p_*.jpg"))[:MAX_PREVIEW_PER_ITER]
        if preview:
            frames, per_frame = preview, PREVIEW_FRAME_S
        else:
            frames = sorted((it_dir / "frames").glob("frame_*.png"))[:MAX_FRAMES_PER_ITER]
            per_frame = min(max((lat.get("engine_s") or 0) / SPEEDUP, 1.0), 8.0) / max(len(frames), 1)
        for fp in frames:
            if should_stop():
                return
            try:
                emit_frame(file_to_jpeg_b64(fp), n)
            except OSError:
                pass
            sleep(per_frame)
        if not frames:
            _pace(lat.get("engine_s"), sleep)
        emit("engine_done", dict(iteration=n, success=it.get("execution_success", False),
                                 n_frames=len(list((it_dir / "frames").glob("frame_*.png"))),
                                 error=None, latency_s=lat.get("engine_s")))

        emit("verifier", dict(iteration=n))
        _pace(lat.get("verifier_s"), sleep)
        # pipeline_run.json stores the hybrid score, pass flag and failed
        # predicates, but not per-predicate agreement -> leave those null
        # instead of inventing them.
        emit("verifier_done", dict(
            iteration=n, score=it.get("verification_score"),
            predicate_agreement=None, predicate_agreement_std=None,
            predicates=None, applicable_predicates=None,
            missing_elements=it.get("missing_elements", []),
            passed=it.get("verification_passed", False),
            threshold=None, backend=None, notes="", latency_s=lat.get("verifier_s"),
        ))

        if idx + 1 < len(iters):
            emit("reflector", dict(iteration=n, missing_elements=it.get("missing_elements", [])))
            _pace(lat.get("reflector_s"), sleep)
            emit("reflector_done", dict(iteration=n,
                                        refinement_context=iters[idx + 1].get("refinement_context") or "",
                                        latency_s=lat.get("reflector_s")))

    emit("done", dict(scenario_id=data["scenario_id"], final_status=data.get("final_status", "unknown"),
                      iterations=len(iters), total_wall_time_s=data.get("total_wall_time_s")))