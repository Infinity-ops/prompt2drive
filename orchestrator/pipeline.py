"""
orchestrator/pipeline.py

Ties the three agents together into the closed-loop Generation ->
Execution -> Verification -> Reflection cycle described in Section 3 of
the paper (Algorithm 1).

LATENCY INSTRUMENTATION (added for ESWEEK Reviewer 2)
------------------------------------------------------
Every agent call is now wrapped in time.perf_counter() timers. The
results are stored in IterationRecord.agent_latencies_s and surfaced in
the persisted pipeline_run.json so that eval/compute_metrics.py can
aggregate them into the systems-evaluation tables the reviewer asked for:

  - Per-agent latency breakdown (Generator / Engine / Verifier / Reflector)
  - Total pipeline wall time
  - Peak memory footprint per iteration (via resource/tracemalloc)

IMPORTANT FRAMING:
Prompt-to-Drive operates at SCENARIO-AUTHORING TIME, not inside the
vehicle control loop. The relevant latency metric is therefore SCENARIO 
THROUGHPUT, not control-loop determinism (Hz).
"""
from __future__ import annotations

import gc
import json
import logging
import resource
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Any

try:  # torch is only used for an optional CUDA cache flush between iterations
    import torch
except ImportError:  # pragma: no cover - UI/replay installs don't ship torch
    torch = None

from engine.carla_runner import CarlaRunner, ExecutionResult
from engine.scenario_spec import ScenarioSpec
from generator.agent import Generator
from verifier.agent import HybridVerificationResult, Verifier
from reflector.agent import Reflector, reflect  # Supports both Class or functional reflection

logger = logging.getLogger("prompt_to_drive.orchestrator")

MAX_ITERATIONS = 3

# Signature of the optional UI hook: status_callback(stage, payload).
# Stages, in order per iteration (iteration numbers in payloads are 1-based):
#   run_start, generator, generator_done, engine, engine_done,
#   verifier, verifier_done, [reflector, reflector_done], ..., done
# The hook is observational only: it is called OUTSIDE every latency timer
# and any exception it raises is swallowed, so passing it (or not) cannot
# change specs, scores, latencies or the persisted pipeline_run.json.
StatusCallback = Callable[[str, dict], None]


def _emit(callback: Optional[StatusCallback], stage: str, payload: dict) -> None:
    if callback is None:
        return
    try:
        callback(stage, payload)
    except Exception:  # noqa: BLE001 - a broken UI must never break a run
        logger.debug("status_callback raised for stage %s", stage, exc_info=True)


# -----------------------------------------------------------------------
# Latency context manager
# -----------------------------------------------------------------------
@contextmanager
def _timer(label: str, store: dict):
    """Measures wall-clock duration of a code block using time.perf_counter()."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[f"{label}_s"] = round(time.perf_counter() - t0, 4)


def _peak_memory_mb() -> float:
    """Returns the peak RSS memory usage of this process in megabytes."""
    try:
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)
    except Exception:
        return 0.0


# -----------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------
@dataclass
class AgentLatencies:
    generator_s:      float = 0.0
    engine_s:         float = 0.0
    verifier_s:       float = 0.0
    reflector_s:      float = 0.0
    total_pipeline_s: float = 0.0


@dataclass
class IterationRecord:
    iteration: int
    spec: dict
    refinement_context: Optional[str]
    execution_success: bool
    verification_score: float
    verification_passed: bool
    missing_elements: list[str]
    latencies: AgentLatencies = field(default_factory=AgentLatencies)
    peak_memory_mb: float = 0.0


@dataclass
class PipelineRun:
    scenario_id: str
    prompt: str
    mode: str
    iterations: list[IterationRecord] = field(default_factory=list)
    final_status: str = "in_progress"
    total_wall_time_s: float = 0.0
    total_generator_s: float = 0.0
    total_engine_s:    float = 0.0
    total_verifier_s:  float = 0.0
    total_reflector_s: float = 0.0
    peak_memory_mb:    float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# -----------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------
class PromptToDrivePipeline:
    """
    The main closed-loop entry point managing three dedicated LLM/VLM models.
    """

    def __init__(
        self,
        generator: Optional[Generator] = None,
        verifier: Optional[Verifier] = None,
        reflector: Optional[Reflector] = None,
        runner: Optional[CarlaRunner] = None,
        max_iterations: int = MAX_ITERATIONS,
        output_root: str = "outputs",
        hardware_profile: Optional[str] = None,
        generator_backend: Optional[Any] = None,
        verifier_backend: Optional[Any] = None,
        reflector_backend: Optional[Any] = None,
    ):
        from engine.hardware_config import get_preferred_towns

        self.hardware_profile = hardware_profile
        self.preferred_towns = None

        if hardware_profile:
            self.preferred_towns = get_preferred_towns(hardware_profile)
            logger.info(
                "Hardware profile '%s' active: restricted to towns %s",
                hardware_profile, self.preferred_towns,
            )

        # Store resolved backends cleanly
        self.generator_backend = generator_backend
        self.verifier_backend = verifier_backend
        self.reflector_backend = reflector_backend

        # Explicitly assign dedicated backends to respective agents
        self.generator = generator or Generator(
            backend=self.generator_backend,
            preferred_towns=self.preferred_towns,
        )
        
        self.verifier = verifier or Verifier(
            backend=self.verifier_backend
        )
        
        # Instantiates explicit Reflector agent if available, else falls back to functional interface
        self.reflector = reflector or (
            Reflector(backend=self.reflector_backend) if 'Reflector' in globals() or reflector_backend else None
        )
        
        self.runner = runner or CarlaRunner()
        self.max_iterations = max_iterations
        self.output_root = Path(output_root)

    def run(self, prompt: str, scenario_id: str, mode: str = "full",
            rng_seed: Optional[int] = None, *,
            status_callback: Optional[StatusCallback] = None,
            live_frame_callback: Optional[Callable[[Any], None]] = None) -> PipelineRun:
        t0_total = time.perf_counter()
        run_record = PipelineRun(scenario_id=scenario_id, prompt=prompt, mode=mode)
        refinement_context = ""
        iterations = 1 if mode == "generator_only" else self.max_iterations
        _emit(status_callback, "run_start", dict(
            scenario_id=scenario_id, prompt=prompt, mode=mode, max_iterations=iterations,
        ))

        for iteration in range(iterations):
            iter_latencies: dict = {}
            iter_t0 = time.perf_counter()
            it_no = iteration + 1

            # ---- Agent 1: Generator ----------------------------------------
            _emit(status_callback, "generator", dict(
                iteration=it_no, refinement_context=refinement_context or None,
            ))
            with _timer("generator", iter_latencies):
                spec = self.generator.generate(
                    prompt, refinement_context=refinement_context
                )

            logger.info(
                "[%s] iter=%d Generator: %.2fs → town=%s weather=%s event=%s",
                scenario_id, iteration, iter_latencies["generator_s"],
                spec.town, spec.weather, spec.event_type,
            )
            _emit(status_callback, "generator_done", dict(
                iteration=it_no, spec=spec.model_dump(), latency_s=iter_latencies["generator_s"],
            ))

            # ---- Agent 2: CARLA Engine -------------------------------------
            iter_dir = self.output_root / scenario_id / f"iter_{iteration}"
            _emit(status_callback, "engine", dict(iteration=it_no, town=spec.town))
            with _timer("engine", iter_latencies):
                exec_result: ExecutionResult = self.runner.run(
                    spec,
                    scenario_id=f"{scenario_id}_iter{iteration}",
                    output_dir=str(iter_dir),
                    live_frame_callback=live_frame_callback,
                    rng_seed=rng_seed,
                )

            logger.info(
                "[%s] iter=%d Engine: %.2fs → success=%s frames=%d",
                scenario_id, iteration, iter_latencies["engine_s"],
                exec_result.success, len(exec_result.frame_paths),
            )
            if status_callback is not None:
                spawned = {}
                try:
                    with open(exec_result.log_path) as fh:
                        _sl = json.load(fh)
                    spawned = dict(vehicles=_sl.get("vehicle_count"),
                                   pedestrian=_sl.get("pedestrian_spawned"))
                except Exception:
                    pass
            _emit(status_callback, "engine_done", dict(
                iteration=it_no, success=exec_result.success,
                n_frames=len(exec_result.frame_paths), error=exec_result.error,
                spawned=spawned if status_callback is not None else None,
                requested=dict(vehicles=spec.num_vehicles, pedestrian=spec.pedestrian,
                               cyclist=spec.cyclist, event=spec.event_type),
                latency_s=iter_latencies["engine_s"],
            ))

            # ---- Agent 3: Verifier (Run honestly even in generator_only) ---
            _emit(status_callback, "verifier", dict(iteration=it_no))
            with _timer("verifier", iter_latencies):
                verification = self.verifier.verify(
                    original_prompt=prompt,
                    spec=spec,
                    frame_paths=exec_result.frame_paths,
                    log_path=exec_result.log_path or "",
                    exec_success=exec_result.success,
                )

            logger.info(
                "[%s] iter=%d Verifier: %.2fs → score=%.2f passed=%s missing=%s",
                scenario_id, iteration, iter_latencies["verifier_s"],
                verification.final_score, verification.overall_pass, verification.missing_elements,
            )
            _emit(status_callback, "verifier_done", dict(
                iteration=it_no,
                score=verification.final_score,
                predicate_agreement=verification.predicate_agreement,
                predicate_agreement_std=verification.predicate_agreement_std,
                predicates=verification.vlm_result.predicates,
                applicable_predicates=verification.applicable_predicates,
                missing_elements=verification.missing_elements,
                n_samples=verification.n_samples,
                log_checks_passed=verification.log_checks_passed,
                log_checks_total=verification.log_checks_total,
                execution_success=verification.execution_success,
                passed=verification.overall_pass,
                threshold=getattr(self.verifier, "threshold", None),
                backend=verification.backend_name,
                notes=verification.vlm_result.notes,
                latency_s=iter_latencies["verifier_s"],
            ))

            # ---- Reflector latency initialization slot --------------------
            iter_latencies["reflector_s"] = 0.0

            # ---- Capture system metrics ------------------------------------
            mem_mb = _peak_memory_mb()
            iter_latencies["total_pipeline_s"] = round(time.perf_counter() - iter_t0, 4)

            # ---- Record current iteration ----------------------------------
            run_record.iterations.append(
                IterationRecord(
                    iteration=iteration,
                    spec=spec.model_dump(),
                    refinement_context=refinement_context or None,
                    execution_success=exec_result.success,
                    verification_score=verification.final_score,
                    verification_passed=verification.overall_pass,
                    missing_elements=verification.missing_elements,
                    latencies=AgentLatencies(**iter_latencies),
                    peak_memory_mb=mem_mb,
                )
            )

            # ---- Exit conditions -------------------------------------------
            if verification.overall_pass:
                run_record.final_status = "passed"
                break

            if mode == "generator_only":
                # Exit immediately after 1 iteration, keeping the real evaluation scores!
                run_record.final_status = "generator_only_complete"
                break

            # ---- Agent 4: Reflector (Runs if another iteration is scheduled) --
            if iteration < iterations - 1:
                _emit(status_callback, "reflector", dict(
                    iteration=it_no, missing_elements=verification.missing_elements,
                ))
                with _timer("reflector", iter_latencies):
                    if self.reflector and hasattr(self.reflector, "reflect"):
                        # Class instance approach
                        refinement_context = self.reflector.reflect(
                            prompt=prompt,
                            verification_result=verification,
                            iteration=iteration + 1,
                            preferred_town=self.preferred_towns
                        )
                    else:
                        # Fallback functional approach utilizing clean reflector_backend parameter
                        refinement_context = reflect(
                            prompt,
                            verification,
                            iteration + 1,
                            preferred_town=self.preferred_towns,
                            backend=self.reflector_backend  # Fed cleanly into the agent module
                        )
                        
                run_record.iterations[-1].latencies.reflector_s = iter_latencies["reflector_s"]
                logger.info(
                    "[%s] iter=%d Reflector: %.2fs",
                    scenario_id, iteration, iter_latencies["reflector_s"],
                )
                _emit(status_callback, "reflector_done", dict(
                    iteration=it_no, refinement_context=refinement_context,
                    latency_s=iter_latencies["reflector_s"],
                ))

            # ---- Mid-Scenario Iteration Memory Safety Flush ----
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

        else:
            run_record.final_status = "iteration_limit_reached"

        # ---- Aggregate systems metrics over lifecycle --------------------
        run_record.total_wall_time_s = round(time.perf_counter() - t0_total, 2)
        run_record.total_generator_s  = round(sum(it.latencies.generator_s for it in run_record.iterations), 4)
        run_record.total_engine_s     = round(sum(it.latencies.engine_s for it in run_record.iterations), 4)
        run_record.total_verifier_s   = round(sum(it.latencies.verifier_s for it in run_record.iterations), 4)
        run_record.total_reflector_s  = round(sum(it.latencies.reflector_s for it in run_record.iterations), 4)
        run_record.peak_memory_mb     = max((it.peak_memory_mb for it in run_record.iterations), default=0.0)

        # ---- Print aggregate telemetry summary ---------------------------
        logger.info(
            "[%s] DONE status=%s | wall=%.1fs | gen=%.1fs eng=%.1fs ver=%.1fs ref=%.1fs | mem=%.0fMB",
            scenario_id, run_record.final_status, run_record.total_wall_time_s,
            run_record.total_generator_s, run_record.total_engine_s,
            run_record.total_verifier_s, run_record.total_reflector_s, run_record.peak_memory_mb,
        )

        self._persist(run_record)
        _emit(status_callback, "done", dict(
            scenario_id=scenario_id,
            final_status=run_record.final_status,
            iterations=len(run_record.iterations),
            total_wall_time_s=run_record.total_wall_time_s,
        ))
        return run_record

    def run_with_pregenerated_spec(self, prompt: str, scenario_id: str, spec: ScenarioSpec, 
                                   mode: str = "full", rng_seed: Optional[int] = None) -> PipelineRun:
        """
        Executes Phase B of the topology-sorted pipeline. Runs the CARLA simulation engine, 
        verifier, and subsequent reflection loops utilizing a pre-built ScenarioSpec allocation.
        """
        t0_total = time.perf_counter()
        run_record = PipelineRun(scenario_id=scenario_id, prompt=prompt, mode=mode)
        refinement_context = ""
        iterations = 1 if mode == "generator_only" else self.max_iterations

        for iteration in range(iterations):
            iter_latencies: dict = {}
            iter_t0 = time.perf_counter()

            # ---- Agent 1: Generator (Skipped on initial loop, runs on Reflected cycles) ----
            if iteration == 0:
                iter_latencies["generator_s"] = 0.0
                current_spec = spec 
            else:
                with _timer("generator", iter_latencies):
                    current_spec = self.generator.generate(
                        prompt, refinement_context=refinement_context
                    )
                logger.info(
                    "[%s] iter=%d Generator: %.2fs → town=%s weather=%s event=%s",
                    scenario_id, iteration, iter_latencies["generator_s"],
                    current_spec.town, current_spec.weather, current_spec.event_type,
                )

            # ---- Agent 2: CARLA Engine -------------------------------------
            iter_dir = self.output_root / scenario_id / f"iter_{iteration}"
            with _timer("engine", iter_latencies):
                exec_result: ExecutionResult = self.runner.run(
                    current_spec,
                    scenario_id=f"{scenario_id}_iter{iteration}",
                    output_dir=str(iter_dir),
                    rng_seed=rng_seed,
                )

            logger.info(
                "[%s] iter=%d Engine: %.2fs → success=%s frames=%d",
                scenario_id, iteration, iter_latencies["engine_s"],
                exec_result.success, len(exec_result.frame_paths),
            )

            # ---- Agent 3: Verifier -----------------------------------------
            with _timer("verifier", iter_latencies):
                verification = self.verifier.verify(
                    original_prompt=prompt,
                    spec=current_spec,
                    frame_paths=exec_result.frame_paths,
                    log_path=exec_result.log_path or "",
                    exec_success=exec_result.success,
                )

            logger.info(
                "[%s] iter=%d Verifier: %.2fs → score=%.2f passed=%s missing=%s",
                scenario_id, iteration, iter_latencies["verifier_s"],
                verification.final_score, verification.overall_pass, verification.missing_elements,
            )

            # ---- Reflector latency initialization slot --------------------
            iter_latencies["reflector_s"] = 0.0

            # ---- Capture system metrics ------------------------------------
            mem_mb = _peak_memory_mb()
            iter_latencies["total_pipeline_s"] = round(time.perf_counter() - iter_t0, 4)

            # ---- Record current iteration ----------------------------------
            run_record.iterations.append(
                IterationRecord(
                    iteration=iteration,
                    spec=current_spec.model_dump(),
                    refinement_context=refinement_context or None,
                    execution_success=exec_result.success,
                    verification_score=verification.final_score,
                    verification_passed=verification.overall_pass,
                    missing_elements=verification.missing_elements,
                    latencies=AgentLatencies(**iter_latencies),
                    peak_memory_mb=mem_mb,
                )
            )

            # ---- Exit conditions -------------------------------------------
            if verification.overall_pass:
                run_record.final_status = "passed"
                break

            if mode == "generator_only":
                run_record.final_status = "generator_only_complete"
                break

            # ---- Agent 4: Reflector ----------------------------------------
            if iteration < iterations - 1:
                with _timer("reflector", iter_latencies):
                    if self.reflector and hasattr(self.reflector, "reflect"):
                        refinement_context = self.reflector.reflect(
                            prompt=prompt,
                            verification_result=verification,
                            iteration=iteration + 1,
                            preferred_town=self.preferred_towns
                        )
                    else:
                        refinement_context = reflect(
                            prompt,
                            verification,
                            iteration + 1,
                            preferred_town=self.preferred_towns,
                            backend=self.reflector_backend
                        )
                        
                run_record.iterations[-1].latencies.reflector_s = iter_latencies["reflector_s"]
                logger.info(
                    "[%s] iter=%d Reflector: %.2fs",
                    scenario_id, iteration, iter_latencies["reflector_s"],
                )

            # ---- Mid-Scenario Iteration Memory Safety Flush ----
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

        else:
            run_record.final_status = "iteration_limit_reached"

        # ---- Aggregate systems metrics over lifecycle --------------------
        run_record.total_wall_time_s = round(time.perf_counter() - t0_total, 2)
        run_record.total_generator_s  = round(sum(it.latencies.generator_s for it in run_record.iterations), 4)
        run_record.total_engine_s     = round(sum(it.latencies.engine_s for it in run_record.iterations), 4)
        run_record.total_verifier_s   = round(sum(it.latencies.verifier_s for it in run_record.iterations), 4)
        run_record.total_reflector_s  = round(sum(it.latencies.reflector_s for it in run_record.iterations), 4)
        run_record.peak_memory_mb     = max((it.peak_memory_mb for it in run_record.iterations), default=0.0)

        # ---- Print aggregate telemetry summary ---------------------------
        logger.info(
            "[%s] DONE status=%s | wall=%.1fs | gen=%.1fs eng=%.1fs ver=%.1fs ref=%.1fs | mem=%.0fMB",
            scenario_id, run_record.final_status, run_record.total_wall_time_s,
            run_record.total_generator_s, run_record.total_engine_s,
            run_record.total_verifier_s, run_record.total_reflector_s, run_record.peak_memory_mb,
        )

        self._persist(run_record)
        return run_record

    def _persist(self, run_record: PipelineRun) -> None:
        out_dir = self.output_root / run_record.scenario_id
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "pipeline_run.json", "w") as f:
            f.write(run_record.to_json())
        logger.info(
            "Scenario %s finished: status=%s iterations=%d",
            run_record.scenario_id, run_record.final_status, len(run_record.iterations),
        )