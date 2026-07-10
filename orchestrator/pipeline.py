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

import json
import logging
import resource
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Any

from engine.carla_runner import CarlaRunner, ExecutionResult
from engine.scenario_spec import ScenarioSpec
from generator.agent import Generator
from verifier.agent import HybridVerificationResult, Verifier
from reflector.agent import Reflector, reflect  # Supports both Class or functional reflection

logger = logging.getLogger("prompt_to_drive.orchestrator")

MAX_ITERATIONS = 3


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
            rng_seed: Optional[int] = None) -> PipelineRun:
        t0_total = time.perf_counter()
        run_record = PipelineRun(scenario_id=scenario_id, prompt=prompt, mode=mode)
        refinement_context = ""
        iterations = 1 if mode == "generator_only" else self.max_iterations

        for iteration in range(iterations):
            iter_latencies: dict = {}
            iter_t0 = time.perf_counter()

            # ---- Agent 1: Generator ----------------------------------------
            with _timer("generator", iter_latencies):
                spec = self.generator.generate(
                    prompt, refinement_context=refinement_context
                )

            logger.info(
                "[%s] iter=%d Generator: %.2fs → town=%s weather=%s event=%s",
                scenario_id, iteration, iter_latencies["generator_s"],
                spec.town, spec.weather, spec.event_type,
            )

            # ---- Agent 2: CARLA Engine -------------------------------------
            iter_dir = self.output_root / scenario_id / f"iter_{iteration}"
            with _timer("engine", iter_latencies):
                exec_result: ExecutionResult = self.runner.run(
                    spec,
                    scenario_id=f"{scenario_id}_iter{iteration}",
                    output_dir=str(iter_dir),
                    rng_seed=rng_seed,
                )

            logger.info(
                "[%s] iter=%d Engine: %.2fs → success=%s frames=%d",
                scenario_id, iteration, iter_latencies["engine_s"],
                exec_result.success, len(exec_result.frame_paths),
            )

            # ---- Agent 3: Verifier (Short-circuited in generator_only mode) --
            if mode == "generator_only":
                logger.info("[%s] iter=%d Skipping VLM verification (generator_only mode)", scenario_id, iteration)
                verification = HybridVerificationResult(
                    overall_pass=True,
                    final_score=1.0,
                    missing_elements=[],
                    vlm_result={"notes": "Skipped in generator_only mode", "intent_alignment_score": 1.0},
                    predicate_agreement=1.0,
                    predicate_agreement_std=0.0,
                    applicable_predicates=[],
                    failed_predicates=[],
                    n_samples=0,
                    log_checks_passed=0,
                    log_checks_total=0,
                    execution_success=exec_result.success
                )
                iter_latencies["verifier_s"] = 0.0
            else:
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
                run_record.final_status = "generator_only_no_retry" if mode == "generator_only" else "passed"
                break

            if mode == "generator_only":
                run_record.final_status = "generator_only_no_retry"
                break

            if not exec_result.success and iteration == iterations - 1:
                run_record.final_status = "execution_failed"
                break

            # ---- Agent 4: Reflector (Runs if another iteration is scheduled) --
            if iteration < iterations - 1:
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