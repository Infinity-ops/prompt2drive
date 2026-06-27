"""
orchestrator/pipeline.py

Ties the three agents together into the closed-loop Generation ->
Execution -> Verification -> Reflection cycle described in Section 3 of
the paper (Algorithm 1).

This module also persists a full PipelineRun record for every scenario,
which is the raw data consumed by eval/compute_metrics.py to produce the
paper's tables and figures.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from engine.carla_runner import CarlaRunner, ExecutionResult
from engine.scenario_spec import ScenarioSpec
from generator.agent import Generator
from reflector.agent import reflect
from verifier.agent import HybridVerificationResult, Verifier
from backends.llm_backend import get_backend

logger = logging.getLogger("prompt_to_drive.orchestrator")

MAX_ITERATIONS = 3


@dataclass
class IterationRecord:
    iteration: int
    spec: dict
    refinement_context: Optional[str]
    execution_success: bool
    verification_score: float
    verification_passed: bool
    missing_elements: list[str]


@dataclass
class PipelineRun:
    scenario_id: str
    prompt: str
    mode: str  # "baseline" | "generator_only" | "full"
    iterations: list[IterationRecord] = field(default_factory=list)
    final_status: str = "in_progress"  # passed | iteration_limit_reached | execution_failed
    total_wall_time_s: float = 0.0

    def to_json(self) -> str:
        return json.dumps(
            {
                **asdict(self),
                "iterations": [asdict(i) for i in self.iterations],
            },
            indent=2,
        )


class PromptToDrivePipeline:
    """
    The main closed-loop entry point.

    mode="full"            -> Generator -> Engine -> Verifier -> Reflector loop
    mode="generator_only"  -> Generator -> Engine -> Verifier, NO reflection retries
    mode="baseline"        -> handled separately via eval/run_baseline.py
    """

    def __init__(
        self,
        generator: Optional[Generator] = None,
        verifier: Optional[Verifier] = None,
        runner: Optional[CarlaRunner] = None,
        max_iterations: int = MAX_ITERATIONS,
        output_root: str = "outputs",
        hardware_profile: Optional[str] = None,
        backend: str = "openai",
    ):
        from engine.hardware_config import get_preferred_towns

        self.hardware_profile = hardware_profile
        self.preferred_towns = None

        if hardware_profile:
            self.preferred_towns = get_preferred_towns(hardware_profile)
            logger.info(
                "Hardware profile '%s' active: restricted to towns %s",
                hardware_profile,
                self.preferred_towns,
            )

        self.backend_name = backend.lower()
        self.backend = get_backend(self.backend_name)

        self.generator = generator or Generator(
            backend=self.backend,
            preferred_towns=self.preferred_towns,
        )

        self.verifier = verifier or Verifier(
            backend=self.backend,
        )

        self.runner = runner or CarlaRunner()
        self.max_iterations = max_iterations
        self.output_root = Path(output_root)

    def run(self, prompt: str, scenario_id: str, mode: str = "full") -> PipelineRun:
        t0 = time.time()
        run_record = PipelineRun(scenario_id=scenario_id, prompt=prompt, mode=mode)
        refinement_context = ""

        iterations = 1 if mode == "generator_only" else self.max_iterations

        for iteration in range(iterations):
            spec = self.generator.generate(prompt, refinement_context=refinement_context)

            iter_dir = self.output_root / scenario_id / f"iter_{iteration}"
            exec_result: ExecutionResult = self.runner.run(
                spec, scenario_id=f"{scenario_id}_iter{iteration}", output_dir=str(iter_dir)
            )

            verification: HybridVerificationResult = self.verifier.verify(
                original_prompt=prompt,
                spec=spec,
                frame_paths=exec_result.frame_paths,
                log_path=exec_result.log_path or "",
                exec_success=exec_result.success,
            )

            run_record.iterations.append(
                IterationRecord(
                    iteration=iteration,
                    spec=spec.model_dump(),
                    refinement_context=refinement_context or None,
                    execution_success=exec_result.success,
                    verification_score=verification.final_score,
                    verification_passed=verification.overall_pass,
                    missing_elements=verification.missing_elements,
                )
            )

            if verification.overall_pass:
                run_record.final_status = "passed"
                break

            if mode == "generator_only":
                run_record.final_status = "generator_only_no_retry"
                break

            if not exec_result.success and iteration == iterations - 1:
                run_record.final_status = "execution_failed"
                break

            # BUG 1 FIX: only call reflect() when there is a next iteration to
            # use the result. Previously reflect() fired on every iteration
            # including the last, logging a spurious "Reflector iteration N"
            # for a generation that would never run, and wasting an API call.
            # The for/else sets final_status="iteration_limit_reached" after
            # the loop exits naturally — reflect on the last iteration cannot
            # change that outcome.
            if iteration < iterations - 1:
                refinement_context = reflect(
                    prompt, verification, iteration + 1, preferred_towns=self.preferred_towns
                )

        else:
            run_record.final_status = "iteration_limit_reached"

        run_record.total_wall_time_s = round(time.time() - t0, 2)
        self._persist(run_record)
        return run_record

    def _persist(self, run_record: PipelineRun) -> None:
        out_dir = self.output_root / run_record.scenario_id
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "pipeline_run.json", "w") as f:
            f.write(run_record.to_json())
        logger.info(
            "Scenario %s finished: status=%s iterations=%d",
            run_record.scenario_id,
            run_record.final_status,
            len(run_record.iterations),
        )