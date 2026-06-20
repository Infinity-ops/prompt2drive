"""
eval/run_full_evaluation.py

Runs the 20-scenario dataset through:
  - "generator_only": Generator -> Engine -> Verifier, no reflection retries
  - "full":            full closed-loop Prompt-to-Drive (Generator -> Engine
                        -> Verifier -> Reflector, up to --max-iterations)

Together with eval/run_baseline.py, this produces the three conditions
compared in Experiment 1 and Experiment 2 of the paper.

Usage:
    python -m eval.run_full_evaluation --dataset scenarios/dataset.json \\
        --mode full --out outputs/full --max-iterations 3

    python -m eval.run_full_evaluation --dataset scenarios/dataset.json \\
        --mode generator_only --out outputs/generator_only
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from orchestrator.pipeline import PromptToDrivePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prompt_to_drive.eval.full")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scenarios/dataset.json")
    parser.add_argument("--mode", choices=["full", "generator_only"], default="full")
    parser.add_argument("--out", default="outputs/full")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--carla-host", default="localhost")
    parser.add_argument("--carla-port", type=int, default=2000)
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text())

    from engine.carla_runner import CarlaRunner
    runner = CarlaRunner(host=args.carla_host, port=args.carla_port)

    pipeline = PromptToDrivePipeline(
        runner=runner,
        max_iterations=args.max_iterations,
        output_root=args.out,
    )

    all_runs = []
    for s in dataset["scenarios"]:
        logger.info("Running scenario %s (mode=%s): %s", s["id"], args.mode, s["prompt"])
        run = pipeline.run(prompt=s["prompt"], scenario_id=s["id"], mode=args.mode)
        all_runs.append(run.scenario_id)
        logger.info("  -> status=%s, final_score=%.2f, iterations=%d",
                    run.final_status,
                    run.iterations[-1].verification_score if run.iterations else 0.0,
                    len(run.iterations))

    logger.info("Done. %d scenarios run, results in %s/<scenario_id>/pipeline_run.json", len(all_runs), args.out)


if __name__ == "__main__":
    main()
