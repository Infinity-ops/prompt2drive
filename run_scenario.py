"""
run_scenario.py

Single-scenario CLI entry point -- the fastest way to try Prompt-to-Drive
on your own prompt against a locally running CARLA server.

Usage:
    python run_scenario.py --prompt "A pedestrian crosses near a junction in rain" \\
        --scenario-id demo01 --mode full

Requires:
    - A running CARLA server (see README "Prerequisites")
    - OPENAI_API_KEY set in your environment or .env file
"""
from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv

from orchestrator.pipeline import PromptToDrivePipeline
from engine.carla_runner import CarlaRunner

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prompt_to_drive.cli")


def main():
    parser = argparse.ArgumentParser(description="Run a single Prompt-to-Drive scenario.")
    parser.add_argument("--prompt", required=True, help="Natural language scenario description.")
    parser.add_argument("--scenario-id", default="demo01")
    parser.add_argument("--mode", choices=["full", "generator_only"], default="full")
    parser.add_argument("--carla-host", default="localhost")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Create a .env file (see .env.example) "
            "or export it in your shell before running."
        )

    runner = CarlaRunner(host=args.carla_host, port=args.carla_port)
    pipeline = PromptToDrivePipeline(
        runner=runner, max_iterations=args.max_iterations, output_root=args.out
    )

    print(f"\nRunning scenario '{args.scenario_id}' in mode='{args.mode}'")
    print(f"Prompt: {args.prompt}\n")

    run = pipeline.run(prompt=args.prompt, scenario_id=args.scenario_id, mode=args.mode)

    print("\n" + "=" * 60)
    print(f"Final status:   {run.final_status}")
    print(f"Iterations:     {len(run.iterations)}")
    if run.iterations:
        last = run.iterations[-1]
        print(f"Final score:    {last.verification_score:.2f}")
        print(f"Passed:         {last.verification_passed}")
        print(f"Missing elems:  {last.missing_elements}")
    print(f"Full record:    {args.out}/{args.scenario_id}/pipeline_run.json")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
