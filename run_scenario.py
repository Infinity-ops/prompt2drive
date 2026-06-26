"""
run_scenario.py

Single-scenario CLI entry point -- the fastest way to try Prompt-to-Drive
on your own prompt against a locally running CARLA server.

Usage:
    python run_scenario.py --prompt "A pedestrian crosses near a junction in rain" \\
        --scenario-id demo01 --mode full

    # Low VRAM systems (avoid crashes on small machines):
    python run_scenario.py --prompt "A pedestrian crosses near a junction in rain" \\
        --hardware-profile low_vram_2gb

Requires:
    - A running CARLA server (see README "Prerequisites")
    - OPENAI_API_KEY set in your environment or .env file

Hardware Profiles:
    - high_performance: Unrestricted town selection (4GB+ VRAM)
    - standard: Mid-range systems (2-4GB VRAM) 
    - low_vram_2gb: Constrained systems - restricts to Town01/Town02 only
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
    parser.add_argument(
        "--hardware-profile",
        choices=["high_performance", "standard", "low_vram_2gb", "low_vram_2gb_signals"],
        default=None,
        help="Hardware constraints for town selection. "
        "'low_vram_2gb' restricts to Town01/Town02 to avoid OOM.",
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "huggingface"],
        default="openai",
        help="LLM backend used for generation and verification.",
    )
    args = parser.parse_args()

    if args.backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Create a .env file or export it before running."
        )

    if args.backend == "huggingface" and not os.getenv("HF_TOKEN"):
        raise SystemExit(
            "HF_TOKEN is not set. Create a .env file or export it before running."
        )

    runner = CarlaRunner(
        host=args.carla_host,
        port=args.carla_port,
        hardware_profile=args.hardware_profile,
    )
    pipeline = PromptToDrivePipeline(
        runner=runner,
        max_iterations=args.max_iterations,
        output_root=args.out,
        hardware_profile=args.hardware_profile,
        backend=args.backend,
    )

    profile_info = ""
    if args.hardware_profile:
        from engine.hardware_config import get_profile_info
        info = get_profile_info(args.hardware_profile)
        profile_info = f" [Hardware: {info['description']}]"

    print(f"\nRunning scenario '{args.scenario_id}' in mode='{args.mode}'{profile_info}")
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
