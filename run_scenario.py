"""
run_scenario.py

Single-scenario CLI entry point for Prompt-to-Drive.

Usage examples
--------------

# Using local Qwen2.5-VL served by vLLM (YOUR SETUP):
python run_scenario.py \\
    --prompt "A pedestrian suddenly crosses near a traffic signal in rainy weather" \\
    --hardware-profile low_vram_2gb_signals \\
    --backend local \\
    --mode full

# Using HuggingFace cloud models:
python run_scenario.py \\
    --prompt "A pedestrian suddenly crosses near a traffic signal in rainy weather" \\
    --hardware-profile low_vram_2gb_signals \\
    --backend huggingface \\
    --mode full

# Full evaluation run (all scenarios):
python -m eval.run_full_evaluation \\
    --dataset scenarios/dataset.json \\
    --mode full \\
    --out outputs/full \\
    --backend local \\
    --hardware-profile low_vram_2gb_signals

# Then compute metrics:
python -m eval.compute_metrics \\
    --full-dir outputs/full \\
    --generator-only-dir outputs/generator_only \\
    --baseline-dir outputs/baseline \\
    --out outputs/metrics.json

Environment variables (set in .env or shell)
--------------------------------------------
# For --backend local (vLLM):
VLLM_BASE_URL=http://localhost:8000/v1          # default
VLLM_VISION_MODEL=Qwen/Qwen2.5-VL-7B-Instruct  # default
VLLM_TEXT_MODEL=Qwen/Qwen2.5-VL-7B-Instruct    # default (same model)

# For --backend huggingface:
HF_TOKEN=hf_...

# For --backend openai:
OPENAI_API_KEY=sk-...
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
        help="Hardware constraints for town selection and camera resolution. "
             "'low_vram_2gb_signals' restricts to Town01 for 2GB VRAM machines.",
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "huggingface", "local"],
        default="openai",
        help=(
            "LLM backend for generation and verification.\n"
            "  openai      -- GPT-4o (requires OPENAI_API_KEY)\n"
            "  huggingface -- HF Inference API (requires HF_TOKEN)\n"
            "  local       -- locally served vLLM (Qwen2.5-VL at localhost:8000)\n"
            "                 Start with: vllm serve ~/models/Qwen2.5-VL-7B-Instruct \\\n"
            "                                 --dtype float16 --max-model-len 8192"
        ),
    )
    args = parser.parse_args()

    # Validate credentials per backend
    if args.backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Add it to your .env file or export it."
        )
    if args.backend == "huggingface" and not (
        os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")
    ):
        raise SystemExit(
            "HF_TOKEN is not set. Add it to your .env file or export it.\n"
            "Generate one at https://huggingface.co/settings/tokens"
        )
    if args.backend == "local":
        base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        logger.info("Using local vLLM backend at %s", base_url)

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