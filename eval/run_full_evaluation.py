"""
eval/run_full_evaluation.py

Runs the 20-scenario dataset through:
  - "generator_only": Generator -> Engine -> Verifier, no reflection retries
  - "full":           full closed-loop Prompt-to-Drive (Generator -> Engine
                        -> Verifier -> Reflector, up to --max-iterations)
"""
from __future__ import annotations

import argparse
import json
import logging
import gc
import torch
from pathlib import Path

from orchestrator.pipeline import PromptToDrivePipeline
from dotenv import load_dotenv

load_dotenv()
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
    
    # --- Split Backend Arguments for all 3 models ---
    parser.add_argument(
        "--text-backend", choices=["openai", "huggingface", "hf", "local", "vllm"], default=None,
        help="LLM backend for the text Generator. Overrides --backend if provided."
    )
    parser.add_argument(
        "--vision-backend", choices=["openai", "huggingface", "hf", "local", "vllm"], default=None,
        help="VLM backend for the vision Verifier. Overrides --backend if provided."
    )
    parser.add_argument(
        "--reflector-backend", choices=["openai", "huggingface", "hf", "local", "vllm"], default=None,
        help="LLM backend for the Reflector agent. Overrides --backend if provided."
    )
    
    # Legacy catch-all fallback
    parser.add_argument(
        "--backend", choices=["openai", "huggingface"], default=None,
        help="Legacy catch-all backend string for all modules.",
    )
    
    parser.add_argument(
        "--hardware-profile", default=None,
        choices=["default", "low_vram_2gb", "headless_ci"],
        help="Use 'low_vram_2gb' for 2GB-VRAM mobile GPUs.",
    )
    args = parser.parse_args()

    # Resolve backends with priority: split flags > legacy catch-all flag
    text_backend = args.text_backend or args.backend
    vision_backend = args.vision_backend or args.backend
    reflector_backend = args.reflector_backend or args.backend

    dataset = json.loads(Path(args.dataset).read_text())

    from engine.carla_runner import CarlaRunner
    runner = CarlaRunner(host=args.carla_host, port=args.carla_port,
                         hardware_profile=args.hardware_profile)

    # Resolve your separate engines using your backend factory methods
    from backends.llm_backend import get_text_backend, get_vision_backend
    
    text_engine = get_text_backend(text_backend) if text_backend else None
    vision_engine = get_vision_backend(vision_backend) if vision_backend else None
    
    # Resolves the reflector using the appropriate backend loader (falling back to text engine style if necessary)
    reflector_engine = get_text_backend(reflector_backend) if reflector_backend else None

    # Pass all three resolved engines downstream into the pipeline
    pipeline = PromptToDrivePipeline(
        runner=runner,
        max_iterations=args.max_iterations,
        output_root=args.out,
        generator_backend=text_engine,
        verifier_backend=vision_engine,
        reflector_backend=reflector_engine,
    )

    all_runs = []
    for s in dataset["scenarios"]:
        logger.info("Running scenario %s (mode=%s): %s", s["id"], args.mode, s["prompt"])
        run = pipeline.run(
            prompt=s["prompt"], scenario_id=s["id"], mode=args.mode
        )
        all_runs.append(run.scenario_id)
        logger.info("  -> status=%s, final_score=%.2f, iterations=%d",
                    run.final_status,
                    run.iterations[-1].verification_score if run.iterations else 0.0,
                    len(run.iterations))
        
        # --- Memory Cleanup Safety Block ---
        # Wipes orphaned object trees and explicitly releases VRAM segments 
        # before the next scenario map loading sequence initializes.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Done. %d scenarios run, results in %s/<scenario_id>/pipeline_run.json", len(all_runs), args.out)


if __name__ == "__main__":
    main()