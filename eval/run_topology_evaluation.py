"""
eval/run_topology_evaluation.py

Optimized Evaluation Entry Point featuring Topology Map Sorting for CARLA.
Supports both standard and stress-test datasets with split agent backends.
"""
import json
import argparse
import logging
import gc
try:  # only used to flush the CUDA cache between scenarios
    import torch
except ImportError:  # pragma: no cover
    torch = None
from pathlib import Path
from collections import defaultdict
from orchestrator.pipeline import PromptToDrivePipeline
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prompt_to_drive.eval.topology")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",           default="scenarios/dataset.json")
    parser.add_argument("--mode",              choices=["full", "generator_only"], default="full")
    parser.add_argument("--out",               default="outputs/full", help="Directory where json logs are placed")
    parser.add_argument("--max-iterations",    type=int, default=3)
    parser.add_argument("--carla-host",        default="localhost")
    parser.add_argument("--carla-port",        type=int, default=2000)
    parser.add_argument("--hardware-profile", default=None, choices=["default", "low_vram_2gb", "headless_ci"])
    
    # Split backends flags
    parser.add_argument("--text-backend",      default=None)
    parser.add_argument("--vision-backend",    default=None)
    parser.add_argument("--reflector-backend", default=None)
    parser.add_argument("--backend",           default=None)
    args = parser.parse_args()

    text_backend = args.text_backend or args.backend
    vision_backend = args.vision_backend or args.backend
    reflector_backend = args.reflector_backend or args.backend

    # Load infrastructure and backends
    from engine.carla_runner import CarlaRunner
    from backends.llm_backend import get_text_backend, get_vision_backend
    
    runner = CarlaRunner(host=args.carla_host, port=args.carla_port, hardware_profile=args.hardware_profile)
    text_engine = get_text_backend(text_backend) if text_backend else None
    vision_engine = get_vision_backend(vision_backend) if vision_backend else None
    reflector_engine = get_text_backend(reflector_backend) if reflector_backend else None

    pipeline = PromptToDrivePipeline(
        runner=runner, max_iterations=args.max_iterations, output_root=args.out,
        generator_backend=text_engine, verifier_backend=vision_engine, reflector_backend=reflector_engine
    )

    with open(args.dataset, "r") as f:
        dataset = json.loads(Path(args.dataset).read_text())
    scenarios = dataset["scenarios"]

    # ---- PHASE A: Batch Text Generation (Discover Towns) ----
    logger.info("--> Starting Phase A: Querying Generator for Town Layouts...")
    phase_b_tasks = []
    for sc in scenarios:
        logger.info("  Generating initial spec template for %s...", sc["id"])
        spec = pipeline.generator.generate(sc["prompt"])
        chosen_town = spec.town if spec.town else "Town01"
        phase_b_tasks.append({"sc": sc, "spec": spec, "town": chosen_town})

    # ---- TOPOLOGY GROUPING ----
    town_groups = defaultdict(list)
    for task in phase_b_tasks:
        town_groups[task["town"]].append(task)

    logger.info("--> Topology Routing complete. Execution map batches:")
    for town, tasks in town_groups.items():
        logger.info("    * %s: %d scenarios slated", town, len(tasks))

    # ---- PHASE B: Batched Simulation Engine Run ----
    logger.info("--> Starting Phase B: Execution Sequence under topology routing...")
    for town, tasks in town_groups.items():
        logger.info("=== Simulating Town Batch: %s ===", town)
        for task in tasks:
            sc = task["sc"]
            logger.info("Running pipeline for %s on %s...", sc["id"], town)
            pipeline.run_with_pregenerated_spec(
                prompt=sc["prompt"], scenario_id=sc["id"], spec=task["spec"], mode=args.mode
            )
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info("All batched evaluation runs completed. Outputs synced to %s", args.out)

if __name__ == "__main__":
    main()
