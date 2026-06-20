"""
eval/run_backend_comparison.py

Runs the full closed-loop Prompt-to-Drive pipeline (mode="full") across
BOTH the OpenAI and Hugging Face backends on the same 20-scenario dataset,
then produces a side-by-side comparison table.

This directly supports the paper's generalizability/robustness argument:
the contribution is the architecture (constrained JSON IR + hybrid
verification + targeted reflection), not a property of one specific
proprietary model. Running the identical pipeline against an open-weight
backend and reporting comparable results is meaningful evidence for that
claim -- and meaningfully strengthens the paper versus reporting a single
backend.

Usage:
    python -m eval.run_backend_comparison \\
        --dataset scenarios/dataset.json \\
        --out outputs/backend_comparison \\
        --max-iterations 3

Requires BOTH OPENAI_API_KEY and HUGGINGFACE_API_TOKEN to be set.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from engine.carla_runner import CarlaRunner
from orchestrator.pipeline import PromptToDrivePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prompt_to_drive.eval.backend_comparison")

BACKENDS = ["openai", "huggingface"]


def run_for_backend(backend: str, dataset: dict, out_root: Path, max_iterations: int,
                     carla_host: str, carla_port: int) -> dict:
    runner = CarlaRunner(host=carla_host, port=carla_port)
    pipeline = PromptToDrivePipeline(
        runner=runner,
        max_iterations=max_iterations,
        output_root=str(out_root / backend),
        backend=backend,
    )

    runs = {}
    for s in dataset["scenarios"]:
        logger.info("[%s] Running %s: %s", backend, s["id"], s["prompt"])
        try:
            run = pipeline.run(prompt=s["prompt"], scenario_id=s["id"], mode="full")
            runs[s["id"]] = run
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] Scenario %s raised an exception: %s", backend, s["id"], exc)
            runs[s["id"]] = None
    return runs


def summarize(runs: dict) -> dict:
    valid = {k: v for k, v in runs.items() if v is not None and v.iterations}
    n = len(runs)
    if not valid:
        return dict(n=n, success_rate=0.0, mean_score=0.0, mean_iterations=0.0)

    success = sum(1 for r in valid.values() if any(it.execution_success for it in r.iterations))
    mean_score = sum(r.iterations[-1].verification_score for r in valid.values()) / len(valid)
    mean_iters = sum(len(r.iterations) for r in valid.values()) / len(valid)
    pass_rate = sum(1 for r in valid.values() if r.final_status == "passed") / len(valid)

    return dict(
        n=n,
        success_rate=round(success / n, 3),
        mean_semantic_score=round(mean_score, 3),
        mean_iterations=round(mean_iters, 2),
        pass_rate=round(pass_rate, 3),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scenarios/dataset.json")
    parser.add_argument("--out", default="outputs/backend_comparison")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--carla-host", default="localhost")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument(
        "--backends", nargs="+", default=BACKENDS, choices=BACKENDS,
        help="Which backends to run. Defaults to both.",
    )
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text())
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = {}
    for backend in args.backends:
        logger.info("=== Running backend: %s ===", backend)
        runs = run_for_backend(backend, dataset, out_root, args.max_iterations,
                                args.carla_host, args.carla_port)
        summary[backend] = summarize(runs)

    summary_path = out_root / "backend_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 70)
    print(f"{'Backend':<15}{'Success%':<12}{'Sem. Align.':<14}{'Pass%':<10}{'Mean Iter':<10}")
    print("-" * 70)
    for backend, s in summary.items():
        print(
            f"{backend:<15}{s['success_rate']*100:<12.1f}{s['mean_semantic_score']:<14.3f}"
            f"{s['pass_rate']*100:<10.1f}{s['mean_iterations']:<10.2f}"
        )
    print("=" * 70)
    print(f"\nFull summary written to {summary_path}")


if __name__ == "__main__":
    main()
