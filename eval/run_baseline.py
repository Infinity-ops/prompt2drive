"""
eval/run_baseline.py

Runs the 20-scenario dataset through the NAIVE baseline: direct, free-form
LLM code generation with no schema constraint and no verification loop.
This is the comparator condition for Experiment 1 (executable scenario
success rate) and Experiment 2 (semantic alignment).

The baseline deliberately has no safety net -- this is what a typical
practitioner gets from "just asking an LLM to write CARLA code."

Usage:
    python -m eval.run_baseline --dataset scenarios/dataset.json --out outputs/baseline
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from generator.agent import generate_baseline_code

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prompt_to_drive.eval.baseline")

CARLA_BOILERPLATE = """
import carla
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
"""


def run_one(prompt: str, scenario_id: str, out_dir: Path, backend: str | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"scenario_id": scenario_id, "prompt": prompt, "backend": backend or "openai"}

    try:
        code = generate_baseline_code(prompt, backend=backend)
        # strip markdown fences if present
        code = code.replace("```python", "").replace("```", "").strip()
        full_script = CARLA_BOILERPLATE + "\n" + code

        script_path = out_dir / "generated.py"
        script_path.write_text(full_script)

        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=60,
        )
        record["execution_success"] = proc.returncode == 0
        record["stdout"] = proc.stdout[-2000:]
        record["stderr"] = proc.stderr[-2000:]
        record["wall_time_s"] = round(time.time() - t0, 2)

    except Exception as exc:  # noqa: BLE001
        record["execution_success"] = False
        record["stderr"] = f"{exc}\n{traceback.format_exc()}"
        record["wall_time_s"] = None

    (out_dir / "result.json").write_text(json.dumps(record, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scenarios/dataset.json")
    parser.add_argument("--out", default="outputs/baseline")
    parser.add_argument(
        "--backend", choices=["openai", "huggingface"], default=None,
        help="Backend for free-form baseline code generation. Defaults to "
             "P2D_TEXT_BACKEND env var, or 'openai'.",
    )
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text())
    out_root = Path(args.out)
    results = []

    for s in dataset["scenarios"]:
        logger.info("Running baseline for %s [backend=%s]", s["id"], args.backend or "openai")
        result = run_one(s["prompt"], s["id"], out_root / s["id"], backend=args.backend)
        results.append(result)

    summary_path = out_root / "baseline_results.json"
    summary_path.write_text(json.dumps(results, indent=2))

    success_rate = sum(r["execution_success"] for r in results) / len(results)
    logger.info("Baseline executable success rate: %.1f%% (%d/%d)",
                success_rate * 100, sum(r["execution_success"] for r in results), len(results))


if __name__ == "__main__":
    main()
