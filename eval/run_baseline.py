from __future__ import annotations

# --- FORCE LOAD ENVIRONMENT FIRST ---
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(override=False) # False ensures your inline command-line variable takes priority!
except ImportError:
    pass

# Ensure it's set fallback if somehow still missing
if not os.environ.get("HF_TEXT_MODEL"):
    os.environ["HF_TEXT_MODEL"] = "Qwen/Qwen2.5-7B-Instruct"
# ------------------------------------

import argparse
import json
import logging
import subprocess
import sys
import time
import traceback
from typing import Optional

from backends.llm_backend import TextBackend
from generator.agent import generate_baseline_code

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prompt_to_drive.eval.baseline")

# Expanded boilerplate to include primitive modules required by common LLM-generated code
CARLA_BOILERPLATE = """import carla
import json
import math
import random
import time

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
"""


def sanitize_and_clean_fences(generated_code: str) -> str:
    """Removes markdown backticks and fences if the baseline LLM wrapped its output."""
    sanitized_lines = []
    for line in generated_code.splitlines():
        if not line.strip().startswith("```"):
            sanitized_lines.append(line)
    return "\n".join(sanitized_lines).strip()


def run_one(prompt: str, scenario_id: str, out_dir: Path, backend: Optional[TextBackend] = None, backend_name: str = "openai") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"scenario_id": scenario_id, "prompt": prompt, "backend": backend_name}

    try:
        # Generate target code using the proper framework backend pipeline
        raw_code = generate_baseline_code(prompt, backend=backend)

        # Sanitize markdown ticks and cleanly combine with extended boilerplate
        clean_code = sanitize_and_clean_fences(raw_code)
        full_script = CARLA_BOILERPLATE + "\n\n" + clean_code

        script_path = out_dir / "generated.py"
        script_path.write_text(full_script)

        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=60,
        )
        record["execution_success"] = proc.returncode == 0
        record["stdout"] = proc.stdout[-2000:] if proc.stdout else ""
        record["stderr"] = proc.stderr[-2000:] if proc.stderr else ""
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
        "--backend", choices=["openai", "huggingface"], default="openai",
        help="Backend for free-form baseline code generation.",
    )
    args = parser.parse_args()

    # Dynamic import of framework backend setup to support environment configurations cleanly
    try:
        from backends.llm_backend import get_backend
        backend_instance = get_backend(args.backend)
    except ImportError:
        # Fallback if specific get_backend mapping isn't global
        logger.warning("Could not dynamically resolve get_backend function. Constructing default container.")
        backend_instance = None

    dataset = json.loads(Path(args.dataset).read_text())
    out_root = Path(args.out)
    results = []

    for s in dataset["scenarios"]:
        logger.info("Running baseline for %s [backend=%s]", s["id"], args.backend)
        result = run_one(
            prompt=s["prompt"], 
            scenario_id=s["id"], 
            out_dir=out_root / s["id"], 
            backend=backend_instance,
            backend_name=args.backend
        )
        results.append(result)

    summary_path = out_root / "baseline_results.json"
    summary_path.write_text(json.dumps(results, indent=2))

    success_rate = sum(r["execution_success"] for r in results) / len(results)
    logger.info("Baseline executable success rate: %.1f%% (%d/%d)",
                success_rate * 100, sum(r["execution_success"] for r in results), len(results))


if __name__ == "__main__":
    main()