"""
eval/expand_dataset.py

Expands the 20 handcrafted core scenarios into a larger benchmark with
multiple seeded executions per scenario, directly addressing the reviewer
critique that N=20 is too small to support quantitative claims with any
statistical power.

What "seed" means here, precisely (read before citing this in the paper):
each seeded variant of a core scenario keeps the SAME natural-language
prompt and the SAME ground-truth labels, but varies:
  - the Generator's sampling seed / temperature draw (so the LLM doesn't
    return an identical ScenarioSpec every time for a non-deterministic
    backend), and
  - the engine's random spawn-point shuffle seed (controls which exact
    spawn points are used for NPCs/pedestrian, within the same town).

This is NOT paraphrase augmentation (we are not asking an LLM to reword
the prompts) and it is NOT a claim of 60 independent scenario *concepts*
-- it is 20 scenario concepts x 3 independent executions each, which
supports computing per-scenario variance and an aggregate mean +/-
std/CI across repeated runs, rather than a single point estimate per
scenario. State this distinction explicitly in the paper's Experimental
Setup section; conflating "60 executions" with "60 distinct scenarios"
would overstate the benchmark's diversity.

Usage:
    python -m eval.expand_dataset \\
        --input scenarios/dataset.json \\
        --output scenarios/dataset_expanded.json \\
        --seeds-per-scenario 3
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def expand(dataset: dict, seeds_per_scenario: int) -> dict:
    expanded_scenarios = []
    for scenario in dataset["scenarios"]:
        for seed_idx in range(seeds_per_scenario):
            variant = copy.deepcopy(scenario)
            variant["id"] = f"{scenario['id']}_seed{seed_idx}"
            variant["core_scenario_id"] = scenario["id"]
            variant["seed_index"] = seed_idx
            variant["rng_seed"] = hash((scenario["id"], seed_idx)) % (2**31)
            expanded_scenarios.append(variant)

    return {
        "dataset_version": dataset.get("dataset_version", "1.0") + "-expanded",
        "description": (
            f"{len(dataset['scenarios'])} core scenarios x {seeds_per_scenario} seeded executions "
            f"each = {len(expanded_scenarios)} total executions. Each seed variant shares its core "
            f"scenario's prompt and ground-truth labels but uses an independent generation/spawn "
            f"seed -- see eval/expand_dataset.py module docstring for the precise definition and "
            f"why this is NOT equivalent to 60 independently authored scenarios."
        ),
        "core_scenario_count": len(dataset["scenarios"]),
        "seeds_per_scenario": seeds_per_scenario,
        "scenarios": expanded_scenarios,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="scenarios/dataset.json")
    parser.add_argument("--output", default="scenarios/dataset_expanded.json")
    parser.add_argument("--seeds-per-scenario", type=int, default=3)
    args = parser.parse_args()

    dataset = json.loads(Path(args.input).read_text())
    expanded = expand(dataset, args.seeds_per_scenario)

    Path(args.output).write_text(json.dumps(expanded, indent=2))
    print(
        f"Expanded {expanded['core_scenario_count']} core scenarios into "
        f"{len(expanded['scenarios'])} seeded executions -> {args.output}"
    )


if __name__ == "__main__":
    main()
