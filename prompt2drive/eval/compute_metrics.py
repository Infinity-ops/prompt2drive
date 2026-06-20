"""
eval/compute_metrics.py

Reads the output of run_baseline.py and run_full_evaluation.py (both
modes) and computes:

  Table 1 -- Executable scenario success rate, by condition
  Table 2 -- Semantic alignment score, by condition and by category
             (weather / actor / event / difficulty)
  Figure 3 -- Reflection convergence: score by iteration for scenarios
              that initially failed verification
  Figure 4 -- Success/alignment comparison across difficulty levels

Usage:
    python -m eval.compute_metrics \\
        --baseline outputs/baseline/baseline_results.json \\
        --generator-only outputs/generator_only \\
        --full outputs/full \\
        --dataset scenarios/dataset.json \\
        --out eval/results

Produces:
    eval/results/table1_success_rate.csv
    eval/results/table2_semantic_alignment.csv
    eval/results/figure3_reflection_convergence.pdf
    eval/results/figure4_difficulty_breakdown.pdf
    eval/results/summary.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prompt_to_drive.eval.metrics")


def _load_pipeline_runs(out_dir: str) -> dict[str, dict]:
    """Loads every <scenario_id>/pipeline_run.json under out_dir."""
    runs = {}
    root = Path(out_dir)
    if not root.exists():
        logger.warning("Directory %s does not exist; skipping.", out_dir)
        return runs
    for run_file in root.glob("*/pipeline_run.json"):
        data = json.loads(run_file.read_text())
        runs[data["scenario_id"]] = data
    return runs


def _load_baseline(baseline_path: str) -> dict[str, dict]:
    p = Path(baseline_path)
    if not p.exists():
        logger.warning("Baseline file %s not found; skipping.", baseline_path)
        return {}
    results = json.loads(p.read_text())
    return {r["scenario_id"]: r for r in results}


def table1_success_rate(baseline, generator_only, full) -> dict:
    def rate(runs: dict, key_success: str) -> float:
        if not runs:
            return float("nan")
        successes = sum(1 for r in runs.values() if r.get(key_success))
        return successes / len(runs)

    def full_success(run: dict) -> bool:
        return any(it["execution_success"] for it in run.get("iterations", []))

    return {
        "Baseline (free-form code gen)": rate(baseline, "execution_success"),
        "Generator-only (JSON schema, no reflection)": (
            sum(1 for r in generator_only.values() if full_success(r)) / len(generator_only)
            if generator_only else float("nan")
        ),
        "Full Prompt-to-Drive (closed loop)": (
            sum(1 for r in full.values() if full_success(r)) / len(full)
            if full else float("nan")
        ),
    }


def table2_semantic_alignment(generator_only, full, dataset) -> dict:
    def mean_score(runs: dict) -> float:
        scores = [r["iterations"][-1]["verification_score"] for r in runs.values() if r.get("iterations")]
        return sum(scores) / len(scores) if scores else float("nan")

    by_id = {s["id"]: s for s in dataset["scenarios"]}

    def breakdown(runs: dict, field: str) -> dict:
        buckets: dict[str, list[float]] = {}
        for sid, run in runs.items():
            meta = by_id.get(sid, {})
            key = meta.get(field, "unknown")
            score = run["iterations"][-1]["verification_score"] if run.get("iterations") else None
            if score is not None:
                buckets.setdefault(str(key), []).append(score)
        return {k: round(sum(v) / len(v), 3) for k, v in buckets.items()}

    return {
        "overall": {
            "generator_only_mean": mean_score(generator_only),
            "full_mean": mean_score(full),
        },
        "by_difficulty": {
            "generator_only": breakdown(generator_only, "difficulty"),
            "full": breakdown(full, "difficulty"),
        },
        "by_weather": {
            "generator_only": breakdown(generator_only, "expected_weather"),
            "full": breakdown(full, "expected_weather"),
        },
        "by_event": {
            "generator_only": breakdown(generator_only, "expected_event"),
            "full": breakdown(full, "expected_event"),
        },
    }


def figure3_reflection_convergence(full: dict, out_path: Path, dataset) -> None:
    by_id = {s["id"]: s for s in dataset["scenarios"]}
    fig, ax = plt.subplots(figsize=(7, 4.5))

    plotted = 0
    for sid, run in full.items():
        iterations = run.get("iterations", [])
        if not iterations or iterations[0]["verification_score"] >= 0.72:
            continue  # only show scenarios that initially failed
        scores = [it["verification_score"] for it in iterations]
        label = by_id.get(sid, {}).get("prompt", sid)[:38] + "..."
        ax.plot(range(len(scores)), scores, marker="o", label=f"{sid}: {label}")
        plotted += 1
        if plotted >= 6:
            break

    ax.axhline(0.72, color="gray", linestyle="--", linewidth=1, label="pass threshold (0.72)")
    ax.set_xlabel("Reflection iteration")
    ax.set_ylabel("Semantic alignment score")
    ax.set_title("Reflection convergence for initially-failing scenarios")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved %s (%d scenarios plotted)", out_path, plotted)


def figure4_difficulty_breakdown(table2: dict, out_path: Path) -> None:
    difficulties = ["easy", "medium", "hard"]
    gen_only = [table2["by_difficulty"]["generator_only"].get(d, 0) for d in difficulties]
    full = [table2["by_difficulty"]["full"].get(d, 0) for d in difficulties]

    x = range(len(difficulties))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - width / 2 for i in x], gen_only, width, label="Generator-only")
    ax.bar([i + width / 2 for i in x], full, width, label="Full Prompt-to-Drive")
    ax.set_xticks(list(x))
    ax.set_xticklabels(difficulties)
    ax.set_ylabel("Mean semantic alignment score")
    ax.set_title("Semantic alignment by scenario difficulty")
    ax.set_ylim(0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="outputs/baseline/baseline_results.json")
    parser.add_argument("--generator-only", default="outputs/generator_only")
    parser.add_argument("--full", default="outputs/full")
    parser.add_argument("--dataset", default="scenarios/dataset.json")
    parser.add_argument("--out", default="eval/results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = json.loads(Path(args.dataset).read_text())
    baseline = _load_baseline(args.baseline)
    generator_only = _load_pipeline_runs(args.generator_only)
    full = _load_pipeline_runs(args.full)

    # Table 1
    t1 = table1_success_rate(baseline, generator_only, full)
    with open(out_dir / "table1_success_rate.csv", "w") as f:
        f.write("condition,success_rate\n")
        for k, v in t1.items():
            f.write(f'"{k}",{v:.3f}\n')
    logger.info("Table 1: %s", t1)

    # Table 2
    t2 = table2_semantic_alignment(generator_only, full, dataset)
    with open(out_dir / "table2_semantic_alignment.csv", "w") as f:
        f.write("category,subcategory,generator_only,full\n")
        f.write(f"overall,-,{t2['overall']['generator_only_mean']:.3f},{t2['overall']['full_mean']:.3f}\n")
        for cat in ["by_difficulty", "by_weather", "by_event"]:
            keys = set(t2[cat]["generator_only"]) | set(t2[cat]["full"])
            for k in sorted(keys):
                go = t2[cat]["generator_only"].get(k, "")
                fu = t2[cat]["full"].get(k, "")
                f.write(f"{cat},{k},{go},{fu}\n")
    logger.info("Table 2 overall: %s", t2["overall"])

    # Figures
    if full:
        figure3_reflection_convergence(full, out_dir / "figure3_reflection_convergence.pdf", dataset)
        figure4_difficulty_breakdown(t2, out_dir / "figure4_difficulty_breakdown.pdf")
    else:
        logger.warning("No 'full' condition runs found -- skipping figures. Run eval/run_full_evaluation.py --mode full first.")

    summary = {"table1_success_rate": t1, "table2_semantic_alignment": t2}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("All metrics written to %s/", out_dir)


if __name__ == "__main__":
    main()
