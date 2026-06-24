"""
eval/compute_metrics.py

Reads the output of run_baseline.py and run_full_evaluation.py (both
modes) and computes:

  Table 1 -- Executable scenario success rate, by condition
  Table 2 -- Semantic alignment (formal predicate agreement), by condition
             and by category (weather / actor / event / difficulty)
  Table 3 -- Reflection recovery rate: first-pass vs. after-reflection
             pass rate, and the fraction of first-pass FAILURES that the
             Reflector subsequently recovers. This is the metric a CAIESA
             reviewer specifically asked for as evidence the Reflector
             does something, rather than the loop "looking" useful.
  Figure 3 -- Reflection convergence: predicate agreement by iteration for
              scenarios that initially failed verification
  Figure 4 -- Semantic alignment comparison across difficulty levels

NOTE ON THE METRIC: "semantic alignment" here is `predicate_agreement` --
the fraction of APPLICABLE formal predicates (weather_correct,
pedestrian_present, vehicles_present, event_observable) that held, per
majority vote across n_samples independent VLM calls. It is NOT a single
opaque VLM-reported float. See docs/SEMANTIC_VERIFICATION_METRIC.md for
the full formal definition.

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
    eval/results/table3_reflection_recovery.csv
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
        scores = [r["iterations"][-1]["predicate_agreement"] for r in runs.values() if r.get("iterations")]
        return sum(scores) / len(scores) if scores else float("nan")

    by_id = {s["id"]: s for s in dataset["scenarios"]}

    def breakdown(runs: dict, field: str) -> dict:
        buckets: dict[str, list[float]] = {}
        for sid, run in runs.items():
            meta = by_id.get(sid, {})
            key = meta.get(field, "unknown")
            score = run["iterations"][-1]["predicate_agreement"] if run.get("iterations") else None
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


def table3_reflection_recovery(full: dict) -> dict:
    """
    The metric a CAIESA reviewer specifically asked for: does the
    Reflector actually do anything, beyond the loop merely "looking"
    closed?

      first_pass_rate       = fraction of scenarios that pass on iteration 0
      after_reflection_rate = fraction of scenarios that pass by their final iteration
      recovery_rate         = of the scenarios that FAILED on iteration 0,
                               what fraction were subsequently recovered
                               (passed) by iteration 1 or 2?

    recovery_rate is undefined (reported as NaN) if no scenario failed on
    the first pass -- that is itself worth noting in the paper, not hidden.
    """
    if not full:
        return dict(first_pass_rate=float("nan"), after_reflection_rate=float("nan"),
                     recovery_rate=float("nan"), n_first_pass_failures=0, n_recovered=0)

    first_pass_pass = [r["iterations"][0]["verification_passed"] for r in full.values() if r.get("iterations")]
    final_pass = [r["iterations"][-1]["verification_passed"] for r in full.values() if r.get("iterations")]

    n = len(first_pass_pass)
    first_pass_rate = sum(first_pass_pass) / n if n else float("nan")
    after_reflection_rate = sum(final_pass) / n if n else float("nan")

    first_pass_failures = [r for r in full.values() if r.get("iterations") and not r["iterations"][0]["verification_passed"]]
    n_failures = len(first_pass_failures)
    n_recovered = sum(1 for r in first_pass_failures if r["iterations"][-1]["verification_passed"])
    recovery_rate = n_recovered / n_failures if n_failures else float("nan")

    return dict(
        first_pass_rate=round(first_pass_rate, 3),
        after_reflection_rate=round(after_reflection_rate, 3),
        recovery_rate=round(recovery_rate, 3) if n_failures else float("nan"),
        n_first_pass_failures=n_failures,
        n_recovered=n_recovered,
    )


def table4_seed_variance(full: dict, dataset: dict) -> dict:
    """
    Aggregates seeded-variant runs (produced by eval/expand_dataset.py)
    back up to their core scenario, reporting mean +/- std of the final
    predicate_agreement score across seeds for each core scenario, plus
    an overall mean +/- 95% CI (normal approximation) across all core
    scenarios. This directly answers the reviewer's statistical-power
    concern: a single 20-scenario point estimate becomes a per-scenario
    distribution over repeated executions.

    Only meaningful when `full` was produced by running
    eval.run_full_evaluation against an EXPANDED dataset (i.e. scenario
    ids look like "s01_seed0", "s01_seed1", ...). If `full` was produced
    against the plain 20-scenario dataset.json, this returns an empty
    result and the caller should skip writing Table 4.
    """
    by_core: dict[str, list[float]] = {}
    for sid, run in full.items():
        core_id = sid.split("_seed")[0] if "_seed" in sid else None
        if core_id is None or not run.get("iterations"):
            continue
        score = run["iterations"][-1]["predicate_agreement"]
        by_core.setdefault(core_id, []).append(score)

    if not by_core:
        return {"applicable": False, "per_scenario": {}, "overall_mean": None, "overall_ci95": None}

    import math
    import statistics

    per_scenario = {}
    for core_id, scores in by_core.items():
        mean = statistics.fmean(scores)
        std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        per_scenario[core_id] = {"mean": round(mean, 4), "std": round(std, 4), "n_seeds": len(scores)}

    all_means = [v["mean"] for v in per_scenario.values()]
    overall_mean = statistics.fmean(all_means)
    overall_std = statistics.pstdev(all_means) if len(all_means) > 1 else 0.0
    # Normal-approximation 95% CI on the mean-of-means across core scenarios.
    n = len(all_means)
    margin = 1.96 * (overall_std / math.sqrt(n)) if n > 1 else 0.0

    return {
        "applicable": True,
        "per_scenario": per_scenario,
        "overall_mean": round(overall_mean, 4),
        "overall_std": round(overall_std, 4),
        "overall_ci95_margin": round(margin, 4),
        "overall_ci95": [round(overall_mean - margin, 4), round(overall_mean + margin, 4)],
        "n_core_scenarios": n,
    }


def figure3_reflection_convergence(full: dict, out_path: Path, dataset) -> None:
    by_id = {s["id"]: s for s in dataset["scenarios"]}
    fig, ax = plt.subplots(figsize=(7, 4.5))

    plotted = 0
    for sid, run in full.items():
        iterations = run.get("iterations", [])
        if not iterations or iterations[0]["predicate_agreement"] >= 0.72:
            continue  # only show scenarios that initially failed
        scores = [it["predicate_agreement"] for it in iterations]
        stds = [it.get("predicate_agreement_std", 0.0) for it in iterations]
        label = by_id.get(sid, {}).get("prompt", sid)[:38] + "..."
        x = range(len(scores))
        ax.errorbar(x, scores, yerr=stds, marker="o", capsize=3, label=f"{sid}: {label}")
        plotted += 1
        if plotted >= 6:
            break

    ax.axhline(0.72, color="gray", linestyle="--", linewidth=1, label="pass threshold (0.72)")
    ax.set_xlabel("Reflection iteration")
    ax.set_ylabel("Predicate agreement (formal semantic score)")
    ax.set_title("Reflection convergence for initially-failing scenarios\n(error bars: std across verifier samples)")
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
    ax.set_ylabel("Mean predicate agreement")
    ax.set_title("Semantic alignment (predicate agreement) by scenario difficulty")
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

    # Table 3 -- reflection recovery rate (the metric reviewers want)
    t3 = table3_reflection_recovery(full)
    with open(out_dir / "table3_reflection_recovery.csv", "w") as f:
        f.write("metric,value\n")
        for k, v in t3.items():
            f.write(f"{k},{v}\n")
    logger.info("Table 3 (reflection recovery): %s", t3)

    # Table 4 -- seed variance / CI (only populated if `full` was run
    # against an expanded, seeded dataset from eval/expand_dataset.py)
    t4 = table4_seed_variance(full, dataset)
    if t4["applicable"]:
        with open(out_dir / "table4_seed_variance.csv", "w") as f:
            f.write("core_scenario_id,mean,std,n_seeds\n")
            for core_id, v in t4["per_scenario"].items():
                f.write(f"{core_id},{v['mean']},{v['std']},{v['n_seeds']}\n")
            f.write(f"\nOVERALL,{t4['overall_mean']},{t4['overall_std']},{t4['n_core_scenarios']}\n")
            f.write(f"95% CI on overall mean: [{t4['overall_ci95'][0]}, {t4['overall_ci95'][1]}]\n")
        logger.info("Table 4 (seed variance): overall_mean=%.3f 95%%CI=%s over %d core scenarios",
                    t4["overall_mean"], t4["overall_ci95"], t4["n_core_scenarios"])
    else:
        logger.info("Table 4 not applicable -- `full` does not look like an expanded/seeded dataset run "
                     "(no '_seedN' scenario ids found). Run eval.expand_dataset + eval.run_full_evaluation "
                     "against the expanded dataset to populate this table.")

    # Figures
    if full:
        figure3_reflection_convergence(full, out_dir / "figure3_reflection_convergence.pdf", dataset)
        figure4_difficulty_breakdown(t2, out_dir / "figure4_difficulty_breakdown.pdf")
    else:
        logger.warning("No 'full' condition runs found -- skipping figures. Run eval/run_full_evaluation.py --mode full first.")

    summary = {
        "table1_success_rate": t1, "table2_semantic_alignment": t2,
        "table3_reflection_recovery": t3, "table4_seed_variance": t4,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("All metrics written to %s/", out_dir)


if __name__ == "__main__":
    main()
