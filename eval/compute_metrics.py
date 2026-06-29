"""
eval/compute_metrics.py

Computes all tables and figures for the Prompt-to-Drive paper.

TABLES
------
Table 1 -- Executable success rate, by condition
Table 2 -- Semantic alignment (predicate agreement), by condition + breakdown
Table 3 -- Failure Taxonomy: per-category failure count and percentage
           (Weather / Actor / Behavior / Environment)
           THIS IS THE TABLE THE REVIEWER SPECIFICALLY ASKED FOR.
           It is computed directly from failed_predicates in each pipeline_run.json.
           No new data collection needed -- it is latent in the existing output.
Table 4 -- Reflection Recovery by Failure Category: what fraction of each
           failure type is repaired by the Reflector across iterations.
Table 5 -- Per-predicate alignment breakdown (weather / actors / behavior / environment)
           "Behavioral predicates remain the dominant source of semantic errors
            despite high environmental consistency." -- this is the scientific claim.
Table 6 -- Seed variance / CI (only if expanded dataset was used)

FIGURES
-------
Figure 1 -- Reflection convergence: predicate agreement by iteration (initially-failing scenarios)
Figure 2 -- Failure distribution: bar chart of failure categories
Figure 3 -- Recovery by failure type: bar chart
Figure 4 -- Semantic alignment by difficulty level

PREDICATE → FAILURE CATEGORY MAPPING
--------------------------------------
This mapping is the core of the failure taxonomy. It lives here, not scattered
across the project, per the reviewer's recommendation.

  weather_correct   → Weather Failure   (wrong atmospheric condition rendered)
  pedestrian_present → Actor Failure    (requested actor missing from scene)
  vehicles_present  → Environment Failure (scene density / context wrong)
  event_observable  → Behavior Failure  (the driving event did not occur / was not visible)

Usage:
    python -m eval.compute_metrics \\
        --baseline  outputs/baseline/baseline_results.json \\
        --generator-only outputs/generator_only \\
        --full outputs/full \\
        --dataset scenarios/dataset.json \\
        --out eval/results

Outputs (all in --out directory):
    table1_success_rate.csv
    table2_semantic_alignment.csv
    table3_failure_taxonomy.csv
    table4_reflection_recovery_by_category.csv
    table5_per_predicate_alignment.csv
    table6_seed_variance.csv  (only if seeded dataset used)
    figure1_reflection_convergence.pdf
    figure2_failure_distribution.pdf
    figure3_recovery_by_failure_type.pdf
    figure4_difficulty_breakdown.pdf
    summary.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prompt_to_drive.eval.metrics")

# -----------------------------------------------------------------------
# Predicate → Failure Category mapping (reviewer's taxonomy)
# "The taxonomy is latent in your code. You're simply not exposing it."
# -----------------------------------------------------------------------
PREDICATE_TO_CATEGORY = {
    "weather_correct":    "Weather",
    "pedestrian_present": "Actor",
    "vehicles_present":   "Environment",
    "event_observable":   "Behavior",
}

ALL_CATEGORIES = ["Weather", "Actor", "Behavior", "Environment"]
ALL_PREDICATES = list(PREDICATE_TO_CATEGORY.keys())


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------
def _load_pipeline_runs(out_dir: str) -> dict[str, dict]:
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


def _last_iter(run: dict) -> dict | None:
    its = run.get("iterations", [])
    return its[-1] if its else None


def _first_iter(run: dict) -> dict | None:
    its = run.get("iterations", [])
    return its[0] if its else None


# -----------------------------------------------------------------------
# Table 1: Executable success rate
# -----------------------------------------------------------------------
def table1_success_rate(baseline: dict, generator_only: dict, full: dict) -> dict:
    def exec_rate(runs: dict) -> float:
        if not runs:
            return float("nan")
        return sum(1 for r in runs.values()
                   if any(it["execution_success"] for it in r.get("iterations", []))) / len(runs)

    def baseline_rate(b: dict) -> float:
        if not b:
            return float("nan")
        return sum(1 for r in b.values() if r.get("execution_success")) / len(b)

    return {
        "Baseline (free-form code gen)":              round(baseline_rate(baseline), 3),
        "Generator-only (schema, no reflection)":     round(exec_rate(generator_only), 3),
        "Full Prompt-to-Drive (closed-loop)":         round(exec_rate(full), 3),
    }


# -----------------------------------------------------------------------
# Table 2: Semantic alignment by condition and breakdown
# -----------------------------------------------------------------------
def table2_semantic_alignment(generator_only: dict, full: dict, dataset: dict) -> dict:
    by_id = {s["id"]: s for s in dataset["scenarios"]}

    def mean_agreement(runs: dict) -> float:
        scores = [_last_iter(r)["verification_score"]
                  for r in runs.values() if _last_iter(r)]
        return round(statistics.fmean(scores), 3) if scores else float("nan")

    def breakdown(runs: dict, field: str) -> dict[str, float]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for sid, run in runs.items():
            meta = by_id.get(sid.split("_seed")[0], {})
            key = str(meta.get(field, "unknown"))
            li = _last_iter(run)
            if li:
                buckets[key].append(li["verification_score"])
        return {k: round(statistics.fmean(v), 3) for k, v in sorted(buckets.items())}

    return {
        "overall": {
            "generator_only": mean_agreement(generator_only),
            "full":           mean_agreement(full),
        },
        "by_difficulty": {
            "generator_only": breakdown(generator_only, "difficulty"),
            "full":           breakdown(full, "difficulty"),
        },
        "by_weather": {
            "generator_only": breakdown(generator_only, "expected_weather"),
            "full":           breakdown(full, "expected_weather"),
        },
        "by_event": {
            "generator_only": breakdown(generator_only, "expected_event"),
            "full":           breakdown(full, "expected_event"),
        },
    }


# -----------------------------------------------------------------------
# Table 3: Failure Taxonomy
# "The taxonomy is latent in your code. You're simply not exposing it."
#
# Counts each predicate failure across ALL iterations of ALL scenarios,
# then maps to failure category. Reports count + percentage.
# -----------------------------------------------------------------------
def table3_failure_taxonomy(full: dict) -> dict:
    """
    For each scenario's FIRST iteration (before reflection), count how many
    times each predicate failed. This gives the raw failure distribution
    BEFORE the Reflector has had a chance to act -- the baseline failure
    profile of the Generator + Engine alone.

    Reports both:
      - first_pass_failures: failures at iteration 0 (pre-reflection)
      - final_failures:      failures at final iteration (post-reflection)
    """
    first_pass_counts: dict[str, int] = defaultdict(int)
    final_counts: dict[str, int] = defaultdict(int)
    n_scenarios = 0

    for run in full.values():
        iters = run.get("iterations", [])
        if not iters:
            continue
        n_scenarios += 1

        # First pass failures (iteration 0)
        first = iters[0]
        for pred in first.get("missing_elements", []):
            cat = PREDICATE_TO_CATEGORY.get(pred, "Unknown")
            first_pass_counts[cat] += 1

        # Final iteration failures
        last = iters[-1]
        for pred in last.get("missing_elements", []):
            cat = PREDICATE_TO_CATEGORY.get(pred, "Unknown")
            final_counts[cat] += 1

    total_first = sum(first_pass_counts.values()) or 1
    total_final = sum(final_counts.values()) or 1

    taxonomy = {}
    for cat in ALL_CATEGORIES:
        fp = first_pass_counts.get(cat, 0)
        fn = final_counts.get(cat, 0)
        taxonomy[cat] = {
            "first_pass_count":      fp,
            "first_pass_pct":        round(100 * fp / total_first, 1),
            "final_count":           fn,
            "final_pct":             round(100 * fn / total_final, 1),
        }

    taxonomy["_meta"] = {
        "n_scenarios": n_scenarios,
        "total_first_pass_failures": total_first,
        "total_final_failures": total_final,
        "note": (
            "first_pass = failures at iteration 0 before Reflector acts; "
            "final = failures remaining after all reflection iterations."
        ),
    }
    return taxonomy


# -----------------------------------------------------------------------
# Table 4: Reflection Recovery by Failure Category
# "For each failure type, what fraction does the Reflector repair?"
# -----------------------------------------------------------------------
def table4_reflection_recovery_by_category(full: dict) -> dict:
    """
    For each failure category:
      - count how many scenarios had that category fail at iteration 0
      - count how many of those scenarios passed at their final iteration
      - recovery_rate = recovered / first_failed

    Also reports overall first_pass_rate, after_reflection_rate, recovery_rate.
    """
    # Per-category tracking
    first_failed: dict[str, list[str]] = defaultdict(list)   # category -> [scenario_ids]
    final_passed_given_first_failed: dict[str, int] = defaultdict(int)

    first_pass_passes = 0
    final_passes = 0
    n = 0

    for sid, run in full.items():
        iters = run.get("iterations", [])
        if not iters:
            continue
        n += 1

        first = iters[0]
        last = iters[-1]

        if first.get("verification_passed"):
            first_pass_passes += 1
        if last.get("verification_passed"):
            final_passes += 1

        # Which categories failed at iteration 0?
        cats_failed_first = set()
        for pred in first.get("missing_elements", []):
            cat = PREDICATE_TO_CATEGORY.get(pred, "Unknown")
            cats_failed_first.add(cat)
            first_failed[cat].append(sid)

        # Did this scenario recover by the final iteration?
        recovered = last.get("verification_passed", False)
        for cat in cats_failed_first:
            if recovered:
                final_passed_given_first_failed[cat] += 1

    recovery: dict[str, dict] = {}
    for cat in ALL_CATEGORIES:
        ff = len(first_failed[cat])
        rec = final_passed_given_first_failed.get(cat, 0)
        recovery[cat] = {
            "first_pass_failures": ff,
            "recovered":           rec,
            "recovery_rate":       round(rec / ff, 3) if ff > 0 else float("nan"),
        }

    recovery["_overall"] = {
        "n_scenarios":            n,
        "first_pass_rate":        round(first_pass_passes / n, 3) if n else float("nan"),
        "after_reflection_rate":  round(final_passes / n, 3) if n else float("nan"),
        "overall_recovery_rate":  (
            round((final_passes - first_pass_passes) /
                  (n - first_pass_passes), 3)
            if n > first_pass_passes else float("nan")
        ),
    }
    return recovery


# -----------------------------------------------------------------------
# Table 5: Per-predicate alignment breakdown
# "Weather 0.98 / Actors 0.81 / Behavior 0.73 / Environment 0.95"
# This is an actual scientific observation.
# -----------------------------------------------------------------------
def table5_per_predicate_alignment(full: dict, generator_only: dict) -> dict:
    """
    For each predicate, compute the fraction of APPLICABLE scenarios where
    it held at the FINAL iteration. Reports separately for generator_only
    and full conditions so the per-predicate improvement from reflection
    is visible.
    """
    def compute(runs: dict) -> dict[str, dict]:
        pred_held: dict[str, list[bool]] = defaultdict(list)
        for run in runs.values():
            li = _last_iter(run)
            if not li:
                continue
            missing = set(li.get("missing_elements", []))
            spec = li.get("spec", {})
            # Determine applicability from spec fields
            applicable = {
                "weather_correct":    True,
                "pedestrian_present": bool(spec.get("pedestrian", False)),
                "vehicles_present":   int(spec.get("num_vehicles", 0)) > 0,
                "event_observable":   True,
            }
            for pred, app in applicable.items():
                if app:
                    pred_held[pred].append(pred not in missing)

        result = {}
        for pred in ALL_PREDICATES:
            vals = pred_held.get(pred, [])
            cat = PREDICATE_TO_CATEGORY[pred]
            result[pred] = {
                "category":    cat,
                "n_applicable": len(vals),
                "pass_rate":   round(statistics.fmean(vals), 3) if vals else float("nan"),
            }
        return result

    return {
        "generator_only": compute(generator_only),
        "full":           compute(full),
    }


# -----------------------------------------------------------------------
# Table 6: Seed variance / CI (only for expanded datasets)
# -----------------------------------------------------------------------
def table6_seed_variance(full: dict) -> dict:
    by_core: dict[str, list[float]] = defaultdict(list)
    for sid, run in full.items():
        if "_seed" not in sid:
            continue
        core_id = sid.split("_seed")[0]
        li = _last_iter(run)
        if li:
            by_core[core_id].append(li["verification_score"])

    if not by_core:
        return {"applicable": False}

    per_scenario = {}
    for core_id, scores in sorted(by_core.items()):
        mean = statistics.fmean(scores)
        std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        per_scenario[core_id] = {"mean": round(mean, 4), "std": round(std, 4), "n": len(scores)}

    all_means = [v["mean"] for v in per_scenario.values()]
    overall_mean = statistics.fmean(all_means)
    overall_std = statistics.pstdev(all_means) if len(all_means) > 1 else 0.0
    n = len(all_means)
    margin = 1.96 * (overall_std / math.sqrt(n)) if n > 1 else 0.0

    return {
        "applicable": True,
        "per_scenario": per_scenario,
        "overall_mean": round(overall_mean, 4),
        "overall_std":  round(overall_std, 4),
        "ci95": [round(overall_mean - margin, 4), round(overall_mean + margin, 4)],
        "n_core_scenarios": n,
    }


# -----------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------
def figure1_reflection_convergence(full: dict, out_path: Path) -> None:
    """Agreement vs. iteration for scenarios that failed on the first pass."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for sid, run in sorted(full.items()):
        iters = run.get("iterations", [])
        if not iters or iters[0].get("verification_passed"):
            continue
        scores = [it["verification_score"] for it in iters]
        x = list(range(len(scores)))
        ax.plot(x, scores, marker="o", label=sid)
        plotted += 1
        if plotted >= 8:
            break

    ax.axhline(0.72, color="red", linestyle="--", linewidth=1.2, label="pass threshold (0.72)")
    ax.set_xlabel("Reflection Iteration")
    ax.set_ylabel("Predicate Agreement Score")
    ax.set_title("Figure 1: Reflection Convergence\n(scenarios that failed iteration 0)")
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 1, 2])
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved %s (%d scenarios plotted)", out_path, plotted)


def figure2_failure_distribution(taxonomy: dict, out_path: Path) -> None:
    """Bar chart of failure category distribution (first-pass failures)."""
    cats = ALL_CATEGORIES
    counts = [taxonomy.get(c, {}).get("first_pass_count", 0) for c in cats]
    pcts = [taxonomy.get(c, {}).get("first_pass_pct", 0) for c in cats]

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(cats, counts, color=colors, edgecolor="white", linewidth=0.8)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Failure Count (first-pass)")
    ax.set_title("Figure 2: Failure Distribution by Category\n(pre-reflection, all scenarios)")
    ax.set_ylim(0, max(counts) * 1.25 if counts else 10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def figure3_recovery_by_failure_type(recovery: dict, out_path: Path) -> None:
    """Bar chart: reflection recovery rate per failure category."""
    cats = ALL_CATEGORIES
    rates = [recovery.get(c, {}).get("recovery_rate", 0) for c in cats]
    counts = [recovery.get(c, {}).get("first_pass_failures", 0) for c in cats]

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(cats, [r if not isinstance(r, float) or not math.isnan(r) else 0 for r in rates],
                  color=colors, edgecolor="white", linewidth=0.8)
    for bar, rate, count in zip(bars, rates, counts):
        label = f"{rate:.0%}" if not (isinstance(rate, float) and math.isnan(rate)) else "N/A"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{label}\n(n={count})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Recovery Rate")
    ax.set_title("Figure 3: Reflection Recovery Rate by Failure Category")
    ax.set_ylim(0, 1.25)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def figure4_per_predicate_alignment(t5: dict, out_path: Path) -> None:
    """
    Per-predicate pass rate for generator_only vs. full.
    This is the figure that supports the scientific claim:
    'Behavioral predicates remain the dominant source of semantic errors
     despite high environmental consistency.'
    """
    pred_labels = {
        "weather_correct":    "Weather\n(correct)",
        "pedestrian_present": "Actor\n(pedestrian)",
        "vehicles_present":   "Environment\n(vehicles)",
        "event_observable":   "Behavior\n(event)",
    }
    preds = ALL_PREDICATES
    go_rates = [t5["generator_only"].get(p, {}).get("pass_rate", 0) or 0 for p in preds]
    full_rates = [t5["full"].get(p, {}).get("pass_rate", 0) or 0 for p in preds]
    labels = [pred_labels[p] for p in preds]

    x = np.arange(len(preds))
    width = 0.35
    colors_go   = "#AEC6E8"
    colors_full = "#1F77B4"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, go_rates,   width, label="Generator-only", color=colors_go,   edgecolor="white")
    ax.bar(x + width / 2, full_rates, width, label="Full (w/ reflection)", color=colors_full, edgecolor="white")

    for xi, (go, fu) in enumerate(zip(go_rates, full_rates)):
        ax.text(xi - width / 2, go + 0.02, f"{go:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(xi + width / 2, fu + 0.02, f"{fu:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Pass Rate (fraction of applicable scenarios)")
    ax.set_title("Figure 4: Per-Predicate Semantic Alignment\n(Generator-only vs. Full Closed-Loop)")
    ax.set_ylim(0, 1.15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out_path)


# -----------------------------------------------------------------------
# CSV helpers
# -----------------------------------------------------------------------
def _write_csv(path: Path, rows: list[list]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline",        default="outputs/baseline/baseline_results.json")
    parser.add_argument("--generator-only",  default="outputs/generator_only")
    parser.add_argument("--full",            default="outputs/full")
    parser.add_argument("--dataset",         default="scenarios/dataset.json")
    parser.add_argument("--out",             default="eval/results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset       = json.loads(Path(args.dataset).read_text())
    baseline      = _load_baseline(args.baseline)
    generator_only = _load_pipeline_runs(args.generator_only)
    full          = _load_pipeline_runs(args.full)

    logger.info("Loaded: baseline=%d, generator_only=%d, full=%d scenarios",
                len(baseline), len(generator_only), len(full))

    # ---- Table 1 -----------------------------------------------------------
    t1 = table1_success_rate(baseline, generator_only, full)
    _write_csv(out_dir / "table1_success_rate.csv",
               [["condition", "success_rate"]] +
               [[k, f"{v:.3f}"] for k, v in t1.items()])
    logger.info("Table 1 (success rate): %s", t1)

    # ---- Table 2 -----------------------------------------------------------
    t2 = table2_semantic_alignment(generator_only, full, dataset)
    rows = [["category", "subcategory", "generator_only", "full"],
            ["overall", "-", t2["overall"]["generator_only"], t2["overall"]["full"]]]
    for cat in ["by_difficulty", "by_weather", "by_event"]:
        keys = sorted(set(t2[cat]["generator_only"]) | set(t2[cat]["full"]))
        for k in keys:
            rows.append([cat, k,
                         t2[cat]["generator_only"].get(k, ""),
                         t2[cat]["full"].get(k, "")])
    _write_csv(out_dir / "table2_semantic_alignment.csv", rows)
    logger.info("Table 2 overall: %s", t2["overall"])

    # ---- Table 3: Failure Taxonomy -----------------------------------------
    t3 = table3_failure_taxonomy(full)
    rows = [["failure_category", "first_pass_count", "first_pass_pct",
             "final_count", "final_pct"]]
    for cat in ALL_CATEGORIES:
        d = t3.get(cat, {})
        rows.append([cat, d.get("first_pass_count", 0), d.get("first_pass_pct", 0),
                     d.get("final_count", 0), d.get("final_pct", 0)])
    _write_csv(out_dir / "table3_failure_taxonomy.csv", rows)
    logger.info("Table 3 (failure taxonomy): %s",
                {c: t3[c] for c in ALL_CATEGORIES if c in t3})

    # ---- Table 4: Recovery by Category -------------------------------------
    t4 = table4_reflection_recovery_by_category(full)
    rows = [["failure_category", "first_pass_failures", "recovered", "recovery_rate"]]
    for cat in ALL_CATEGORIES:
        d = t4.get(cat, {})
        rows.append([cat, d.get("first_pass_failures", 0),
                     d.get("recovered", 0), d.get("recovery_rate", "nan")])
    rows.append(["OVERALL",
                 t4["_overall"]["n_scenarios"],
                 "-",
                 t4["_overall"]["overall_recovery_rate"]])
    _write_csv(out_dir / "table4_reflection_recovery_by_category.csv", rows)
    logger.info("Table 4 (recovery by category): %s",
                {c: t4[c] for c in ALL_CATEGORIES if c in t4})
    logger.info("Table 4 overall: %s", t4["_overall"])

    # ---- Table 5: Per-predicate alignment ----------------------------------
    t5 = table5_per_predicate_alignment(full, generator_only)
    rows = [["predicate", "category", "generator_only_pass_rate", "full_pass_rate",
             "n_applicable_go", "n_applicable_full"]]
    for pred in ALL_PREDICATES:
        go = t5["generator_only"].get(pred, {})
        fu = t5["full"].get(pred, {})
        rows.append([pred, PREDICATE_TO_CATEGORY[pred],
                     go.get("pass_rate", "nan"), fu.get("pass_rate", "nan"),
                     go.get("n_applicable", 0), fu.get("n_applicable", 0)])
    _write_csv(out_dir / "table5_per_predicate_alignment.csv", rows)
    logger.info("Table 5 (per-predicate): generator_only=%s  full=%s",
                {p: t5["generator_only"].get(p, {}).get("pass_rate") for p in ALL_PREDICATES},
                {p: t5["full"].get(p, {}).get("pass_rate") for p in ALL_PREDICATES})

    # ---- Table 6: Seed variance (optional) ---------------------------------
    t6 = table6_seed_variance(full)
    if t6["applicable"]:
        rows = [["core_scenario_id", "mean", "std", "n_seeds"]]
        for cid, v in t6["per_scenario"].items():
            rows.append([cid, v["mean"], v["std"], v["n"]])
        rows.append(["OVERALL", t6["overall_mean"], t6["overall_std"], t6["n_core_scenarios"]])
        rows.append([f"95%CI", t6["ci95"][0], "to", t6["ci95"][1]])
        _write_csv(out_dir / "table6_seed_variance.csv", rows)
        logger.info("Table 6: overall=%.3f 95%%CI=%s", t6["overall_mean"], t6["ci95"])
    else:
        logger.info("Table 6 skipped (no seeded runs found; run expand_dataset.py first)")

    # ---- Figures -----------------------------------------------------------
    if full:
        figure1_reflection_convergence(full, out_dir / "figure1_reflection_convergence.pdf")
        figure2_failure_distribution(t3, out_dir / "figure2_failure_distribution.pdf")
        figure3_recovery_by_failure_type(t4, out_dir / "figure3_recovery_by_failure_type.pdf")
        figure4_per_predicate_alignment(t5, out_dir / "figure4_per_predicate_alignment.pdf")
    else:
        logger.warning("No full-condition runs found. Run eval/run_full_evaluation.py --mode full first.")

    # ---- Summary JSON ------------------------------------------------------
    summary = {
        "table1_success_rate":               t1,
        "table2_semantic_alignment":          t2,
        "table3_failure_taxonomy":            {c: t3[c] for c in ALL_CATEGORIES if c in t3},
        "table4_reflection_recovery":         {c: t4[c] for c in ALL_CATEGORIES if c in t4},
        "table4_overall_recovery":            t4["_overall"],
        "table5_per_predicate_alignment":     {
            p: {"go": t5["generator_only"].get(p, {}).get("pass_rate"),
                "full": t5["full"].get(p, {}).get("pass_rate")}
            for p in ALL_PREDICATES
        },
        "table6_seed_variance":               t6,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("All metrics written to %s/", out_dir)


if __name__ == "__main__":
    main()