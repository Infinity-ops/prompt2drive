"""
eval/compute_metrics.py

Computes all tables and figures for the Prompt-to-Drive paper, updated to fully
account for grounded stress-test conditions and multi-actor validation metrics.

TABLES
------
Table 1 -- Executable success rate, by condition
Table 2 -- Semantic alignment (predicate agreement), by condition + breakdown
Table 3 -- Failure Taxonomy: per-category failure count and percentage
Table 4 -- Reflection Recovery by Failure Category
Table 5 -- Per-predicate alignment breakdown
Table 6 -- Stress Split Variance / Performance Gap (Updated for non-expanded dataset)
Table 7 -- System Latency Breakdown
Table 8 -- Stress Benchmark Alignment Breakdown by Challenge Category (New)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # No GUI environment required
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prompt_to_drive.eval.metrics")

# Map the newly introduced stress-test failure conditions to their categories
PREDICATE_TO_CATEGORY = {
    "weather_correct":           "Weather",
    "fog_density_violation":     "Weather",   # Added for extreme weather stress evaluations
    "pedestrian_present":        "Actor",
    "sightline_occlusion_error": "Actor",     # Added for blocking truck visibility stress evaluations
    "vehicles_present":          "Environment",
    "event_observable":          "Behavior",
}

ALL_CATEGORIES = ["Weather", "Actor", "Behavior", "Environment"]
ALL_PREDICATES = list(PREDICATE_TO_CATEGORY.keys())


def _load_pipeline_runs(out_dir: str) -> dict[str, dict]:
    runs = {}
    root = Path(out_dir)
    if not root.exists():
        logger.warning("Directory %s does not exist; skipping.", out_dir)
        return runs
    for run_file in root.glob("*/pipeline_run.json"):
        try:
            data = json.loads(run_file.read_text())
            runs[data["scenario_id"]] = data
        except Exception as e:
            logger.error("Failed to load pipeline run file %s: %s", run_file, e)
    return runs


def _load_baseline(baseline_path: str) -> dict[str, dict]:
    p = Path(baseline_path)
    if not p.exists():
        logger.warning("Baseline file %s not found; skipping.", baseline_path)
        return {}
    try:
        data = json.loads(p.read_text())
        if isinstance(data, list):
            return {r["scenario_id"]: r for r in data}
        return data
    except Exception as e:
        logger.error("Failed to load baseline file %s: %s", baseline_path, e)
        return {}


def _last_iter(run: dict) -> dict | None:
    its = run.get("iterations", [])
    return its[-1] if its else None


def table1_success_rate(baseline: dict, generator_only: dict, full: dict) -> dict:
    def exec_rate(runs: dict) -> float:
        if not runs:
            return float("nan")
        success_count = sum(1 for r in runs.values() if any(it.get("execution_success", False) for it in r.get("iterations", [])))
        return success_count / len(runs)

    def baseline_rate(b: dict) -> float:
        if not b:
            return float("nan")
        return sum(1 for r in b.values() if r.get("execution_success", False)) / len(b)

    return {
        "Baseline (free-form code gen)":             round(baseline_rate(baseline), 3),
        "Generator-only (schema, no reflection)":     round(exec_rate(generator_only), 3),
        "Full Prompt-to-Drive (closed-loop)":          round(exec_rate(full), 3),
    }


def table2_semantic_alignment(generator_only: dict, full: dict, dataset: dict) -> dict:
    by_id = {s["id"]: s for s in dataset["scenarios"]}

    def mean_agreement(runs: dict) -> float:
        scores = []
        for r in runs.values():
            li = _last_iter(r)
            if li and "verification_score" in li:
                scores.append(li["verification_score"])
        return round(statistics.fmean(scores), 3) if scores else float("nan")

    def breakdown(runs: dict, field: str) -> dict[str, float]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for sid, run in runs.items():
            core_id = sid.split("_seed")[0]
            meta = by_id.get(core_id, {})
            key = str(meta.get(field, "unknown"))
            li = _last_iter(run)
            if li and "verification_score" in li:
                buckets[key].append(li["verification_score"])
        return {k: round(statistics.fmean(v), 3) for k, v in sorted(buckets.items())}

    return {
        "overall": {
            "generator_only": mean_agreement(generator_only),
            "full": mean_agreement(full)
        },
        "by_difficulty": {
            "generator_only": breakdown(generator_only, "difficulty"),
            "full": breakdown(full, "difficulty")
        },
        "by_weather": {
            "generator_only": breakdown(generator_only, "expected_weather"),
            "full": breakdown(full, "expected_weather")
        },
        "by_event": {
            "generator_only": breakdown(generator_only, "expected_event"),
            "full": breakdown(full, "expected_event")
        },
    }


def compute_true_failure_and_recovery(full: dict) -> tuple[dict, dict]:
    first_pass_counts = defaultdict(int)
    final_counts = defaultdict(int)
    first_failed_sids = defaultdict(set)
    recovered_counts = defaultdict(int)
    n_scenarios = 0

    for sid, run in full.items():
        iters = run.get("iterations", [])
        if not iters:
            continue
        n_scenarios += 1

        first_missing = set(iters[0].get("missing_elements", []))
        final_missing = set(iters[-1].get("missing_elements", []))

        for pred in first_missing:
            cat = PREDICATE_TO_CATEGORY.get(pred, "Environment")
            first_pass_counts[cat] += 1
            first_failed_sids[cat].add(sid)

        for pred in final_missing:
            cat = PREDICATE_TO_CATEGORY.get(pred, "Environment")
            final_counts[cat] += 1

        for pred in first_missing:
            cat = PREDICATE_TO_CATEGORY.get(pred, "Environment")
            if pred not in final_missing:
                recovered_counts[cat] += 1

    total_first = sum(first_pass_counts.values()) or 1
    total_final = sum(final_counts.values()) or 1

    taxonomy = {}
    recovery = {}

    for cat in ALL_CATEGORIES:
        fp = first_pass_counts[cat]
        fn = final_counts[cat]
        rec = recovered_counts[cat]
        ff_len = len(first_failed_sids[cat])

        taxonomy[cat] = {
            "first_pass_count": fp,
            "first_pass_pct":   round(100 * fp / total_first, 1),
            "final_count":      fn,
            "final_pct":        round(100 * fn / total_final, 1)
        }
        recovery[cat] = {
            "first_pass_failures": ff_len,
            "recovered":           rec,
            "recovery_rate":       round(rec / ff_len, 3) if ff_len > 0 else float("nan")
        }

    first_pass_passes = sum(1 for r in full.values() if r.get("iterations", [{}])[0].get("verification_passed", False))
    final_passes = sum(1 for r in full.values() if r.get("iterations", [{}])[-1].get("verification_passed", False))

    taxonomy["_meta"] = {
        "n_scenarios": n_scenarios,
        "total_first_pass_failures": total_first,
        "total_final_failures": total_final
    }
    
    recovery["_overall"] = {
        "n_scenarios":           n_scenarios,
        "first_pass_rate":       round(first_pass_passes / n_scenarios, 3) if n_scenarios else float("nan"),
        "after_reflection_rate": round(final_passes / n_scenarios, 3) if n_scenarios else float("nan"),
        "overall_recovery_rate": round((final_passes - first_pass_passes) / (n_scenarios - first_pass_passes), 3) if n_scenarios > first_pass_passes else float("nan")
    }

    return taxonomy, recovery


def table5_per_predicate_alignment(generator_only: dict, full: dict, dataset: dict) -> dict:
    by_id = {s["id"]: s for s in dataset["scenarios"]}

    def compute_condition(runs: dict) -> dict[str, dict]:
        pred_held = defaultdict(list)
        for sid, run in runs.items():
            li = _last_iter(run)
            if not li:
                continue
            
            core_id = sid.split("_seed")[0]
            gt_meta = by_id.get(core_id, {})
            missing = set(li.get("missing_elements", []))

            # FIXED: Set explicit dictionary evaluation logic safely checking keys instead of using hasattr()
            applicable = {
                "weather_correct":           True,
                "fog_density_violation":     "fog_density" in gt_meta or "fog" in gt_meta.get("expected_weather", "").lower(),
                "pedestrian_present":        gt_meta.get("pedestrian", False) or gt_meta.get("ground_truth_predicates", {}).get("pedestrian_present", False),
                "sightline_occlusion_error": gt_meta.get("blocking_actor", False) or "occlusion" in gt_meta.get("stress_category", "").lower(),
                "vehicles_present":          gt_meta.get("num_vehicles", 0) > 0 or gt_meta.get("ground_truth_predicates", {}).get("vehicles_present", True),
                "event_observable":          True
            }

            for pred, app in applicable.items():
                if app:
                    pred_held[pred].append(pred not in missing)

        return {
            pred: {
                "category":     PREDICATE_TO_CATEGORY[pred],
                "n_applicable": len(pred_held[pred]),
                "pass_rate":    round(statistics.fmean(pred_held[pred]), 3) if pred_held[pred] else float("nan")
            } for pred in ALL_PREDICATES
        }

    return {"generator_only": compute_condition(generator_only), "full": compute_condition(full)}


def table6_stress_variance(low_stress_runs: dict, hard_stress_runs: dict) -> dict:
    """
    REPURPOSED FOR SPLIT DATASET: Measures performance differences and variance 
    between the 25 Low-Stress scenarios and 10 Hard-Stress scenarios.
    """
    low_scores = []
    hard_scores = []

    for run in low_stress_runs.values():
        li = _last_iter(run)
        if li and "verification_score" in li:
            low_scores.append(li["verification_score"])

    for run in hard_stress_runs.values():
        li = _last_iter(run)
        if li and "verification_score" in li:
            hard_scores.append(li["verification_score"])

    return {
        "applicable": True,
        "low_stress_scenarios": {
            "n": len(low_scores),
            "mean": round(statistics.fmean(low_scores), 4) if low_scores else 0.0,
            "std": round(statistics.pstdev(low_scores), 4) if len(low_scores) > 1 else 0.0
        },
        "hard_stress_scenarios": {
            "n": len(hard_scores),
            "mean": round(statistics.fmean(hard_scores), 4) if hard_scores else 0.0,
            "std": round(statistics.pstdev(hard_scores), 4) if len(hard_scores) > 1 else 0.0
        },
        "performance_gap_mean": round(statistics.fmean(low_scores) - statistics.fmean(hard_scores), 4) if low_scores and hard_scores else 0.0
    }


def table7_latency_breakdown(runs: dict) -> dict:
    if not runs:
        return {"n_scenarios": 0, "avg_generator_s": 0.0, "avg_engine_s": 0.0, "avg_verifier_s": 0.0, "avg_reflector_s": 0.0, "avg_wall_time_s": 0.0}
        
    gens, engs, vers, refs, walls = [], [], [], [], []
    for run in runs.values():
        gens.append(run.get("total_generator_s", 0.0))
        engs.append(run.get("total_engine_s", 0.0))
        vers.append(run.get("total_verifier_s", 0.0))
        refs.append(run.get("total_reflector_s", 0.0))
        walls.append(run.get("total_wall_time_s", 0.0))
        
    return {
        "n_scenarios":      len(runs),
        "avg_generator_s":  round(statistics.fmean(gens), 2),
        "avg_engine_s":     round(statistics.fmean(engs), 2),
        "avg_verifier_s":   round(statistics.fmean(vers), 2),
        "avg_reflector_s":  round(statistics.fmean(refs), 2),
        "avg_wall_time_s":  round(statistics.fmean(walls), 2),
    }


def table8_stress_analysis(stress_baseline: dict, stress_gen_only: dict, stress_full: dict, stress_dataset: dict) -> list[list]:
    by_id = {s["id"]: s for s in stress_dataset["scenarios"]}
    
    def compute_slice(runs: dict) -> dict[str, list[float]]:
        scores = defaultdict(list)
        for sid, run in runs.items():
            meta = by_id.get(sid, {})
            cat = meta.get("stress_category", "unknown")
            li = _last_iter(run) if "iterations" in run else run
            
            val = li.get("verification_score") if isinstance(li, dict) else None
            if val is not None:
                scores[cat].append(val)
        return {k: round(statistics.fmean(v), 3) for k, v in scores.items() if v}

    b_scores = compute_slice(stress_baseline)
    go_scores = compute_slice(stress_gen_only)
    f_scores = compute_slice(stress_full)
    
    all_cats = sorted(list(set(list(b_scores.keys()) + list(go_scores.keys()) + list(f_scores.keys()))))
    
    rows = [["stress_category", "baseline_score", "generator_only_score", "full_closed_loop_score"]]
    for c in all_cats:
        rows.append([
            c, 
            b_scores.get(c, float("nan")), 
            go_scores.get(c, float("nan")), 
            f_scores.get(c, float("nan"))
        ])
    return rows


# ---- Visualizations ----
def figure1_reflection_convergence(full: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for sid, run in sorted(full.items()):
        iters = run.get("iterations", [])
        if not iters or iters[0].get("verification_passed", False):
            continue
        scores = [it["verification_score"] for it in iters if "verification_score" in it]
        if scores:
            ax.plot(list(range(len(scores))), scores, marker="o", linewidth=1.5, markersize=4, label=sid)
            plotted += 1
        if plotted >= 8:
            break
    ax.set_xlabel("Reflection Iteration", fontsize=10)
    ax.set_ylabel("Predicate Agreement Score", fontsize=10)
    ax.set_title("Figure 1: Reflection Convergence Loop Trends", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)
    if plotted > 0:
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def figure2_failure_distribution(taxonomy: dict, out_path: Path) -> None:
    cats = ALL_CATEGORIES
    counts = [taxonomy.get(c, {}).get("first_pass_count", 0) for c in cats]
    pcts = [taxonomy.get(c, {}).get("first_pass_pct", 0) for c in cats]
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(cats, counts, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"], edgecolor="black", alpha=0.85)
    
    for bar, pct in zip(bars, pcts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, height + 0.1, f"{pct:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)
        
    ax.set_ylabel("Failure Frequency Count", fontsize=10)
    ax.set_title("Figure 2: Initial Phase Failure Distribution", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def figure3_recovery_by_failure_type(recovery: dict, out_path: Path) -> None:
    cats = ALL_CATEGORIES
    rates = [recovery.get(c, {}).get("recovery_rate", 0.0) for c in cats]
    clean_rates = [0.0 if math.isnan(r) else r for r in rates]
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(cats, clean_rates, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"], edgecolor="black", alpha=0.85)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, height + 0.02, f"{height*100:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)
        
    ax.set_ylabel("Closed-Loop Recovery Rate", fontsize=10)
    ax.set_title("Figure 3: Semantic Recovery Rate by Category", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def figure4_per_predicate_alignment(t5: dict, out_path: Path) -> None:
    preds = ALL_PREDICATES
    
    label_mappings = {"weather_correct": "Weather", "fog_density_violation": "Fog Stress", "pedestrian_present": "Actor Baseline", "sightline_occlusion_error": "Occlusion Stress", "vehicles_present": "Environment", "event_observable": "Behavior"}
    display_labels = [label_mappings[p] for p in preds]
    
    go_rates = [t5["generator_only"].get(p, {}).get("pass_rate", 0.0) or 0.0 for p in preds]
    full_rates = [t5["full"].get(p, {}).get("pass_rate", 0.0) or 0.0 for p in preds]
    
    x = np.arange(len(preds))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, go_rates, width, label="Generator-only", color="#AEC6E8", edgecolor="black", alpha=0.9)
    ax.bar(x + width/2, full_rates, width, label="Full (w/ reflection)", color="#1F77B4", edgecolor="black", alpha=0.9)
    
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=8, rotation=15)
    ax.set_ylabel("Predicate Alignment Pass Rate", fontsize=10)
    ax.set_title("Figure 4: Comparative Per-Predicate Alignment Optimization", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _write_csv(path: Path, rows: list[list]) -> None:
    try:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    except Exception as e:
        logger.error("Failed to write CSV file %s: %s", path, e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline",        default="outputs/baseline/baseline_results.json")
    parser.add_argument("--generator-only",  default="outputs/generator_only")
    parser.add_argument("--full",            default="outputs/full")
    parser.add_argument("--dataset",         default="scenarios/dataset.json")
    parser.add_argument("--stress-baseline", default="outputs/stress_baseline/baseline_results.json")
    parser.add_argument("--stress-gen-only", default="outputs/stress_generator_only")
    parser.add_argument("--stress-full",     default="outputs/stress_full")
    parser.add_argument("--stress-dataset",  default="scenarios/stress_dataset.json")
    parser.add_argument("--out",             default="eval/results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = json.loads(Path(args.dataset).read_text())
    baseline = _load_baseline(args.baseline)
    generator_only = _load_pipeline_runs(args.generator_only)
    full = _load_pipeline_runs(args.full)

    logger.info("Loaded Main Dataset: baseline=%d, generator_only=%d, full=%d scenarios", len(baseline), len(generator_only), len(full))

    # ---- Table 1 Execution Rate ----
    t1 = table1_success_rate(baseline, generator_only, full)
    logger.info("Table 1 (success rate): %s", t1)
    _write_csv(out_dir / "table1_success_rate.csv", [["condition", "success_rate"]] + [[k, f"{v:.3f}"] for k, v in t1.items()])

    # ---- Table 2 Semantic Alignment Mean Scores ----
    t2 = table2_semantic_alignment(generator_only, full, dataset)
    logger.info("Table 2 overall: %s", t2["overall"])
    
    t2_rows = [["dimension", "group", "generator_only", "full"]]
    t2_rows.append(["overall", "all", t2["overall"]["generator_only"], t2["overall"]["full"]])
    for slice_name in ["by_difficulty", "by_weather", "by_event"]:
        for group_k in t2[slice_name]["generator_only"]:
            go_v = t2[slice_name]["generator_only"][group_k]
            f_v = t2[slice_name]["full"].get(group_k, float("nan"))
            t2_rows.append([slice_name, group_k, go_v, f_v])
    _write_csv(out_dir / "table2_semantic_alignment.csv", t2_rows)

    # ---- Table 3 & 4 Failure Recovery Logic ----
    t3, t4 = compute_true_failure_and_recovery(full)
    logger.info("Table 3 (failure taxonomy): %s", {c: t3[c] for c in ALL_CATEGORIES if c in t3})
    logger.info("Table 4 (recovery by category): %s", {c: t4[c] for c in ALL_CATEGORIES if c in t4})

    t3_rows = [["failure_category", "first_pass_count", "first_pass_pct", "final_count", "final_pct"]]
    for cat in ALL_CATEGORIES:
        t3_rows.append([cat, t3[cat]["first_pass_count"], t3[cat]["first_pass_pct"], t3[cat]["final_count"], t3[cat]["final_pct"]])
    _write_csv(out_dir / "table3_failure_taxonomy.csv", t3_rows)

    t4_rows = [["failure_category", "first_pass_failures", "recovered", "recovery_rate"]]
    for cat in ALL_CATEGORIES:
        t4_rows.append([cat, t4[cat]["first_pass_failures"], t4[cat]["recovered"], t4[cat]["recovery_rate"]])
    _write_csv(out_dir / "table4_reflection_recovery_by_category.csv", t4_rows)

    # ---- Table 5 Per-Predicate Denominator Guard ----
    t5 = table5_per_predicate_alignment(generator_only, full, dataset)

    # ---- New Table 8 Evaluation Path (Stress Test Cross Breakdown) ----
    stress_dataset_path = Path(args.stress_dataset)
    stress_full = {}
    if stress_dataset_path.exists():
        stress_dataset = json.loads(stress_dataset_path.read_text())
        s_baseline = _load_baseline(args.stress_baseline)
        s_gen_only = _load_pipeline_runs(args.stress_gen_only)
        stress_full = _load_pipeline_runs(args.stress_full)
        
        logger.info("Processing Grounded Stress Dataset Slices...")
        t8_rows = table8_stress_analysis(s_baseline, s_gen_only, stress_full, stress_dataset)
        _write_csv(out_dir / "table8_stress_category_breakdown.csv", t8_rows)

    # ---- Table 6 Stress Split Variance Check ----
    # Compares the 25 standard scenarios (full) vs the 10 hard stress scenarios (stress_full)
    t6 = table6_stress_variance(full, stress_full)
    
    t6_rows = [
        ["dataset_split", "n_scenarios", "mean_score", "std_dev"],
        ["low_stress_scenarios", t6["low_stress_scenarios"]["n"], t6["low_stress_scenarios"]["mean"], t6["low_stress_scenarios"]["std"]],
        ["hard_stress_scenarios", t6["hard_stress_scenarios"]["n"], t6["hard_stress_scenarios"]["mean"], t6["hard_stress_scenarios"]["std"]],
        ["performance_gap_mean", "-", t6["performance_gap_mean"], "-"]
    ]
    _write_csv(out_dir / "table6_stress_split_variance.csv", t6_rows)

    # ---- Table 7 Latency Breakdown Metrics ----
    t7 = table7_latency_breakdown(full)
    t7_rows = [
        ["metric", "seconds"],
        ["avg_generator_s", t7["avg_generator_s"]],
        ["avg_engine_s", t7["avg_engine_s"]],
        ["avg_verifier_s", t7["avg_verifier_s"]],
        ["avg_reflector_s", t7["avg_reflector_s"]],
        ["avg_wall_time_s", t7["avg_wall_time_s"]],
    ]
    _write_csv(out_dir / "table7_system_latency.csv", t7_rows)

    # ---- Figures Sequence ----
    if full:
        figure1_reflection_convergence(full, out_dir / "figure1_reflection_convergence.pdf")
        figure2_failure_distribution(t3, out_dir / "figure2_failure_distribution.pdf")
        figure3_recovery_by_failure_type(t4, out_dir / "figure3_recovery_by_failure_type.pdf")
        figure4_per_predicate_alignment(t5, out_dir / "figure4_per_predicate_alignment.pdf")

    # ---- Summary JSON Sync ----
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
        "table6_stress_split_variance":       t6,
        "table7_system_latency":              t7
    }
    (out_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2))
    logger.info("All metrics successfully synchronized to %s/", out_dir)


if __name__ == "__main__":
    main()