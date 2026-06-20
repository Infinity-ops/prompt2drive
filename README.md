# Prompt-to-Drive

**A verification-aware, closed-loop framework for AI-driven scenario synthesis in autonomous driving simulation.**

Prompt-to-Drive converts natural-language driving scenario descriptions into executable CARLA simulations, automatically verifies whether the resulting simulation semantically matches the original intent using a vision-language model, and iteratively refines the scenario when it doesn't — closing the loop between generative AI and trustworthy simulation-driven validation.

This repository is the reference implementation accompanying the paper:

> **Prompt-to-Drive: Closed-Loop AI-Driven Scenario Synthesis and Semantic Verification for Trustworthy Autonomous Driving Simulation**
> Submitted to CAIESA — ESWEEK Workshop on Challenges in AI-Based Embedded Systems Applications, Barcelona, 2026.

```
┌─────────────────┐     ┌───────────┐     ┌─────────────┐     ┌──────────────────┐
│ "A pedestrian    │────▶│ Generator │────▶│ ScenarioSpec │────▶│ CARLA Simulator  │
│  crosses near a  │     │  (LLM)    │     │   (JSON)     │     │  (deterministic) │
│  signal in rain" │     └───────────┘     └─────────────┘     └────────┬─────────┘
└─────────────────┘            ▲                                       │
                                │                                       ▼
                          ┌─────┴──────┐                       ┌───────────────┐
                          │ Reflector  │◀──────────────────────│   Verifier     │
                          │ (targeted  │   refinement_context  │ (VLM + log     │
                          │ correction)│   on failure           │  hybrid score) │
                          └────────────┘                       └───────────────┘
```

---

## Why this exists

Direct "ask an LLM to write CARLA code" pipelines hallucinate: non-compilable scripts, wrong weather, missing actors, events that never actually occur in the rendered simulation. Existing scenario-generation tools execute scenarios but rarely check whether what *ran* matches what was *asked for*. Prompt-to-Drive treats that check — semantic alignment between natural language intent and simulation output — as a first-class, measurable engineering requirement.

Three agents make this work:

| Agent | Role | Key idea |
|---|---|---|
| **Generator** | NL prompt → structured `ScenarioSpec` JSON | Constrained schema, not free-form code — bounds the hallucination surface |
| **Verifier** | Simulation output → semantic alignment score | Hybrid: VLM visual reasoning + deterministic state-log validation |
| **Reflector** | Failed verification → targeted refinement instructions | Root-cause mapping, not blind retry — corrects only the fields that failed |

---

## Repository layout

```
prompt2drive/
├── engine/                 # Deterministic CARLA execution layer
│   ├── scenario_spec.py    #   ScenarioSpec schema (the JSON intermediate representation)
│   └── carla_runner.py     #   ScenarioSpec -> CARLA API calls, frame + log capture
├── generator/
│   └── agent.py            # Agent 1: LLM prompt -> ScenarioSpec (+ free-form baseline generator)
├── verifier/
│   └── agent.py            # Agent 2: hybrid VLM + log semantic verification
├── reflector/
│   └── agent.py            # Agent 3: targeted root-cause refinement
├── orchestrator/
│   └── pipeline.py         # Closed-loop orchestration (Algorithm 1 in the paper)
├── scenarios/
│   └── dataset.json        # 20-scenario evaluation set with ground-truth labels
├── eval/
│   ├── run_baseline.py         # Experiment comparator: free-form code gen, no schema/loop
│   ├── run_full_evaluation.py  # Runs generator_only and full conditions
│   └── compute_metrics.py      # Produces Table 1, Table 2, Figure 3, Figure 4
├── tests/
│   └── test_core.py        # Unit tests (no CARLA / no API key required)
├── run_scenario.py         # Single-scenario CLI entry point
├── requirements.txt
├── .env.example
└── README.md
```

---

## Prerequisites

You said you already have CARLA installed — good, that's the only heavy dependency. You will additionally need:

1. **Python 3.10+**
2. **The CARLA Python API wheel matching your CARLA version**, installed into your virtualenv:
   ```bash
   pip install <CARLA_ROOT>/PythonAPI/carla/dist/carla-<version>-cp310-cp310-<platform>.whl
   ```
   (This repo never bundles or pins a `carla` package in `requirements.txt` because the correct wheel depends on your local CARLA build.)
3. **An OpenAI API key** (used by the Generator's structured outputs and the Verifier's vision calls). Other LLM/VLM providers can be swapped in by replacing the `OpenAI()` client in `generator/agent.py` and `verifier/agent.py` — the rest of the system is provider-agnostic.

---

## Setup

```bash
git clone https://github.com/<your-username>/prompt2drive.git
cd prompt2drive

python3.10 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install <CARLA_ROOT>/PythonAPI/carla/dist/carla-<version>-...whl

cp .env.example .env
# edit .env and paste your OPENAI_API_KEY
```

### Sanity check (no CARLA server needed)

```bash
pytest tests/ -v
```

Expected output: 7 tests pass. These exercise the `ScenarioSpec` schema validation, the Reflector's correction-lookup logic, and the Verifier's score arithmetic — all pure Python, no network calls, no CARLA connection. If this fails, something is wrong with your Python environment, not your CARLA setup.

---

## Running a single scenario

Start your CARLA server first, in a separate terminal (you already know how to do this):

```bash
# e.g. ./CarlaUE4.sh -RenderOffScreen   (headless, recommended for speed)
```

Then, from the repo root with your venv active:

```bash
python run_scenario.py \
  --prompt "A pedestrian suddenly crosses near a traffic signal in rainy weather." \
  --scenario-id demo01 \
  --mode full
```

### What to expect

- Console output streams the Generator's chosen `ScenarioSpec`, the simulation running in CARLA (you'll see it in the CARLA window if not headless), and the Verifier's score.
- A typical run takes **30–90 seconds per iteration** (dominated by simulation duration + one or two API calls), and up to 3 iterations if reflection is triggered.
- Final printed summary looks like:

  ```
  ============================================================
  Final status:   passed
  Iterations:     1
  Final score:    0.86
  Passed:         True
  Missing elems:  []
  Full record:    outputs/demo01/pipeline_run.json
  ============================================================
  ```

- If verification fails on the first attempt, you'll see a second iteration begin automatically with a `[REFLECTION]` block visible in the logs (`logging.INFO` level) explaining exactly what's being corrected and why.
- All artifacts (frames, state logs, the full `pipeline_run.json` trace) are saved under `outputs/<scenario_id>/`.

Try `--mode generator_only` to see the same prompt run through the schema-constrained Generator and Engine **without** the verification/reflection loop — useful for understanding what the closed loop actually buys you.

---

## Reproducing the paper's experiments

This is a three-step pipeline; run them in order.

### Step 1 — Baseline (free-form code generation, no schema, no loop)

```bash
python -m eval.run_baseline --dataset scenarios/dataset.json --out outputs/baseline
```

**What this does:** for each of the 20 scenarios, asks the LLM to write raw CARLA Python directly (the way most people would first try this), executes it in a subprocess against your running CARLA server, and records whether it ran without crashing.

**Expect:** this step is the fastest (no simulation loop, just code-gen + execution attempt) but has the lowest success rate — this is intentional; it's the naive comparator. Typical run time: 3–8 minutes for 20 scenarios.

### Step 2 — Generator-only condition (schema-constrained, no reflection)

```bash
python -m eval.run_full_evaluation \
  --dataset scenarios/dataset.json \
  --mode generator_only \
  --out outputs/generator_only
```

**Expect:** noticeably higher executable success than the baseline (the JSON schema alone removes most code-level hallucination), but semantic alignment scores will show clear room for improvement on harder scenarios — this is the gap the Reflector exists to close. Run time: roughly (20 scenarios × ~15–25s simulated duration each) plus one Generator + one Verifier API call per scenario — typically 15–25 minutes total.

### Step 3 — Full closed-loop Prompt-to-Drive

```bash
python -m eval.run_full_evaluation \
  --dataset scenarios/dataset.json \
  --mode full \
  --out outputs/full \
  --max-iterations 3
```

**Expect:** this is the slowest step since failing scenarios re-run up to 3 times. Most scenarios converge within 1–2 iterations. Run time: typically 30–60 minutes for 20 scenarios, depending on how many require reflection.

### Step 4 — Compute metrics and generate figures

```bash
python -m eval.compute_metrics \
  --baseline outputs/baseline/baseline_results.json \
  --generator-only outputs/generator_only \
  --full outputs/full \
  --dataset scenarios/dataset.json \
  --out eval/results
```

**Output files (all in `eval/results/`):**

| File | Contents |
|---|---|
| `table1_success_rate.csv` | Executable success rate, 3 conditions |
| `table2_semantic_alignment.csv` | Semantic alignment score, overall + broken down by difficulty/weather/event |
| `figure3_reflection_convergence.pdf` | Score-by-iteration line chart for initially-failing scenarios — the paper's key figure |
| `figure4_difficulty_breakdown.pdf` | Bar chart comparing generator-only vs. full system across difficulty tiers |
| `summary.json` | All computed numbers in one machine-readable file |

If you only have `outputs/full` (skipped baseline/generator-only), the script will skip the unavailable comparisons gracefully and still produce what it can — check the console warnings for what was skipped.

---

## Extending the framework

- **Swap the LLM/VLM provider:** replace the `OpenAI()` client instantiation in `generator/agent.py` and `verifier/agent.py`. The rest of the pipeline (schema, engine, orchestrator) is provider-agnostic.
- **Add new event types or weather presets:** extend the `Literal` types in `engine/scenario_spec.py` and add a matching entry to `WEATHER_PRESETS` in `engine/carla_runner.py`, plus a correction rule in `reflector/agent.py`'s `ELEMENT_CORRECTIONS` table.
- **Add new scenarios to the evaluation set:** append entries to `scenarios/dataset.json` following the existing schema — `id`, `prompt`, `difficulty`, `expected_weather`, `expected_actors`, `expected_event`.
- **Tune the hybrid verification weights:** `SCORE_WEIGHTS` in `verifier/agent.py` (defaults: VLM 0.5 / log 0.3 / execution 0.2). If you find your VLM is less reliable on your scenes, lower its weight and raise the log-channel weight.
- **Change the pass threshold:** `DEFAULT_THRESHOLD = 0.72` in `verifier/agent.py`, or pass `threshold=` to `Verifier(...)`.

---

## Known limitations

- The Verifier's accuracy is bounded by the underlying VLM's ability to interpret synthetic, rendered scenes — performance on photorealistic real-world imagery may differ.
- The `ScenarioSpec` schema intentionally trades expressiveness for safety: scenarios outside the enumerated `event_type` / `weather` / `town` values cannot be represented without extending the schema.
- The Reflector cannot resolve scenarios that are fundamentally inexpressible within the current schema (e.g., maneuvers CARLA's NPC behavior trees don't support); these are marked `iteration_limit_reached` and excluded from semantic-alignment metrics but retained in executable-success metrics.
- API costs: each Verifier call sends up to 3 images to a vision-capable model. All Generator and Verifier calls are cached locally (`shelve`, keyed by content hash) to avoid redundant spend on repeated runs.

---

## Citing this work

```bibtex
@inproceedings{promptodrive2026,
  title     = {Prompt-to-Drive: Closed-Loop AI-Driven Scenario Synthesis and Semantic Verification for Trustworthy Autonomous Driving Simulation},
  booktitle = {Proceedings of the ESWEEK Workshop on Challenges in AI-Based Embedded Systems Applications (CAIESA)},
  series    = {IFIP Advances in Information and Communication Technology},
  publisher = {Springer},
  year      = {2026}
}
```

---

## License

MIT — see [LICENSE](LICENSE). Contributions welcome via pull request.
