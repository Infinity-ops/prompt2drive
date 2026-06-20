# Prompt-to-Drive

**A verification-aware, closed-loop framework for AI-driven scenario synthesis in autonomous driving simulation.**

Prompt-to-Drive converts natural-language driving scenario descriptions into executable CARLA simulations, automatically verifies whether the resulting simulation semantically matches the original intent using a vision-language model, and iteratively refines the scenario when it doesn't — closing the loop between generative AI and trustworthy simulation-driven validation.

This repository (`prompt2drive`) is the reference implementation accompanying the paper:

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

The Generator and Verifier sit behind a pluggable backend interface, supporting both **OpenAI** (proprietary, GPT-4o) and **Hugging Face** (open-weight models via the Inference API) — see [Backends and cost](#backends-and-cost) below.

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
├── backends/
│   └── llm_backend.py      # Pluggable OpenAI / Hugging Face backend abstraction
├── engine/                 # Deterministic CARLA execution layer
│   ├── scenario_spec.py    #   ScenarioSpec schema (the JSON intermediate representation)
│   └── carla_runner.py     #   ScenarioSpec -> CARLA API calls, frame + log capture
├── generator/
│   └── agent.py            # Agent 1: prompt -> ScenarioSpec (+ free-form baseline generator)
├── verifier/
│   └── agent.py            # Agent 2: hybrid VLM + log semantic verification
├── reflector/
│   └── agent.py            # Agent 3: targeted root-cause refinement
├── orchestrator/
│   └── pipeline.py         # Closed-loop orchestration (Algorithm 1 in the paper)
├── scenarios/
│   └── dataset.json        # 20-scenario evaluation set with ground-truth labels
├── eval/
│   ├── run_baseline.py             # Experiment comparator: free-form code gen, no schema/loop
│   ├── run_full_evaluation.py      # Runs generator_only and full conditions, single backend
│   ├── run_backend_comparison.py   # Runs full conditions across BOTH backends side-by-side
│   └── compute_metrics.py          # Produces Table 1, Table 2, Figure 3, Figure 4
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
3. **API credentials for at least one backend** — see the next section for what you actually need and what it costs.

---

## Backends and cost

Prompt-to-Drive does not require OpenAI specifically. The Generator and Verifier are built against a small backend interface (`backends/llm_backend.py`), with two implementations shipped:

| | OpenAI backend | Hugging Face backend |
|---|---|---|
| **Setup** | `OPENAI_API_KEY` (pay-as-you-go) | `HUGGINGFACE_API_TOKEN` (free tier works for many models; a **PRO account raises rate limits and unlocks more hosted endpoints**, which is relevant since you already have one) |
| **Generator model (default)** | `gpt-4o-2024-08-06` | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| **Verifier model (default)** | `gpt-4o` (vision) | `Qwen/Qwen2-VL-7B-Instruct` |
| **Structured output** | Native (`response_format`) — very reliable JSON | Prompted JSON + a repair parser (open-weight models occasionally wrap JSON in prose) |
| **Cost** | Per-token, billed by OpenAI. For the full 20-scenario eval (3 iterations max, ~3 images per verification call) budget roughly **$3–6 USD total**. | Covered by your existing HF Pro subscription / free inference quota for most of the default models above. No separate API key purchase needed. |

**Do you need to pay for OPENAI_API_KEY?** Yes — OpenAI's API is billed separately from ChatGPT Plus/Pro and has no meaningful free tier beyond a small initial trial credit for new accounts. You will need to add a payment method and the usage above (a few dollars for the full evaluation) will be billed to it.

**Should you use OpenAI, Hugging Face, or both for the paper?** Both — and the repository is built to make that easy:

- Using **only Hugging Face** (since you already have HF Pro) is a completely valid, zero-marginal-cost way to run every experiment in this repo. Set `P2D_TEXT_BACKEND=huggingface` and `P2D_VISION_BACKEND=huggingface` in your `.env` and skip OpenAI entirely.
- Using **both** and reporting them side-by-side (`eval/run_backend_comparison.py`) is the stronger choice for the paper specifically: it converts "we built a pipeline around GPT-4o" into "we built a backend-agnostic architecture, and report consistent results across a proprietary and an open-weight stack" — a genuine generalizability claim that CAIESA reviewers (an embedded-systems / CPS audience, not an LLM-leaderboard audience) tend to value, since portability across model providers and licensing regimes is itself a practical deployment concern for safety-critical CPS tooling.
- If you only have budget/time for one, **Hugging Face alone is sufficient** to support every claim in the paper except the cross-backend robustness one. Drop that one paragraph from the Discussion section if you skip the comparison run.

To select a backend, set environment variables in `.env` (see `.env.example`) or pass `--backend openai` / `--backend huggingface` on any eval script or `run_scenario.py`.

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
# edit .env: set P2D_TEXT_BACKEND / P2D_VISION_BACKEND and the matching key(s)
```

### Sanity check (no CARLA server needed)

```bash
pytest tests/ -v
```

Expected output: 13 tests pass. These exercise the `ScenarioSpec` schema validation, the Reflector's correction-lookup logic, the Verifier's score arithmetic, and the backend abstraction's JSON-repair parser — all pure Python, no network calls, no CARLA connection. If this fails, something is wrong with your Python environment, not your CARLA setup or your API keys.

---

## Running a single scenario

Start your CARLA server first, in a separate terminal:

```bash
# e.g. ./CarlaUE4.sh -RenderOffScreen   (headless, recommended for speed)
```

Then, from the repo root with your venv active:

```bash
python run_scenario.py \
  --prompt "A pedestrian suddenly crosses near a traffic signal in rainy weather." \
  --scenario-id demo01 \
  --mode full \
  --backend huggingface
```

(Omit `--backend` to use whatever `P2D_TEXT_BACKEND` / `P2D_VISION_BACKEND` are set to in your `.env`, which defaults to `openai` if unset.)

### What to expect

- Console output streams the Generator's chosen `ScenarioSpec`, the simulation running in CARLA (you'll see it in the CARLA window if not headless), and the Verifier's score.
- A typical run takes **30–90 seconds per iteration** (dominated by simulation duration + one or two backend calls), and up to 3 iterations if reflection is triggered.
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

This is a four-step pipeline; run them in order. Add `--backend huggingface` to any command to run that step under the Hugging Face backend instead of the default.

### Step 1 — Baseline (free-form code generation, no schema, no loop)

```bash
python -m eval.run_baseline --dataset scenarios/dataset.json --out outputs/baseline
```

**What this does:** for each of the 20 scenarios, asks the backend to write raw CARLA Python directly (the way most people would first try this), executes it in a subprocess against your running CARLA server, and records whether it ran without crashing.

**Expect:** this step is the fastest (no simulation loop, just code-gen + execution attempt) but has the lowest success rate — this is intentional; it's the naive comparator. Typical run time: 3–8 minutes for 20 scenarios.

### Step 2 — Generator-only condition (schema-constrained, no reflection)

```bash
python -m eval.run_full_evaluation \
  --dataset scenarios/dataset.json \
  --mode generator_only \
  --out outputs/generator_only
```

**Expect:** noticeably higher executable success than the baseline (the JSON schema alone removes most code-level hallucination), but semantic alignment scores will show clear room for improvement on harder scenarios — this is the gap the Reflector exists to close. Run time: roughly 15–25 minutes total.

### Step 3 — Full closed-loop Prompt-to-Drive

```bash
python -m eval.run_full_evaluation \
  --dataset scenarios/dataset.json \
  --mode full \
  --out outputs/full \
  --max-iterations 3
```

**Expect:** this is the slowest step since failing scenarios re-run up to 3 times. Most scenarios converge within 1–2 iterations. Run time: typically 30–60 minutes for 20 scenarios.

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

### Optional Step 5 — Side-by-side backend comparison

If you want the cross-backend robustness result discussed above:

```bash
python -m eval.run_backend_comparison \
  --dataset scenarios/dataset.json \
  --out outputs/backend_comparison \
  --max-iterations 3
```

This runs the full closed-loop pipeline once under OpenAI and once under Hugging Face on the identical 20-scenario set and prints a side-by-side table:

```
======================================================================
Backend        Success%    Sem. Align.   Pass%     Mean Iter
----------------------------------------------------------------------
openai         92.0        0.847         85.0      1.65
huggingface    84.0        0.781         75.0      2.10
======================================================================
```

(Numbers above are illustrative placeholders — your actual run will populate `outputs/backend_comparison/backend_comparison_summary.json` with real results.) Run only one backend with `--backends huggingface` if you'd rather not spend OpenAI credits.

---

## Extending the framework

- **Add a new backend:** implement `TextBackend` and/or `VisionBackend` from `backends/llm_backend.py` and register it in `get_backend()`. The Generator, Verifier, orchestrator, and all eval scripts work unchanged against any conforming backend.
- **Add new event types or weather presets:** extend the `Literal` types in `engine/scenario_spec.py` and add a matching entry to `WEATHER_PRESETS` in `engine/carla_runner.py`, plus a correction rule in `reflector/agent.py`'s `ELEMENT_CORRECTIONS` table.
- **Add new scenarios to the evaluation set:** append entries to `scenarios/dataset.json` following the existing schema — `id`, `prompt`, `difficulty`, `expected_weather`, `expected_actors`, `expected_event`.
- **Tune the hybrid verification weights:** `SCORE_WEIGHTS` in `verifier/agent.py` (defaults: VLM 0.5 / log 0.3 / execution 0.2).
- **Change the pass threshold:** `P2D_VERIFICATION_THRESHOLD` in `.env`, or pass `threshold=` to `Verifier(...)`.

---

## Known limitations

- The Verifier's accuracy is bounded by the underlying VLM's ability to interpret synthetic, rendered scenes — performance on photorealistic real-world imagery may differ, and open-weight VLMs may lag GPT-4o on this specific task (this is exactly what `run_backend_comparison.py` measures).
- The `ScenarioSpec` schema intentionally trades expressiveness for safety: scenarios outside the enumerated `event_type` / `weather` / `town` values cannot be represented without extending the schema.
- The Reflector cannot resolve scenarios that are fundamentally inexpressible within the current schema; these are marked `iteration_limit_reached` and excluded from semantic-alignment metrics but retained in executable-success metrics.
- Open-weight models accessed via the Hugging Face Inference API are more prone to wrapping JSON output in prose than OpenAI's structured outputs; `backends/llm_backend.py` includes a repair parser for this, but parse failures are still possible and are logged + scored as a verification failure rather than silently retried.
- All Generator and Verifier calls are cached locally (`shelve`, keyed by backend name + content hash) to avoid redundant spend/inference on repeated runs.

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
