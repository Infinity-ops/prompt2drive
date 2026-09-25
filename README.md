# Prompt-to-Drive

**A verification-aware, closed-loop framework for AI-driven scenario synthesis in autonomous driving simulation.**

Prompt-to-Drive converts natural-language driving scenario descriptions into executable CARLA simulations, automatically verifies whether the resulting simulation semantically matches the original intent using a vision-language model, and iteratively refines the scenario when it doesn't — closing the loop between generative AI and trustworthy simulation-driven validation.

This repository (`prompt2drive`) is the reference implementation accompanying the paper:

> **Prompt-to-Drive: Closed-Loop Semantic Verification for Trustworthy AI-Synthesized Autonomous Driving Scenarios**


<details>
<summary><b>Abstract</b></summary>

Simulation-driven validation is central to the development of software-defined vehicles and autonomous driving systems, yet authoring diverse, semantically precise test scenarios at scale remains a manual bottleneck. Large language models can synthesize CARLA scenarios from natural-language descriptions, but existing approaches treat generation and execution as an open loop: the resulting simulation is rarely checked against the intent that produced it, and execution-success metrics do not capture whether the scenario is semantically correct. This creates a verification gap that undermines simulation as evidence in a safety case. We present Prompt-to-Drive, a closed-loop framework comprising a Generator agent that maps prompts to a constrained, schema-validated scenario representation; a deterministic CARLA execution engine; a Verifier agent that combines vision-language reasoning over rendered frames with deterministic simulation-log checks into a hybrid semantic alignment score; and a Reflector agent that performs targeted, root-cause-specific regeneration when verification fails. The framework is backend-agnostic, supporting both proprietary and open-weight model providers, and includes an interactive interface exposing per-agent pipeline status and alignment scores for inspection. We evaluate on a 20-scenario natural-language benchmark spanning five weather conditions, six driving events, and three difficulty levels against free-form and schema-only baselines. *[Results pending — see Evidence Ledger in the project notes; this placeholder is intentionally left unfilled until real experiments are run against a live CARLA instance.]* We discuss the framework's scope, hardware constraints, and limitations for trustworthy simulation-based CPS validation.

</details>

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
│   ├── carla_runner.py     #   ScenarioSpec -> CARLA API calls, frame + log capture
│   └── hardware_profiles.py#   Resolution/quality/actor-density presets for modest GPUs
├── generator/
│   └── agent.py            # Agent 1: prompt -> ScenarioSpec (+ free-form baseline generator)
├── verifier/
│   └── agent.py            # Agent 2: hybrid VLM + log semantic verification
├── reflector/
│   └── agent.py            # Agent 3: targeted root-cause refinement
├── orchestrator/
│   └── pipeline.py         # Closed-loop orchestration (Algorithm 1 in the paper)
├── ui/
│   └── app.py               # Interactive pygame UI: live agent status + semantic score panel
├── webui/                  # Browser UI (FastAPI + WebSocket): live or replay of recorded runs
├── demo_runs/              # Recorded runs shipped with the public replay demo
├── scenarios/
│   └── dataset.json        # 20-scenario evaluation set with ground-truth labels
├── eval/
│   ├── run_baseline.py             # Experiment comparator: free-form code gen, no schema/loop
│   ├── run_full_evaluation.py      # Runs generator_only and full conditions, single backend
│   ├── run_backend_comparison.py   # Runs full conditions across BOTH backends side-by-side
│   └── compute_metrics.py          # Produces Table 1, Table 2, Figure 3, Figure 4
├── tests/
│   └── test_core.py        # Unit tests (no CARLA / no API key required)
├── docs/
│   └── TROUBLESHOOTING_LOW_VRAM.md  # Segfault diagnosis for low-VRAM GPUs (e.g. Quadro M620)
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

## Hardware profiles (for modest/older GPUs)

If you're running on a laptop or mobile workstation GPU with limited VRAM (e.g., a Quadro M620, 2GB), CARLA's default settings will commonly segfault. Use the `low_vram_2gb` profile, which lowers camera resolution, caps concurrent NPC vehicles, slows the simulation tick rate, and restricts scenarios to CARLA's smallest maps:

```bash
python run_scenario.py --prompt "..." --hardware-profile low_vram_2gb
```

This alone does not fix server-side CARLA crashes — you must also launch the CARLA server itself with matching flags (`-opengl -quality-level=Low -RenderOffScreen`). **See [docs/TROUBLESHOOTING_LOW_VRAM.md](docs/TROUBLESHOOTING_LOW_VRAM.md) for the full diagnosis and fix, including exactly why the Vulkan renderer is the most common segfault cause on this GPU class.** If you run your evaluation under reduced settings, report this explicitly in your paper's Experimental Setup — it's a disclosed constraint, not something to hide, and is itself relevant context for a CPS/embedded-systems audience.

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

## Running on constrained hardware

CARLA's own recommended minimum is a GPU with roughly 6GB VRAM. If you're running on a mobile workstation GPU in the 2GB-VRAM class (e.g. an NVIDIA Quadro M620, GeForce 940MX/950M-class, or similar Maxwell-era mobile chips), you will likely hit out-of-memory errors or unusable frame rates at default settings. This repo includes a low-spec mode to make a first end-to-end pass realistic on that hardware.

**1. Launch CARLA itself in low-spec mode:**

```bash
./CarlaUE4.sh -quality-level=Low -RenderOffScreen -ResX=480 -ResY=270
```

`-RenderOffScreen` skips the display compositor entirely (you don't need to see the CARLA window — the engine captures frames programmatically). `-quality-level=Low` disables most post-processing and shadow/reflection passes, which are the most VRAM-hungry parts of the renderer.

**2. Enable low-spec mode in `.env`:**

```bash
P2D_LOWSPEC_MODE=1
P2D_CAMERA_WIDTH=480
P2D_CAMERA_HEIGHT=270
```

With `P2D_LOWSPEC_MODE=1`, the Generator clamps every scenario to `Town01`/`Town02` (CARLA's smallest maps — far fewer background buildings and draw calls than Town03–05) and caps `num_vehicles` at 8, regardless of what the LLM would otherwise choose. This bounds render load without changing the semantic content of the scenario.

**3. Use the reduced 8-scenario subset for your first pass:**

```bash
python -m eval.run_baseline --dataset scenarios/dataset_lowspec.json --out outputs/baseline
python -m eval.run_full_evaluation --dataset scenarios/dataset_lowspec.json --mode generator_only --out outputs/generator_only
python -m eval.run_full_evaluation --dataset scenarios/dataset_lowspec.json --mode full --out outputs/full
python -m eval.compute_metrics --dataset scenarios/dataset_lowspec.json --baseline outputs/baseline/baseline_results.json --generator-only outputs/generator_only --full outputs/full --out eval/results_lowspec
```

`scenarios/dataset_lowspec.json` is an 8-scenario subset of the full dataset, chosen to avoid the busier Town03/04/05 maps. Once this completes successfully end-to-end, you'll know your hardware/driver setup works, and you can decide whether to attempt the full 20-scenario `dataset.json` (which will take noticeably longer and push your GPU harder) or report results on the 8-scenario subset with that scoping explicitly stated in your paper's experimental setup section — this is an honest, defensible choice given your hardware, not a shortcut to hide.

**If CARLA still crashes or is unusably slow even at Low quality:** your GPU is the bottleneck for the simulation step specifically — the Generator and Verifier API calls don't need a GPU at all. A practical fallback is running CARLA on a free Google Colab GPU runtime, or a few hours of a rented cloud GPU (RunPod, Lambda, Vast.ai), for the simulation/capture step only, while everything else (dataset, agents, metrics, writing) stays on your laptop.

---

## Reproducing the paper's experiments

This is a four-step pipeline; run them in order. Add `--backend huggingface` to any command to run that step under the Hugging Face backend instead of the default.

### Step 1 — Baseline (free-form code generation, no schema, no loop)

```bash
python -m eval.run_baseline --dataset scenarios/dataset.json --out outputs/baseline
```

**What this does:** for each of the 20 scenarios, asks the backend to write raw CARLA Python directly (the way most people would first try this), executes it in a subprocess against your running CARLA server, and records whether it ran without crashing. The baseline's prompt engineering (spawn safety, actor limits, color handling) is adapted from an earlier hand-tuned prototype rather than written as a deliberately weak strawman — a free-form baseline that nobody would actually use in practice would be a less defensible comparator than one reflecting genuine prior effort to make open-loop generation work well.

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
| `table2_semantic_alignment.csv` | Semantic alignment score (formal predicate agreement — see below), overall + broken down by difficulty/weather/event |
| `table3_reflection_recovery.csv` | First-pass pass rate, after-reflection pass rate, and recovery rate (fraction of first-pass failures the Reflector subsequently fixes) — the metric that directly evidences the Reflector does something |
| `table4_seed_variance.csv` | Per-core-scenario mean ± std and an overall 95% CI across seeded executions — only populated if `--full` points at results from the expanded/seeded dataset (see Optional Step 4b below) |
| `figure3_reflection_convergence.pdf` | Score-by-iteration line chart for initially-failing scenarios — the paper's key figure |
| `figure4_difficulty_breakdown.pdf` | Bar chart comparing generator-only vs. full system across difficulty tiers |
| `summary.json` | All computed numbers in one machine-readable file |

**On the semantic alignment metric:** this is `predicate_agreement` — the fraction of applicable, named, boolean predicates (`weather_correct`, `pedestrian_present`, `vehicles_present`, `event_observable`) that held under majority vote across `P2D_VERIFIER_N_SAMPLES` (default 3) independent VLM calls per verification, not a single opaque holistic score from one VLM call. See [docs/SEMANTIC_VERIFICATION_METRIC.md](docs/SEMANTIC_VERIFICATION_METRIC.md) for the full formal definition and worked example.

### Optional Step 4b — Expand the benchmark for statistical power

The 20 handcrafted scenarios alone support only a single point estimate per scenario. To report mean ± std / a confidence interval rather than a bare 20-sample point estimate, expand each core scenario into multiple seeded executions:

```bash
python -m eval.expand_dataset \
  --input scenarios/dataset.json \
  --output scenarios/dataset_expanded.json \
  --seeds-per-scenario 3
```

This produces 60 total executions (20 core scenarios × 3 seeds), where each seed varies the engine's NPC/pedestrian spawn-point shuffle (and, for non-deterministic backends, the Generator's sampling draw) while keeping the same prompt and ground-truth labels. **This is 20 scenario concepts executed 3 times each, not 60 independently authored scenarios — state this distinction explicitly in your paper's Experimental Setup section.** Re-run Steps 1–4 with `--dataset scenarios/dataset_expanded.json` to populate Table 4.

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

## Interactive UI

### Web UI (recommended)

```bash
pip install -r requirements-ui.txt
python -m webui.server --mode live --backend local --hardware-profile low_vram_2gb
# open http://127.0.0.1:8080
```

Type a scenario, press **Generate**, and watch each agent report in turn: **Agent 1 Generator** shows the `ScenarioSpec` it produced, the **CARLA Engine** streams the frames it captures, **Agent 2 Verifier** shows each predicate, the predicate agreement and the hybrid score against the threshold, and **Agent 3 Reflector** shows the exact correction it sends back for the next iteration. Every live run is saved under `outputs/ui/` and can be replayed without CARLA (`--mode replay`), which is how the public demo works. Full guide: [docs/WEB_UI.md](docs/WEB_UI.md).

### Desktop (pygame) UI

```bash
python -m ui.app --backend huggingface --hardware-profile low_vram_2gb
```

A single-window pygame app: type a scenario prompt, press Enter, and watch it move through the four pipeline stages in real time — a status panel shows **Generator → Engine → Verifier → Reflector** with a live indicator per stage (pending / active / done / failed), the live CARLA camera feed streams into the window as the scenario executes, and once verification completes you see the **semantic alignment score as a percentage with a pass/fail checkmark**, plus the specific missing elements if it failed and which iteration the Reflector is now correcting.

All the same flags as `run_scenario.py` apply (`--backend`, `--hardware-profile`, `--carla-host`, `--carla-port`, `--max-iterations`). This is a genuinely useful demo for a paper Q&A or a GitHub README GIF, separate from the eval pipeline used to produce the paper's actual results tables.

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
  title     = {Prompt-to-Drive: Closed-Loop Semantic Verification for Trustworthy AI-Synthesized Autonomous Driving Scenarios},
  booktitle = {Proceedings of the ESWEEK Workshop on Challenges in AI-Based Embedded Systems Applications (CAIESA)},
  series    = {IFIP Advances in Information and Communication Technology},
  publisher = {Springer},
  year      = {2026}
}
```

---

## License

MIT — see [LICENSE](LICENSE). Contributions welcome via pull request.
