# Evaluation

The paper's numbers come from preserved traces (`outputs/<condition>/<scenario_id>/pipeline_run.json`)
processed by one script, `eval/compute_metrics.py`. The per-verifier results it produced are
checked in under `eval/paper_results/llama32/`, `qwen25/` and `internvl25/`.

Each trace also records what actually ran: the ScenarioSpec, the map loaded and the hardware
profile, in `iter_*/state_log.json`. Check those files rather than assuming a setting.

## 0. Setup

```bash
pip install -r requirements.txt
pip install <CARLA_ROOT>/PythonAPI/carla/dist/carla-0.9.16-cp310-*.whl
cp .env.example .env        # OPENAI_API_KEY for the Generator/Reflector, VLLM_* for the Verifier
```

Start CARLA, then a vLLM server for the verifier you want to evaluate. The model name in
`.env` (`VLLM_VISION_MODEL`) must equal the server's `--served-model-name`.

| Verifier | vLLM command (single 24 GB GPU next to CARLA) |
|---|---|
| Qwen2.5-VL-7B-AWQ | `vllm serve <path>/Qwen2.5-VL-7B-Instruct-AWQ --served-model-name Qwen/Qwen2.5-VL-7B-Instruct --max-model-len 4096 --gpu-memory-utilization 0.45 --max-num-seqs 1 --max-num-batched-tokens 4096 --enforce-eager --limit-mm-per-prompt '{"image":3}'` |
| InternVL2.5-8B-AWQ | `vllm serve <path>/InternVL2_5-8B-AWQ --served-model-name InternVL2_5-8B --max-model-len 4096 --gpu-memory-utilization 0.45 --max-num-seqs 1 --max-num-batched-tokens 4096 --trust-remote-code --limit-mm-per-prompt '{"image":3}'` |
| Llama-3.2-11B-Vision (W4A16) | `vllm serve <path>/Llama-3.2-11B-Vision-Instruct-W4A16 --served-model-name ./Llama-3.2-11B-Vision-Instruct --tokenizer <path>/Llama-3.2-11B-Vision-Instruct --max-model-len 4096 --gpu-memory-utilization 0.50 --max-num-seqs 1 --max-num-batched-tokens 4096 --enforce-eager --limit-mm-per-prompt '{"image":3}'` |

If vLLM fails with a FlashInfer sampler error, prefix the command with `VLLM_USE_FLASHINFER_SAMPLER=0`.

## 1. Free-form baseline

```bash
python -m eval.run_baseline --dataset scenarios/dataset.json --out outputs/baseline
python -m eval.run_baseline --dataset scenarios/stress_dataset.json --out outputs/stress_baseline
```

## 2. Generator-only (schema, one verification pass, no reflection)

```bash
python -m eval.run_topology_evaluation --mode generator_only \
  --text-backend openai --vision-backend local \
  --dataset scenarios/dataset.json --out outputs/generator_only
python -m eval.run_topology_evaluation --mode generator_only \
  --text-backend openai --vision-backend local \
  --dataset scenarios/stress_dataset.json --out outputs/stress_generator_only
```

## 3. Full closed loop, once per verifier

```bash
# with the Qwen server running and VLLM_VISION_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
python -m eval.run_topology_evaluation --mode full --max-iterations 3 \
  --text-backend openai --reflector-backend openai --vision-backend local \
  --dataset scenarios/dataset.json --out outputs/full_cloud_reflector_run_QwenVL25_AWQ
python -m eval.run_topology_evaluation --mode full --max-iterations 3 \
  --text-backend openai --reflector-backend openai --vision-backend local \
  --dataset scenarios/stress_dataset.json --out outputs/stress_run_QwenVL25_AWQ
```

Repeat with the InternVL and Llama servers, changing the output directory names.
`eval/run_topology_evaluation.py` groups scenarios by map to reduce map reloads.

## 4. Metrics and figures

```bash
python -m eval.compute_metrics \
  --baseline outputs/baseline/baseline_results.json \
  --generator-only outputs/generator_only \
  --full outputs/full_cloud_reflector_run_QwenVL25_AWQ \
  --stress-baseline outputs/stress_baseline/baseline_results.json \
  --stress-gen-only outputs/stress_generator_only \
  --stress-full outputs/stress_run_QwenVL25_AWQ \
  --out eval/paper_results/qwen25

python -m eval.generate_fig6      # SAS distribution across the three verifiers
```

## Caching

`Generator.generate()` caches each ScenarioSpec on disk, keyed by prompt, refinement context and
hardware profile. The Verifier cache is keyed per backend. Delete `.cache/` for a cold run.
Otherwise later runs reuse earlier specs, which is also why Generator latency is near zero for
the second and third verifier.

## Metric definition

This document defines the semantic-alignment score exactly as implemented in
`verifier/agent.py`. Every score in `pipeline_run.json` (`verification_score`)
and every table produced by `eval/compute_metrics.py` uses this definition.

### 1. Predicates

For a scenario with specification `spec` (an instance of `ScenarioSpec`) and a
completed simulation run producing frames `F = {f_1, ..., f_k}` and a state
log `L`, define four boolean predicates evaluated against the rendered
frames:

| Predicate | Definition |
|---|---|
| `weather_correct(F, spec)` | The visually rendered weather condition in `F` matches `spec.weather`. |
| `pedestrian_present(F, spec)` | A pedestrian actor is visible in `F`. |
| `vehicles_present(F, spec)` | The number of vehicle actors visible in `F` is approximately consistent with `spec.num_vehicles`. |
| `event_observable(F, spec)` | The event described by `spec.event_type` (e.g. crossing, lane change, emergency braking) is observably occurring or has occurred across `F`. |

Each predicate is evaluated by a vision-language model `M` conditioned on the
original natural-language prompt `p`, `spec`, and the sampled frames `F`, and
returns a value in `{True, False}`.

### 2. Applicability

Not every predicate is meaningful for every scenario. Define the applicability
function `A: predicate -> {True, False}` per scenario:

```
A(weather_correct)      = True                         (always applicable)
A(pedestrian_present)   = spec.pedestrian               (only if a pedestrian was requested)
A(vehicles_present)     = spec.num_vehicles > 0         (only if vehicles were requested)
A(event_observable)     = True                          (always applicable)
```

The applicable predicate set for a scenario is:

```
Applicable(spec) = { q in {weather_correct, pedestrian_present, vehicles_present, event_observable}
                      : A(q) = True }
```

A predicate that is not applicable is excluded from scoring entirely — it is
neither a pass nor a failure, since the prompt made no claim about it.

### 3. Predicate Agreement (the primary semantic alignment score)

For a single VLM evaluation `j`, define the per-sample score:

```
score_j = ( Σ_{q in Applicable(spec)} 1[q holds in sample j] ) / |Applicable(spec)|
```

This is the fraction of *applicable* predicates that the model judged to
hold, in a single pass. It is fully explainable: for any `score_j`, the
specific predicates that failed can be listed and shown to a human reviewer.

### 4. Repeated Sampling and Uncertainty

Because VLM judgments are not perfectly stable across repeated calls (even at
temperature 0, across different backends/models), the Verifier draws `n`
independent samples (default `n = 3`, configurable via
`P2D_VERIFIER_N_SAMPLES`) of the same frame set and prompt. The reported
predicate agreement is:

```
predicate_agreement      = mean(score_1, ..., score_n)
predicate_agreement_std  = population stddev(score_1, ..., score_n)
```

The **final predicate set** used for downstream decisions (Reflector input,
failed-predicate reporting) is determined by **majority vote** per predicate
across the `n` samples:

```
final(q) = True   if  |{ j : q holds in sample j }| > n/2
           False  otherwise
```

`predicate_agreement_std` is reported alongside every score in the paper's
tables and figures as the metric's own uncertainty estimate — analogous to
reporting a confidence interval rather than a bare point estimate. A high std
on a given scenario is itself a meaningful finding (it indicates the VLM's
judgment is unstable for that scenario, which is worth discussing rather than
averaging away).

### 5. Hybrid Final Score

The predicate agreement score is combined with two deterministic channels
into the final score used for the pass/fail decision and the Reflector
trigger:

```
S_final = w_vlm * predicate_agreement + w_log * S_log + w_exec * S_exec

where:
  S_log  = (number of deterministic state-log checks passed) / (total checks)
  S_exec = 1 if the simulation executed without error, else 0
  w_vlm = 0.5, w_log = 0.3, w_exec = 0.2   (see verifier/agent.py SCORE_WEIGHTS)
```

A scenario **passes verification** if `S_final >= threshold` (default 0.75,
configurable via `P2D_VERIFICATION_THRESHOLD`).

Both components are stored per iteration: `predicate_agreement` (the
explainable, VLM-only part) and `S_final` (which also includes the state-log
and execution evidence, and decides pass/fail).

### 6. Reflection Recovery Rate

To directly answer "does the Reflector do anything," in addition to the score
trajectory across iterations, report:

```
first_pass_rate        = P(scenario passes on iteration 0)
after_reflection_rate  = P(scenario passes by its final iteration)
recovery_rate          = P(scenario passes by final iteration | scenario failed on iteration 0)
```

`recovery_rate` isolates exactly the population the Reflector is responsible
for — scenarios that failed initially — and reports what fraction of *those*
it subsequently fixes. This is computed by `eval/compute_metrics.py`'s
`table3_reflection_recovery()` and is the single number that most directly
supports (or undermines) the claim that targeted reflection, specifically,
is doing useful work — as opposed to the loop merely running more iterations.

### 7. What this metric does NOT claim

- It is not a formal/temporal-logic verification in the runtime-verification
  sense (e.g. LTL monitoring over a continuous actor trace). It is a
  VLM-judged, frame-sampled, majority-voted boolean predicate evaluation. This
  is a meaningfully weaker but more tractable form of verification, and the
  paper should not describe it with stronger formal-methods language than it
  earns.
- Predicate definitions are coarse (four predicates) and do not capture
  finer-grained spatial/temporal relationships (e.g. exact crossing distance,
  precise timing relative to a traffic light phase). This is an explicit
  scope limitation, not an oversight, and should be stated as such in the
  paper's Limitations section.
- `predicate_agreement_std` measures *sampling* instability in the VLM's
  judgment, not the VLM's *calibration* (i.e. it does not tell you whether a
  consistently-given answer is actually correct against ground truth — that
  requires the separate human-labeled ground-truth comparison described in
  the evaluation plan, which is a distinct, still-outstanding validation
  step).
