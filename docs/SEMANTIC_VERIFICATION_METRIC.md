# Semantic Verification Metric — Formal Definition

This document is the canonical definition of the "semantic alignment" metric
used throughout Prompt-to-Drive's evaluation, intended to be adapted directly
into the paper's Methods / System Model section. It exists because an earlier
version of this metric was a single, unexplained float reported by a
vision-language model ("how well does this match, 0 to 1?"), and a reviewer
correctly identified that as inadequate evidence for a trustworthiness claim.
This document describes what replaced it.

## 1. Predicates

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

## 2. Applicability

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

## 3. Predicate Agreement (the primary semantic alignment score)

For a single VLM evaluation `j`, define the per-sample score:

```
score_j = ( Σ_{q in Applicable(spec)} 1[q holds in sample j] ) / |Applicable(spec)|
```

This is the fraction of *applicable* predicates that the model judged to
hold, in a single pass. It is fully explainable: for any `score_j`, the
specific predicates that failed can be listed and shown to a human reviewer.

## 4. Repeated Sampling and Uncertainty

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

## 5. Hybrid Final Score

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

A scenario **passes verification** if `S_final >= threshold` (default 0.72,
configurable via `P2D_VERIFICATION_THRESHOLD`).

**Reporting note for the paper:** report `predicate_agreement` as the primary
semantic alignment metric in all tables, since it is the explainable,
auditable component; report `S_final` only where the pass/fail decision
itself is the subject (e.g. Table 1's success-rate definition), and be
explicit in the Methods section that `S_final` additionally incorporates
deterministic execution/log evidence, not just the VLM channel.

## 6. Reflection Recovery Rate

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

## 7. What this metric does NOT claim

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
