# Reproducing the evaluation

The paper's numbers come from preserved traces (`outputs/<condition>/<scenario_id>/pipeline_run.json`)
processed by one script, `eval/compute_metrics.py`. The per-verifier results it produced are
checked in under `eval/results_llama32/`, `eval/results_qwen25/` and `eval/results_intern25/`.

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
  --out eval/results_qwen25

python -m eval.generate_fig6      # SAS distribution across the three verifiers
```

## Caching

`Generator.generate()` caches each ScenarioSpec on disk, keyed by prompt, refinement context and
hardware profile. The Verifier cache is keyed per backend. Delete `.cache/` for a cold run.
Otherwise later runs reuse earlier specs, which is also why Generator latency is near zero for
the second and third verifier.
