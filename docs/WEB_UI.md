# Web UI

A browser front-end for the closed loop. Type a scenario, press **Generate**,
and each stage reports as it runs:

| Step | What the page shows |
|---|---|
| Agent 1 Generator | The validated `ScenarioSpec` (weather, map, vehicles, pedestrian, event, duration) and whether it used a Reflector correction |
| CARLA Engine | The camera frames it captures, streamed live, then success and frame count |
| Agent 2 Verifier | Each applicable predicate (holds / fails), predicate agreement ± std over the VLM samples, state-log checks, and the hybrid score against the pass threshold |
| Agent 3 Reflector | The exact correction text sent back to the Generator for the next iteration |

Iterations get their own tab, so you can step back through a run after it
finishes.

## Two modes

| | `live` | `replay` |
|---|---|---|
| Runs | the real `PromptToDrivePipeline`, same code path as `run_scenario.py` | recorded runs from `pipeline_run.json` + saved frames |
| Needs | CARLA server, a backend (vLLM / OpenAI / HF) | nothing but `requirements-ui.txt` |
| Safe to put on the internet | no (each click spends GPU time or API credit) | yes |
| New prompts | yes | only prompts that were recorded |

Live runs are written to `outputs/ui/<run_id>/`, which is exactly the format
replay reads, so everything you run live can be replayed later.

## Run it locally (live)

```bash
source .venv/bin/activate
pip install -r requirements.txt -r requirements-ui.txt

# terminal 1: CARLA (low-VRAM flags from docs/TROUBLESHOOTING_LOW_VRAM.md)
./CarlaUE4.sh -quality-level=Low -RenderOffScreen -opengl

# terminal 2: vLLM, if you use --backend local
vllm serve <model> --gpu-memory-utilization 0.65 --max-num-seqs 1 --enforce-eager

# terminal 3: the UI
python -m webui.server --mode live --backend local --hardware-profile low_vram_2gb
# open http://127.0.0.1:8080
```

Options: `--verifier-backend` (use a different backend for the Verifier only),
`--max-iterations`, `--carla-host/--carla-port`, `--out`, `--port`.

The header shows whether CARLA is reachable. If it is not, Generate reports it
instead of hanging.

## Replay locally

```bash
python -m webui.server --mode replay --runs-dir outputs/ui
```

## Record runs for the public demo

1. Run a handful of prompts in live mode (a pass on the first iteration, and
   at least one that the Reflector fixes, show the loop best).
2. Copy the run folders you want to publish:
   ```bash
   cp -r outputs/ui/ui_20260925_101500 demo_runs/
   ```
   Frames are PNG; keep a few runs to keep the repo small.
3. Check it: `python -m webui.server --mode replay --runs-dir demo_runs`.

Recorded runs store the hybrid score, pass/fail and the failed predicates,
not per-predicate agreement, so replay shows only what was recorded.

## Publish the replay demo

The `Dockerfile` builds a replay-only image (no CARLA, GPU or keys):

```bash
docker build -t prompt2drive-demo .
docker run -p 8080:8080 prompt2drive-demo
```

Any Docker host that supports WebSockets and sets `$PORT` can run it (for
example Render: New Web Service, pick the GitHub repo, runtime Docker).

## Exposing live mode (not recommended publicly)

The server binds to `127.0.0.1` and refuses to bind elsewhere in live mode
unless an access token is set:

```bash
export P2D_UI_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(24))")
python -m webui.server --mode live --host 0.0.0.0 ...
# share http://<host>:8080/?token=<token>
```

Only one scenario runs at a time (CARLA is a single shared world); other
viewers watch the same run.

## How it plugs into the pipeline

`PromptToDrivePipeline.run()` takes two optional keyword arguments:

```python
pipeline.run(prompt, scenario_id,
             status_callback=lambda stage, payload: ...,
             live_frame_callback=lambda carla_image: ...)
```

Stages per iteration: `generator`, `generator_done`, `engine`, `engine_done`,
`verifier`, `verifier_done`, then `reflector`, `reflector_done` if another
iteration follows; the run is wrapped in `run_start` and `done`.

The hook is observational: it is called outside every latency timer, and any
exception it raises is swallowed. `tests/test_webui.py` checks that
`pipeline_run.json` is identical with and without it, so evaluation numbers
are unaffected. Without the callbacks, `run()` behaves exactly as before.
