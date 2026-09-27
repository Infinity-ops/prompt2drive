"""
webui/server.py

Browser front-end for Prompt-to-Drive. Type a prompt, press Generate, and
watch each agent of the closed loop report in turn:

    Agent 1 Generator  ->  CARLA Engine  ->  Agent 2 Verifier  ->  Agent 3 Reflector  (loop)

with the camera frames the Engine captures streamed into the page.

Two modes
---------
live    Runs the real PromptToDrivePipeline in-process against a CARLA
        server (same code path as run_scenario.py). Needs CARLA + a backend.
replay  Re-streams recorded runs (outputs/<id>/pipeline_run.json + frames).
        Needs no CARLA, GPU or API key -> safe to host publicly.

Run from the repo root:
    python -m webui.server --mode live --backend local --hardware-profile low_vram_2gb
    python -m webui.server --mode replay --runs-dir demo_runs

Design notes
------------
* One job at a time. CARLA is a single shared world, so concurrent runs
  would corrupt each other; extra requests get a clear "busy" error.
* Every connected browser sees the same run (useful for a demo projector
  plus a laptop). Late joiners receive the current run's event history.
* The pipeline runs on a worker thread; events cross into asyncio through
  one queue and one broadcaster task, so message order is preserved.
* Binds to 127.0.0.1 by default. Live mode spends API credit / GPU time
  per click, so exposing it needs P2D_UI_TOKEN (see docs/WEB_UI.md).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import secrets
import socket
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from webui import replay as replay_mod
from webui.frames import carla_image_to_jpeg_b64

load_dotenv()
logger = logging.getLogger("prompt_to_drive.webui")

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_PROMPT_CHARS = 500
# Profiles valid in BOTH engine/hardware_profiles.py and engine/hardware_config.py
PIPELINE_PROFILES = ("low_vram_2gb", "low_vram_2gb_signals")


@dataclass
class UIConfig:
    mode: str = "live"                       # "live" | "replay"
    backend: str = "openai"
    verifier_backend: Optional[str] = None   # defaults to backend
    hardware_profile: Optional[str] = None
    pin_town: Optional[str] = None           # demo safety: never switch maps
    threshold: Optional[float] = None        # UI-only override of the Verifier pass threshold
    carla_host: str = "localhost"
    carla_port: int = 2000
    max_iterations: int = 3
    output_root: str = "outputs/ui"
    runs_dir: str = "outputs/ui"
    token: Optional[str] = None
    dataset: str = "scenarios/dataset.json"
    extra: dict = field(default_factory=dict)


def carla_reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    """TCP probe of the CARLA RPC port (ported from test_UI/start_carla_sim.py)."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _clean_prompt(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    return " ".join(text.split())[:MAX_PROMPT_CHARS]


class Hub:
    """Owns the connected sockets, the single job slot and the event history."""

    def __init__(self, cfg: UIConfig):
        self.cfg = cfg
        self.clients: set[WebSocket] = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.queue: Optional[asyncio.Queue] = None
        self.history: list[dict] = []
        self.last_frame: Optional[dict] = None
        self.busy = threading.Lock()
        self.stop_flag = threading.Event()
        self.current_iteration = 0
        self.current_scenario: Optional[str] = None
        self.preview_count: dict[int, int] = {}
        self._pipeline = None

    # ---------------- thread -> asyncio bridge ----------------
    def publish(self, msg: dict) -> None:
        if msg["type"] == "event":
            self.history.append(msg)
        elif msg["type"] == "frame":
            self.last_frame = msg
        if self.loop and self.queue:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, msg)

    def emit(self, stage: str, payload: dict) -> None:
        if "iteration" in payload:
            self.current_iteration = payload["iteration"]
        if stage == "run_start":
            self.current_scenario = payload.get("scenario_id")
            self.preview_count = {}
        self.publish({"type": "event", "stage": stage, "payload": payload, "t": time.time()})

    def emit_frame(self, jpeg_b64: str, iteration: Optional[int] = None) -> None:
        it = iteration or self.current_iteration
        self.publish({"type": "frame", "iteration": it, "jpeg": jpeg_b64})
        if self.cfg.mode == "live" and self.current_scenario:
            self._save_preview(jpeg_b64, it)

    def _save_preview(self, jpeg_b64: str, iteration: int) -> None:
        """Keep the UI preview frames next to the run (iter_N/preview/), so a
        replay of this run later shows the same smooth clip. The Verifier's
        frames in iter_N/frames/ are untouched."""
        try:
            k = self.preview_count.get(iteration, 0)
            self.preview_count[iteration] = k + 1
            d = Path(self.cfg.output_root) / self.current_scenario / f"iter_{iteration - 1}" / "preview"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"p_{k:04d}.jpg").write_bytes(base64.b64decode(jpeg_b64))
        except Exception:  # noqa: BLE001 - preview is best-effort
            logger.debug("preview save failed", exc_info=True)

    async def broadcaster(self) -> None:
        assert self.queue is not None
        while True:
            msg = await self.queue.get()
            text = json.dumps(msg, default=str)
            dead = []
            for ws in list(self.clients):
                try:
                    await ws.send_text(text)
                except Exception:  # noqa: BLE001
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)

    # ---------------- jobs ----------------
    def _build_pipeline(self):
        # Imported lazily so replay deployments don't need carla/torch/SDKs.
        from backends.llm_backend import get_backend
        from engine.carla_runner import CarlaRunner
        from orchestrator.pipeline import PromptToDrivePipeline

        cfg = self.cfg
        gen_backend = get_backend(cfg.backend)
        ver_backend = get_backend(cfg.verifier_backend) if cfg.verifier_backend else gen_backend
        runner = CarlaRunner(host=cfg.carla_host, port=cfg.carla_port, hardware_profile=cfg.hardware_profile)
        pipeline = PromptToDrivePipeline(
            runner=runner,
            max_iterations=cfg.max_iterations,
            output_root=cfg.output_root,
            hardware_profile=cfg.hardware_profile if cfg.hardware_profile in PIPELINE_PROFILES else None,
            generator_backend=gen_backend,
            verifier_backend=ver_backend,
            reflector_backend=gen_backend,
        )
        if cfg.threshold is not None:
            # Raising tau is the paper's "forced-reflection" condition: the
            # Reflector then engages whenever any check fails. UI-only knob.
            pipeline.verifier.threshold = cfg.threshold
            logger.info("Verifier pass threshold set to %.2f for this UI session.", cfg.threshold)
        if cfg.pin_town:
            # Every spec is clamped to one town, so the Engine never reloads a
            # map between runs. UI-only knob; evaluation scripts don't use it.
            pipeline.preferred_towns = (cfg.pin_town,)
            pipeline.generator.preferred_towns = (cfg.pin_town,)
            logger.info("Pinned to %s: map reloads between runs are disabled.", cfg.pin_town)
        return pipeline

    def _on_carla_frame(self, image) -> None:
        try:
            self.emit_frame(carla_image_to_jpeg_b64(image))
        except Exception:  # noqa: BLE001
            logger.debug("frame encode failed", exc_info=True)

    def _job(self, prompt: str) -> None:
        try:
            self.history = []
            self.last_frame = None
            self.stop_flag.clear()
            if self.cfg.mode == "replay":
                run = replay_mod.find_run(self.cfg.runs_dir, prompt)
                if run is None:
                    self.emit("error", {"message": (
                        "This demo replays recorded runs only. Pick one of the recorded prompts, "
                        "or run the UI locally in live mode with CARLA to try a new prompt.")})
                    return
                replay_mod.replay_run(run, self.emit, self.emit_frame, should_stop=self.stop_flag.is_set)
                return

            if not carla_reachable(self.cfg.carla_host, self.cfg.carla_port):
                self.emit("error", {"message": (
                    f"CARLA is not reachable at {self.cfg.carla_host}:{self.cfg.carla_port}. "
                    "Start the CARLA server, then press Generate again.")})
                return
            if self._pipeline is None:
                self._pipeline = self._build_pipeline()
            scenario_id = time.strftime("ui_%Y%m%d_%H%M%S")
            self._pipeline.run(
                prompt=prompt, scenario_id=scenario_id, mode="full",
                status_callback=self.emit, live_frame_callback=self._on_carla_frame,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("pipeline job failed")
            self.emit("error", {"message": f"{type(exc).__name__}: {exc}"})
        finally:
            self.busy.release()

    def start(self, prompt: str) -> Optional[str]:
        if not self.busy.acquire(blocking=False):
            return "A scenario is already running. Wait for it to finish, then generate again."
        threading.Thread(target=self._job, args=(prompt,), daemon=True, name="p2d-job").start()
        return None


def _examples(cfg: UIConfig) -> list[dict]:
    if cfg.mode == "replay":
        return [r.summary() for r in replay_mod.list_runs(cfg.runs_dir)]
    try:
        data = json.loads((REPO_ROOT / cfg.dataset).read_text())
        return [{"scenario_id": s["id"], "prompt": s["prompt"], "difficulty": s.get("difficulty")}
                for s in data["scenarios"][:8]]
    except (OSError, ValueError, KeyError):
        return []


def create_app(cfg: UIConfig) -> FastAPI:
    hub = Hub(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.loop = asyncio.get_running_loop()
        hub.queue = asyncio.Queue()
        task = asyncio.create_task(hub.broadcaster())
        yield
        hub.stop_flag.set()
        task.cancel()

    app = FastAPI(title="Prompt-to-Drive UI", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.hub = hub

    def authorized(token: Optional[str]) -> bool:
        return not cfg.token or (token is not None and secrets.compare_digest(token, cfg.token))

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/config")
    def config(token: Optional[str] = None) -> JSONResponse:
        if not authorized(token):
            return JSONResponse({"error": "token required"}, status_code=401)
        reachable = cfg.mode == "live" and carla_reachable(cfg.carla_host, cfg.carla_port)
        return JSONResponse(dict(
            mode=cfg.mode,
            backend=cfg.backend if cfg.mode == "live" else None,
            verifier_backend=(cfg.verifier_backend or cfg.backend) if cfg.mode == "live" else None,
            hardware_profile=cfg.hardware_profile,
            max_iterations=cfg.max_iterations,
            threshold=cfg.threshold,
            carla=dict(host=cfg.carla_host, port=cfg.carla_port, reachable=reachable),
            examples=_examples(cfg),
            max_prompt_chars=MAX_PROMPT_CHARS,
            busy=hub.busy.locked(),
        ))

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        if not authorized(ws.query_params.get("token")):
            await ws.close(code=4401)
            return
        await ws.accept()
        hub.clients.add(ws)
        await ws.send_text(json.dumps({"type": "hello", "busy": hub.busy.locked(),
                                       "history": hub.history, "last_frame": hub.last_frame}, default=str))
        try:
            while True:
                msg = await ws.receive_json()
                kind = msg.get("type")
                if kind == "generate":
                    prompt = _clean_prompt(msg.get("prompt"))
                    if len(prompt) < 5:
                        await ws.send_json({"type": "error", "message": "Describe the scenario in a few words first."})
                        continue
                    err = hub.start(prompt)
                    if err:
                        await ws.send_json({"type": "error", "message": err})
                elif kind == "stop" and cfg.mode == "replay":
                    hub.stop_flag.set()
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            logger.debug("websocket closed", exc_info=True)
        finally:
            hub.clients.discard(ws)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser(description="Prompt-to-Drive web UI")
    p.add_argument("--mode", choices=["live", "replay"], default=os.getenv("P2D_UI_MODE", "live"))
    p.add_argument("--backend", choices=["openai", "huggingface", "local"],
                   default=os.getenv("P2D_TEXT_BACKEND", "openai"))
    p.add_argument("--verifier-backend", choices=["openai", "huggingface", "local"], default=None,
                   help="Backend for the Verifier only (defaults to --backend).")
    p.add_argument("--hardware-profile", choices=list(PIPELINE_PROFILES), default=None)
    p.add_argument("--pin-town", choices=["Town01", "Town02"], default=None,
                   help="Clamp every scenario to one town so CARLA never switches maps (demo safety).")
    p.add_argument("--threshold", type=float, default=None,
                   help="Verifier pass threshold for this UI session (paper default 0.75; 1.0 = every check must pass).")
    p.add_argument("--carla-host", default="localhost")
    p.add_argument("--carla-port", type=int, default=2000)
    p.add_argument("--max-iterations", type=int, default=int(os.getenv("P2D_MAX_ITERATIONS", "3")))
    p.add_argument("--out", default="outputs/ui", help="Where live runs are written (replayable later).")
    p.add_argument("--runs-dir", default=os.getenv("P2D_UI_RUNS_DIR", "outputs/ui"),
                   help="Recorded runs to replay in --mode replay.")
    p.add_argument("--host", default=os.getenv("P2D_UI_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.getenv("PORT", os.getenv("P2D_UI_PORT", "8080"))))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = UIConfig(
        mode=args.mode, backend=args.backend, verifier_backend=args.verifier_backend,
        hardware_profile=args.hardware_profile, pin_town=args.pin_town, threshold=args.threshold, carla_host=args.carla_host, carla_port=args.carla_port,
        max_iterations=args.max_iterations, output_root=args.out, runs_dir=args.runs_dir,
        token=os.getenv("P2D_UI_TOKEN") or None,
    )

    if cfg.mode == "live":
        if cfg.backend == "openai" and not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set (see .env.example), or use --backend local/huggingface.")
        if cfg.backend == "huggingface" and not (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")):
            raise SystemExit("HF_TOKEN is not set (see .env.example).")
        if args.host not in ("127.0.0.1", "localhost") and not cfg.token:
            raise SystemExit("Refusing to expose live mode without P2D_UI_TOKEN: every click spends "
                             "API credit and GPU time. Set P2D_UI_TOKEN or bind to 127.0.0.1.")
    else:
        n = len(replay_mod.list_runs(cfg.runs_dir))
        logger.info("Replay mode: %d recorded run(s) in %s", n, cfg.runs_dir)
        if n == 0:
            logger.warning("No recorded runs found. Copy outputs/ui/<run_id>/ folders into %s.", cfg.runs_dir)

    print(f"\n  Prompt-to-Drive UI ({cfg.mode}) -> http://{args.host}:{args.port}/"
          + (f"?token={cfg.token}" if cfg.token else "") + "\n")
    uvicorn.run(create_app(cfg), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()