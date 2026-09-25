"""
ui/app.py

A single-process pygame UI for Prompt-to-Drive: type a natural-language
scenario prompt, watch it move through Generator -> Engine -> Verifier ->
Reflector, see the live camera feed from CARLA, and see the semantic
alignment score (with a pass/fail checkmark) update per iteration.

Design note vs. the older two-process / socket-based prototype this
project started from: that design split a pygame UI process from a
simulator process connected over a raw TCP socket with a custom framing
protocol (1-byte type + 4-byte length + pickled/JPEG payload), reconnect
logic, etc. That plumbing is real and works, but it solves a problem this
version doesn't have: here, the UI and the CARLA client live in the SAME
process, with the pipeline run on a background thread and frames handed
to the UI thread via an in-memory queue. This removes an entire class of
socket-reconnect bugs at the cost of being a single process -- the right
tradeoff for a research demo, even though the socket-based version was a
reasonable design for a different deployment shape.

Run:
    python -m ui.app --backend huggingface --hardware-profile low_vram_2gb

Requires a running CARLA server (see README "Prerequisites").
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pygame
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends.llm_backend import get_backend  # noqa: E402
from engine.carla_runner import CarlaRunner  # noqa: E402
from orchestrator.pipeline import PipelineRun, PromptToDrivePipeline  # noqa: E402

load_dotenv()

WINDOW_W, WINDOW_H = 1000, 760
SIM_VIEW_RECT = pygame.Rect(30, 70, 640, 360)
PANEL_RECT = pygame.Rect(690, 70, 280, 360)

AGENT_STAGES = ["generator", "engine", "verifier", "reflector"]

COLORS = dict(
    bg=(24, 24, 26),
    panel_bg=(34, 34, 38),
    text=(235, 235, 235),
    dim=(150, 150, 155),
    accent=(90, 160, 255),
    success=(110, 220, 130),
    fail=(235, 95, 95),
    pending=(90, 90, 95),
    active=(255, 200, 80),
    input_active=(90, 160, 255),
    input_inactive=(70, 70, 74),
)


class PipelineUIState:
    """Thread-safe-ish shared state between the pipeline-running background
    thread and the pygame render loop. Reads/writes are simple enough
    (single producer, single consumer per field) that a lock per-field
    would be overkill; the frame queue is the one place we use a real
    queue.Queue for safety."""

    def __init__(self):
        self.frame_queue: "queue.Queue" = queue.Queue(maxsize=2)
        self.current_surface: Optional[pygame.Surface] = None

        self.running_pipeline = False
        self.current_iteration = 0
        self.agent_stage_status: dict[str, str] = {s: "pending" for s in AGENT_STAGES}
        self.last_score: Optional[float] = None
        self.last_predicate_agreement: Optional[float] = None
        self.last_predicate_std: Optional[float] = None
        self.last_passed: Optional[bool] = None
        self.last_missing: list[str] = []
        self.last_notes: str = ""
        self.final_status: Optional[str] = None
        self.status_message = "Ready. Type a scenario prompt below and press Enter."
        self.status_color = COLORS["text"]

    def reset_for_new_run(self):
        self.running_pipeline = True
        self.current_iteration = 0
        self.agent_stage_status = {s: "pending" for s in AGENT_STAGES}
        self.last_score = None
        self.last_predicate_agreement = None
        self.last_predicate_std = None
        self.last_passed = None
        self.last_missing = []
        self.last_notes = ""
        self.final_status = None


def carla_image_to_surface(image) -> pygame.Surface:
    """Converts a carla.Image (BGRA raw buffer) to a pygame.Surface."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    array = array[:, :, :3][:, :, ::-1]  # BGRA -> RGB
    return pygame.surfarray.make_surface(array.swapaxes(0, 1))


class PromptToDriveUI:
    def __init__(self, backend: Optional[str], hardware_profile: Optional[str],
                 carla_host: str, carla_port: int, max_iterations: int, output_root: str):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Prompt-to-Drive — Closed-Loop Scenario Verification")
        self.font = pygame.font.Font(None, 20)
        self.font_small = pygame.font.Font(None, 16)
        self.title_font = pygame.font.Font(None, 30)

        self.state = PipelineUIState()
        self.input_text = ""
        self.input_active = True
        self.input_rect = pygame.Rect(30, 460, 940, 40)
        self.send_button = pygame.Rect(30, 510, 120, 36)

        runner = CarlaRunner(host=carla_host, port=carla_port, hardware_profile=hardware_profile)
        backend_instance = get_backend(backend or os.getenv("P2D_TEXT_BACKEND", "openai"))
        self.pipeline = PromptToDrivePipeline(
            runner=runner, max_iterations=max_iterations, output_root=output_root,
            hardware_profile=hardware_profile,
            generator_backend=backend_instance,
            verifier_backend=backend_instance,
            reflector_backend=backend_instance,
        )
        self.last_run: Optional[PipelineRun] = None
        self.running = True

    # ------------------------------------------------------------------ #
    # Pipeline thread + callbacks
    # ------------------------------------------------------------------ #
    def _on_frame(self, carla_image) -> None:
        try:
            surface = carla_image_to_surface(carla_image)
            if self.state.frame_queue.full():
                self.state.frame_queue.get_nowait()
            self.state.frame_queue.put_nowait(surface)
        except Exception as exc:  # noqa: BLE001
            print(f"[UI] frame conversion error: {exc}")

    def _on_status(self, stage: str, payload: dict) -> None:
        s = self.state
        if stage == "generator":
            s.current_iteration = payload["iteration"]
            s.agent_stage_status = {st: "pending" for st in AGENT_STAGES}
            s.agent_stage_status["generator"] = "active"
            s.status_message = f"Iteration {s.current_iteration}: Generator synthesizing ScenarioSpec..."
        elif stage == "generator_done":
            s.agent_stage_status["generator"] = "done"
        elif stage == "engine":
            s.agent_stage_status["engine"] = "active"
            s.status_message = f"Iteration {s.current_iteration}: running scenario in CARLA..."
        elif stage == "engine_done":
            s.agent_stage_status["engine"] = "done" if payload["success"] else "fail"
        elif stage == "verifier":
            s.agent_stage_status["verifier"] = "active"
            s.status_message = f"Iteration {s.current_iteration}: Verifier scoring semantic alignment..."
        elif stage == "verifier_done":
            s.agent_stage_status["verifier"] = "done"
            s.last_score = payload["score"]
            s.last_predicate_agreement = payload["predicate_agreement"]
            s.last_predicate_std = payload["predicate_agreement_std"]
            s.last_passed = payload["passed"]
            s.last_missing = payload["missing_elements"]
            s.last_notes = payload["notes"]
        elif stage == "reflector":
            s.agent_stage_status["reflector"] = "active"
            s.status_message = (
                f"Verification failed (predicate agreement {s.last_predicate_agreement:.0%} "
                f"± {s.last_predicate_std:.2f}) — Reflector building targeted correction..."
            )
        elif stage == "reflector_done":
            s.agent_stage_status["reflector"] = "done"
        elif stage == "done":
            s.final_status = payload["final_status"]
            s.running_pipeline = False
            if payload["final_status"] == "passed":
                s.status_message = f"Passed — predicate agreement {s.last_predicate_agreement:.0%}. Ready for a new prompt."
            else:
                s.status_message = f"Finished with status '{payload['final_status']}'. Ready for a new prompt."

    def _run_pipeline_thread(self, prompt: str) -> None:
        self.state.reset_for_new_run()
        scenario_id = f"ui_{int(time.time())}"
        try:
            self.last_run = self.pipeline.run(
                prompt=prompt,
                scenario_id=scenario_id,
                mode="full",
                status_callback=self._on_status,
                live_frame_callback=self._on_frame,
            )
        except Exception as exc:  # noqa: BLE001
            self.state.status_message = f"Pipeline error: {exc}"
            self.state.status_color = COLORS["fail"]
            self.state.running_pipeline = False

    def submit_prompt(self) -> None:
        prompt = self.input_text.strip()
        if not prompt or self.state.running_pipeline:
            return
        threading.Thread(target=self._run_pipeline_thread, args=(prompt,), daemon=True).start()

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def draw(self) -> None:
        self.screen.fill(COLORS["bg"])

        title = self.title_font.render("Prompt-to-Drive", True, COLORS["text"])
        self.screen.blit(title, (30, 20))
        subtitle = self.font_small.render(
            "Closed-loop AI scenario synthesis and semantic verification for CARLA", True, COLORS["dim"]
        )
        self.screen.blit(subtitle, (200, 28))

        self._draw_sim_view()
        self._draw_agent_panel()
        self._draw_input()
        self._draw_status_bar()

        pygame.display.flip()

    def _draw_sim_view(self) -> None:
        pygame.draw.rect(self.screen, (12, 12, 14), SIM_VIEW_RECT)
        pygame.draw.rect(self.screen, (70, 70, 75), SIM_VIEW_RECT, 1)

        try:
            surface = self.state.frame_queue.get_nowait()
            self.state.current_surface = surface
        except queue.Empty:
            pass

        if self.state.current_surface is not None:
            scaled = pygame.transform.scale(self.state.current_surface, SIM_VIEW_RECT.size)
            self.screen.blit(scaled, SIM_VIEW_RECT.topleft)
        else:
            label = self.font.render("No live frame yet — run a scenario", True, COLORS["dim"])
            self.screen.blit(label, label.get_rect(center=SIM_VIEW_RECT.center))

    def _draw_agent_panel(self) -> None:
        pygame.draw.rect(self.screen, COLORS["panel_bg"], PANEL_RECT)
        pygame.draw.rect(self.screen, (70, 70, 75), PANEL_RECT, 1)

        y = PANEL_RECT.top + 12
        header = self.font.render(f"Iteration {self.state.current_iteration}", True, COLORS["text"])
        self.screen.blit(header, (PANEL_RECT.left + 12, y))
        y += 30

        stage_labels = {
            "generator": "1. Generator (prompt → spec)",
            "engine": "2. Engine (CARLA execution)",
            "verifier": "3. Verifier (semantic check)",
            "reflector": "4. Reflector (correction)",
        }
        status_colors = {
            "pending": COLORS["pending"],
            "active": COLORS["active"],
            "done": COLORS["success"],
            "fail": COLORS["fail"],
        }
        status_symbols = {"pending": "○", "active": "◐", "done": "●", "fail": "✕"}

        for stage in AGENT_STAGES:
            status = self.state.agent_stage_status.get(stage, "pending")
            color = status_colors[status]
            symbol = status_symbols[status]
            line = f"{symbol}  {stage_labels[stage]}"
            surf = self.font_small.render(line, True, color)
            self.screen.blit(surf, (PANEL_RECT.left + 12, y))
            y += 26

        y += 10
        pygame.draw.line(self.screen, (70, 70, 75), (PANEL_RECT.left + 12, y), (PANEL_RECT.right - 12, y))
        y += 16

        # Semantic alignment score with checkmark
        if self.state.last_predicate_agreement is not None:
            pct = self.state.last_predicate_agreement
            std = self.state.last_predicate_std or 0.0
            passed = self.state.last_passed
            score_color = COLORS["success"] if passed else COLORS["fail"]
            checkmark = "✓" if passed else "✗"
            score_line = f"{checkmark}  Predicate agreement: {pct:.0%} ± {std:.2f}"
            surf = self.font.render(score_line, True, score_color)
            self.screen.blit(surf, (PANEL_RECT.left + 12, y))
            y += 22

            hybrid_line = f"   (hybrid score incl. log+exec: {self.state.last_score:.0%})"
            hybrid_surf = self.font_small.render(hybrid_line, True, COLORS["dim"])
            self.screen.blit(hybrid_surf, (PANEL_RECT.left + 12, y))
            y += 22

            # Simple progress bar
            bar_rect = pygame.Rect(PANEL_RECT.left + 12, y, PANEL_RECT.width - 24, 14)
            pygame.draw.rect(self.screen, (50, 50, 54), bar_rect)
            fill_w = int(bar_rect.width * min(max(pct, 0), 1))
            pygame.draw.rect(self.screen, score_color, (bar_rect.left, bar_rect.top, fill_w, bar_rect.height))
            pygame.draw.rect(self.screen, (90, 90, 95), bar_rect, 1)
            y += 26

            if self.state.last_missing:
                miss_label = self.font_small.render("Missing:", True, COLORS["dim"])
                self.screen.blit(miss_label, (PANEL_RECT.left + 12, y))
                y += 18
                for elem in self.state.last_missing[:4]:
                    line = self.font_small.render(f"  • {elem}", True, COLORS["fail"])
                    self.screen.blit(line, (PANEL_RECT.left + 12, y))
                    y += 16
        else:
            placeholder = self.font_small.render("No verification yet.", True, COLORS["dim"])
            self.screen.blit(placeholder, (PANEL_RECT.left + 12, y))

    def _draw_input(self) -> None:
        label = self.font_small.render(
            'Scenario prompt (e.g. "A pedestrian crosses near a signal in rain"):',
            True, COLORS["dim"],
        )
        self.screen.blit(label, (30, 438))

        color = COLORS["input_active"] if self.input_active else COLORS["input_inactive"]
        pygame.draw.rect(self.screen, (18, 18, 20), self.input_rect)
        pygame.draw.rect(self.screen, color, self.input_rect, 2)
        text_surf = self.font.render(self.input_text, True, COLORS["text"])
        self.screen.blit(text_surf, (self.input_rect.x + 8, self.input_rect.y + 10))

        disabled = self.state.running_pipeline
        btn_color = COLORS["pending"] if disabled else COLORS["accent"]
        pygame.draw.rect(self.screen, btn_color, self.send_button)
        btn_label = "Running..." if disabled else "Run"
        btn_surf = self.font.render(btn_label, True, (10, 10, 10))
        self.screen.blit(btn_surf, btn_surf.get_rect(center=self.send_button.center))

    def _draw_status_bar(self) -> None:
        bar_rect = pygame.Rect(0, WINDOW_H - 30, WINDOW_W, 30)
        pygame.draw.rect(self.screen, (16, 16, 18), bar_rect)
        surf = self.font_small.render(self.state.status_message, True, COLORS["text"])
        self.screen.blit(surf, (10, WINDOW_H - 22))

    # ------------------------------------------------------------------ #
    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.send_button.collidepoint(event.pos):
                    self.submit_prompt()
                self.input_active = self.input_rect.collidepoint(event.pos)
            elif event.type == pygame.KEYDOWN and self.input_active:
                if event.key == pygame.K_RETURN:
                    self.submit_prompt()
                elif event.key == pygame.K_BACKSPACE:
                    self.input_text = self.input_text[:-1]
                elif event.unicode.isprintable() and len(self.input_text) < 300:
                    self.input_text += event.unicode
        return True

    def run(self) -> None:
        clock = pygame.time.Clock()
        while self.running:
            self.running = self.handle_events()
            self.draw()
            clock.tick(30)
        pygame.quit()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prompt-to-Drive interactive UI")
    parser.add_argument("--backend", choices=["openai", "huggingface", "local"], default=None)
    parser.add_argument("--hardware-profile", default=None,
                         choices=["low_vram_2gb", "low_vram_2gb_signals"])
    parser.add_argument("--carla-host", default="localhost")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--out", default="outputs/ui")
    args = parser.parse_args()

    backend_needed = args.backend or os.getenv("P2D_TEXT_BACKEND", "openai")
    if backend_needed == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. See .env.example.")
    if backend_needed == "huggingface" and not os.getenv("HUGGINGFACE_API_TOKEN"):
        raise SystemExit("HUGGINGFACE_API_TOKEN is not set. See .env.example.")

    app = PromptToDriveUI(
        backend=args.backend,
        hardware_profile=args.hardware_profile,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        max_iterations=args.max_iterations,
        output_root=args.out,
    )
    app.run()


if __name__ == "__main__":
    main()
