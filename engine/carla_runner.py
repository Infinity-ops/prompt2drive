"""
engine/carla_runner.py

Deterministic execution engine: converts a validated ScenarioSpec into
CARLA API calls, runs the simulation in synchronous mode, and captures
frames + a structured state log used by the Verifier agent.

This module deliberately contains NO LLM calls and NO free-form code
execution. Every action it can take is enumerated below. This is what
makes Prompt-to-Drive's generation pipeline safe and reproducible: the
LLM only ever chooses *values* for a fixed set of fields; this engine is
the only thing that ever touches the CARLA API.
"""
from __future__ import annotations

import json
import logging
import random
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from engine.scenario_spec import ScenarioSpec

logger = logging.getLogger("prompt_to_drive.engine")

WEATHER_PRESETS = {
    "clear": dict(cloudiness=10, precipitation=0, precipitation_deposits=0,
                  fog_density=0, sun_altitude_angle=70),
    "rain": dict(cloudiness=80, precipitation=80, precipitation_deposits=50,
                 fog_density=0, sun_altitude_angle=40),
    "fog": dict(cloudiness=60, precipitation=0, precipitation_deposits=0,
                fog_density=60, fog_distance=15, sun_altitude_angle=50),
    "night": dict(cloudiness=20, precipitation=0, precipitation_deposits=0,
                  fog_density=0, sun_altitude_angle=-30),
    "cloudy": dict(cloudiness=70, precipitation=0, precipitation_deposits=0,
                   fog_density=0, sun_altitude_angle=55),
}

# Fallback spawn point indices known to be valid per town for pedestrian
# proximity / junction proximity. Used if the requested index is occupied.
FALLBACK_SPAWN_INDICES = [0, 3, 5, 8, 12, 20, 35]


@dataclass
class ExecutionResult:
    scenario_id: str
    success: bool
    frame_paths: list[str] = field(default_factory=list)
    log_path: Optional[str] = None
    error: Optional[str] = None
    duration_s: float = 0.0


class CarlaRunner:
    """
    Wraps a CARLA client connection and executes a ScenarioSpec.

    Usage:
        runner = CarlaRunner(host="localhost", port=2000)
        result = runner.run(spec, scenario_id="s001", output_dir="outputs/s001")
    """

    def __init__(self, host: str = "localhost", port: int = 2000, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._client = None
        self._world = None

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #
    def connect(self):
        import carla  # imported lazily so the rest of the repo is importable
        # without a local CARLA install (e.g. for running unit tests / CI)

        self._client = carla.Client(self.host, self.port)
        self._client.set_timeout(self.timeout)
        self._world = self._client.get_world()

        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 Hz simulation tick
        self._world.apply_settings(settings)
        logger.info("Connected to CARLA at %s:%s", self.host, self.port)

    def disconnect(self):
        if self._world is not None:
            settings = self._world.get_settings()
            settings.synchronous_mode = False
            self._world.apply_settings(settings)

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def run(self, spec: ScenarioSpec, scenario_id: str, output_dir: str) -> ExecutionResult:
        import carla

        out = Path(output_dir)
        frames_dir = out / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        state_log: dict = {
            "scenario_id": scenario_id,
            "spec": spec.model_dump(),
            "weather_api_called": None,
            "pedestrian_spawned": False,
            "vehicle_count": 0,
            "actors": [],
            "execution_completed": False,
            "errors": [],
        }

        actors_to_destroy = []
        camera = None
        t0 = time.time()

        try:
            if self._client is None:
                self.connect()

            # 1. Load town if different from current map
            current_map = self._world.get_map().name.split("/")[-1]
            if current_map != spec.town:
                self._world = self._client.load_world(spec.town)
                settings = self._world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                self._world.apply_settings(settings)

            # 2. Apply weather
            weather_params = carla.WeatherParameters(**WEATHER_PRESETS[spec.weather])
            self._world.set_weather(weather_params)
            state_log["weather_api_called"] = spec.weather

            blueprint_library = self._world.get_blueprint_library()
            spawn_points = self._world.get_map().get_spawn_points()

            # 3. Spawn ego vehicle + camera
            ego_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
            ego_spawn = self._safe_spawn_point(spawn_points, spec.spawn_point_index)
            ego_vehicle = self._world.spawn_actor(ego_bp, ego_spawn)
            ego_vehicle.set_autopilot(True)
            actors_to_destroy.append(ego_vehicle)
            state_log["actors"].append({"type": "ego_vehicle", "id": ego_vehicle.id})

            camera_bp = blueprint_library.find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", "960")
            camera_bp.set_attribute("image_size_y", "540")
            camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
            camera = self._world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
            actors_to_destroy.append(camera)

            captured_frames: dict[int, "carla.Image"] = {}

            def _on_image(image):
                captured_frames[image.frame] = image

            camera.listen(_on_image)

            # 4. Spawn NPC vehicles
            vehicle_bps = blueprint_library.filter("vehicle.*")
            npc_spawn_points = [p for i, p in enumerate(spawn_points) if i != spec.spawn_point_index]
            random.shuffle(npc_spawn_points)
            for i in range(min(spec.num_vehicles, len(npc_spawn_points))):
                bp = random.choice(vehicle_bps)
                npc = self._world.try_spawn_actor(bp, npc_spawn_points[i])
                if npc is not None:
                    npc.set_autopilot(True)
                    actors_to_destroy.append(npc)
                    state_log["vehicle_count"] += 1

            # 5. Spawn pedestrian if requested
            if spec.pedestrian:
                walker_bp = random.choice(blueprint_library.filter("walker.pedestrian.*"))
                walker_loc = ego_spawn.location
                walker_loc.x += random.uniform(-8, 8)
                walker_loc.y += random.uniform(-8, 8)
                walker_transform = carla.Transform(walker_loc)
                walker = self._world.try_spawn_actor(walker_bp, walker_transform)
                if walker is not None:
                    actors_to_destroy.append(walker)
                    state_log["pedestrian_spawned"] = True
                    state_log["actors"].append({"type": "pedestrian", "id": walker.id})
                    # simple crossing controller
                    controller_bp = blueprint_library.find("controller.ai.walker")
                    controller = self._world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
                    actors_to_destroy.append(controller)
                    controller.start()
                    controller.go_to_location(self._world.get_random_location_from_navigation())
                    controller.set_max_speed(1.4)

            # 6. Tick the simulation, capturing frames every ~2s
            ticks = int(spec.duration_s / 0.05)
            capture_every = int(2.0 / 0.05)
            saved_frames = []

            for t in range(ticks):
                self._world.tick()
                if t % capture_every == 0:
                    # give the camera callback a moment to populate
                    frame_id = self._world.get_snapshot().frame
                    time.sleep(0.02)
                    if frame_id in captured_frames:
                        img = captured_frames[frame_id]
                        fp = frames_dir / f"frame_{len(saved_frames):03d}.png"
                        img.save_to_disk(str(fp))
                        saved_frames.append(str(fp))

            state_log["execution_completed"] = True
            camera.stop()

            log_path = out / "state_log.json"
            with open(log_path, "w") as f:
                json.dump(state_log, f, indent=2)

            return ExecutionResult(
                scenario_id=scenario_id,
                success=True,
                frame_paths=saved_frames,
                log_path=str(log_path),
                duration_s=time.time() - t0,
            )

        except Exception as exc:  # noqa: BLE001 - we want to capture and log everything
            tb = traceback.format_exc()
            state_log["errors"].append(str(exc))
            logger.error("Scenario %s failed: %s\n%s", scenario_id, exc, tb)

            log_path = out / "state_log.json"
            with open(log_path, "w") as f:
                json.dump(state_log, f, indent=2)

            return ExecutionResult(
                scenario_id=scenario_id,
                success=False,
                log_path=str(log_path),
                error=str(exc),
                duration_s=time.time() - t0,
            )

        finally:
            for actor in actors_to_destroy:
                try:
                    actor.destroy()
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _safe_spawn_point(self, spawn_points, index: int):
        """Returns a spawn point, falling back to known-good indices if the
        requested one is out of range."""
        if 0 <= index < len(spawn_points):
            return spawn_points[index]
        for fallback in FALLBACK_SPAWN_INDICES:
            if fallback < len(spawn_points):
                logger.warning("spawn_point_index %s invalid, falling back to %s", index, fallback)
                return spawn_points[fallback]
        return spawn_points[0]
