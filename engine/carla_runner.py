"""
engine/carla_runner.py

Deterministic execution engine for Prompt-to-Drive.

KEY FIXES IN THIS VERSION
--------------------------
1. TM PORT CONFLICT (the current crash)
   CARLA's Traffic Manager RPC server binds to port 8000 by default.
   vLLM also uses port 8000. They conflict → "bind error".
   Fix: get_trafficmanager(TM_PORT=8100) then set_autopilot(True, TM_PORT=8100).

2. PRE-TICKS BEFORE ACTOR OPERATIONS
   Walker controller.start() and set_autopilot() must only be called
   after the actor has been registered in a world tick. Missing ticks
   cause "trying to operate on a destroyed actor" core dumps.

3. DESTROY ORDER
   controller.stop() → controller.destroy() → walker → npcs →
   camera.stop() → camera.destroy() → ego
   Camera must be destroyed before ego (it is attached to ego).

4. FRAME SAVING
   The camera callback needs up to 30 retries × 20ms = 0.6s to deliver
   a frame. The original code only waited 20ms → no frames saved.
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
from engine.hardware_profiles import HardwareProfile, get_profile

logger = logging.getLogger("prompt_to_drive.engine")

# Port for CARLA's Traffic Manager RPC server.
# Must NOT be 8000 when vLLM is running (vLLM default is 8000).
TM_PORT = 8100

WEATHER_PRESETS = {
    "clear":  dict(cloudiness=10,  precipitation=0,  precipitation_deposits=0,
                   fog_density=0,  sun_altitude_angle=70),
    "rain":   dict(cloudiness=80,  precipitation=80, precipitation_deposits=50,
                   fog_density=0,  sun_altitude_angle=40),
    "fog":    dict(cloudiness=60,  precipitation=0,  precipitation_deposits=0,
                   fog_density=60, fog_distance=15,  sun_altitude_angle=50),
    "night":  dict(cloudiness=20,  precipitation=0,  precipitation_deposits=0,
                   fog_density=0,  sun_altitude_angle=-30),
    "cloudy": dict(cloudiness=70,  precipitation=0,  precipitation_deposits=0,
                   fog_density=0,  sun_altitude_angle=55),
}

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
    def __init__(self, host: str = "localhost", port: int = 2000,
                 timeout: float = 30.0, hardware_profile: Optional[str] = None):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.profile: HardwareProfile = get_profile(hardware_profile)
        self._client = None
        self._world = None
        self._tm = None  # Traffic Manager — kept alive across iterations

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #
    def connect(self):
        import carla
        self._client = carla.Client(self.host, self.port)
        self._client.set_timeout(self.timeout)
        self._world = self._client.get_world()

        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.profile.fixed_delta_seconds
        self._world.apply_settings(settings)

        # Fix 1: bind TM to TM_PORT (8100) to avoid collision with vLLM on 8000
        self._tm = self._client.get_trafficmanager(TM_PORT)
        self._tm.set_synchronous_mode(True)

        logger.info("Connected to CARLA at %s:%s [hardware_profile=%s, TM_PORT=%d]",
                    self.host, self.port, self.profile.name, TM_PORT)

    def disconnect(self):
        if self._world is not None:
            try:
                settings = self._world.get_settings()
                settings.synchronous_mode = False
                self._world.apply_settings(settings)
            except Exception:
                pass

    def _world_alive(self) -> bool:
        try:
            if self._world is None:
                return False
            self._world.get_snapshot()
            return True
        except Exception:
            return False

    def _safe_spawn_point(self, spawn_points, index: int):
        if 0 <= index < len(spawn_points):
            return spawn_points[index]
        for fallback in FALLBACK_SPAWN_INDICES:
            if fallback < len(spawn_points):
                logger.warning("spawn_point_index %s invalid, falling back to %s", index, fallback)
                return spawn_points[fallback]
        return spawn_points[0]

    def _safe_destroy(self, actor) -> None:
        if actor is None:
            return
        try:
            if actor.is_alive:
                actor.destroy()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def run(self, spec: ScenarioSpec, scenario_id: str, output_dir: str,
            live_frame_callback: Optional[callable] = None,
            rng_seed: Optional[int] = None) -> ExecutionResult:

        if rng_seed is not None:
            random.seed(rng_seed)
        import carla

        out = Path(output_dir)
        frames_dir = out / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        state_log: dict = {
            "scenario_id": scenario_id,
            "spec": spec.model_dump(),
            "hardware_profile": self.profile.name,
            "weather_api_called": None,
            "pedestrian_spawned": False,
            "vehicle_count": 0,
            "actors": [],
            "execution_completed": False,
            "errors": [],
        }

        ego_vehicle = None
        camera = None
        npc_list: list = []
        walker_actor = None
        walker_controller = None
        callback_active = True
        t0 = time.time()

        try:
            if not self._world_alive():
                self.connect()

            # ---- Step 1: Map -----------------------------------------------
            logger.info("[%s] Step 1/7: checking current map", scenario_id)
            try:
                current_map = self._world.get_map().name.split("/")[-1]
            except RuntimeError:
                logger.warning("[%s] World handle stale; reconnecting.", scenario_id)
                self.connect()
                current_map = self._world.get_map().name.split("/")[-1]

            if current_map != spec.town:
                logger.info("[%s] Step 1/7: current map '%s' != '%s' -- reloading",
                            scenario_id, current_map, spec.town)
                pre = self._world.get_settings()
                pre.synchronous_mode = False
                self._world.apply_settings(pre)

                logger.info("[%s] Loading map %s (up to 180s on constrained hardware).",
                            scenario_id, spec.town)
                self._client.set_timeout(180.0)
                self._world = self._client.load_world(spec.town)
                time.sleep(2)
                self._world = self._client.get_world()
                self._client.set_timeout(self.timeout)

                settings = self._world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.profile.fixed_delta_seconds
                self._world.apply_settings(settings)

                # Re-acquire TM after map reload — the old handle is stale
                self._tm = self._client.get_trafficmanager(TM_PORT)
                self._tm.set_synchronous_mode(True)

                logger.info("[%s] Step 1/7: map reload complete, synchronous mode re-applied",
                            scenario_id)
            else:
                logger.info("[%s] Step 1/7: map already correct, no reload needed", scenario_id)

            state_log["map_loaded"] = spec.town

            # ---- Step 2: Weather -------------------------------------------
            logger.info("[%s] Step 2/7: applying weather '%s'", scenario_id, spec.weather)
            self._world.set_weather(carla.WeatherParameters(**WEATHER_PRESETS[spec.weather]))
            state_log["weather_api_called"] = spec.weather

            # ---- Step 3: Blueprint library & spawn points ------------------
            logger.info("[%s] Step 3/7: fetching blueprint library and spawn points", scenario_id)
            blueprint_library = self._world.get_blueprint_library()
            spawn_points = self._world.get_map().get_spawn_points()

            # ---- Step 4: Ego vehicle ---------------------------------------
            logger.info("[%s] Step 4/7: spawning ego vehicle", scenario_id)
            ego_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
            ego_spawn = self._safe_spawn_point(spawn_points, spec.spawn_point_index)
            ego_vehicle = self._world.try_spawn_actor(ego_bp, ego_spawn)
            if ego_vehicle is None:
                raise RuntimeError(
                    f"Failed to spawn ego vehicle at spawn_point_index={spec.spawn_point_index}. "
                    "Spawn point may be occupied — try a different index."
                )
            state_log["actors"].append({"type": "ego_vehicle", "id": ego_vehicle.id})

            # Fix 2a: tick BEFORE set_autopilot so the TM has the actor registered
            self._world.tick()
            # Fix 1: pass TM_PORT to set_autopilot so it uses our TM, not port 8000
            ego_vehicle.set_autopilot(True, TM_PORT)

            # ---- Step 5: Camera --------------------------------------------
            logger.info("[%s] Step 5/7: spawning camera (%dx%d)", scenario_id,
                        self.profile.camera_width, self.profile.camera_height)
            camera_bp = blueprint_library.find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", str(self.profile.camera_width))
            camera_bp.set_attribute("image_size_y", str(self.profile.camera_height))
            camera_transform = carla.Transform(
                carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)
            )
            camera = self._world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

            captured_frames: dict[int, carla.Image] = {}

            def _on_image(image):
                if not callback_active:
                    return
                try:
                    captured_frames[image.frame] = image
                    # Bound memory use on 2GB VRAM hardware
                    if len(captured_frames) > 20:
                        del captured_frames[min(captured_frames.keys())]
                except Exception:
                    pass

            camera.listen(_on_image)

            # ---- Step 6: NPC vehicles --------------------------------------
            logger.info("[%s] Step 6/7: spawning NPC vehicles", scenario_id)
            vehicle_bps = blueprint_library.filter("vehicle.*")
            npc_spawn_pts = [p for i, p in enumerate(spawn_points)
                             if i != spec.spawn_point_index]
            random.shuffle(npc_spawn_pts)
            effective_nveh = min(spec.num_vehicles, self.profile.max_vehicles)
            if effective_nveh < spec.num_vehicles:
                logger.info("Hardware profile caps num_vehicles at %d (spec requested %d)",
                            self.profile.max_vehicles, spec.num_vehicles)

            for i in range(min(effective_nveh, len(npc_spawn_pts))):
                npc = self._world.try_spawn_actor(random.choice(vehicle_bps), npc_spawn_pts[i])
                if npc is not None:
                    npc_list.append(npc)
                    state_log["vehicle_count"] += 1

            logger.info("[%s] Step 6/7: spawned %d/%d requested NPC vehicles",
                        scenario_id, state_log["vehicle_count"], effective_nveh)

            if npc_list:
                # Fix 2b: tick before set_autopilot on NPCs
                self._world.tick()
                for npc in npc_list:
                    try:
                        if npc.is_alive:
                            # Fix 1: pass TM_PORT here too
                            npc.set_autopilot(True, TM_PORT)
                    except RuntimeError as e:
                        logger.warning("[%s] NPC autopilot failed (skipping): %s", scenario_id, e)

            # ---- Step 6b: Pedestrian ---------------------------------------
            if spec.pedestrian:
                logger.info("[%s] Step 6/7: spawning pedestrian", scenario_id)
                walker_bp = random.choice(blueprint_library.filter("walker.pedestrian.*"))

                ego_t = ego_vehicle.get_transform()
                fwd = ego_t.get_forward_vector()
                loc = carla.Location(
                    x=ego_t.location.x + fwd.x * 12.0 + random.uniform(-2.0, 2.0),
                    y=ego_t.location.y + fwd.y * 12.0 + random.uniform(-2.0, 2.0),
                    z=ego_t.location.z + 0.5,
                )
                walker_actor = self._world.try_spawn_actor(walker_bp, carla.Transform(loc))

                if walker_actor is not None:
                    state_log["pedestrian_spawned"] = True
                    state_log["actors"].append({"type": "pedestrian", "id": walker_actor.id})

                    # Fix 2c: tick to register walker BEFORE spawning its controller
                    self._world.tick()

                    ctrl_bp = blueprint_library.find("controller.ai.walker")
                    walker_controller = self._world.spawn_actor(
                        ctrl_bp, carla.Transform(), attach_to=walker_actor
                    )

                    # Fix 2d: tick to register controller BEFORE calling start()
                    # Skipping this tick is the "destroyed actor" core dump cause
                    self._world.tick()

                    try:
                        walker_controller.start()
                        target = self._world.get_random_location_from_navigation()
                        if target is not None:
                            walker_controller.go_to_location(target)
                        walker_controller.set_max_speed(1.4)
                    except RuntimeError as e:
                        logger.warning("[%s] Walker controller failed: %s", scenario_id, e)

                    logger.info("[%s] Step 6/7: pedestrian spawned and walking", scenario_id)
                else:
                    logger.warning("[%s] Step 6/7: pedestrian spawn returned None (location blocked)",
                                   scenario_id)

            # ---- Step 7: Tick loop -----------------------------------------
            logger.info("[%s] Step 7/7: starting tick loop (duration=%ds, tick=%.2fs)",
                        scenario_id, spec.duration_s, self.profile.fixed_delta_seconds)

            ticks = int(spec.duration_s / self.profile.fixed_delta_seconds)
            capture_every = max(
                1, int(self.profile.capture_every_s / self.profile.fixed_delta_seconds)
            )
            saved_frames = []
            execution_failed = False

            for t in range(ticks):
                if not ego_vehicle.is_alive:
                    logger.error("[%s] Ego vehicle destroyed at tick %d; aborting.", scenario_id, t)
                    execution_failed = True
                    break

                try:
                    self._world.tick()
                except RuntimeError as e:
                    logger.error("[%s] Tick failure at tick %d: %s", scenario_id, t, e)
                    execution_failed = True
                    break

                if t % capture_every == 0:
                    frame_id = self._world.get_snapshot().frame
                    # Fix 4: wait up to 0.6s (30 × 20ms) for the camera callback
                    # The original code only waited 20ms — not enough on slow hardware
                    for _ in range(30):
                        if frame_id in captured_frames:
                            break
                        time.sleep(0.02)

                    if frame_id in captured_frames:
                        img = captured_frames[frame_id]
                        fp = frames_dir / f"frame_{len(saved_frames):03d}.png"
                        try:
                            img.save_to_disk(str(fp))
                            saved_frames.append(str(fp))
                        except RuntimeError:
                            pass
                        if live_frame_callback is not None:
                            try:
                                live_frame_callback(img)
                            except Exception as exc:
                                state_log["errors"].append(str(exc))
                                logger.error("Live callback failed: %s", exc)
                    else:
                        logger.warning("[%s] Timeout waiting for frame %d", scenario_id, frame_id)

            state_log["execution_completed"] = not execution_failed
            if execution_failed:
                raise RuntimeError("Tick loop terminated early; see per-tick log entries above.")

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

        except Exception as exc:
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
            # Fix 3: strict teardown order
            # 1. Disable callback, stop camera stream
            callback_active = False
            if camera is not None:
                try:
                    camera.stop()
                except Exception:
                    pass
            time.sleep(0.15)  # let the stream drain

            # 2. Stop + destroy walker controller before walker
            if walker_controller is not None:
                try:
                    if walker_controller.is_alive:
                        walker_controller.stop()
                except Exception:
                    pass
                self._safe_destroy(walker_controller)

            # 3. Destroy walker after its controller
            self._safe_destroy(walker_actor)

            # 4. Destroy NPC vehicles
            for npc in npc_list:
                self._safe_destroy(npc)

            # 5. Destroy camera before ego (camera is attached to ego;
            #    ego.destroy() auto-destroys attached actors, making our
            #    explicit camera.destroy() hit a dead actor otherwise)
            self._safe_destroy(camera)

            # 6. Ego last
            self._safe_destroy(ego_vehicle)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _safe_spawn_point(self, spawn_points, index: int):
        if 0 <= index < len(spawn_points):
            return spawn_points[index]
        for fallback in FALLBACK_SPAWN_INDICES:
            if fallback < len(spawn_points):
                logger.warning("spawn_point_index %s invalid, falling back to %s", index, fallback)
                return spawn_points[fallback]
        return spawn_points[0]