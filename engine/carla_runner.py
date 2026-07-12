"""
engine/carla_runner.py

Deterministic execution engine for Prompt-to-Drive.
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

TM_PORT = 8100

WEATHER_PRESETS = {
    "clear":  dict(cloudiness=10,  precipitation=0,  precipitation_deposits=0, fog_density=0,  sun_altitude_angle=70),
    "rain":   dict(cloudiness=80,  precipitation=80, precipitation_deposits=50, fog_density=0,  sun_altitude_angle=40),
    "fog":    dict(cloudiness=60,  precipitation=0,  precipitation_deposits=0, fog_density=60, fog_distance=15,  sun_altitude_angle=50),
    "night":  dict(cloudiness=20,  precipitation=0,  precipitation_deposits=0, fog_density=0,  sun_altitude_angle=-30),
    "cloudy": dict(cloudiness=70,  precipitation=0,  precipitation_deposits=0, fog_density=0,  sun_altitude_angle=55),
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
        self._tm = None 

    def connect(self):
        import carla
        self._client = carla.Client(self.host, self.port)
        self._client.set_timeout(self.timeout)
        self._world = self._client.get_world()

        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.profile.fixed_delta_seconds
        self._world.apply_settings(settings)

        self._tm = self._client.get_trafficmanager(TM_PORT)
        self._tm.set_synchronous_mode(True)

        logger.info("Connected to CARLA at %s:%s [hardware_profile=%s, TM_PORT=%d]",
                    self.host, self.port, self.profile.name, TM_PORT)

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

            self._tm = self._client.get_trafficmanager(TM_PORT)
            self._tm.set_synchronous_mode(True)

            # ---- Step 1: Map Reload / Clean Slate ----
            try:
                current_map = self._world.get_map().name.split("/")[-1]
            except RuntimeError:
                logger.warning("[%s] World handle stale; reconnecting.", scenario_id)
                self.connect()
                current_map = self._world.get_map().name.split("/")[-1]

            if current_map != spec.town:
                logger.info("[%s] Map mismatch. Forcing clean map reload.", scenario_id)
                pre = self._world.get_settings()
                pre.synchronous_mode = False
                self._world.apply_settings(pre)
                
                for actor in self._world.get_actors().filter('*sensor*'):
                    try:
                        if actor.is_alive:
                            actor.stop()
                            actor.destroy()
                    except Exception: pass
                for actor in self._world.get_actors().filter('*vehicle*'): self._safe_destroy(actor)
                for actor in self._world.get_actors().filter('*walker*'): self._safe_destroy(actor)
                
                time.sleep(2.0)
                self._client.set_timeout(180.0)
                self._world = self._client.load_world(spec.town)
                time.sleep(2.0)
                self._world = self._client.get_world()
                self._client.set_timeout(self.timeout)

                settings = self._world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.profile.fixed_delta_seconds
                self._world.apply_settings(settings)
            else:
                all_actors = self._world.get_actors()
                for actor in all_actors:
                    if any(x in actor.type_id for x in ['vehicle.', 'walker.', 'sensor.']):
                        try:
                            if 'sensor.' in actor.type_id:
                                actor.stop()
                            actor.destroy()
                        except Exception: pass
                self._world.tick()
                time.sleep(0.5)

            state_log["map_loaded"] = spec.town

            # ---- Step 2: Weather Overrides ----
            weather_params = WEATHER_PRESETS[spec.weather].copy()
            if hasattr(spec, 'fog_density') and spec.fog_density is not None:
                weather_params['fog_density'] = spec.fog_density
            if hasattr(spec, 'fog_distance') and spec.fog_distance is not None:
                weather_params['fog_distance'] = spec.fog_distance

            logger.info("[%s] Applying weather '%s' with adjustments: %s", scenario_id, spec.weather, weather_params)
            self._world.set_weather(carla.WeatherParameters(**weather_params))
            state_log["weather_api_called"] = spec.weather

            # ---- Step 3: Blueprint Library ----
            blueprint_library = self._world.get_blueprint_library()
            spawn_points = self._world.get_map().get_spawn_points()

            # ---- Step 4: Ego Vehicle ----
            ego_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
            ego_spawn = self._safe_spawn_point(spawn_points, spec.spawn_point_index)
            ego_vehicle = self._world.try_spawn_actor(ego_bp, ego_spawn)
            if ego_vehicle is None:
                raise RuntimeError(f"Failed to spawn ego vehicle at index {spec.spawn_point_index}")
            
            state_log["actors"].append({"type": "ego_vehicle", "id": ego_vehicle.id})
            self._world.tick()
            ego_vehicle.set_autopilot(True, TM_PORT)
            
            # SAFEGUARD: Ensure Traffic Manager drives at standard operational speed limits
            self._tm.set_desired_speed(ego_vehicle, 30.0) 

            # ---- Step 5: Camera Setup ----
            camera_bp = blueprint_library.find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", str(self.profile.camera_width))
            camera_bp.set_attribute("image_size_y", str(self.profile.camera_height))
            camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
            camera = self._world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

            captured_frames: dict[int, carla.Image] = {}
            def _on_image(image):
                if not callback_active: return
                try:
                    captured_frames[image.frame] = image
                    if len(captured_frames) > 20:
                        del captured_frames[min(captured_frames.keys())]
                except Exception: pass
            camera.listen(_on_image)

            # ---- Step 6: Spawn Background NPC Vehicles ----
            logger.info("[%s] Spawning %d baseline NPC vehicles...", scenario_id, spec.num_vehicles)
            vehicle_bps = blueprint_library.filter("vehicle.*")
            available_spawns = [sp for i, sp in enumerate(spawn_points) if i != spec.spawn_point_index]
            random.shuffle(available_spawns)

            spawned_count = 0
            for sp in available_spawns:
                if spawned_count >= spec.num_vehicles:
                    break
                bp = random.choice(vehicle_bps)
                if bp.has_attribute('color'):
                    color = random.choice(bp.get_attribute('color').recommend_values)
                    bp.set_attribute('color', color)
                
                npc_v = self._world.try_spawn_actor(bp, sp)
                if npc_v is not None:
                    npc_v.set_autopilot(True, TM_PORT)
                    npc_list.append(npc_v)
                    state_log["actors"].append({"type": "npc_vehicle", "id": npc_v.id})
                    spawned_count += 1

            # ---- Step 6b: Handle Stress Blocking Actor Occlusion ----
            if hasattr(spec, 'blocking_actor') and spec.blocking_actor:
                occlusion_dist = getattr(spec, 'occlusion_distance', 8.0)
                logger.info("[%s] Spawning stress blocking actor (occlusion distance: %sm)", scenario_id, occlusion_dist)
                
                ego_t = ego_vehicle.get_transform()
                fwd = ego_t.get_forward_vector()
                blocker_bp = blueprint_library.filter("vehicle.carlamotors.carlacola")[0]
                blocker_loc = carla.Location(
                    x=ego_t.location.x + fwd.x * occlusion_dist,
                    y=ego_t.location.y + fwd.y * occlusion_dist,
                    z=ego_t.location.z + 0.5
                )
                blocker_v = self._world.try_spawn_actor(blocker_bp, carla.Transform(blocker_loc, ego_t.rotation))
                if blocker_v is not None:
                    blocker_v.set_simulate_physics(True)
                    npc_list.append(blocker_v)
                    state_log["actors"].append({"type": "blocking_truck", "id": blocker_v.id})
                    
                    # Prevent Traffic Manager from stopping completely before encountering the visibility occlusion point
                    self._tm.ignore_vehicles_percentage(ego_vehicle, 100.0)

            state_log["vehicle_count"] = len(npc_list)

            # ---- Step 6c: Pedestrian Logic ----
            if spec.pedestrian:
                logger.info("[%s] Spawning pedestrian (No-Nav Velocity Mode)", scenario_id)
                self._world.tick()
                try:
                    walker_bp = random.choice(blueprint_library.filter("walker.pedestrian.*"))
                    ego_t = ego_vehicle.get_transform()
                    fwd = ego_t.get_forward_vector()
                    right = ego_t.get_right_vector()
                    
                    spawn_loc = carla.Location(
                        x=ego_t.location.x + fwd.x * 12.0 + right.x * 2.5,
                        y=ego_t.location.y + fwd.y * 12.0 + right.y * 2.5,
                        z=ego_t.location.z + 1.2
                    )
                    spawn_rot = carla.Rotation(pitch=0.0, yaw=ego_t.rotation.yaw + 180.0, roll=0.0)
                    walker_actor = self._world.try_spawn_actor(walker_bp, carla.Transform(spawn_loc, spawn_rot))

                    if walker_actor is not None:
                        state_log["pedestrian_spawned"] = True
                        self._world.tick()
                        cross_velocity = carla.Vector3D(x=-fwd.x * 1.5 - right.x * 2.0, y=-fwd.y * 1.5 - right.y * 2.0, z=0.0)
                        walker_actor.set_target_velocity(cross_velocity)
                except Exception as e:
                    logger.error("[%s] Unexpected exception during pedestrian sequence: %s", scenario_id, e)

            # ---- Step 7: Simulation Tick Loop ----
            logger.info("[%s] Starting tick loop execution", scenario_id)
            ticks = int(spec.duration_s / self.profile.fixed_delta_seconds)
            capture_every = max(1, int(self.profile.capture_every_s / self.profile.fixed_delta_seconds))
            saved_frames = []
            execution_failed = False

            for t in range(ticks):
                if not ego_vehicle.is_alive:
                    execution_failed = True
                    break
                try:
                    self._world.tick()
                except RuntimeError:
                    execution_failed = True
                    break

                if t % capture_every == 0:
                    frame_id = self._world.get_snapshot().frame
                    for _ in range(30):
                        if frame_id in captured_frames: break
                        time.sleep(0.02)

                    if frame_id in captured_frames:
                        img = captured_frames[frame_id]
                        fp = frames_dir / f"frame_{len(saved_frames):03d}.png"
                        try:
                            img.save_to_disk(str(fp))
                            saved_frames.append(str(fp))
                        except RuntimeError: pass
                        if live_frame_callback is not None:
                            try: live_frame_callback(img)
                            except Exception: pass

            state_log["execution_completed"] = not execution_failed
            log_path = out / "state_log.json"
            with open(log_path, "w") as f:
                json.dump(state_log, f, indent=2)

            return ExecutionResult(
                scenario_id=scenario_id,
                success=not execution_failed,
                frame_paths=saved_frames,
                log_path=str(log_path),
                duration_s=time.time() - t0,
            )

        except Exception as exc:
            state_log["errors"].append(str(exc))
            log_path = out / "state_log.json"
            with open(log_path, "w") as f:
                json.dump(state_log, f, indent=2)
            return ExecutionResult(
                scenario_id=scenario_id, success=False, log_path=str(log_path),
                error=str(exc), duration_s=time.time() - t0,
            )

        finally:
            callback_active = False
            if camera is not None:
                try: camera.stop()
                except Exception: pass
            time.sleep(0.15) 

            if walker_controller is not None:
                try:
                    if walker_controller.is_alive: walker_controller.stop()
                except Exception: pass
                self._safe_destroy(walker_controller)

            self._safe_destroy(walker_actor)
            for npc in npc_list: self._safe_destroy(npc)
            self._safe_destroy(camera)
            self._safe_destroy(ego_vehicle)

            if self._world is not None:
                try: self._world.tick()
                except Exception: pass