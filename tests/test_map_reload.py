"""Order of operations in CarlaRunner._reload_map, checked with a fake carla module."""
import sys
import types

from engine.carla_runner import CarlaRunner


class _Log(list):
    def add(self, x):
        self.append(x)


def _fake_env(log):
    carla = types.ModuleType("carla")
    carla.command = types.SimpleNamespace(DestroyActor=lambda i: ("destroy", i))

    class Settings:
        synchronous_mode = True
        fixed_delta_seconds = 0.1

    class Actor:
        def __init__(self, i, t):
            self.id, self.type_id, self.is_alive = i, t, True

        def set_autopilot(self, on, port):
            log.add(f"autopilot {self.type_id}={on}")

    class Actors(list):
        def filter(self, pat):
            return [a for a in self if a.type_id.startswith(pat.rstrip("*"))]

    class World:
        def get_actors(self):
            return Actors([Actor(1, "sensor.camera.rgb"), Actor(2, "vehicle.tesla.model3"),
                           Actor(3, "walker.pedestrian.0001"), Actor(4, "traffic.traffic_light")])

        def get_settings(self):
            return Settings()

        def apply_settings(self, s):
            log.add(f"world sync={s.synchronous_mode}")

        def wait_for_tick(self):
            log.add("wait_for_tick")

        def tick(self):
            log.add("tick")

    class TM:
        def set_synchronous_mode(self, v):
            log.add(f"tm sync={v}")

        def shut_down(self):
            log.add("tm shut_down")

    class Client:
        def set_timeout(self, t):
            pass

        def apply_batch_sync(self, cmds, do_tick):
            log.add(f"destroy {sorted(i for _, i in cmds)}")

        def load_world(self, town):
            log.add(f"load {town}")
            return World()

        def get_trafficmanager(self, port):
            return TM()

    return carla, World(), TM(), Client()


def test_reload_shuts_down_tm_before_load_and_restores_sync(monkeypatch):
    log = _Log()
    carla, world, tm, client = _fake_env(log)
    monkeypatch.setitem(sys.modules, "carla", carla)
    monkeypatch.setattr("engine.carla_runner.time.sleep", lambda s: None)
    r = CarlaRunner(hardware_profile="low_vram_2gb")
    r._world, r._tm, r._client = world, tm, client
    r._reload_map("Town02", "t")
    assert log == [
        "autopilot vehicle.tesla.model3=False", "tick",   # unregister from TM first
        "tm sync=False", "world sync=False",
        "destroy [1, 2, 3]",            # traffic lights are map actors, never destroyed
        "tm shut_down",                 # no Traffic Manager may exist during the load
        "load Town02", "wait_for_tick",
        "world sync=True", "tm sync=True", "tick",
    ]
