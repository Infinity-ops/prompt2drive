"""
tools/carla_reload_test.py

Exercises the REAL Engine (engine/carla_runner.py) with fixed ScenarioSpecs
that alternate between Town01 and Town02, so every run forces a map reload.
No LLM, no vLLM -- only CARLA. This reproduces the crash seen after
"Map mismatch. Forcing clean map reload." and verifies the fix.

    python tools/carla_reload_test.py            # 6 runs, 5 map switches
    python tools/carla_reload_test.py --runs 10
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.carla_runner import CarlaRunner  # noqa: E402
from engine.scenario_spec import ScenarioSpec  # noqa: E402

SPECS = [
    dict(weather="rain", town="Town01", num_vehicles=3, pedestrian=True, event_type="crossing",
         duration_s=12, spawn_point_index=6),
    dict(weather="fog", town="Town02", num_vehicles=3, pedestrian=False, event_type="red_light_violation",
         duration_s=12, spawn_point_index=8),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--profile", default="low_vram_2gb")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    runner = CarlaRunner(hardware_profile=args.profile)
    for i in range(args.runs):
        spec = ScenarioSpec(**SPECS[i % 2])
        res = runner.run(spec, scenario_id=f"reload_test_{i}", output_dir=f"outputs/reload_test/run_{i}")
        print(f"[reload-test] run {i + 1}/{args.runs} {spec.town}: success={res.success} "
              f"frames={len(res.frame_paths)} error={res.error}", flush=True)
        if not res.success:
            sys.exit(1)
    print(f"[reload-test] PASS: {args.runs} runs, {args.runs - 1} map switches, server still up")


if __name__ == "__main__":
    main()
