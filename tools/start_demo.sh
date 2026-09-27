#!/usr/bin/env bash
# tools/start_demo.sh -- start CARLA + the web UI in an order that never
# switches maps while a Traffic Manager exists (the cause of the segfaults).
#
#   1. kill any old CARLA and start a fresh server
#   2. load the demo town with a plain client (no Traffic Manager exists yet)
#   3. start the web UI pinned to that town, so no run ever reloads a map
#
# vLLM must already be running (see docs/WEB_UI.md). Usage:
#   tools/start_demo.sh              # Town01
#   TOWN=Town02 tools/start_demo.sh
#   tools/start_demo.sh --threshold 1.0   # extra flags go to webui.server
set -euo pipefail

CARLA_ROOT="${CARLA_ROOT:-$HOME/Documents/Narmada/CARLA_0.9.16}"
TOWN="${TOWN:-Town01}"
PORT="${CARLA_PORT:-2000}"
cd "$(dirname "$0")/.."

echo "[demo] stopping old CARLA processes"
pkill -9 -f CarlaUE4 2>/dev/null || true
sleep 3

echo "[demo] starting CARLA (log: /tmp/carla_server.log)"
( cd "$CARLA_ROOT" && nohup ./CarlaUE4.sh -quality-level=Low -ResX=800 -ResY=600 -RenderOffScreen \
    -carla-rpc-port="$PORT" > /tmp/carla_server.log 2>&1 & )

echo "[demo] waiting for CARLA on port $PORT, then loading $TOWN (no Traffic Manager yet)"
python - "$PORT" "$TOWN" <<'PY'
import sys, time
import carla
port, town = int(sys.argv[1]), sys.argv[2]
deadline = time.time() + 180
while True:
    try:
        c = carla.Client("localhost", port); c.set_timeout(5.0)
        c.get_server_version(); break
    except RuntimeError:
        if time.time() > deadline:
            sys.exit("[demo] CARLA did not come up in 180 s -- see /tmp/carla_server.log")
        time.sleep(3)
c.set_timeout(180.0)
if c.get_world().get_map().name.split("/")[-1] != town:
    c.load_world(town).wait_for_tick()
print(f"[demo] CARLA ready on {town}")
PY

echo "[demo] starting web UI pinned to $TOWN"
exec python -m webui.server --mode live --backend "${BACKEND:-local}" \
    --hardware-profile low_vram_2gb --pin-town "$TOWN" --max-iterations 3 "$@"