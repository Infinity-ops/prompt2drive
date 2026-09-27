"""
tools/carla_load_matrix.py

Finds out WHAT makes client.load_world() segfault, instead of guessing.
Each case runs in its own subprocess (a segfault only kills that case) and
switches to the other town, with no vehicles, sensors or LLM involved:

  A  load_world only                       (no Traffic Manager in the process)
  B  TM created, synchronous -> load_world (the ORIGINAL Engine sequence)
  C  TM created, then TM+world async -> load_world (the current Engine sequence)
  D  TM created, then tm.shut_down() -> load_world

It first prints the client (pip) and server CARLA versions: a mismatch
between them is itself a classic cause of libcarla segfaults.

    python tools/carla_load_matrix.py
"""
import subprocess
import sys
import time

HOST, PORT, TM_PORT = "localhost", 2000, 8100


def other_town(client):
    cur = client.get_world().get_map().name.split("/")[-1]
    return "Town02" if cur == "Town01" else "Town01"


def run_case(case: str) -> None:
    import carla

    client = carla.Client(HOST, PORT)
    client.set_timeout(60.0)
    world = client.get_world()
    target = other_town(client)

    if case in "BCD":
        s = world.get_settings()
        s.synchronous_mode, s.fixed_delta_seconds = True, 0.1
        world.apply_settings(s)
        tm = client.get_trafficmanager(TM_PORT)
        tm.set_synchronous_mode(True)
        world.tick()
    if case == "C":
        tm.set_synchronous_mode(False)
        s = world.get_settings()
        s.synchronous_mode, s.fixed_delta_seconds = False, None
        world.apply_settings(s)
    if case == "D":
        tm.set_synchronous_mode(False)
        s = world.get_settings()
        s.synchronous_mode, s.fixed_delta_seconds = False, None
        world.apply_settings(s)
        tm.shut_down()

    client.set_timeout(180.0)
    world = client.load_world(target)
    world.wait_for_tick()
    print(f"loaded {target}", flush=True)


def server_alive() -> bool:
    import carla
    try:
        c = carla.Client(HOST, PORT)
        c.set_timeout(5.0)
        c.get_server_version()
        return True
    except RuntimeError:
        return False


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        run_case(sys.argv[2])
        return

    import carla
    c = carla.Client(HOST, PORT)
    c.set_timeout(10.0)
    cv, sv = c.get_client_version(), c.get_server_version()
    print(f"client (pip) CARLA: {cv}\nserver CARLA:       {sv}")
    if cv.split("-")[0] != sv.split("-")[0]:
        print("!! VERSION MISMATCH: install the client wheel that ships with this server "
              "(CARLA_0.9.16/PythonAPI/carla/dist/) -- this alone can explain the segfaults.")

    for case, label in [("A", "load_world only, no TM"),
                        ("B", "TM synchronous during load (original)"),
                        ("C", "TM + world async before load (current)"),
                        ("D", "TM shut_down() before load")]:
        p = subprocess.run([sys.executable, "-X", "faulthandler", __file__, "--case", case],
                           capture_output=True, text=True, timeout=400)
        if p.returncode == 0:
            result = "PASS"
        elif p.returncode in (-11, 139):
            result = "SEGFAULT (client)"
        else:
            last = (p.stderr.strip().splitlines() or ["?"])[-1]
            result = f"FAIL rc={p.returncode}: {last[:120]}"
        alive = server_alive()
        print(f"[matrix] {case}  {label:<42} {result}   server alive: {alive}", flush=True)
        if not alive:
            print("[matrix] CARLA server is down; restart it and rerun to test the remaining cases.")
            break
        time.sleep(3)


if __name__ == "__main__":
    main()
