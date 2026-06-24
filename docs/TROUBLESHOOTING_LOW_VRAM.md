# Troubleshooting: CARLA segfault on low-VRAM GPUs (Quadro M620 / Dell Precision 3530)

## Why this happens

The Quadro M620 is a Maxwell-generation mobile GPU with **2GB of dedicated VRAM**.
CARLA's default settings (Epic-quality rendering, Vulkan renderer, 1920x1080+
resolution, dozens of rendered actors) routinely exceed what 2GB of VRAM can
hold, and the engine segfaults rather than degrading gracefully when a VRAM
allocation fails. This is a known class of failure across CARLA versions on
GPUs below roughly 4GB VRAM, and gets worse on 0.9.14+ as default scene
complexity has grown.

This is **not a bug in Prompt-to-Drive's Python code** — it crashes inside the
CARLA server binary (`CarlaUE4.sh` / `CarlaUE4.exe`) itself, before your Python
client even finishes connecting in many cases. Fixing it means changing how
the CARLA *server* is launched, plus capping what Prompt-to-Drive asks it to
render.

## Step 1 — Launch the CARLA server with low-VRAM flags

```bash
./CarlaUE4.sh -quality-level=Low -ResX=800 -ResY=600 -opengl -RenderOffScreen
```

What each flag does and why it matters on this GPU:

| Flag | Effect | Why it matters here |
|---|---|---|
| `-quality-level=Low` | Disables most post-processing, shadows, reflections | This is what actually eats VRAM — far more than resolution alone |
| `-ResX=800 -ResY=600` | Caps the server's own render resolution | Default is often 1920x1080+, which a 2GB card cannot comfortably double-buffer alongside the scene |
| `-opengl` | Forces the OpenGL renderer instead of Vulkan | **This is usually the single most important flag.** The M620's Vulkan driver support is the most common segfault trigger on this exact GPU class. OpenGL is older but far more stable on Maxwell mobile parts. |
| `-RenderOffScreen` | Disables the spectator window | Saves additional VRAM and compositor overhead; you don't need to see the CARLA window for automated evaluation runs |

**Try `-opengl` first, on its own, if you want to isolate the fix.** In most
reports of this failure mode on similar Maxwell mobile GPUs, switching the
renderer from Vulkan to OpenGL alone resolves the segfault.

## Step 2 — Check your NVIDIA driver version

```bash
nvidia-smi
```

Maxwell-generation mobile GPUs (M-series Quadro) are on NVIDIA's legacy driver
branch. If `nvidia-smi` reports a driver version above ~470.x on Linux, some
Vulkan extensions CARLA's newer Unreal Engine builds expect may not be fully
supported — another reason `-opengl` is the safer choice on this hardware
rather than chasing driver updates that may not exist for this GPU generation.

## Step 3 — Use the low_vram_2gb hardware profile in Prompt-to-Drive

This repository ships a hardware profile tuned for exactly this GPU class.
It does NOT fix server-side segfaults by itself (Step 1 does that) — it
reduces what the Python client *asks* the server to render, which matters
once the server is stable and you're trying to avoid frame-capture timeouts
or slow ticks during longer evaluation runs.

```bash
python run_scenario.py \
  --prompt "..." \
  --hardware-profile low_vram_2gb \
  --backend huggingface
```

or for the full evaluation sweep:

```bash
python -m eval.run_full_evaluation \
  --dataset scenarios/dataset.json \
  --mode full \
  --hardware-profile low_vram_2gb \
  --out outputs/full
```

What this profile changes (`engine/hardware_profiles.py`):

| Setting | Default | low_vram_2gb | Why |
|---|---|---|---|
| Camera resolution | 960x540 | 480x270 | Smaller frame buffer = less VRAM per camera sensor |
| Max NPC vehicles | up to 20 (per scenario) | capped at 6 | Each rendered vehicle adds geometry + physics load |
| Simulation tick rate | 20 Hz (0.05s) | 10 Hz (0.1s) | Halves the render-and-physics workload per second |
| Frame capture interval | every 2.0s | every 3.0s | Fewer disk writes / encode operations during the run |
| Preferred towns | all 5 | Town01, Town02 | These are CARLA's smallest, least geometry-dense maps |

## Step 4 — If it still crashes

1. **Reduce further**: try `-quality-level=Low -ResX=640 -ResY=480` and the
   `headless_ci` profile (even smaller footprint than `low_vram_2gb`).
2. **Check available VRAM at crash time** with `nvidia-smi` running in a
   second terminal while you launch CARLA — if VRAM usage climbs to within
   ~100-200MB of the 2GB ceiling right before the crash, that confirms VRAM
   exhaustion as the cause (as opposed to a driver/Vulkan issue, which tends
   to crash immediately on launch rather than climbing first).
3. **Try an older CARLA version.** CARLA 0.9.13 has measurably lower baseline
   VRAM requirements than 0.9.14+. If you have flexibility on CARLA version,
   downgrading is a legitimate option for this hardware — note the version
   in your paper's reproducibility section either way, since CARLA version is
   a relevant experimental detail regardless of which you use.
4. **Close every other GPU-using application** (browser hardware acceleration,
   other Electron apps, a second monitor doing GPU compositing) before
   launching CARLA — on a 2GB card, a browser alone can consume 200-400MB,
   which is a meaningful fraction of your budget.

## What this means for your paper

If you end up running your full evaluation under these reduced settings
(lower resolution, fewer NPCs, only Town01/Town02, 10Hz physics), **report
this explicitly in your Experimental Setup section** — it is a legitimate,
disclosed constraint, not something to hide. Something like:

> "Experiments were conducted on a Dell Precision 3530 mobile workstation
> (Intel Core i7-8850H, NVIDIA Quadro M620, 2GB VRAM) using CARLA's
> OpenGL renderer at reduced quality settings (Low quality level, 480×270
> camera resolution, maximum 6 concurrent NPC vehicles, 10Hz simulation
> tick rate) due to VRAM constraints on the available hardware."

This is honest, specific, and — importantly for a CPS/embedded-systems
audience — actually relevant context, since resource constraints are
exactly what CAIESA reviewers care about. A modest-hardware evaluation,
clearly disclosed, is not a weakness to hide; an undisclosed one would be.
