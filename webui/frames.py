"""
webui/frames.py

Frame encoding for the browser UI. The JPEG approach (quality ~70, RGB, no
alpha) is carried over from the earlier test_UI simulator, which streamed
CARLA camera frames to a pygame client over a socket. Here the frame goes
over a WebSocket as base64 instead of a custom 1-byte-type + 4-byte-length
TCP framing, so the browser can drop it straight into an <img>.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

JPEG_QUALITY = 70
MAX_WIDTH = 960  # never ship frames wider than this to the browser


def _to_jpeg_b64(img: Image.Image) -> str:
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.width > MAX_WIDTH:
        img = img.resize((MAX_WIDTH, round(img.height * MAX_WIDTH / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def carla_image_to_jpeg_b64(image) -> str:
    """carla.Image (BGRA raw buffer) -> base64 JPEG."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]  # BGRA -> RGB
    return _to_jpeg_b64(Image.fromarray(np.ascontiguousarray(array)))


def file_to_jpeg_b64(path: str | Path) -> str:
    """PNG/JPEG on disk (e.g. outputs/<id>/iter_0/frames/frame_000.png) -> base64 JPEG."""
    with Image.open(path) as img:
        return _to_jpeg_b64(img.copy())
