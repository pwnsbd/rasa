"""Lazy, background-loaded singleton for monocular depth estimation
(Depth Anything V2 Small) — see generation.py's DepthGradientStrategy for
the actual use: a continuous, depth-driven strength/controlnet_scale blend
instead of segmentation.py's binary subject/background mask, for photos
without one clear subject (landscapes, group shots, product shots). This
is the one thing a flat 2D filter has no equivalent of at all — it has no
notion of the photo's actual scene depth to react to.

Mirrors pipeline_manager.py's lazy-singleton pattern (_status dict,
ensure_loading_started, background thread from app.py's lifespan, a
get_*_blocking-style call) — same shape, a much smaller model. No new pip
dependency: `transformers.pipeline("depth-estimation", ...)` and the
depth_anything modeling code are both already present in the pinned
transformers==4.46.3 (confirmed by direct import, not assumed).

Runs on CPU deliberately, not GPU: the SDXL pipeline already runs the
12GB card near its VRAM ceiling (see pipeline_manager.py's
enable_model_cpu_offload, added specifically because of that pressure).
Depth Anything V2 Small is small/fast enough on CPU that it isn't worth
reopening that fight.
"""
from __future__ import annotations

import os
import threading
import time

from PIL import Image, ImageFilter

import paths

# Same guard as pipeline_manager.py, set independently and idempotently
# (via setdefault) so it's correct regardless of which of the two modules
# happens to import transformers/huggingface_hub first.
os.environ.setdefault("HF_HOME", str(paths.models_dir() / "hf-cache"))

_lock = threading.Lock()
_estimator = None
_status: dict = {"state": "idle", "detail": None}  # idle | loading | ready | error

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def status() -> dict:
    return dict(_status)


def _load() -> None:
    global _estimator
    try:
        from transformers import pipeline

        _status.update(state="loading", detail="Downloading/loading depth model (~100MB, first run only)…")
        _estimator = pipeline("depth-estimation", model=DEPTH_MODEL_ID, device=-1)  # CPU — see module docstring
        _status.update(state="ready", detail=None)
    except Exception as e:  # noqa: BLE001 — record the failure so status()/callers can surface it, not crash the sidecar
        _status.update(state="error", detail=str(e))


def ensure_loading_started() -> None:
    with _lock:
        if _status["state"] == "idle":
            _status.update(state="loading", detail="Starting…")
            threading.Thread(target=_load, daemon=True, name="depth-loader").start()


def get_depth_map(image: Image.Image, timeout_s: float = 120.0) -> Image.Image:
    """Blocks the CALLING thread until the depth model is ready, or raises.
    Only call this from a synchronous context — see pipeline_manager.py's
    get_pipeline_blocking for why (Starlette's threadpool).

    Returns an "L" mode PIL image, same size as `image`: the pipeline's own
    postprocessing already min-max normalizes the raw predicted depth to
    0..255, so this needs no separate normalization step. Depth Anything's
    convention: larger value (brighter) = closer to the camera (relative
    inverse depth/disparity, not metric distance — standard for this model
    family, matching its own visualization convention).
    """
    ensure_loading_started()
    deadline = time.time() + timeout_s
    while _status["state"] == "loading":
        if time.time() > deadline:
            raise TimeoutError("Depth model is still loading after the timeout")
        time.sleep(0.2)
    if _status["state"] == "error":
        raise RuntimeError(f"Depth model failed to load: {_status['detail']}")

    result = _estimator(image.convert("RGB"))
    return result["depth"]


def depth_to_alpha_mask(depth_image: Image.Image, invert: bool = False) -> Image.Image:
    """Pure function, no model involved — shapes a depth map (from
    get_depth_map, or a synthetic one in tests) into the smooth mask
    generation.py's DepthGradientStrategy composites two generation passes
    with (same "L" mode convention as segmentation.get_subject_mask's
    return: 255 -> the first/"near" pass, 0 -> the second/"far" pass).

    invert=True flips near/far — the one-line fix if a real run ever shows
    the effect backwards (background staying crisp, foreground stylizing)
    once this gets tested against an actual photo.

    A light Gaussian blur smooths the mask so the near/far transition reads
    as gradual — raw depth is usually already fairly smooth, but this
    avoids a mottled-looking blend at fine edges (hair, foliage), the same
    goal segmentation.py's alpha matting serves for subject cutouts, just
    via a simpler mechanism since there's no hard cutout boundary here to
    feather, only a continuous gradient to soften.
    """
    depth_image = depth_image.convert("L")
    if invert:
        from PIL import ImageOps

        depth_image = ImageOps.invert(depth_image)

    blur_radius = max(1.0, max(depth_image.size) / 100)
    return depth_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
