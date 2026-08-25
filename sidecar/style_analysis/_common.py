"""Shared helpers for style_analysis/*.py."""
from __future__ import annotations

import numpy as np
from PIL import Image

ANALYSIS_MAX_DIM = 384  # long edge, within the spec's suggested 256-512px range


def analysis_resize(image: Image.Image) -> Image.Image:
    """Downscales to the fixed analysis resolution. All four analyzers call
    this first so their outputs are comparable regardless of the source
    image's original size, and so analysis stays cheap independent of
    generation resolution.
    """
    img = image.convert("RGB")
    img.thumbnail((ANALYSIS_MAX_DIM, ANALYSIS_MAX_DIM))
    return img


def to_gray_array(image: Image.Image) -> np.ndarray:
    """float64 grayscale array in 0..1, from an already-resized image.
    float64 specifically — this installed OpenCV build's optimized filter
    path (cv2.Laplacian/Sobel with a float32 source into a CV_64F
    destination) throws "Unsupported combination of source format" on some
    ops; float64 in avoids the mismatch entirely. Analysis images are small
    (384px), so the extra precision costs nothing that matters.
    """
    return np.asarray(image.convert("L"), dtype=np.float64) / 255.0


def normalize01(value: float, lo: float, hi: float) -> float:
    """Clamped linear rescale of `value` from [lo, hi] to [0, 1]. `lo`/`hi`
    are empirical bounds picked per-metric (see each analyzer) from
    inspecting real values on the synthetic test images in sidecar/tests/
    — not derived from a formula, since these are relative descriptors, not
    physically normalized quantities.
    """
    if hi <= lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))
