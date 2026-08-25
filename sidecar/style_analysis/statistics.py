"""General style statistics (spec Phase 5) — deliberately small. Just
measurable, deterministic descriptors; no attempt at "abstraction" or any
other concept without a reliable measurable proxy (left as None, per the
spec explicitly allowing that).
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from essence_models import StyleStatistics
from style_analysis._common import analysis_resize, normalize01, to_gray_array

LOCAL_CONTRAST_MAX = 0.09  # mean local std (5x5 window) on a 0..1 image — empirical bound, see style_analysis/_common.py


def analyze_style_statistics(image: Image.Image) -> StyleStatistics:
    img = analysis_resize(image)
    gray = to_gray_array(img)
    gray_u8 = (gray * 255).astype(np.uint8)

    # Edge density: fraction of pixels Canny marks as an edge. Otsu-derived
    # thresholds (via median-based heuristic) rather than fixed constants,
    # so it adapts reasonably across differently-exposed references.
    median = float(np.median(gray_u8))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(gray_u8, lower, upper)
    edge_density = float((edges > 0).mean())

    # Local contrast: mean of local standard deviation over 5x5 windows —
    # via the standard box-filter trick (E[x^2] - E[x]^2), cheaper than a
    # sliding-window std computed directly.
    mean = cv2.boxFilter(gray, ddepth=-1, ksize=(5, 5))
    mean_sq = cv2.boxFilter(gray * gray, ddepth=-1, ksize=(5, 5))
    local_var = np.clip(mean_sq - mean * mean, 0, None)
    local_contrast = normalize01(float(np.sqrt(local_var).mean()), 0.0, LOCAL_CONTRAST_MAX)

    return StyleStatistics(
        abstraction=None,
        edge_density=round(edge_density, 4),
        local_contrast=round(local_contrast, 4),
    )
