"""Dominant color palette extraction (spec Phase 2). Classical CV, no
neural model: LAB clustering, not raw RGB, since LAB clusters group
perceptually similar colors rather than clustering on the R/G/B axes'
uneven perceptual weighting.

    image -> resize for analysis -> RGB -> LAB -> k-means -> N clusters
          -> cluster weights + per-cluster average RGB swatch
"""
from __future__ import annotations

import colorsys

import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans2
from skimage.color import rgb2lab

from essence_models import ColorSwatch, PaletteProfile
from style_analysis._common import analysis_resize, normalize01

NUM_CLUSTERS = 6  # within the spec's suggested 5-8 dominant colors
KMEANS_SEED = 0  # deterministic output, per the spec's "keep the extraction deterministic"

# Empirical bounds for normalize01 — see _common.py's docstring on where
# these numbers come from. L*a*b* contrast (std of L*) and warm/cool
# (mean of a*+b*) don't have a natural 0..1 range, so these are picked
# from inspecting real values on the test images in sidecar/tests/.
CONTRAST_L_STD_MAX = 35.0
TEMPERATURE_AXIS_MAX = 40.0  # mean(a*)+mean(b*) rarely exceeds this in practice


def extract_palette(image: Image.Image) -> PaletteProfile:
    img = analysis_resize(image)
    rgb01 = np.asarray(img, dtype=np.float64) / 255.0
    lab = rgb2lab(rgb01)

    h, w, _ = lab.shape
    lab_pixels = lab.reshape(-1, 3)
    rgb_pixels = (rgb01.reshape(-1, 3) * 255).astype(np.uint8)

    rng = np.random.default_rng(KMEANS_SEED)
    # kmeans2 raises if a cluster goes empty under some inits; seeding once
    # and retrying with a fresh seed on that specific failure is simpler
    # than hand-rolling k-means++ retries.
    for attempt in range(5):
        try:
            centroids, labels = kmeans2(lab_pixels, NUM_CLUSTERS, seed=int(rng.integers(0, 2**31)), minit="++")
            break
        except Exception:  # noqa: BLE001 — degenerate init, retry with a new seed
            if attempt == 4:
                raise

    swatches = []
    for cluster_idx in range(NUM_CLUSTERS):
        member_mask = labels == cluster_idx
        count = int(member_mask.sum())
        if count == 0:
            continue
        weight = count / len(labels)
        # Average the *actual* RGB pixels in this cluster for the swatch
        # color, not the LAB centroid converted back to RGB — avoids
        # out-of-gamut LAB->RGB round-trip artifacts.
        avg_rgb = rgb_pixels[member_mask].mean(axis=0).round().astype(int)
        r, g, b = (int(c) for c in avg_rgb)
        swatches.append(ColorSwatch(hex=f"#{r:02X}{g:02X}{b:02X}", rgb=(r, g, b), weight=round(float(weight), 4)))

    swatches.sort(key=lambda s: s.weight, reverse=True)

    l_channel = lab_pixels[:, 0]
    a_channel = lab_pixels[:, 1]
    b_channel = lab_pixels[:, 2]

    # Saturation/luminance via HSV mean, not LAB — matches how a person
    # would describe "how saturated/bright is this image" more directly
    # than LAB's perceptual-uniformity-oriented axes would.
    hsv_pixels = np.array([colorsys.rgb_to_hsv(*p) for p in rgb01.reshape(-1, 3)])
    mean_saturation = float(hsv_pixels[:, 1].mean())
    mean_luminance = float(hsv_pixels[:, 2].mean())

    # Warm/cool: LAB's a* (green-red) and b* (yellow-blue) axes are the
    # standard proxy — positive skews red/yellow (warm), negative skews
    # green/blue (cool).
    temperature_raw = float(a_channel.mean() + b_channel.mean())
    temperature = float(np.clip(temperature_raw / TEMPERATURE_AXIS_MAX, -1.0, 1.0))

    contrast = normalize01(float(l_channel.std()), 0.0, CONTRAST_L_STD_MAX)

    return PaletteProfile(
        dominant_colors=swatches,
        mean_saturation=round(mean_saturation, 4),
        mean_luminance=round(mean_luminance, 4),
        temperature=round(temperature, 4),
        contrast=round(contrast, 4),
    )
