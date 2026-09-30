"""Essence-reactive stroke texture — a generation-time pass that overlays a
faint directional "grain" on the stylized result, oriented per-region
according to the essence's own measured stroke pattern (style_analysis/
stroke.py's per-cell theta/coherence field, persisted at Distillation time
as stroke_field.npz — see essence_store.py's _run_analyzers and
load_stroke_field). Everything else generation.py exposes (intensity,
color_preservation, blend_mode) applies the same adjustment everywhere
regardless of which essence was dropped; this is the first control that
varies spatially with the *specific* essence's own stroke field instead of
reacting to one flat number for the whole photo.

Classical technique, not a new model: a small bank of Gabor kernels (the
standard tool for oriented texture analysis *and* synthesis — same
structure-tensor family of math stroke.py already runs to measure
orientation in the first place) built at a handful of discrete angles,
applied to the result's own luminance, and composited per-pixel using
whichever bin the upsampled field says that pixel's local region belongs
to, weighted by how confidently (coherence) that region was oriented to
begin with. cv2.getGaborKernel's `theta` and stroke.py's `theta` are both
"orientation of the structure-tensor's dominant axis" in the same
convention (verified empirically: filtering random noise with
getGaborKernel(theta=T) and re-measuring it with analyze_stroke_field
round-trips to dominant_angle ~= T) — so a bin's kernel built at that bin's
own angle reproduces stripes oriented the same way the essence's strokes
were measured, no conversion needed.

Not yet validated against a real photo the way color_transfer.py's
preserve_original_color was (that fix was reported against a specific
observed bug) — this is new, so every call site defaults amount to 0.0
(off) until it earns real feedback, same honesty as generation.py's
DepthGradientStrategy defaults.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from skimage.color import lab2rgb, rgb2lab

NUM_BINS = 8  # discrete angle buckets spanning the mod-pi (-pi/2, pi/2] range a stroke axis lives in
KERNEL_SIZE = 9
GABOR_SIGMA = 2.2
GABOR_LAMBDA = 6.0
GABOR_GAMMA = 0.6
# Empirical: keeps the effect a visible grain rather than a wash at
# amount=1.0, tuned by eye against the synthetic test patterns below —
# same "not yet validated against a real photo" caveat as the module
# docstring.
L_BOOST_SCALE = 18.0


def _gabor_bank(num_bins: int = NUM_BINS) -> list[np.ndarray]:
    return [
        cv2.getGaborKernel(
            (KERNEL_SIZE, KERNEL_SIZE), GABOR_SIGMA, b * np.pi / num_bins, GABOR_LAMBDA, GABOR_GAMMA, 0, ktype=cv2.CV_64F
        )
        for b in range(num_bins)
    ]


def apply_stroke_texture(image: Image.Image, theta: np.ndarray, coherence: np.ndarray, amount: float = 1.0) -> Image.Image:
    """Returns a copy of `image` with a faint directional grain overlaid,
    oriented per-region to `theta`/`coherence` (the essence's own raw
    stroke field, at whatever grid resolution style_analysis/stroke.py
    produced it — upsampled here to match `image`'s size). `amount` (0..1)
    scales the effect's strength; 0.0 is a no-op, returned as a plain copy
    to match preserve_original_color's own zero-amount behavior.
    """
    amount = min(1.0, max(0.0, amount))
    if amount == 0.0:
        return image.convert("RGB").copy()

    w, h = image.size
    gray = np.asarray(image.convert("L"), dtype=np.float64) / 255.0

    # Upsample the coarse per-cell field to the image's resolution. theta is
    # mod-pi (an axis, not a direction — see stroke.py's module docstring),
    # so it's resized via its doubled-angle unit-vector components, the same
    # trick analyze_stroke_field's own curvature calculation uses, rather
    # than interpolating the raw angle values directly (which would break
    # at the +-pi/2 wraparound, averaging e.g. +89 deg and -89 deg to ~0
    # instead of the ~90 deg they actually agree on).
    cos2 = cv2.resize(np.cos(2 * theta).astype(np.float64), (w, h), interpolation=cv2.INTER_LINEAR)
    sin2 = cv2.resize(np.sin(2 * theta).astype(np.float64), (w, h), interpolation=cv2.INTER_LINEAR)
    theta_full = 0.5 * np.arctan2(sin2, cos2)
    coherence_full = cv2.resize(coherence.astype(np.float64), (w, h), interpolation=cv2.INTER_LINEAR)

    # _gabor_bank's kernels live at absolute angles [0, pi) (bin b is built
    # at b*pi/num_bins), while theta_full is stroke.py's own (-pi/2, pi/2]
    # convention -- wrap into [0, pi) via mod pi (an axis has no direction,
    # so e.g. -45 deg and 135 deg are the same bin) before bucketing, rather
    # than assuming the two ranges line up directly.
    bins = _gabor_bank()
    bin_width = np.pi / len(bins)
    theta_wrapped = np.mod(theta_full, np.pi)
    bin_idx = np.clip((theta_wrapped / bin_width).astype(int), 0, len(bins) - 1)

    ridge = np.zeros_like(gray)
    for b, kernel in enumerate(bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
        ridge[mask] = filtered[mask]

    # Centered so the effect adds/subtracts grain around zero rather than
    # uniformly brightening or darkening the whole image.
    ridge = ridge - ridge.mean()
    l_boost = ridge * coherence_full * amount * L_BOOST_SCALE

    # Applied in LAB's L channel only, same convention as
    # color_transfer.py's preserve_original_color — a luminance-domain
    # effect that doesn't disturb hue/saturation.
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    lab = rgb2lab(rgb)
    lab[..., 0] = np.clip(lab[..., 0] + l_boost, 0.0, 100.0)
    out = np.clip(lab2rgb(lab), 0.0, 1.0)
    return Image.fromarray((out * 255).round().astype(np.uint8))
