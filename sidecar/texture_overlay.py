"""Essence-as-material texture overlay — a second, non-diffusion apply
"engine" alongside the SDXL/IP-Adapter/ControlNet restyling path
(apply_essence in generation.py). Composites the essence reference's own
surface relief onto the target photo via bump/emboss shading, keeping the
target's content, composition, and every fine detail completely untouched.

Why a second engine, not a slider on the existing one: reported directly
against a real attempt — dropping a pure-texture reference (a glass-surface
photo, no real "brushwork" or palette to speak of) through the normal
diffusion path produced a rougher, content-drifted result, because that
path *regenerates* the whole image from scratch (guided by structure +
style), which is the wrong operation for "keep this photo exactly as it is,
just print it onto/behind this material." No amount of tuning `strength`/
`controlnet_scale` closes that gap — diffusion is stochastic-generative by
construction and can't guarantee the pixel fidelity a non-generative filter
gives for free. A classical bump-map overlay is the right tool for
literally-a-material essences (glass, canvas, paper grain, fabric weave);
the diffusion path remains the right tool for genuinely painterly ones.

Reuses relight.py's Gradient Domain Rendering shading (render_diffuse_frame
— Wang, Gonen & Akleman, see relight.py's own module docstring for the
full citation) directly, but *not* its depth_to_normal_field: that
function's gradient scaling (GRADIENT_SENSITIVITY) is tuned for MiDaS
depth maps specifically, which have a fairly consistent gradient-magnitude
distribution scene to scene. A material reference photo's contrast varies
enormously by source — a bold high-contrast pattern and a soft, subtly
bumpy surface (e.g. glass under gentle light) need very different absolute
scaling to read as the *same* relief strength once applied. Reusing a
fixed sensitivity meant a soft/low-contrast reference came out barely
visible even at full strength (reported directly: "maxed the overlay but I
see not much of overlay" — measured ~4x weaker gradient signal than a
higher-contrast reference at the same settings). _height_to_normal_field
below auto-normalizes each texture's own gradient magnitude to its own
95th-percentile before scaling, so relief strength reads consistently
regardless of the source photo's absolute contrast.

What differs from relight.py's own use of render_diffuse_frame: that
module blends between a darkened copy and the plain original (dark..lit),
appropriate for a GIF frame that should never look brighter than the
source. A single static overlay wants the opposite property — a texture
with *no* relief anywhere (a flat field, cd == 0.5 everywhere) must leave
the target completely unchanged, not uniformly dimmed. So here the two
control images are a symmetric darkened/brightened pair around the
target's own brightness (image * (1 - relief_strength) .. image * (1 +
relief_strength)); at cd == 0.5 that blend reproduces the target exactly,
and only actual relief (cd departing from 0.5) pushes brightness up or
down from there — the essence's bump pattern reading as physically molded
onto the target, with zero effect where the essence itself is flat.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from relight import render_diffuse_frame

# -3*pi/4: classic top-left bevel/emboss light angle, the same convention
# most image editors default to for a "texturizer"/emboss filter.
DEFAULT_LIGHT_ANGLE = -2.356194490192345
# Symmetric +/- brightness swing (as a fraction of the target's own
# brightness) at full relief; gentler than relight.py's GIF darken factor
# since this is meant to read as a subtle material relief, not a dramatic
# light sweep. Meaningful now that _height_to_normal_field auto-normalizes
# per texture -- before that fix this number couldn't be trusted to mean
# the same thing for two different reference photos.
DEFAULT_RELIEF_STRENGTH = 0.35
DEFAULT_FLATTEN = 0.85
# Percentile (not the raw max) used to normalize gradient magnitude -- a
# handful of noisy/outlier pixels (e.g. a single sharp speck) shouldn't
# compress the rest of the field toward zero the way normalizing by the
# absolute max would.
GRADIENT_PERCENTILE = 92


def _height_to_normal_field(height_image: Image.Image, flatten: float = DEFAULT_FLATTEN) -> np.ndarray:
    """Like relight.py's depth_to_normal_field, but self-normalizing — see
    module docstring for why a fixed sensitivity constant doesn't work
    across arbitrarily different reference photos the way it does for a
    consistent family of depth maps.
    """
    height = np.asarray(height_image.convert("L"), dtype=np.float64) / 255.0
    gx = cv2.Sobel(height, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(height, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    # Percentile of the *nonzero* magnitudes only -- a texture that's
    # mostly flat with relief concentrated in a small area (a sparse
    # pattern on an otherwise smooth material, or the synthetic hard-edge
    # test fixtures) would otherwise pull a percentile computed over every
    # pixel down toward zero, discarding the very relief being measured.
    nonzero = magnitude[magnitude > 1e-6]
    if nonzero.size == 0:
        return np.zeros((*height.shape, 2))
    scale = np.percentile(nonzero, GRADIENT_PERCENTILE)
    if scale < 1e-6:
        return np.zeros((*height.shape, 2))
    x = np.clip(-gx / scale, -1.0, 1.0) * flatten
    y = np.clip(-gy / scale, -1.0, 1.0) * flatten
    return np.stack([x, y], axis=-1)


def _tile_to_size(texture: Image.Image, width: int, height: int) -> Image.Image:
    """Repeats `texture` as a tiled pattern to cover (width, height). A
    physical material (glass, canvas, paper) is a repeating swatch, not a
    single shape to be stretched — stretching a small reference across a
    much larger target would blur its fine relief into mush, the opposite
    of what a material overlay needs.
    """
    tex = texture.convert("L")
    tw, th = tex.size
    tiles_x = -(-width // tw)  # ceil division
    tiles_y = -(-height // th)
    canvas = Image.new("L", (tw * tiles_x, th * tiles_y))
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            canvas.paste(tex, (tx * tw, ty * th))
    return canvas.crop((0, 0, width, height))


def apply_texture_overlay(
    image: Image.Image,
    texture: Image.Image,
    amount: float = 1.0,
    light_angle: float = DEFAULT_LIGHT_ANGLE,
    relief_strength: float = DEFAULT_RELIEF_STRENGTH,
) -> Image.Image:
    """Returns a copy of `image` with `texture`'s own surface relief
    embossed onto it. `amount` (0..1) blends between the untouched original
    (0.0) and the full effect (1.0); 0.0 is a no-op, returned as a plain
    copy, same convention as color_transfer.py/stroke_texture.py's own
    zero-amount behavior. A texture with no measurable relief (a flat
    field) is also effectively a no-op regardless of amount — see module
    docstring.
    """
    amount = min(1.0, max(0.0, amount))
    image = image.convert("RGB")
    if amount == 0.0:
        return image.copy()

    w, h = image.size
    tiled = _tile_to_size(texture, w, h)
    vector_field = _height_to_normal_field(tiled)

    lit = np.asarray(image, dtype=np.float64)
    bright = lit * (1.0 + relief_strength)
    dark = lit * (1.0 - relief_strength)
    shaded = render_diffuse_frame(vector_field, dark, bright, light_angle)

    out = lit + (shaded - lit) * amount
    return Image.fromarray(np.clip(out, 0, 255).round().astype(np.uint8), mode="RGB")
