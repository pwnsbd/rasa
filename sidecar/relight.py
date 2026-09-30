"""Relight GIF export — animates a generated image with a sweeping light,
producing a looping GIF, from the depth map alone. No new model.

Adapted from "Gradient Domain Rendering" (Wang, Youyou; Gonen, Ozgur;
Akleman, Ergun — Texas A&M, https://people.it.tamu.edu/~ergun/research/2Drendering/):
a classical technique for relighting flat 2D artwork from a 2D vector field
(≈ the x/y channels of a normal map) plus two artist-painted "control
images" (fully-shadowed / fully-lit), diffuse shading a single per-pixel dot
product. The paper itself never animates anything — it's a static-image
relighting technique — but its cheapness (no Poisson solve, explicitly
GPU/shader-friendly, and their own claim that even the most expensive part,
ambient occlusion, "can be computed in real-time") is exactly what makes
sweeping the light direction over many frames a fast way to get a looping
GIF out of one still image.

Two adaptations from the paper, since there's no artist in the loop here:
  - The 2D vector field is derived from the depth map already computed
    elsewhere in this app (depth.py) via its own image gradient — the
    standard, non-neural way to get a normal-map-like field from a height
    field — rather than hand-painted. This is exactly the kind of
    imprecise, not-truly-3D-consistent field the paper says its shading
    math tolerates (see its own w="flattening" parameter, used here too).
  - The two control images (DI0 = fully shadowed, DI1 = fully lit) are
    generated automatically: DI1 is the image itself (already correct),
    DI0 is a darkened version of it — not hand-painted.

v1 scope: diffuse shading only (the paper's Eq. 4 + Eq. 2). Ambient
occlusion and directional shadows are real, described techniques in the
same paper, both legitimate richness for a v2 — not required for a visibly
working sweeping-light effect, and both added real complexity (AO is a
neighborhood convolution; shadows need a directional line-integral
convolution recomputed per light angle).

A full 2π sweep of the light angle across all frames guarantees a perfectly
seamless loop for free — frame 0 and the (nonexistent) frame N are the same
light angle, so there's no jump-cut to hide and no loop-closure blending
step needed.
"""
from __future__ import annotations

import math
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FRAMES = 36
DEFAULT_DURATION_MS = 50
DEFAULT_DARKEN_FACTOR = 0.35  # DI0 = image * this factor
DEFAULT_FLATTEN = 0.5  # the paper's own recommended default for w, an imprecise/approximate vector field
GRADIENT_SENSITIVITY = 4.0  # scales normalized-depth gradients before clipping to [-1, 1] — tuned for a visible but not exaggerated effect

# A plain hyphen, not an em-dash: PIL's default bitmap font (ImageFont.load_default)
# has limited glyph coverage and renders "—" as a broken/missing-character box —
# caught by actually rendering a frame and looking at it, not assumed to work.
CREDIT_TEXT = "Gradient Domain Rendering - Wang, Gonen & Akleman"


def depth_to_normal_field(depth_image: Image.Image, flatten: float = DEFAULT_FLATTEN) -> np.ndarray:
    """Derives a 2D vector field (H, W, 2) from a depth map, standing in for
    the paper's hand-painted normal-map-like field. Treats the depth map's
    own pixel value as a height field z(u,v) (depth.py's convention:
    brighter = closer = "taller" toward the viewer) and takes its image
    gradient — the standard way to get a normal field from a height field:
    for z(x,y), the (unnormalized) surface normal's horizontal components
    are (-dz/dx, -dz/dy) (from the cross product of the surface's two
    tangent vectors), which is what's computed here via Sobel.

    Gradients are clipped rather than per-pixel-normalized to unit length —
    normalizing would amplify noise in near-flat regions (dividing a
    near-zero vector by its near-zero magnitude). Clipping instead means
    flat regions correctly fall toward (0, 0) — which the paper's own
    shading equation treats as a neutral c=0.5 regardless of light
    direction — while steep depth transitions (edges/silhouettes) get a
    stronger, correctly-directional field.
    """
    depth = np.asarray(depth_image.convert("L"), dtype=np.float64) / 255.0
    gx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
    x = np.clip(-gx * GRADIENT_SENSITIVITY, -1.0, 1.0) * flatten
    y = np.clip(-gy * GRADIENT_SENSITIVITY, -1.0, 1.0) * flatten
    return np.stack([x, y], axis=-1)


def render_diffuse_frame(vector_field: np.ndarray, dark: np.ndarray, lit: np.ndarray, light_angle: float) -> np.ndarray:
    """The paper's Eq. 4 (diffuse shading parameter from a 2D vector field
    and a parallel light direction) and Eq. 2 (final image as a blend of
    two control images by that parameter), directly. `dark`/`lit` are
    float64 (H, W, 3) arrays — DI0/DI1 in the paper's terms. Returns a
    float64 (H, W, 3) array (not yet cast to uint8 — see make_relight_gif).
    """
    light_x, light_y = math.cos(light_angle), math.sin(light_angle)
    x = vector_field[..., 0]
    y = vector_field[..., 1]
    cd = np.clip(0.5 * (x * light_x + y * light_y) + 0.5, 0.0, 1.0)
    cd3 = cd[..., None]  # broadcast the per-pixel scalar over the RGB channels
    return dark * (1 - cd3) + lit * cd3


def _burn_credit(frame: Image.Image) -> Image.Image:
    """Small, unobtrusive text credit in the bottom-right corner — present
    specifically because a GIF is the kind of file that gets shared out of
    context (chat, social media), where the connection back to the
    technique this feature is built on would otherwise be lost entirely.
    Stroke outline (not just a plain fill) so it stays legible against
    whatever the image's own bottom-right corner happens to look like.
    """
    frame = frame.convert("RGB")
    draw = ImageDraw.Draw(frame)
    font_size = max(10, min(16, frame.width // 40))
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:  # older Pillow without the size= param
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), CREDIT_TEXT, font=font, stroke_width=1)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    margin = max(4, font_size // 3)
    pos = (frame.width - text_w - margin, frame.height - text_h - margin)
    draw.text(pos, CREDIT_TEXT, font=font, fill=(255, 255, 255, 220), stroke_width=1, stroke_fill=(0, 0, 0, 200))
    return frame


def make_relight_gif(
    image: Image.Image,
    depth_image: Image.Image,
    frames: int = DEFAULT_FRAMES,
    duration_ms: int = DEFAULT_DURATION_MS,
    darken_factor: float = DEFAULT_DARKEN_FACTOR,
) -> bytes:
    """Renders a looping GIF of `image` under a light sweeping a full circle,
    using `depth_image` (same convention/size as depth.py's get_depth_map
    output) to derive the shading field. DI1 (lit) is the image itself;
    DI0 (dark) is the same image scaled by darken_factor — see module
    docstring for why these are reasonable automatic stand-ins for the
    paper's artist-painted control images.
    """
    image = image.convert("RGB")
    if depth_image.size != image.size:
        depth_image = depth_image.resize(image.size)

    vector_field = depth_to_normal_field(depth_image)
    lit = np.asarray(image, dtype=np.float64)
    dark = lit * darken_factor

    gif_frames = []
    for i in range(frames):
        angle = 2 * math.pi * i / frames  # full sweep -> frame 0 and the implicit next frame match exactly, a seamless loop for free
        frame_arr = np.clip(render_diffuse_frame(vector_field, dark, lit, angle), 0, 255).astype(np.uint8)
        gif_frames.append(_burn_credit(Image.fromarray(frame_arr, mode="RGB")))

    buf = BytesIO()
    gif_frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return buf.getvalue()
