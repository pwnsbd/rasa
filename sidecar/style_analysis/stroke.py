"""Stroke / directional-flow analysis (spec Phase 4) — the most important
of the four analyzers per the spec. Classical structure-tensor approach:
for each cell of a coarse grid, estimate a dominant local orientation and
how confidently oriented (vs. isotropic) that cell is, then aggregate into
global directionality/curvature/coherence/density scores plus a debug
visualization (short arrows over the reference image) so the extraction
can be visually sanity-checked — the spec is explicit that this needs
human inspection, not just trusting the numbers.

Structure tensor recipe (standard, e.g. Bigun & Granlund): for gradients
Ix, Iy, the local tensor is [[Ix*Ix, Ix*Iy], [Ix*Iy, Iy*Iy]] integrated
(here: averaged) over a window. Its dominant eigenvector's angle is the
local orientation; the eigenvalue ratio gives a coherence measure.
Equivalently and what's used below: theta = 0.5*atan2(2*Ixy, Ixx-Iyy),
coherence = sqrt((Ixx-Iyy)^2 + 4*Ixy^2) / (Ixx+Iyy).
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from essence_models import StrokeProfile
from style_analysis._common import analysis_resize, to_gray_array

GRID_SIZE = 24  # cells per side for the orientation field — within the spec's suggested 32x32/64x64 range, on the smaller side to keep the debug visualization legible


def _cell_orientation_field(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (theta, coherence, magnitude), each shaped (GRID_SIZE, GRID_SIZE).
    theta is in radians, range (-pi/2, pi/2] (orientation is undirected —
    a stroke pointing left and one pointing right are the same orientation).
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    ixx, iyy, ixy = gx * gx, gy * gy, gx * gy

    h, w = gray.shape
    cell_h, cell_w = h / GRID_SIZE, w / GRID_SIZE

    theta = np.zeros((GRID_SIZE, GRID_SIZE))
    coherence = np.zeros((GRID_SIZE, GRID_SIZE))
    magnitude = np.zeros((GRID_SIZE, GRID_SIZE))

    for row in range(GRID_SIZE):
        y0, y1 = int(row * cell_h), max(int(row * cell_h) + 1, int((row + 1) * cell_h))
        for col in range(GRID_SIZE):
            x0, x1 = int(col * cell_w), max(int(col * cell_w) + 1, int((col + 1) * cell_w))
            sxx, syy, sxy = ixx[y0:y1, x0:x1].mean(), iyy[y0:y1, x0:x1].mean(), ixy[y0:y1, x0:x1].mean()
            theta[row, col] = 0.5 * np.arctan2(2 * sxy, sxx - syy)
            denom = sxx + syy
            coherence[row, col] = np.sqrt((sxx - syy) ** 2 + 4 * sxy ** 2) / denom if denom > 1e-8 else 0.0
            magnitude[row, col] = np.hypot(gx[y0:y1, x0:x1], gy[y0:y1, x0:x1]).mean()

    return theta, coherence, magnitude


def _render_orientation_field(base: Image.Image, theta: np.ndarray, coherence: np.ndarray) -> Image.Image:
    """Debug visualization — one short line per grid cell, angled at that
    cell's dominant orientation, length/opacity scaled by how confident
    (coherent) that orientation is. Exists so stroke extraction can be
    visually inspected, not just trusted numerically (spec Phase 4).
    """
    viz = base.convert("RGB").copy()
    draw = ImageDraw.Draw(viz, "RGBA")
    h, w = viz.size[1], viz.size[0]
    cell_h, cell_w = h / GRID_SIZE, w / GRID_SIZE
    max_len = min(cell_h, cell_w) * 0.45

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            conf = float(coherence[row, col])
            if conf < 0.08:  # skip near-isotropic cells — an arrow there would be meaningless noise
                continue
            cy, cx = (row + 0.5) * cell_h, (col + 0.5) * cell_w
            angle = float(theta[row, col])
            length = max_len * min(1.0, conf * 1.5)
            dx, dy = np.cos(angle) * length, np.sin(angle) * length
            alpha = int(80 + 150 * min(1.0, conf))
            draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=(255, 200, 60, alpha), width=2)

    return viz


def analyze_stroke(image: Image.Image) -> tuple[StrokeProfile, Image.Image]:
    img = analysis_resize(image)
    gray = to_gray_array(img)
    theta, coherence, magnitude = _cell_orientation_field(gray)

    # Global directionality: average the doubled-angle unit vectors
    # (cos 2θ, sin 2θ), weighted by each cell's coherence, then take the
    # magnitude of that mean vector. Doubling the angle makes orientation
    # (mod π, since a stroke has no "direction", just an axis) behave like
    # a proper mod-2π angle for averaging — cells that agree on axis
    # reinforce each other; cells pointing in unrelated directions cancel
    # out. High = one dominant axis across the whole image (e.g. all
    # horizontal); low = directions vary a lot (chaotic, or many axes).
    weights = coherence
    vx = float((np.cos(2 * theta) * weights).mean())
    vy = float((np.sin(2 * theta) * weights).mean())
    directionality = float(np.clip(np.hypot(vx, vy) / (weights.mean() + 1e-8), 0.0, 1.0))

    # Global coherence: mean of per-cell coherence — "how confidently
    # oriented is each mark", independent of whether marks *agree* with
    # each other globally. A swirl scores high here (each local patch is
    # strongly oriented) but low on directionality (the axis keeps
    # rotating) — that distinction is exactly what tells swirls apart from
    # chaotic/photographic texture (low on both).
    global_coherence = float(np.clip(coherence.mean(), 0.0, 1.0))

    # Curvature: how much orientation changes between neighboring cells,
    # via the spatial gradient of the doubled-angle field (handles the
    # mod-π wraparound correctly since it's a complex/vector field, not a
    # raw angle difference).
    complex_field = weights * np.exp(1j * 2 * theta)
    dfield_y = np.abs(np.diff(complex_field, axis=0))
    dfield_x = np.abs(np.diff(complex_field, axis=1))
    curvature = float(np.clip((dfield_y.mean() + dfield_x.mean()) / 2 * 3.0, 0.0, 1.0))  # *3.0: empirical scale, see module docstring

    # Density: how much of the image has stroke-like (high-gradient)
    # content at all, vs. flat/blank — a photographic texture with weak,
    # everywhere-present gradients still scores lower here than bold
    # brushwork.
    density = float(np.clip(magnitude.mean() / 0.25, 0.0, 1.0))  # /0.25: empirical scale, see module docstring

    viz = _render_orientation_field(img, theta, coherence)

    profile = StrokeProfile(
        directionality=round(directionality, 4),
        curvature=round(curvature, 4),
        coherence=round(global_coherence, 4),
        density=round(density, 4),
        orientation_map_path=None,  # filled in by essence.py once it knows the essence_id/output path
    )
    return profile, viz
