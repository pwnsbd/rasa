"""Stroke/directional-flow orientation analysis on synthetic patterns with
known, deliberately different directional characteristics (spec Phase 20 /
Phase 4's own verification requirement — visually confirmed via the debug
visualization during development, these tests pin the numeric behavior).
"""
import numpy as np
from PIL import Image, ImageDraw

from style_analysis.stroke import analyze_stroke


def _line_image(angle_deg: float, spacing: int = 14) -> Image.Image:
    img = Image.new("L", (300, 300), 255)
    d = ImageDraw.Draw(img)
    rad = np.radians(angle_deg)
    dx, dy = np.cos(rad), np.sin(rad)
    for offset in range(-400, 400, spacing):
        x0 = 150 + offset * (-dy) - 400 * dx
        y0 = 150 + offset * dx - 400 * dy
        x1 = 150 + offset * (-dy) + 400 * dx
        y1 = 150 + offset * dx + 400 * dy
        d.line([(x0, y0), (x1, y1)], fill=0, width=3)
    return img.convert("RGB")


def _circular_image() -> Image.Image:
    img = Image.new("L", (300, 300), 255)
    d = ImageDraw.Draw(img)
    for r in range(10, 150, 12):
        d.ellipse([150 - r, 150 - r, 150 + r, 150 + r], outline=0, width=3)
    return img.convert("RGB")


def _random_dots_image(seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.new("L", (300, 300), 255)
    d = ImageDraw.Draw(img)
    for _ in range(400):
        x, y = rng.integers(0, 300, 2)
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=0)
    return img.convert("RGB")


def test_straight_line_patterns_score_high_directionality_low_curvature():
    for angle in (0, 45, 90):
        profile, _ = analyze_stroke(_line_image(angle))
        assert profile.directionality > 0.8, f"angle={angle}"
        assert profile.curvature < 0.15, f"angle={angle}"


def test_circular_pattern_scores_low_directionality_high_curvature():
    profile, _ = analyze_stroke(_circular_image())
    assert profile.directionality < 0.2
    assert profile.curvature > 0.4


def test_circular_pattern_still_locally_coherent():
    # Each arc segment is itself strongly oriented, even though the axis
    # rotates around the circle — this is exactly what distinguishes a
    # swirl from chaotic/isotropic noise (next test).
    profile, _ = analyze_stroke(_circular_image())
    assert profile.coherence > 0.5


def test_random_dots_score_low_on_everything():
    profile, _ = analyze_stroke(_random_dots_image())
    assert profile.directionality < 0.3
    assert profile.coherence < 0.3


def test_visualization_is_same_size_as_input():
    profile, viz = analyze_stroke(_line_image(30))
    assert viz.size == (300, 300)
