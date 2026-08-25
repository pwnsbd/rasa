"""Palette extraction correctness (spec Phase 20)."""
from PIL import Image, ImageDraw

from style_analysis.palette import extract_palette


def _color_block_image(splits: list[tuple[float, tuple[int, int, int]]]) -> Image.Image:
    """Builds a 300x300 image with horizontal bands of the given (share, rgb)
    pairs, in order, share summing to 1.0.
    """
    img = Image.new("RGB", (300, 300))
    d = ImageDraw.Draw(img)
    y = 0
    for share, color in splits:
        band_h = round(300 * share)
        d.rectangle([0, y, 300, y + band_h], fill=color)
        y += band_h
    return img


def test_dominant_color_weights_match_known_ratios():
    # 50% red / 30% blue / 20% yellow — spec's own example ratios.
    img = _color_block_image([(0.5, (255, 0, 0)), (0.3, (0, 0, 255)), (0.2, (255, 255, 0))])
    profile = extract_palette(img)

    by_hex = {s.hex: s.weight for s in profile.dominant_colors}
    assert set(by_hex.keys()) == {"#FF0000", "#0000FF", "#FFFF00"}
    assert abs(by_hex["#FF0000"] - 0.5) < 0.03
    assert abs(by_hex["#0000FF"] - 0.3) < 0.03
    assert abs(by_hex["#FFFF00"] - 0.2) < 0.03


def test_weights_sum_to_approximately_one():
    img = _color_block_image([(0.5, (255, 0, 0)), (0.3, (0, 0, 255)), (0.2, (255, 255, 0))])
    profile = extract_palette(img)
    assert abs(sum(s.weight for s in profile.dominant_colors) - 1.0) < 0.01


def test_solid_color_image_has_zero_contrast_and_full_saturation():
    img = Image.new("RGB", (200, 200), (200, 30, 30))
    profile = extract_palette(img)
    assert profile.contrast < 0.05  # no variation at all
    assert profile.mean_saturation > 0.7  # a strong red is highly saturated


def test_warm_image_scores_positive_temperature_cool_scores_negative():
    warm = Image.new("RGB", (200, 200), (230, 90, 30))  # orange
    cool = Image.new("RGB", (200, 200), (30, 90, 230))  # blue
    assert extract_palette(warm).temperature > 0
    assert extract_palette(cool).temperature < 0


def test_is_deterministic():
    img = _color_block_image([(0.5, (255, 0, 0)), (0.3, (0, 0, 255)), (0.2, (255, 255, 0))])
    p1 = extract_palette(img)
    p2 = extract_palette(img)
    assert [s.hex for s in p1.dominant_colors] == [s.hex for s in p2.dominant_colors]
    assert [s.weight for s in p1.dominant_colors] == [s.weight for s in p2.dominant_colors]
