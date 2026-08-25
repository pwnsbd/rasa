"""Pure math/compositing behind the Cauldron (essence_store.blend_essences)
— _blend_colors and _composite_weighted_thumbnail. blend_essences itself
needs the real SDXL pipeline (like extract_essence, no direct unit test
today) — these are the parts of it that don't.
"""
from PIL import Image

from essence_store import _blend_colors, _composite_weighted_thumbnail, THUMBNAIL_SIZE


def test_equal_weights_average_evenly():
    result = _blend_colors([(255, 0, 0), (0, 255, 0), (0, 0, 255)], [1, 1, 1])
    # Roughly a third of each channel's max, evenly.
    assert all(70 < c < 100 for c in result)


def test_zero_weight_ingredient_has_no_influence():
    heavy = _blend_colors([(255, 0, 0), (0, 0, 255)], [1.0, 0.0])
    assert heavy == (255, 0, 0)


def test_dominant_weight_pulls_result_toward_it():
    balanced = _blend_colors([(255, 0, 0), (0, 0, 255)], [0.5, 0.5])
    lopsided = _blend_colors([(255, 0, 0), (0, 0, 255)], [0.9, 0.1])
    assert lopsided[0] > balanced[0]  # more red-weighted -> more red
    assert lopsided[2] < balanced[2]  # correspondingly less blue


def test_single_ingredient_returns_its_own_color():
    result = _blend_colors([(12, 34, 56)], [1.0])
    assert result == (12, 34, 56)


def test_output_stays_in_byte_range():
    result = _blend_colors([(300, -10, 128)], [1.0])  # out-of-range input, still must clamp
    assert all(0 <= c <= 255 for c in result)


def _solid(color: tuple[int, int, int], size=(200, 100)) -> Image.Image:
    return Image.new("RGB", size, color)


def test_thumbnail_strip_widths_match_weight_ratios():
    red, blue = _solid((255, 0, 0)), _solid((0, 0, 255))
    thumb = _composite_weighted_thumbnail([red, blue], [0.25, 0.75])
    assert thumb.size == THUMBNAIL_SIZE
    w = THUMBNAIL_SIZE[0]
    left_pixel = thumb.getpixel((2, THUMBNAIL_SIZE[1] // 2))
    right_pixel = thumb.getpixel((w - 3, THUMBNAIL_SIZE[1] // 2))
    assert left_pixel[0] > left_pixel[2]  # left side reads red
    assert right_pixel[2] > right_pixel[0]  # right side reads blue
    # 25/75 split -> the boundary should land noticeably left of center.
    boundary_zone = thumb.crop((w // 2 - 5, 0, w // 2 + 5, THUMBNAIL_SIZE[1]))
    assert boundary_zone.getpixel((0, 0))[2] > boundary_zone.getpixel((0, 0))[0]  # already blue by center


def test_single_ingredient_fills_the_whole_thumbnail():
    green = _solid((0, 255, 0))
    thumb = _composite_weighted_thumbnail([green], [1.0])
    assert thumb.getpixel((0, 0))[1] == 255
    assert thumb.getpixel((THUMBNAIL_SIZE[0] - 1, 0))[1] == 255


def test_negligible_weight_ingredient_is_skipped_entirely():
    # blue is last in the list AND has zero weight — without excluding
    # negligible-weight ingredients before assigning "last" (which absorbs
    # rounding remainder), blue would still fill the remaining canvas
    # despite contributing nothing to the blend.
    red, blue = _solid((255, 0, 0)), _solid((0, 0, 255))
    thumb = _composite_weighted_thumbnail([red, blue], [1.0, 0.0])
    assert thumb.size == THUMBNAIL_SIZE
    assert thumb.getpixel((THUMBNAIL_SIZE[0] - 1, 0)) == (255, 0, 0)
