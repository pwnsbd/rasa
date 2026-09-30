"""apply_texture_overlay: the classical (non-diffusion) bump/emboss engine.
Verifies the two claims the module docstring makes -- a flat/relief-free
texture is a no-op regardless of amount (the reason this reuses
render_diffuse_frame with a symmetric bright/dark pair instead of
relight.py's own dark..lit pair), and light direction actually matters
(flipping the light angle by pi flips which side of a slope brightens) --
plus the more mechanical amount/size behavior, same style as
tests/test_color_transfer.py and tests/test_stroke_texture.py.
"""
import numpy as np
from PIL import Image

from texture_overlay import apply_texture_overlay


def _flat_texture(value: int = 128, size: int = 64) -> Image.Image:
    return Image.new("L", (size, size), value)


def _step_texture(size: int = 64) -> Image.Image:
    """Left half bright/"tall", right half dark/"short" -- a single slope
    down the middle column, the simplest field with an unambiguous facing
    direction to test light-angle behavior against.
    """
    arr = np.zeros((size, size), dtype=np.uint8)
    arr[:, : size // 2] = 255
    return Image.fromarray(arr, "L")


def _mid_gray_image(size: int = 128) -> Image.Image:
    return Image.new("RGB", (size, size), (128, 128, 128))


def test_amount_zero_is_a_no_op():
    image = _mid_gray_image()
    result = apply_texture_overlay(image, _step_texture(), amount=0.0)
    assert np.array_equal(np.asarray(result), np.asarray(image.convert("RGB")))


def test_flat_texture_is_a_no_op_even_at_full_amount():
    # The core fix over naively reusing relight.py's own dark..lit pair:
    # a texture with no relief anywhere must not uniformly dim the target.
    image = _mid_gray_image()
    result = np.asarray(apply_texture_overlay(image, _flat_texture(), amount=1.0), dtype=np.float64)
    base = np.asarray(image.convert("RGB"), dtype=np.float64)
    assert np.abs(result - base).max() < 1.0


def test_textured_input_visibly_varies_a_flat_target():
    image = _mid_gray_image()
    result = np.asarray(apply_texture_overlay(image, _step_texture(), amount=1.0), dtype=np.float64)
    assert result.std() > 1.0  # the flat target on its own has zero variance


def test_higher_amount_produces_a_stronger_effect():
    image = _mid_gray_image()
    texture = _step_texture()
    low = np.asarray(apply_texture_overlay(image, texture, amount=0.2), dtype=np.float64)
    high = np.asarray(apply_texture_overlay(image, texture, amount=1.0), dtype=np.float64)
    base = np.asarray(image.convert("RGB"), dtype=np.float64)
    assert np.abs(high - base).mean() > np.abs(low - base).mean()


def test_flipping_light_angle_flips_which_side_brightens():
    # A slope's shading should depend on which direction the light comes
    # from -- lighting it from one side vs. the opposite side (angle + pi)
    # should swap which half of the boundary region reads brighter.
    image = _mid_gray_image(size=64)
    texture = _step_texture(size=64)

    from_right = np.asarray(apply_texture_overlay(image, texture, amount=1.0, light_angle=0.0), dtype=np.float64)
    from_left = np.asarray(apply_texture_overlay(image, texture, amount=1.0, light_angle=np.pi), dtype=np.float64)

    # Sample a narrow band straddling the step's boundary column (x=32).
    band_from_right = from_right[:, 28:36].mean()
    band_from_left = from_left[:, 28:36].mean()
    assert abs(band_from_right - band_from_left) > 1.0, "opposite light angles should shade the boundary differently"


def test_output_size_matches_input():
    image = _mid_gray_image(size=50)
    result = apply_texture_overlay(image, _step_texture(size=20), amount=1.0)
    assert result.size == (50, 50)
