"""relight.py's pure math: depth_to_normal_field and render_diffuse_frame.
make_relight_gif itself is an integration of both plus GIF encoding — not
unit tested directly (same philosophy as extract_essence/apply_essence
having no direct test: this is glue over already-tested pieces), but
exercised for real in test_make_relight_gif_produces_a_looping_gif below
since it needs no model/GPU, just PIL/numpy/cv2, all cheap.
"""
import numpy as np
from PIL import Image

from relight import depth_to_normal_field, make_relight_gif, render_diffuse_frame


def test_flat_depth_gives_zero_field():
    flat = Image.new("L", (64, 64), 128)
    field = depth_to_normal_field(flat)
    assert np.allclose(field, 0.0, atol=1e-6)


def test_depth_ramp_gives_consistently_signed_field():
    # Brighter = closer (depth.py's convention). A left-to-right ramp means
    # depth increases with x, i.e. dz/dx > 0 -> the derived x-component
    # should be consistently negative (see depth_to_normal_field's docstring
    # on the -dz/dx, -dz/dy convention).
    w, h = 64, 64
    ramp = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    field = depth_to_normal_field(Image.fromarray(ramp, mode="L"))
    interior = field[8:-8, 8:-8, 0]  # avoid border artifacts from the Sobel kernel
    assert np.all(interior < 0)


def test_field_respects_flatten_scale():
    w, h = 64, 64
    ramp = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    img = Image.fromarray(ramp, mode="L")
    full = depth_to_normal_field(img, flatten=1.0)
    half = depth_to_normal_field(img, flatten=0.5)
    assert np.allclose(half, full * 0.5, atol=1e-6)


def test_light_aligned_with_field_gives_lit_result():
    field = np.zeros((4, 4, 2))
    field[..., 0] = 1.0  # every pixel's field points right
    dark = np.zeros((4, 4, 3))
    lit = np.full((4, 4, 3), 255.0)
    result = render_diffuse_frame(field, dark, lit, light_angle=0.0)  # light also points right
    assert np.all(result > 250)


def test_light_opposite_field_gives_dark_result():
    field = np.zeros((4, 4, 2))
    field[..., 0] = 1.0
    dark = np.zeros((4, 4, 3))
    lit = np.full((4, 4, 3), 255.0)
    result = render_diffuse_frame(field, dark, lit, light_angle=np.pi)  # light points left, opposite the field
    assert np.all(result < 5)


def test_zero_field_gives_midpoint_regardless_of_light_angle():
    field = np.zeros((4, 4, 2))
    dark = np.zeros((4, 4, 3))
    lit = np.full((4, 4, 3), 200.0)
    for angle in (0.0, 1.5, 3.9, 5.5):
        result = render_diffuse_frame(field, dark, lit, angle)
        assert np.allclose(result, 100.0)


def test_output_is_clipped_to_0_255_range():
    field = np.zeros((4, 4, 2))
    field[..., 0] = 1.0
    dark = np.zeros((4, 4, 3))
    lit = np.full((4, 4, 3), 255.0)
    result = render_diffuse_frame(field, dark, lit, light_angle=0.0)
    assert result.min() >= 0.0 and result.max() <= 255.0


def test_make_relight_gif_produces_a_looping_gif():
    image = Image.new("RGB", (48, 48), (180, 100, 60))
    depth = Image.new("L", (48, 48), 0)
    for x in range(48):
        for y in range(48):
            depth.putpixel((x, y), min(255, x * 5))  # a real ramp, not flat, so the field isn't degenerate

    gif_bytes = make_relight_gif(image, depth, frames=6, duration_ms=40)
    assert gif_bytes[:6] in (b"GIF87a", b"GIF89a")

    from io import BytesIO

    reopened = Image.open(BytesIO(gif_bytes))
    assert getattr(reopened, "n_frames", 1) == 6
    assert reopened.info.get("loop") == 0
