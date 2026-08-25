"""preserve_original_color: keeps the stylized image's luminance/contrast
but restores the original photo's hue — the fix for a reported real-world
case where a strongly-colored essence tinted an entire photo toward its hue.
"""
import numpy as np
import pytest
from PIL import Image

from color_transfer import preserve_original_color


def _half_split_image(left: tuple[int, int, int], right: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGB", (100, 100))
    img.paste(left, (0, 0, 50, 100))
    img.paste(right, (50, 0, 100, 100))
    return img


def test_restores_original_hue_per_region():
    # Original: distinctly red left half, blue right half.
    original = _half_split_image((200, 30, 30), (30, 30, 200))
    # Stylized: as if the essence's own purple tint took over everywhere —
    # this is exactly the reported bug ("recent image became purple").
    stylized = _half_split_image((160, 40, 160), (150, 45, 150))

    result = preserve_original_color(stylized, original)
    arr = np.asarray(result, dtype=np.float64)
    left, right = arr[:, :25], arr[:, 75:]

    assert left[..., 0].mean() > left[..., 2].mean()  # left reads red-ish again
    assert right[..., 2].mean() > right[..., 0].mean()  # right reads blue-ish again


def test_preserves_stylized_luminance_contrast():
    # Same hue both sides so only luminance differs -- a stand-in for the
    # essence's texture/contrast contribution, which should survive.
    original = _half_split_image((120, 120, 120), (120, 120, 120))
    stylized = _half_split_image((220, 220, 220), (40, 40, 40))  # bright vs. dark

    result = preserve_original_color(stylized, original)
    arr = np.asarray(result, dtype=np.float64)
    left_brightness = arr[:, :25].mean()
    right_brightness = arr[:, 75:].mean()
    assert left_brightness > right_brightness + 30  # the stylized brightness gap survives


def test_rejects_mismatched_sizes():
    a = Image.new("RGB", (50, 50))
    b = Image.new("RGB", (60, 60))
    with pytest.raises(ValueError):
        preserve_original_color(a, b)


def test_output_size_matches_input():
    a = Image.new("RGB", (64, 48), (10, 20, 30))
    b = Image.new("RGB", (64, 48), (200, 100, 50))
    result = preserve_original_color(a, b)
    assert result.size == (64, 48)
