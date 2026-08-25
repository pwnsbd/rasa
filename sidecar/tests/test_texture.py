"""Texture metric ordering across simple, well-understood cases (spec Phase
20). Assertions check relative ordering, not exact values — these are
relative descriptors, not physically calibrated measurements (per the
spec's own framing).
"""
import numpy as np
from PIL import Image

from style_analysis.texture import analyze_texture


def _flat_image() -> Image.Image:
    return Image.new("RGB", (256, 256), (128, 128, 128))


def _checkerboard_image(cell=16) -> Image.Image:
    arr = np.zeros((256, 256), dtype=np.uint8)
    for y in range(0, 256, cell):
        for x in range(0, 256, cell):
            if ((x // cell) + (y // cell)) % 2 == 0:
                arr[y:y + cell, x:x + cell] = 255
    return Image.fromarray(arr).convert("RGB")


def _noise_image(seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def test_flat_has_lowest_roughness():
    flat = analyze_texture(_flat_image())
    checker = analyze_texture(_checkerboard_image())
    noise = analyze_texture(_noise_image())
    assert flat.roughness < checker.roughness
    assert flat.roughness < noise.roughness


def test_flat_has_near_zero_detail_density():
    flat = analyze_texture(_flat_image())
    assert flat.detail_density < 0.05


def test_noise_has_more_high_frequency_energy_than_flat():
    flat = analyze_texture(_flat_image())
    noise = analyze_texture(_noise_image())
    assert noise.high_frequency_energy > flat.high_frequency_energy


def test_checkerboard_is_more_repetitive_than_noise():
    checker = analyze_texture(_checkerboard_image())
    noise = analyze_texture(_noise_image())
    assert checker.repetition > noise.repetition
