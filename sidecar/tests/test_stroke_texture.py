"""apply_stroke_texture: the essence-reactive directional grain pass.
Verifies the core claim the module docstring makes — that feeding it a
stroke field oriented at angle T produces an output whose own measured
orientation (via style_analysis.stroke.analyze_stroke_field, the same
analyzer that produced the field in the first place) comes back close to
T — plus the more mechanical amount/coherence/no-op behavior, same style as
tests/test_color_transfer.py's amount tests.
"""
import numpy as np
import pytest
from PIL import Image

from stroke_texture import apply_stroke_texture
from style_analysis.stroke import analyze_stroke_field

FIELD_SIZE = 24  # matches style_analysis/stroke.py's own GRID_SIZE, but not load-bearing here -- apply_stroke_texture upsamples whatever shape it's given


def _uniform_field(angle_rad: float, coherence: float = 1.0, size: int = FIELD_SIZE) -> tuple[np.ndarray, np.ndarray]:
    return np.full((size, size), angle_rad), np.full((size, size), coherence)


def _flat_gray_image(size: int = 200) -> Image.Image:
    return Image.new("RGB", (size, size), (128, 128, 128))


def _noise_image(size: int = 200, seed: int = 0) -> Image.Image:
    # The Gabor bank is a zero-mean bandpass filter -- convolving it against
    # a perfectly flat image (no spectral energy anywhere) always yields
    # exactly zero, regardless of orientation, so a meaningful test needs a
    # base with actual texture for the effect to imprint onto, same as the
    # real use case (applying grain atop the SDXL result, not a blank canvas).
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def test_amount_zero_is_a_no_op():
    theta, coherence = _uniform_field(np.pi / 2)
    image = _noise_image()
    result = apply_stroke_texture(image, theta, coherence, amount=0.0)
    assert np.array_equal(np.asarray(result), np.asarray(image.convert("RGB")))


@pytest.mark.parametrize("angle_deg", [0, 45, 90, -45])
def test_output_orientation_matches_the_input_field(angle_deg):
    # The round-trip claim the module docstring makes: a field uniformly
    # oriented at angle T should produce an output whose own measured
    # dominant_angle (re-run through the same analyzer that produces the
    # field in the first place) lands close to T.
    theta_in = np.radians(angle_deg)
    theta, coherence = _uniform_field(theta_in)
    result = apply_stroke_texture(_noise_image(), theta, coherence, amount=1.0)

    profile, _viz, _theta_out, _coh_out = analyze_stroke_field(result)
    assert profile.directionality > 0.7, "a uniform input field should produce a strongly directional output"
    assert profile.dominant_angle is not None
    # Compare on the circle (mod pi) since -90 deg and +90 deg are the same axis.
    diff = abs(profile.dominant_angle - theta_in)
    diff = min(diff, np.pi - diff)
    assert diff < np.radians(10), f"angle={angle_deg}: measured {np.degrees(profile.dominant_angle):.1f} deg"


def test_higher_amount_produces_a_stronger_effect():
    theta, coherence = _uniform_field(np.pi / 4)
    image = _noise_image()
    low = np.asarray(apply_stroke_texture(image, theta, coherence, amount=0.2), dtype=np.float64)
    high = np.asarray(apply_stroke_texture(image, theta, coherence, amount=1.0), dtype=np.float64)
    base = np.asarray(image.convert("RGB"), dtype=np.float64)
    assert np.abs(high - base).mean() > np.abs(low - base).mean()


def test_zero_coherence_mutes_the_effect_almost_entirely():
    theta, coherence = _uniform_field(np.pi / 4, coherence=0.0)
    image = _noise_image()
    result = np.asarray(apply_stroke_texture(image, theta, coherence, amount=1.0), dtype=np.float64)
    base = np.asarray(image.convert("RGB"), dtype=np.float64)
    assert np.abs(result - base).mean() < 1.0


def test_output_size_matches_input():
    theta, coherence = _uniform_field(0.0)
    image = _flat_gray_image(size=64)
    result = apply_stroke_texture(image, theta, coherence, amount=1.0)
    assert result.size == (64, 64)
