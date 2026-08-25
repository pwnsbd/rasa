"""depth_to_alpha_mask: pure mask-shaping math, no model involved (see
depth.py's module docstring — get_depth_map itself needs the real Depth
Anything model, same as pipeline_manager.py has no direct unit test today).
Synthetic "L" mode depth images, same style as tests/test_stroke.py's
synthetic line/circle patterns.
"""
import numpy as np
from PIL import Image

from depth import depth_to_alpha_mask


def _half_split_depth(left: int, right: int) -> Image.Image:
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[:, :50] = left
    arr[:, 50:] = right
    return Image.fromarray(arr, mode="L")


def test_bright_near_side_stays_bright_after_smoothing():
    # Left = near (bright, per Depth Anything's convention), right = far (dark).
    depth_image = _half_split_depth(left=220, right=30)
    mask = depth_to_alpha_mask(depth_image)
    arr = np.asarray(mask)
    assert arr[:, :20].mean() > arr[:, 80:].mean()


def test_invert_flips_near_and_far():
    depth_image = _half_split_depth(left=220, right=30)
    normal = np.asarray(depth_to_alpha_mask(depth_image))
    inverted = np.asarray(depth_to_alpha_mask(depth_image, invert=True))
    assert normal[:, :20].mean() > normal[:, 80:].mean()
    assert inverted[:, :20].mean() < inverted[:, 80:].mean()


def test_smooths_a_hard_edge():
    # A hard step between the two halves should come out gradual, not a
    # razor edge — check there's a real transition zone near the boundary
    # rather than a single-pixel jump.
    depth_image = _half_split_depth(left=255, right=0)
    mask = depth_to_alpha_mask(depth_image)
    arr = np.asarray(mask, dtype=np.float64)
    near_boundary = arr[:, 45:55]
    # A smoothed transition has real variance across this thin strip; a
    # perfectly hard edge left untouched would be all-255 then all-0 with a
    # single-column jump, and outputs from a Gaussian blur applied over it
    # necessarily spread that jump across the strip's mean.
    assert 20 < near_boundary.mean() < 235


def test_output_same_size_as_input():
    depth_image = _half_split_depth(left=200, right=50)
    mask = depth_to_alpha_mask(depth_image)
    assert mask.size == depth_image.size


def test_accepts_non_l_mode_input():
    # get_depth_map's real caller always hands this an "L" image, but the
    # function itself should tolerate anything Pillow can .convert("L").
    rgb = Image.new("RGB", (64, 64), (180, 180, 180))
    mask = depth_to_alpha_mask(rgb)
    assert mask.mode == "L"
