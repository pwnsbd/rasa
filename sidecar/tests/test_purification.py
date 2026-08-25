"""Multi-crop style purification's pure math (_purify_embedding) and crop
sampling (_sample_crops) — see essence_store.py's module docstring for the
full rationale. _extract_purified_embedding itself needs the real SDXL
pipeline (like extract_essence, no direct unit test today) — these are the
parts of it that don't.
"""
import numpy as np
import torch
from PIL import Image

from essence_store import _purify_embedding, _sample_crops


def test_strength_zero_equals_plain_mean():
    samples = [torch.tensor([1.0, 5.0]), torch.tensor([3.0, -5.0]), torch.tensor([2.0, 10.0])]
    result = _purify_embedding(samples, strength=0.0)
    expected = torch.stack(samples).mean(dim=0)
    assert torch.allclose(result, expected)


def test_high_variance_dimension_gets_down_weighted_relative_to_stable_one():
    # dim0 constant across every sample (pure "style" signal); dim1 swings
    # wildly but still averages to the same value (pure "content" signal).
    samples = [
        torch.tensor([10.0, 0.0]),
        torch.tensor([10.0, 20.0]),
        torch.tensor([10.0, 0.0]),
        torch.tensor([10.0, 20.0]),
        torch.tensor([10.0, 10.0]),
        torch.tensor([10.0, 10.0]),
    ]
    result = _purify_embedding(samples, strength=0.4)
    assert result[0].item() == 10.0  # zero variance -> zero down-weight
    assert result[1].item() < 10.0  # max variance among these samples -> full strength down-weight
    assert abs(result[1].item() - 6.0) < 1e-4  # 10 * (1 - 0.4)


def test_degenerate_identical_samples_returns_mean_unchanged():
    same = torch.tensor([1.0, 2.0, 3.0])
    samples = [same.clone() for _ in range(5)]
    result = _purify_embedding(samples, strength=0.4)
    assert torch.allclose(result, same)


def test_single_sample_returns_it_unchanged():
    only = torch.tensor([4.0, -1.0, 9.0])
    result = _purify_embedding([only], strength=0.4)
    assert torch.allclose(result, only)


def test_output_dtype_matches_input():
    samples = [torch.tensor([1.0, 2.0], dtype=torch.float16), torch.tensor([3.0, 0.0], dtype=torch.float16)]
    result = _purify_embedding(samples)
    assert result.dtype == torch.float16


def test_output_shape_matches_sample_shape():
    samples = [torch.rand(2, 3), torch.rand(2, 3), torch.rand(2, 3)]
    result = _purify_embedding(samples)
    assert result.shape == (2, 3)


def _gradient_image(size=(200, 200)) -> Image.Image:
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    arr[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    return Image.fromarray(arr)


def test_returns_requested_count():
    crops = _sample_crops(_gradient_image(), 5, seed=1)
    assert len(crops) == 5


def test_crop_size_is_half_of_original():
    crops = _sample_crops(_gradient_image((100, 100)), 3, seed=1)
    assert all(c.size == (50, 50) for c in crops)


def test_tiny_image_does_not_crash():
    crops = _sample_crops(_gradient_image((3, 3)), 2, seed=1)
    assert len(crops) == 2


def test_same_seed_is_deterministic():
    img = _gradient_image()
    a = _sample_crops(img, 4, seed=42)
    b = _sample_crops(img, 4, seed=42)
    for ca, cb in zip(a, b):
        assert list(ca.get_flattened_data()) == list(cb.get_flattened_data())


def test_different_seed_produces_different_crops():
    img = _gradient_image()
    a = _sample_crops(img, 4, seed=1)
    b = _sample_crops(img, 4, seed=2)
    assert any(list(ca.get_flattened_data()) != list(cb.get_flattened_data()) for ca, cb in zip(a, b))
