"""mask_content_correlated: apply-time target-aware content masking's pure
math — see content_mask.py's module docstring. No model involved (both
inputs here are hand-built tensors, not real CLIP embeddings).
"""
import torch

from content_mask import mask_content_correlated


def test_positively_correlated_dimension_gets_down_weighted():
    # dim0: essence and target both strongly positive -> high correlation, should be suppressed.
    # dim1: essence positive, target ~0 -> no correlation, should stay untouched.
    essence = torch.tensor([10.0, 10.0])
    target = torch.tensor([10.0, 0.0])
    result = mask_content_correlated(essence, target, strength=0.25)
    assert result[0].item() < essence[0].item()
    assert result[1].item() == essence[1].item()


def test_strongest_correlation_gets_full_strength_down_weight():
    essence = torch.tensor([4.0, 4.0, 4.0])
    target = torch.tensor([0.0, 2.0, 4.0])  # dim2 has the strongest positive correlation
    result = mask_content_correlated(essence, target, strength=0.25)
    assert abs(result[2].item() - 4.0 * (1 - 0.25)) < 1e-4
    assert result[0].item() == 4.0  # zero correlation -> untouched


def test_negative_correlation_is_not_treated_as_content_evidence():
    # essence and target disagree in sign -> clamped to zero correlation, no down-weight,
    # same treatment as a dimension with no correlation at all.
    essence = torch.tensor([5.0, 5.0])
    target = torch.tensor([-5.0, 0.0])
    result = mask_content_correlated(essence, target, strength=0.5)
    assert torch.allclose(result, essence)


def test_zero_target_embedding_leaves_essence_unchanged():
    essence = torch.tensor([1.0, 2.0, 3.0])
    target = torch.zeros(3)
    result = mask_content_correlated(essence, target)
    assert torch.allclose(result, essence)


def test_strength_zero_leaves_essence_unchanged():
    essence = torch.tensor([1.0, 9.0, 3.0])
    target = torch.tensor([2.0, 9.0, 1.0])
    result = mask_content_correlated(essence, target, strength=0.0)
    assert torch.allclose(result, essence)


def test_output_dtype_matches_essence_input():
    essence = torch.tensor([1.0, 2.0], dtype=torch.float16)
    target = torch.tensor([1.0, 0.5], dtype=torch.float16)
    result = mask_content_correlated(essence, target)
    assert result.dtype == torch.float16


def test_output_shape_matches_input():
    essence = torch.rand(2, 4)
    target = torch.rand(2, 4)
    result = mask_content_correlated(essence, target)
    assert result.shape == essence.shape
