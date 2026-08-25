"""Apply-time target-aware content masking — a second, complementary line
of defense against essence content leaking into generation, alongside
essence_store.py's distillation-time multi-crop purification (which cleans
an Essence once, from the reference alone, before any target exists).

Adapted from MaskST (arXiv:2502.07466, ICLR 2025): that paper identifies
IP-Adapter embedding dimensions to mask by clustering the element-wise
product of the style reference's features and a *text* description of the
desired content. Rasa's pipeline has no text prompt anywhere — apply_essence
runs with prompt="" throughout — so there's no content-text embedding to
correlate against. This substitutes the *target photo's own* CLIP-image
embedding instead: dimensions where the essence's embedding strongly agrees
with the target's own content embedding are the dimensions most likely
restating/fighting over content the target's own structure (already held in
place by the Tile ControlNet — see pipeline_manager.py) already covers,
rather than contributing pure style.

This is NOT a reimplementation of RB-Modulation's Attention Feature
Aggregation module (arXiv:2405.17401, ICLR 2025) — that operates inside the
denoising attention layers via a full stochastic-optimal-control
reformulation of the reverse diffusion process, a substantially larger
undertaking. This is the same *category* of idea — training-free, apply-time
content/style decoupling — done at the embedding level instead, the same
level essence_store.py's purification already operates at, and cheap
because of it: one extra CLIP encode per apply, not a change to the
denoising loop itself.
"""
from __future__ import annotations

import torch

TARGET_AWARE_MASKING_STRENGTH = 0.25  # gentler than distillation-time purification's 0.4 — this compounds on an
# already-purified embedding, and the per-apply correlation signal (one target image, not several crops) is noisier.


def mask_content_correlated(
    essence_embed: torch.Tensor,
    target_embed: torch.Tensor,
    strength: float = TARGET_AWARE_MASKING_STRENGTH,
) -> torch.Tensor:
    """Down-weights dimensions of `essence_embed` that correlate strongly
    with `target_embed` — same soft-mask shape as
    essence_store.py's _purify_embedding (a down-weight, never a hard
    zero), deliberately: this is a second pass of the same kind of
    adjustment, not a different philosophy.

    Correlation is the element-wise product of the two (both from the same
    CLIP-encoder + IP-Adapter projector, see
    essence_store.extract_ip_adapter_embedding, so their dimensions line
    up meaningfully). Only *positive* correlation counts as evidence of
    content overlap — a dimension where the essence and the target push in
    the same direction is "restating" something; a dimension where they
    disagree or one is near zero isn't evidence of shared content either
    way, so it's left alone rather than penalized.
    """
    dtype = essence_embed.dtype
    e = essence_embed.float()
    t = target_embed.float()

    correlation = torch.clamp(e * t, min=0)
    c_min, c_max = correlation.min(), correlation.max()
    if (c_max - c_min).item() < 1e-8:
        return essence_embed  # no meaningful (positive, varying) correlation signal — nothing to mask

    c_norm = (correlation - c_min) / (c_max - c_min)
    weight = 1.0 - strength * c_norm
    return (e * weight).to(dtype)
