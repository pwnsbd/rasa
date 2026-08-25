"""Applying an Essence to a target photo (spec §2.2b) — the actual SDXL
img2img + InstantStyle IP-Adapter + Tile ControlNet pipeline call, plus
subject-isolated strength blending. Split out of what used to be a single
essence.py; see essence_store.py for extraction/persistence/schema, which
this module reads from (load_embedding) but never writes to.

Real implementation: SDXL + InstantStyle + a Tile ControlNet (see
pipeline_manager.py for why SDXL rather than the spec's originally-suggested
Flux, and why ControlNet is needed alongside IP-Adapter — plain img2img
`strength` alone let style-driven regeneration drift too much of the
target's own content/layout away, which is exactly the failure the
InstantStyle authors' own follow-up paper, InstantStyle-Plus, fixes with a
Tile ControlNet).

Generation strategies (single-pass, subject-isolated two-pass, depth-driven
two-pass) are each their own class behind GenerationStrategy so future
experiments (a palette-guided pass — explicitly future work, not started)
add a new class here instead of another branch in apply_essence.
apply_essence itself just picks a strategy and shapes its result into the
API response.
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from PIL import Image

import color_transfer
import content_mask
import depth
import essence_store
import pipeline_manager
import segmentation
from imaging import to_data_url

WORKING_MAX_DIM = 1024  # SDXL's native resolution; img2img input is resized to this

# Tuned empirically after adding the Tile ControlNet (see pipeline_manager.py):
# with ControlNet holding structure, strength can run much higher than the
# pre-ControlNet 0.55 without content drift — tested up to 0.85 with zero
# observed drift across both flat-color and textured synthetic images, so
# defaults sit there to favor a visible style shift. Both are exposed as
# optional /apply request overrides (see app.py's ApplyRequest) for further
# tuning without a code change.
DEFAULT_STRENGTH = 0.85
DEFAULT_GUIDANCE = 5.0
DEFAULT_STEPS = 30
NEGATIVE_PROMPT = "lowres, blurry, bad anatomy, worst quality, low quality, watermark, text"

# DepthGradientStrategy defaults — an initial, reasoned starting point (near
# sits close to the non-face subject suggestion in segmentation.py, far sits
# looser than the flat single-pass default so the depth effect actually
# reads), not yet validated against a real photo the way suggest_subject_params
# was. Both ends are exposed as optional /apply overrides (see app.py) for
# tuning without a code change, same pattern as the subject-region params.
DEPTH_NEAR_STRENGTH = 0.45
DEPTH_NEAR_CONTROLNET_SCALE = 0.90
DEPTH_FAR_STRENGTH = 0.95
DEPTH_FAR_CONTROLNET_SCALE = 0.65


def _resize_working(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((WORKING_MAX_DIM, WORKING_MAX_DIM))
    # SDXL wants multiple-of-8 dimensions.
    w, h = (d - d % 8 for d in img.size)
    return img.resize((max(w, 8), max(h, 8)))


def _run_generation(pipe, embeds, working, strength, controlnet_scale, steps, generator=None):
    result = pipe(
        prompt="",
        negative_prompt=NEGATIVE_PROMPT,
        image=working,
        control_image=working,
        controlnet_conditioning_scale=controlnet_scale,
        ip_adapter_image_embeds=embeds,
        strength=strength,
        guidance_scale=DEFAULT_GUIDANCE,
        num_inference_steps=steps,
        generator=generator,
    )
    return result.images[0]


def _two_pass_blend(pipe, embeds, working, steps, params_a, params_b, mask, label_a="A", label_b="B"):
    """Shared two-pass-plus-composite machinery, used by both
    SubjectIsolatedStrategy (binary subject mask) and DepthGradientStrategy
    (continuous depth mask) — extracted so the identical pattern isn't
    duplicated between them.

    Generates two full-frame passes at params_a/params_b = (strength,
    controlnet_scale), sharing one seeded generator: without that, the two
    outputs diverge in grain/noise/color balance independent of the
    strength difference, which reads as a mismatch at the mask boundary
    even though the mask itself is well-feathered/smooth. Composites with
    Image.composite(pass_a, pass_b, mask) — pass_a shows where mask is
    bright (255), pass_b where it's dark (0), same convention PIL's own
    composite uses. label_a/label_b are just for the timing prints.
    """
    strength_a, controlnet_scale_a = params_a
    strength_b, controlnet_scale_b = params_b

    seed = int.from_bytes(os.urandom(4), "big")
    gen_a = torch.Generator(device=pipe._execution_device).manual_seed(seed)
    gen_b = torch.Generator(device=pipe._execution_device).manual_seed(seed)

    t_a = time.perf_counter()
    pass_a = _run_generation(pipe, embeds, working, strength_a, controlnet_scale_a, steps, gen_a)
    print(f"[apply] pass {label_a}: {time.perf_counter() - t_a:.1f}s")

    t_b = time.perf_counter()
    pass_b = _run_generation(pipe, embeds, working, strength_b, controlnet_scale_b, steps, gen_b)
    print(f"[apply] pass {label_b}: {time.perf_counter() - t_b:.1f}s")

    t_composite = time.perf_counter()
    final = Image.composite(pass_a, pass_b, mask)
    print(f"[apply] composite: {(time.perf_counter() - t_composite) * 1000:.0f}ms")
    return final


@dataclass
class GenerationResult:
    final: Image.Image
    subject_detected: bool = False
    face_detected: bool = False
    suggested_subject_strength: float | None = None
    suggested_subject_controlnet_scale: float | None = None


class GenerationStrategy(ABC):
    @abstractmethod
    def run(self, pipe, embeds, working: Image.Image, steps: int) -> GenerationResult:
        ...


class SinglePassStrategy(GenerationStrategy):
    """The plain case: one img2img+ControlNet pass over the whole frame at a
    single strength/controlnet_scale. Used directly when blend_mode is
    "none", and as SubjectIsolatedStrategy's own fallback when no subject
    can be segmented from the target.
    """

    def __init__(self, strength: float, controlnet_scale: float):
        self.strength = strength
        self.controlnet_scale = controlnet_scale

    def run(self, pipe, embeds, working, steps):
        t = time.perf_counter()
        final = _run_generation(pipe, embeds, working, self.strength, self.controlnet_scale, steps)
        print(f"[apply] single pass: {time.perf_counter() - t:.1f}s")
        return GenerationResult(final=final)


class SubjectIsolatedStrategy(GenerationStrategy):
    """Subject-isolated strength blending (see segmentation.py): when a
    subject is segmented from the target, generation runs *twice* — once at
    the background strength/controlnet_scale over the whole frame, once at a
    lower, tighter subject strength/controlnet_scale (suggested by face
    detection within the subject region, or overridden) — then composites
    the two with the subject's soft feathered mask. Both passes share one
    seeded generator: without that, the two outputs diverge in grain/noise/
    color balance independent of the strength difference, which reads as a
    mismatch at the mask boundary even though the mask itself is
    well-feathered. Falls back to SinglePassStrategy when no distinguishable
    subject is found.
    """

    def __init__(
        self,
        bg_strength: float,
        bg_controlnet_scale: float,
        subject_strength_override: float | None = None,
        subject_controlnet_scale_override: float | None = None,
    ):
        self.bg_strength = bg_strength
        self.bg_controlnet_scale = bg_controlnet_scale
        self.subject_strength_override = subject_strength_override
        self.subject_controlnet_scale_override = subject_controlnet_scale_override

    def run(self, pipe, embeds, working, steps):
        t_seg = time.perf_counter()
        mask = segmentation.get_subject_mask(working)
        print(f"[apply] segmentation: {(time.perf_counter() - t_seg) * 1000:.0f}ms (subject_detected={mask is not None})")

        if mask is None:
            return SinglePassStrategy(self.bg_strength, self.bg_controlnet_scale).run(pipe, embeds, working, steps)

        t_face = time.perf_counter()
        face_detected = segmentation.detect_face(working, mask)
        print(f"[apply] face detection: {(time.perf_counter() - t_face) * 1000:.0f}ms (face_detected={face_detected})")

        suggested_strength, suggested_controlnet_scale = segmentation.suggest_subject_params(face_detected)
        actual_subject_strength = (
            self.subject_strength_override if self.subject_strength_override is not None else suggested_strength
        )
        actual_subject_controlnet_scale = (
            self.subject_controlnet_scale_override
            if self.subject_controlnet_scale_override is not None
            else suggested_controlnet_scale
        )

        final = _two_pass_blend(
            pipe,
            embeds,
            working,
            steps,
            params_a=(actual_subject_strength, actual_subject_controlnet_scale),
            params_b=(self.bg_strength, self.bg_controlnet_scale),
            mask=mask,
            label_a="B (subject)",
            label_b="A (background)",
        )

        return GenerationResult(
            final=final,
            subject_detected=True,
            face_detected=face_detected,
            suggested_subject_strength=suggested_strength,
            suggested_subject_controlnet_scale=suggested_controlnet_scale,
        )


class DepthGradientStrategy(GenerationStrategy):
    """Continuous depth-driven blending (see depth.py) — the thing a flat
    2D filter has no way to do at all, since it has no notion of the
    photo's actual scene depth. Instead of a binary subject/background
    split, foreground (near-camera) content is generated at a tighter,
    more-preserved strength/controlnet_scale and background (far) content
    at a looser, more-stylized one, blended with a smooth continuous depth
    mask rather than a hard cutout — so stylization visibly deepens with
    distance instead of jumping at a subject boundary. Reuses the exact
    same two-pass-plus-shared-seed machinery as SubjectIsolatedStrategy
    (_two_pass_blend): same generation cost, no third pass.

    For photos without one clear rembg-segmentable subject (landscapes,
    group shots, product shots) — SubjectIsolatedStrategy would just fall
    back to a flat single pass on these; this gives them a real alternative
    instead.

    Takes an already-computed depth map (via the constructor) rather than
    calling depth.get_depth_map itself — apply_essence now computes depth
    unconditionally (see its own docstring: every creation gets a depth map
    for the Media Page's parallax effect, not just depth-blend-mode ones),
    so this strategy would otherwise redo that work a second time. Falls
    back to computing its own if none was given (a standalone/future direct
    use, or a test), so it still works correctly on its own.
    """

    def __init__(
        self,
        near_strength: float,
        near_controlnet_scale: float,
        far_strength: float,
        far_controlnet_scale: float,
        depth_map: Image.Image | None = None,
    ):
        self.near_strength = near_strength
        self.near_controlnet_scale = near_controlnet_scale
        self.far_strength = far_strength
        self.far_controlnet_scale = far_controlnet_scale
        self.depth_map = depth_map

    def run(self, pipe, embeds, working, steps):
        depth_map = self.depth_map if self.depth_map is not None else depth.get_depth_map(working)
        mask = depth.depth_to_alpha_mask(depth_map)

        final = _two_pass_blend(
            pipe,
            embeds,
            working,
            steps,
            params_a=(self.near_strength, self.near_controlnet_scale),
            params_b=(self.far_strength, self.far_controlnet_scale),
            mask=mask,
            label_a="near",
            label_b="far",
        )

        return GenerationResult(final=final)


def apply_essence(
    essence_id: str,
    target_image_path: str,
    steps: int = DEFAULT_STEPS,
    strength: float | None = None,
    controlnet_scale: float | None = None,
    blend_mode: str = "subject",
    subject_strength: float | None = None,
    subject_controlnet_scale: float | None = None,
    depth_near_strength: float | None = None,
    depth_near_controlnet_scale: float | None = None,
    depth_far_strength: float | None = None,
    depth_far_controlnet_scale: float | None = None,
    preserve_color: bool = True,
    compute_depth: bool = True,
    content_aware_masking: bool = True,
) -> dict:
    """Runs the real SDXL img2img + InstantStyle IP-Adapter + Tile ControlNet
    pipeline: the target photo is used both as the img2img init image and as
    the ControlNet's control image (the Tile ControlNet's "Tile Var" /
    image-variation mode wants just the plain resized image, no edge/blur
    preprocessing — see pipeline_manager.py), restyled toward the essence's
    embedding. The ControlNet holds structure throughout denoising — this is
    what actually keeps content/layout intact; `strength` alone couldn't
    (see the module + pipeline_manager docstrings).

    Picks a GenerationStrategy (above) and shapes its result into the API
    response. Returns only the original and final frames (not a
    per-diffusion-step sequence) — the Main Stage's crossfade still runs
    over its own fixed duration regardless (spec §4.2.1's animation-
    generation decoupling), it just has one hop instead of several for now.
    True progressive previews (decoding intermediate latents during
    generation) are a natural follow-up, not yet implemented.

    blend_mode: "subject" (default — rembg subject/face-aware two-pass
    blending, already validated on portraits), "depth" (continuous
    depth-driven two-pass blending for photos without one clear subject —
    see DepthGradientStrategy/depth.py), or "none" (flat single pass).
    Any unrecognized value falls back to "subject", the existing default.

    preserve_color (default True — reported directly against a real run
    where a strongly-colored essence tinted the whole photo toward its hue):
    restores the target's original color post-generation, keeping only the
    stylized result's luminance/texture. See color_transfer.py. False uses
    the essence's own color untouched, same as the pipeline's original
    behavior.

    compute_depth (default True): estimates a depth map for the target
    (see depth.py) so every creation can drive the Media Page's parallax
    hover effect, not just blend_mode="depth" ones — previously this only
    ran inside DepthGradientStrategy. An escape hatch, not a UI toggle, for
    anyone who wants to skip the added cost; DepthGradientStrategy still
    computes its own if this is False but blend_mode="depth" is requested.

    content_aware_masking (default True): a second, complementary pass
    against essence content leaking into generation, on top of
    essence_store.py's distillation-time multi-crop purification — down-
    weights embedding dimensions that correlate with the *target* photo's
    own content (see content_mask.py; adapted from MaskST, arXiv:2502.07466,
    substituting the target's own CLIP embedding for that paper's content
    text prompt, since this pipeline has none). One extra, cheap CLIP encode
    per apply — not a diffusion pass. Escape hatch, not a UI toggle.
    """
    t_load = time.perf_counter()
    pipe = pipeline_manager.get_pipeline_blocking()
    embeds = essence_store.load_embedding(essence_id, pipe._execution_device)

    target = Image.open(target_image_path)
    working = _resize_working(target)
    print(f"[apply] load target: {(time.perf_counter() - t_load) * 1000:.0f}ms")

    if content_aware_masking:
        t_mask = time.perf_counter()
        target_embed = essence_store.extract_ip_adapter_embedding(pipe, working)
        embeds = [content_mask.mask_content_correlated(embeds[0], target_embed)]
        print(f"[apply] content-aware masking: {(time.perf_counter() - t_mask) * 1000:.0f}ms")

    depth_map = None
    if compute_depth:
        t_depth = time.perf_counter()
        depth_map = depth.get_depth_map(working)
        print(f"[apply] depth estimation: {(time.perf_counter() - t_depth) * 1000:.0f}ms")

    bg_strength = strength if strength is not None else DEFAULT_STRENGTH
    bg_controlnet_scale = (
        controlnet_scale if controlnet_scale is not None else pipeline_manager.CONTROLNET_CONDITIONING_SCALE
    )

    strategy: GenerationStrategy
    if blend_mode == "depth":
        strategy = DepthGradientStrategy(
            depth_near_strength if depth_near_strength is not None else DEPTH_NEAR_STRENGTH,
            depth_near_controlnet_scale if depth_near_controlnet_scale is not None else DEPTH_NEAR_CONTROLNET_SCALE,
            depth_far_strength if depth_far_strength is not None else DEPTH_FAR_STRENGTH,
            depth_far_controlnet_scale if depth_far_controlnet_scale is not None else DEPTH_FAR_CONTROLNET_SCALE,
            depth_map=depth_map,
        )
    elif blend_mode == "none":
        strategy = SinglePassStrategy(bg_strength, bg_controlnet_scale)
    else:  # "subject" — the default, and the fallback for any unrecognized value
        strategy = SubjectIsolatedStrategy(bg_strength, bg_controlnet_scale, subject_strength, subject_controlnet_scale)

    result = strategy.run(pipe, embeds, working, steps)

    final = result.final
    if preserve_color:
        t_color = time.perf_counter()
        final = color_transfer.preserve_original_color(final, working)
        print(f"[apply] preserve_color: {(time.perf_counter() - t_color) * 1000:.0f}ms")

    return {
        "steps": [to_data_url(working), to_data_url(final)],
        "final": to_data_url(final),
        "depth_map": to_data_url(depth_map) if depth_map is not None else None,
        "subject_detected": result.subject_detected,
        "face_detected": result.face_detected,
        "suggested_subject_strength": result.suggested_subject_strength,
        "suggested_subject_controlnet_scale": result.suggested_subject_controlnet_scale,
    }
