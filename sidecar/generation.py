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

apply_essence's `mode` param picks between this whole diffusion pipeline
("restyle", the default) and a second, entirely non-diffusion apply engine
("texture_overlay" — see texture_overlay.py) for essences that are really
a material (glass, canvas, paper grain) rather than a painterly style,
where regenerating the image via diffusion only introduces unwanted
content drift a classical bump/emboss overlay doesn't have.
"""
from __future__ import annotations

import os
import statistics
import threading
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
import stroke_texture
import texture_overlay
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
NEGATIVE_PROMPT = pipeline_manager.NEGATIVE_PROMPT

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


SLOW_STEP_S = 6.0  # sec/step above which the UI warns that GPU memory is likely full/spilling


class ProgressTracker:
    """Thread-safe live progress of the current /apply, polled via
    GET /apply/progress while POST /apply blocks (they run in separate
    threadpool threads). sec_per_step is the rolling median of the last 3
    step durations, so one slow outlier doesn't flip the slow flag."""

    def __init__(self):
        self._lock = threading.Lock()
        self._reset_locked()

    def _reset_locked(self):
        self._active = False
        self._pass_index = 0
        self._pass_count = 1
        self._step = 0
        self._total_steps = 0
        self._durations: list[float] = []
        self._last_t: float | None = None

    def begin(self, pass_count: int = 1):
        with self._lock:
            self._reset_locked()
            self._active = True
            self._pass_count = pass_count

    def set_pass_count(self, pass_count: int):
        with self._lock:
            self._pass_count = pass_count

    def start_pass(self, now: float | None = None):
        with self._lock:
            self._active = True
            self._pass_index += 1
            self._step = 0
            self._total_steps = 0
            self._durations = []
            self._last_t = time.monotonic() if now is None else now

    def on_step(self, step: int, total_steps: int, now: float | None = None):
        """`step` is the 1-based number of steps completed."""
        t = time.monotonic() if now is None else now
        with self._lock:
            if self._last_t is not None:
                self._durations = (self._durations + [t - self._last_t])[-3:]
            self._last_t = t
            self._step = step
            self._total_steps = total_steps

    def finish(self):
        with self._lock:
            self._active = False

    def snapshot(self) -> dict:
        with self._lock:
            if not self._active:
                return {"active": False}
            sps = statistics.median(self._durations) if self._durations else None
            return {
                "active": True,
                "pass_index": self._pass_index,
                "pass_count": self._pass_count,
                "step": self._step,
                "total_steps": self._total_steps,
                "sec_per_step": sps,
                "slow": sps is not None and sps > SLOW_STEP_S,
            }


progress = ProgressTracker()


def _run_generation(pipe, embeds, working, strength, controlnet_scale, steps, generator=None):
    cached = pipeline_manager.cached_prompt_embeds()
    # Cached embeds (computed once at load for prompt="" + NEGATIVE_PROMPT)
    # skip both text encoders on every apply; fall back to strings if absent.
    prompt_kwargs = dict(cached) if cached else {"prompt": "", "negative_prompt": NEGATIVE_PROMPT}

    def _on_step_end(_pipe, i, _t, callback_kwargs):
        progress.on_step(i + 1, getattr(_pipe, "num_timesteps", steps))
        return callback_kwargs

    progress.start_pass()
    result = pipe(
        **prompt_kwargs,
        callback_on_step_end=_on_step_end,
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
    gen_a = torch.Generator(device=pipeline_manager.compute_device()).manual_seed(seed)
    gen_b = torch.Generator(device=pipeline_manager.compute_device()).manual_seed(seed)

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
            progress.set_pass_count(1)
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
    blend_mode: str = "none",
    subject_strength: float | None = None,
    subject_controlnet_scale: float | None = None,
    depth_near_strength: float | None = None,
    depth_near_controlnet_scale: float | None = None,
    depth_far_strength: float | None = None,
    depth_far_controlnet_scale: float | None = None,
    color_preservation: float = 1.0,
    stroke_amount: float = 0.0,
    compute_depth: bool = False,
    content_aware_masking: bool = False,
    mode: str = "restyle",
    texture_overlay_amount: float = 1.0,
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

    color_preservation (0..1, default 1.0 — reported directly against a
    real run where a strongly-colored essence tinted the whole photo toward
    its hue): how much of the target's original color to restore
    post-generation, keeping only the stylized result's luminance/texture
    at the high end. See color_transfer.py. 1.0 is the pipeline's original
    all-or-nothing "on" behavior; 0.0 lets the essence's own color through
    untouched (the original "off"); values between are a continuous blend.

    stroke_amount (0..1, default 0.0 — new and not yet validated against a
    real photo the way color_preservation was, so it defaults off): overlays
    a faint directional grain post-generation, oriented per-region to the
    essence's own measured stroke pattern (see stroke_texture.py) rather
    than applying the same adjustment everywhere the way every control above
    this one does. Silently a no-op when the essence has no persisted
    stroke field — a blended (Cauldron) Essence, one saved before this
    existed, or one whose stroke analysis failed at Distillation time (see
    essence_store.py's _run_analyzers) — same "missing analysis just means
    no effect" honesty as everywhere else this data is optional.

    compute_depth (default False — changed from an earlier True default:
    reported directly against a real run as a silent, un-toggleable cost —
    the blend_mode Subject/Depth/Off selector is a *different* setting and
    turning it to "Off"/"Subject" does nothing to this): estimates a depth
    map for the target (see depth.py) so a creation can drive the Media
    Page's parallax hover effect, independent of blend_mode. With the
    default False, parallax simply won't be available unless this is
    explicitly requested (e.g. a future UI toggle, or a direct API call) —
    an honest trade until this earns a real switch instead of an invisible
    tax. DepthGradientStrategy still computes its own depth map on the fly
    when blend_mode="depth" is explicitly requested, regardless of this flag
    — that map just doesn't get returned/persisted for parallax unless this
    is also on.

    content_aware_masking (default False — same reasoning as compute_depth
    above: a real, reported performance cost with no way to turn it off
    short of a direct API call). A second, complementary pass against
    essence content leaking into generation, on top of essence_store.py's
    distillation-time multi-crop purification — down-weights embedding
    dimensions that correlate with the *target* photo's own content (see
    content_mask.py; adapted from MaskST, arXiv:2502.07466, substituting
    the target's own CLIP embedding for that paper's content text prompt,
    since this pipeline has none). One extra, cheap CLIP encode
    per apply — not a diffusion pass. Escape hatch, not a UI toggle.

    mode ("restyle", the default, or "texture_overlay"): picks which of two
    entirely different apply *engines* runs, not another blend of the same
    one. "restyle" is everything above this paragraph — the real diffusion
    pipeline. "texture_overlay" skips the pipeline (and the model-readiness
    requirement — see app.py's apply_endpoint) entirely and instead
    composites the essence's own persisted texture swatch onto the target
    as a classical bump/emboss overlay (see texture_overlay.py), preserving
    the target's content with total fidelity — the right tool for an
    essence that's really a *material* (glass, canvas, paper grain) rather
    than a painterly style, where diffusion regeneration only introduces
    unwanted content drift. Every strength/blend_mode/color_preservation/
    stroke_amount param above is ignored in this mode. Raises ValueError if
    the essence has no persisted texture source (a blended Essence, or one
    saved before this existed).

    texture_overlay_amount (0..1, default 1.0, only used when mode ==
    "texture_overlay"): how strong the overlay reads — 0.0 leaves the
    target untouched, 1.0 is the full effect. See texture_overlay.py.
    """
    if mode == "texture_overlay":
        texture = essence_store.load_texture_source(essence_id)
        if texture is None:
            raise ValueError(
                f"Essence {essence_id} has no texture source (a blended Essence, or one saved before this existed) "
                "-- texture_overlay mode needs one"
            )
        target = Image.open(target_image_path)
        working = _resize_working(target)
        final = texture_overlay.apply_texture_overlay(working, texture, amount=texture_overlay_amount)
        return {
            "steps": [to_data_url(working), to_data_url(final)],
            "final": to_data_url(final),
            "depth_map": None,
            "subject_detected": False,
            "face_detected": False,
            "suggested_subject_strength": None,
            "suggested_subject_controlnet_scale": None,
        }

    t_load = time.perf_counter()
    pipe = pipeline_manager.get_pipeline_blocking()
    embeds = essence_store.load_embedding(essence_id, pipeline_manager.compute_device())

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

    progress.begin(pass_count=1 if blend_mode == "none" else 2)

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

    try:
        result = strategy.run(pipe, embeds, working, steps)
    finally:
        progress.finish()

    final = result.final
    if color_preservation > 0.0:
        t_color = time.perf_counter()
        final = color_transfer.preserve_original_color(final, working, amount=color_preservation)
        print(f"[apply] color_preservation={color_preservation:.2f}: {(time.perf_counter() - t_color) * 1000:.0f}ms")

    if stroke_amount > 0.0:
        stroke_field = essence_store.load_stroke_field(essence_id)
        if stroke_field is not None:
            t_stroke = time.perf_counter()
            theta, coherence = stroke_field
            final = stroke_texture.apply_stroke_texture(final, theta, coherence, amount=stroke_amount)
            print(f"[apply] stroke_amount={stroke_amount:.2f}: {(time.perf_counter() - t_stroke) * 1000:.0f}ms")

    return {
        "steps": [to_data_url(working), to_data_url(final)],
        "final": to_data_url(final),
        "depth_map": to_data_url(depth_map) if depth_map is not None else None,
        "subject_detected": result.subject_detected,
        "face_detected": result.face_detected,
        "suggested_subject_strength": result.suggested_subject_strength,
        "suggested_subject_controlnet_scale": result.suggested_subject_controlnet_scale,
    }
