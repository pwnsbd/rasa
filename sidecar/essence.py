"""Essence extraction + application (spec §2.1, §2.2b, §7).

Real implementation: SDXL + InstantStyle + a Tile ControlNet (see
pipeline_manager.py for why SDXL rather than the spec's originally-suggested
Flux, and why ControlNet is needed alongside IP-Adapter — plain img2img
`strength` alone let style-driven regeneration drift too much of the
target's own content/layout away, which is exactly the failure the
InstantStyle authors' own follow-up paper, InstantStyle-Plus, fixes with a
Tile ControlNet). An earlier "mock-palette" version of this file (dominant
color + duotone tint, no real model) stood in while the pipeline contracts
below were being built — see git history if useful as reference.

`StyleExtractor.extract(reference_image) -> embedding` and
`StyleExtractor.apply(embedding, target_image) -> conditioning` (spec §2.2b)
map onto `extract_essence` and `apply_essence` below.

Essence-on-disk schema (spec §7 + the structured-Essence extension —
see essence_models.py):
    essences/<id>/
        meta.json             EssenceMeta (id, name, technique, created_at,
                               color, version, palette, texture, stroke,
                               style_statistics)
        thumbnail.png          source reference, resized
        embedding.safetensors  the real IP-Adapter image embedding
        embedding.json         UI-only color/tone stats (shelf badge, bottle
                                pour animation color) — NOT used for the
                                actual style transfer
        stroke_map.png         stroke orientation-field debug visualization,
                                when stroke analysis succeeded

Structured analysis (palette/texture/stroke/style_statistics) is
Distillation-time only: computed once in extract_essence below, saved to
meta.json, and never recomputed — apply_essence loads the embedding and
does not touch style_analysis/ at all. None of it feeds generation yet
(spec's own explicit sequencing: verify extraction before deciding how it
influences diffusion).
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

import paths
import pipeline_manager
import segmentation
from essence_models import EssenceMeta
from style_analysis.palette import extract_palette
from style_analysis.statistics import analyze_style_statistics
from style_analysis.stroke import analyze_stroke
from style_analysis.texture import analyze_texture

TECHNIQUE = "instantstyle-sdxl-controlnet-v1"
ESSENCE_SCHEMA_VERSION = 2
THUMBNAIL_SIZE = (160, 160)
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


def _to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    import base64

    buf = BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


def _saturation(rgb: tuple[int, int, int]) -> float:
    r, g, b = (c / 255 for c in rgb)
    hi, lo = max(r, g, b), min(r, g, b)
    return 0.0 if hi == 0 else (hi - lo) / hi


def _dominant_color(img: Image.Image) -> tuple[int, int, int]:
    """UI-only accent color (shelf bottle badge, Distillation Room's pour
    animation) — has no bearing on the actual style transfer, which is
    driven entirely by the real IP-Adapter embedding below. Ranks by
    saturation among colors covering a meaningful share of the image so a
    vivid accent wins over a duller majority background (sky, wall, a white
    page).
    """
    small = img.convert("RGB").resize((64, 64))
    quantized = small.quantize(colors=8, method=Image.MEDIANCUT)
    palette = quantized.getpalette()
    color_counts = quantized.getcolors()
    total = sum(count for count, _ in color_counts)

    def rgb_of(index: int) -> tuple[int, int, int]:
        return tuple(palette[index * 3: index * 3 + 3])

    MIN_SHARE = 0.02
    candidates = [c for c in color_counts if c[0] / total >= MIN_SHARE] or color_counts
    best = max(candidates, key=lambda c: _saturation(rgb_of(c[1])))
    return rgb_of(best[1])


def _resize_working(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((WORKING_MAX_DIM, WORKING_MAX_DIM))
    # SDXL wants multiple-of-8 dimensions.
    w, h = (d - d % 8 for d in img.size)
    return img.resize((max(w, 8), max(h, 8)))


PREVIEW_MAX_DIM = 1600  # generous — this is a display preview, not the working-resolution copy used for generation


def preview_data_url(image_path: str) -> str:
    """Decodes any Pillow-openable image (including HEIC/HEIF via paths.py's
    pillow-heif registration) and re-encodes it as a PNG data: URL. Exists
    for electron/main.js's image:readAsDataUrl to call for formats Chromium
    has no <img> codec for at all (HEIC/HEIF) or won't reliably display from
    a data: URL (TIFF) — see that handler's BROWSER_DISPLAYABLE_EXTENSIONS.
    """
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((PREVIEW_MAX_DIM, PREVIEW_MAX_DIM))
    return _to_data_url(img)


def _run_analyzers(src: Image.Image, out_dir: Path) -> dict:
    """Runs the four style_analysis/ analyzers, each independently
    try/excepted — per the spec, one failing (e.g. stroke) must never block
    Essence creation, since the IP-Adapter embedding remains the one
    load-bearing component. Returns a dict of the field name -> profile (or
    None on failure), ready to spread into EssenceMeta. Timing for each is
    printed — see the module docstring on why (dev-visible, not a new
    logging framework).
    """
    results: dict = {"palette": None, "texture": None, "stroke": None, "style_statistics": None}

    for field, fn in (
        ("palette", lambda: extract_palette(src)),
        ("texture", lambda: analyze_texture(src)),
        ("style_statistics", lambda: analyze_style_statistics(src)),
    ):
        start = time.perf_counter()
        try:
            results[field] = fn()
            print(f"[essence] {field}: {(time.perf_counter() - start) * 1000:.0f}ms")
        except Exception as e:  # noqa: BLE001 — analysis failure must not block Essence creation
            print(f"[essence] {field} FAILED ({(time.perf_counter() - start) * 1000:.0f}ms): {e}")

    start = time.perf_counter()
    try:
        stroke_profile, stroke_viz = analyze_stroke(src)
        stroke_viz.save(out_dir / "stroke_map.png")
        stroke_profile.orientation_map_path = "stroke_map.png"
        results["stroke"] = stroke_profile
        print(f"[essence] stroke: {(time.perf_counter() - start) * 1000:.0f}ms")
    except Exception as e:  # noqa: BLE001
        print(f"[essence] stroke FAILED ({(time.perf_counter() - start) * 1000:.0f}ms): {e}")

    return results


def extract_essence(reference_image_path: str, name: str | None = None) -> dict:
    src = Image.open(reference_image_path)
    dominant = _dominant_color(src)

    pipe = pipeline_manager.get_pipeline_blocking()
    # `_execution_device`, not `.device` — under enable_model_cpu_offload
    # (see pipeline_manager.py) components idle on CPU until their hook
    # moves them to GPU for their turn, so `.device` doesn't reliably name
    # the GPU. `_execution_device` is what the pipeline's own __call__ uses
    # internally for this exact reason.
    embed_start = time.perf_counter()
    embeds = pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_image=[src.convert("RGB")],
        ip_adapter_image_embeds=None,
        device=pipe._execution_device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    print(f"[essence] ip_adapter_embed: {(time.perf_counter() - embed_start) * 1000:.0f}ms")

    essence_id = uuid.uuid4().hex[:12]
    out_dir = paths.essences_dir() / essence_id
    out_dir.mkdir(parents=True, exist_ok=True)

    thumb = src.convert("RGB").copy()
    thumb.thumbnail(THUMBNAIL_SIZE)
    thumb.save(out_dir / "thumbnail.png")

    # embeds is a list (one entry per loaded IP-Adapter — we only load one)
    # of a single tensor shaped (2, 1, tokens, dim): [uncond, cond] stacked.
    # Moved to CPU before saving — safetensors needs a portable on-disk
    # format regardless of which device extracted it (also lets
    # _load_embedding pick the right device back up at apply time).
    save_file({"ip_adapter_embed": embeds[0].contiguous().cpu()}, out_dir / "embedding.safetensors")

    (out_dir / "embedding.json").write_text(json.dumps({"dominant_color": list(dominant)}, indent=2))

    analysis = _run_analyzers(src, out_dir)

    meta = EssenceMeta(
        id=essence_id,
        name=name or Path(reference_image_path).stem,
        technique=TECHNIQUE,
        created_at=datetime.now(timezone.utc).isoformat(),
        color=dominant,
        version=ESSENCE_SCHEMA_VERSION,
        **analysis,
    )
    (out_dir / "meta.json").write_text(meta.model_dump_json(indent=2))

    return _meta_response(meta, thumb)


def _meta_response(meta: EssenceMeta, thumb: Image.Image | None) -> dict:
    """Shapes an EssenceMeta into the dict extract_essence/list_essences
    return: flat top-level fields the frontend/shelf already consumes
    (id/name/technique/created_at/color/version), plus a nested `analysis`
    block for the new structured data — kept separate so old response
    consumers are unaffected, and so raw tensors/orientation arrays never
    leave the backend (orientation_map_path is just a filename string).
    """
    data = meta.model_dump()
    analysis = {
        "palette": data.pop("palette"),
        "texture": data.pop("texture"),
        "stroke": data.pop("stroke"),
        "style_statistics": data.pop("style_statistics"),
    }
    return {**data, "analysis": analysis, "thumbnail": _to_data_url(thumb) if thumb else None}


def list_essences() -> list[dict]:
    out = []
    root = paths.essences_dir()
    for d in root.iterdir():
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        # model_validate, not raw dict access — an essence saved before this
        # schema existed has none of the new keys; every new field is
        # Optional with a default (see essence_models.py), so it loads
        # cleanly with analysis fields set to None rather than crashing or
        # needing a migration.
        meta = EssenceMeta.model_validate(json.loads(meta_path.read_text()))
        thumb_path = d / "thumbnail.png"
        thumb = Image.open(thumb_path) if thumb_path.exists() else None
        out.append(_meta_response(meta, thumb))
    out.sort(key=lambda e: e["created_at"], reverse=True)
    return out


def export_debug_analysis(essence_id: str, out_dir: str | Path) -> Path:
    """Dev-only debug export (spec Phase 21) — dumps the reference thumbnail,
    each analyzer's JSON, the stroke visualization, and the full essence.json
    to `out_dir` for inspection. Not wired to any user-facing endpoint;
    call directly (e.g. from a Python shell) when verifying extraction.
    """
    src_dir = paths.essences_dir() / essence_id
    if not src_dir.is_dir():
        raise FileNotFoundError(essence_id)
    meta = EssenceMeta.model_validate(json.loads((src_dir / "meta.json").read_text()))

    dest = Path(out_dir) / essence_id
    dest.mkdir(parents=True, exist_ok=True)

    thumb_path = src_dir / "thumbnail.png"
    if thumb_path.exists():
        Image.open(thumb_path).convert("RGB").save(dest / "reference.jpg", quality=90)

    (dest / "palette.json").write_text(meta.palette.model_dump_json(indent=2) if meta.palette else "null")
    (dest / "texture.json").write_text(meta.texture.model_dump_json(indent=2) if meta.texture else "null")
    (dest / "stroke.json").write_text(meta.stroke.model_dump_json(indent=2) if meta.stroke else "null")
    (dest / "essence.json").write_text(meta.model_dump_json(indent=2))

    stroke_map = src_dir / "stroke_map.png"
    if stroke_map.exists():
        Image.open(stroke_map).save(dest / "stroke_visualization.png")

    return dest


def _load_embedding(essence_id: str, device):
    path = paths.essences_dir() / essence_id / "embedding.safetensors"
    if not path.exists():
        raise FileNotFoundError(essence_id)
    tensors = load_file(path, device=str(device))
    return [tensors["ip_adapter_embed"]]


def delete_essence(essence_id: str) -> None:
    """Removes an Essence's on-disk folder entirely. Does not touch any Media
    Page creations already made with it — those keep their saved image and
    essence_name (a label, not a live reference) even if the essence they
    came from is later deleted.
    """
    root = paths.essences_dir().resolve()
    out_dir = (root / essence_id).resolve()
    # essence_id ultimately comes from an HTTP path param — guard against a
    # "../.." id escaping essences_dir before it ever reaches rmtree.
    if not out_dir.is_relative_to(root) or not out_dir.is_dir():
        raise FileNotFoundError(essence_id)
    shutil.rmtree(out_dir)


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


def apply_essence(
    essence_id: str,
    target_image_path: str,
    steps: int = DEFAULT_STEPS,
    strength: float | None = None,
    controlnet_scale: float | None = None,
    isolate_subject: bool = True,
    subject_strength: float | None = None,
    subject_controlnet_scale: float | None = None,
) -> dict:
    """Runs the real SDXL img2img + InstantStyle IP-Adapter + Tile ControlNet
    pipeline: the target photo is used both as the img2img init image and as
    the ControlNet's control image (the Tile ControlNet's "Tile Var" /
    image-variation mode wants just the plain resized image, no edge/blur
    preprocessing — see pipeline_manager.py), restyled toward the essence's
    embedding. The ControlNet holds structure throughout denoising — this is
    what actually keeps content/layout intact; `strength` alone couldn't
    (see the module + pipeline_manager docstrings).

    Subject-isolated strength blending (see segmentation.py): when a subject
    is segmented from the target, generation runs *twice* — once at the
    background strength/controlnet_scale over the whole frame, once at a
    lower, tighter subject strength/controlnet_scale (suggested by face
    detection within the subject region, or overridden) — then composites
    the two with the subject's soft feathered mask. Both passes share one
    seeded generator: without that, the two outputs diverge in grain/noise/
    color balance independent of the strength difference, which reads as a
    mismatch at the mask boundary even though the mask itself is
    well-feathered. `isolate_subject=False`, or no distinguishable subject
    found, falls back to the original single-pass behavior.

    Returns only the original and final frames (not a per-diffusion-step
    sequence) — the Main Stage's crossfade still runs over its own fixed
    duration regardless (spec §4.2.1's animation-generation decoupling), it
    just has one hop instead of several for now. True progressive previews
    (decoding intermediate latents during generation) are a natural
    follow-up, not yet implemented.
    """
    t_load = time.perf_counter()
    pipe = pipeline_manager.get_pipeline_blocking()
    embeds = _load_embedding(essence_id, pipe._execution_device)

    target = Image.open(target_image_path)
    working = _resize_working(target)
    print(f"[apply] load target: {(time.perf_counter() - t_load) * 1000:.0f}ms")

    bg_strength = strength if strength is not None else DEFAULT_STRENGTH
    bg_controlnet_scale = (
        controlnet_scale if controlnet_scale is not None else pipeline_manager.CONTROLNET_CONDITIONING_SCALE
    )

    t_seg = time.perf_counter()
    mask = segmentation.get_subject_mask(working) if isolate_subject else None
    subject_detected = mask is not None
    print(f"[apply] segmentation: {(time.perf_counter() - t_seg) * 1000:.0f}ms (subject_detected={subject_detected})")

    face_detected = False
    suggested_strength = suggested_controlnet_scale = None

    if subject_detected:
        t_face = time.perf_counter()
        face_detected = segmentation.detect_face(working, mask)
        print(f"[apply] face detection: {(time.perf_counter() - t_face) * 1000:.0f}ms (face_detected={face_detected})")

        suggested_strength, suggested_controlnet_scale = segmentation.suggest_subject_params(face_detected)
        actual_subject_strength = subject_strength if subject_strength is not None else suggested_strength
        actual_subject_controlnet_scale = (
            subject_controlnet_scale if subject_controlnet_scale is not None else suggested_controlnet_scale
        )

        seed = int.from_bytes(os.urandom(4), "big")
        gen_a = torch.Generator(device=pipe._execution_device).manual_seed(seed)
        gen_b = torch.Generator(device=pipe._execution_device).manual_seed(seed)

        t_a = time.perf_counter()
        pass_a = _run_generation(pipe, embeds, working, bg_strength, bg_controlnet_scale, steps, gen_a)
        print(f"[apply] pass A (background): {time.perf_counter() - t_a:.1f}s")

        t_b = time.perf_counter()
        pass_b = _run_generation(
            pipe, embeds, working, actual_subject_strength, actual_subject_controlnet_scale, steps, gen_b
        )
        print(f"[apply] pass B (subject): {time.perf_counter() - t_b:.1f}s")

        t_composite = time.perf_counter()
        final = Image.composite(pass_b, pass_a, mask)
        print(f"[apply] composite: {(time.perf_counter() - t_composite) * 1000:.0f}ms")
    else:
        t_single = time.perf_counter()
        final = _run_generation(pipe, embeds, working, bg_strength, bg_controlnet_scale, steps)
        print(f"[apply] single pass: {time.perf_counter() - t_single:.1f}s")

    return {
        "steps": [_to_data_url(working), _to_data_url(final)],
        "final": _to_data_url(final),
        "subject_detected": subject_detected,
        "face_detected": face_detected,
        "suggested_subject_strength": suggested_strength,
        "suggested_subject_controlnet_scale": suggested_controlnet_scale,
    }
