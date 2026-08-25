"""Essence persistence, schema, and analysis (spec §2.1, §7, + the structured-
Essence extension). Split out of what used to be a single essence.py once
that file grew to cover extraction, disk storage, *and* generation — this
module owns everything except actually running the diffusion pipeline
against a target photo. See generation.py for that (apply_essence and the
GenerationStrategy classes), which imports load_embedding from here.

Essence-on-disk schema (spec §7 + the structured-Essence extension — see
essence_models.py):
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
meta.json, and never recomputed — generation.py's apply_essence loads the
embedding and does not touch style_analysis/ at all. None of it feeds
generation yet (spec's own explicit sequencing: verify extraction before
deciding how it influences diffusion).

blend_essences (the Cauldron) is a second way to create an Essence —
distilling one from a weighted mix of other Essences and/or fresh reference
photos, instead of a single reference. It saves the exact same on-disk
shape (a blended Essence is a completely normal Essence afterward — usable,
deletable, blendable again), just with technique=BLEND_TECHNIQUE and no
analysis fields populated (see that function's docstring for why).

Multi-crop style purification (_extract_purified_embedding): every fresh
reference photo that becomes part of an Essence — extract_essence's own
reference, or an image ingredient in a Cauldron blend — is embedded not
just once as a whole image but also from several overlapping crops of
itself, then combined by _purify_embedding so embedding dimensions that
stay stable no matter which crop you sampled (texture, palette, brushwork
— true of the whole piece) are kept, while dimensions that swing depending
on which crop you took (whatever specific object happened to be in that
region) get softly down-weighted. This is a from-a-single-image adaptation
of the same content/style disentanglement idea behind recent style-transfer
research — MaskST (arXiv:2502.07466, ICLR 2025) masks embedding dimensions
correlated with a *text* description of content; Rasa's pipeline has no
text prompt anywhere (apply_essence always runs prompt=""), so this uses
variance across crops of the *same* reference as the content signal
instead — no text, no target image, no training data needed, works at
distillation time exactly like the rest of this module already does.
"""
from __future__ import annotations

import json
import random
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file, save_file

import paths
import pipeline_manager
from essence_models import BlendIngredientInfo, EssenceMeta
from imaging import to_data_url
from style_analysis.palette import extract_palette
from style_analysis.statistics import analyze_style_statistics
from style_analysis.stroke import analyze_stroke
from style_analysis.texture import analyze_texture

TECHNIQUE = "instantstyle-sdxl-controlnet-v1"
BLEND_TECHNIQUE = "instantstyle-sdxl-controlnet-v1-blend"  # the Cauldron (blend_essences) — see its docstring
ESSENCE_SCHEMA_VERSION = 2
THUMBNAIL_SIZE = (160, 160)
PREVIEW_MAX_DIM = 1600  # generous — this is a display preview, not the working-resolution copy generation.py uses

# Multi-crop style purification (see module docstring). NUM_PURIFICATION_CROPS
# additional crops are sampled alongside the whole image itself, so every
# extraction combines this many samples total. PURIFICATION_STRENGTH is the
# maximum down-weight applied to the single highest-variance dimension (a
# soft mask, not a hard zero — see _purify_embedding); 0 would disable
# purification entirely (equivalent to the old plain-mean behavior), 1
# would fully zero out the most content-like dimensions. CROP_SEED is fixed
# rather than random so the same reference photo always purifies the same
# way — determinism matters here the same way it did for palette extraction
# (see tests/test_palette.py's test_is_deterministic).
NUM_PURIFICATION_CROPS = 5
PURIFICATION_STRENGTH = 0.4
CROP_SEED = 0


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


def preview_data_url(image_path: str) -> str:
    """Decodes any Pillow-openable image (including HEIC/HEIF via paths.py's
    pillow-heif registration) and re-encodes it as a PNG data: URL. Exists
    for electron/main.js's image:readAsDataUrl to call for formats Chromium
    has no <img> codec for at all (HEIC/HEIF) or won't reliably display from
    a data: URL (TIFF) — see that handler's BROWSER_DISPLAYABLE_EXTENSIONS.
    """
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((PREVIEW_MAX_DIM, PREVIEW_MAX_DIM))
    return to_data_url(img)


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


def extract_ip_adapter_embedding(pipe, image: Image.Image):
    """Runs the real CLIP-image-encoder + IP-Adapter projector on `image`,
    returning the raw (2, 1, tokens, dim) embedding tensor — [uncond, cond]
    stacked, the shape prepare_ip_adapter_image_embeds always returns for a
    single loaded IP-Adapter. Public (no leading underscore) — used within
    this module by _extract_purified_embedding (extract_essence,
    blend_essences), and by generation.py's apply-time content-aware
    masking to embed the *target* photo the same way (see content_mask.py).

    `_execution_device`, not `.device` — under enable_model_cpu_offload
    (see pipeline_manager.py) components idle on CPU until their hook moves
    them to GPU for their turn, so `.device` doesn't reliably name the GPU.
    `_execution_device` is what the pipeline's own __call__ uses internally
    for this exact reason.
    """
    embeds = pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_image=[image.convert("RGB")],
        ip_adapter_image_embeds=None,
        device=pipe._execution_device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    return embeds[0]


def _sample_crops(image: Image.Image, count: int, seed: int = CROP_SEED) -> list[Image.Image]:
    """Deterministic (seeded), overlapping, half-scale crops of `image` for
    multi-crop style purification (see module docstring). Half-scale: large
    enough to still show meaningful local texture, small enough to actually
    vary in content from crop to crop — a 95%-scale crop is basically the
    whole image again, with nothing for _purify_embedding's variance signal
    to find.
    """
    rng = random.Random(seed)
    w, h = image.size
    crop_w, crop_h = max(1, w // 2), max(1, h // 2)
    crops = []
    for _ in range(count):
        x0 = rng.randint(0, max(0, w - crop_w))
        y0 = rng.randint(0, max(0, h - crop_h))
        crops.append(image.crop((x0, y0, x0 + crop_w, y0 + crop_h)))
    return crops


def _purify_embedding(samples: list, strength: float = PURIFICATION_STRENGTH):
    """Combines several embedding samples of the same reference (the whole
    image plus a few crops of it — see _sample_crops) into one purified
    embedding: the per-dimension mean, down-weighted by that dimension's own
    variance across the samples.

    A dimension that stays roughly the same no matter which crop it came
    from is describing something true of the whole piece regardless of
    which part you're looking at — texture, palette, brushwork: style. A
    dimension that swings a lot from crop to crop is reacting to whatever
    specific thing happened to be in that particular region: content. This
    is a from-a-single-image adaptation of the disentanglement idea behind
    MaskST (arXiv:2502.07466) — see the module docstring for why that
    paper's own text-conditioned masking doesn't port directly onto a
    pipeline with no text prompt.

    Soft down-weight, not a hard mask: even the single highest-variance
    dimension only loses `strength` of its influence, never all of it —
    this is a noisier signal (5-6 crops of one image) than a text- or
    dataset-conditioned mask, so a gentle touch is the honest choice.
    Computed in float32 regardless of the samples' own dtype (fp16 std can
    be numerically unstable) and cast back before returning.
    """
    dtype = samples[0].dtype
    stacked = torch.stack([s.float() for s in samples], dim=0)
    mean = stacked.mean(dim=0)

    if len(samples) < 2:
        return mean.to(dtype)  # nothing to compare against — no variance signal, no purification possible

    std = stacked.std(dim=0)
    std_min, std_max = std.min(), std.max()
    if (std_max - std_min).item() < 1e-8:
        return mean.to(dtype)  # degenerate: identical (or near-identical) across every sample, nothing to down-weight

    std_norm = (std - std_min) / (std_max - std_min)
    weight = 1.0 - strength * std_norm
    return (mean * weight).to(dtype)


def _extract_purified_embedding(pipe, image: Image.Image):
    """Multi-crop style purification, end to end: embeds the whole image
    plus NUM_PURIFICATION_CROPS crops of it, then combines them via
    _purify_embedding. Used by extract_essence and by blend_essences' image
    ingredients — anywhere a fresh reference photo becomes part of an
    Essence (an already-saved Essence loaded as a blend ingredient skips
    this — it was purified once already, at its own distillation time).
    """
    samples = [extract_ip_adapter_embedding(pipe, image)]
    for crop in _sample_crops(image, NUM_PURIFICATION_CROPS):
        samples.append(extract_ip_adapter_embedding(pipe, crop))
    return _purify_embedding(samples)


def extract_essence(reference_image_path: str, name: str | None = None) -> dict:
    src = Image.open(reference_image_path)
    dominant = _dominant_color(src)

    pipe = pipeline_manager.get_pipeline_blocking()
    embed_start = time.perf_counter()
    embed_tensor = _extract_purified_embedding(pipe, src)
    print(
        f"[essence] ip_adapter_embed (purified, {1 + NUM_PURIFICATION_CROPS} samples): "
        f"{(time.perf_counter() - embed_start) * 1000:.0f}ms"
    )

    essence_id = uuid.uuid4().hex[:12]
    out_dir = paths.essences_dir() / essence_id
    out_dir.mkdir(parents=True, exist_ok=True)

    thumb = src.convert("RGB").copy()
    thumb.thumbnail(THUMBNAIL_SIZE)
    thumb.save(out_dir / "thumbnail.png")

    # Moved to CPU before saving — safetensors needs a portable on-disk
    # format regardless of which device extracted it (also lets
    # load_embedding pick the right device back up at apply time).
    save_file({"ip_adapter_embed": embed_tensor.contiguous().cpu()}, out_dir / "embedding.safetensors")

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
    return {**data, "analysis": analysis, "thumbnail": to_data_url(thumb) if thumb else None}


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


def load_embedding(essence_id: str, device):
    """Used by generation.py at apply time — public (no leading underscore)
    since it's a cross-module entry point now, not a private helper.
    """
    path = paths.essences_dir() / essence_id / "embedding.safetensors"
    if not path.exists():
        raise FileNotFoundError(essence_id)
    tensors = load_file(path, device=str(device))
    return [tensors["ip_adapter_embed"]]


def _blend_colors(colors: list[tuple[int, int, int]], weights: list[float]) -> tuple[int, int, int]:
    """Pure weighted per-channel average, rounded/clamped to 0..255. Used by
    blend_essences for the new Essence's shelf-badge/pour-animation color —
    same role _dominant_color plays for a single-reference Essence.
    """
    total = sum(weights) or 1.0

    def clamp(v: float) -> int:
        return max(0, min(255, round(v)))

    return tuple(
        clamp(sum(c[ch] * w for c, w in zip(colors, weights)) / total) for ch in range(3)
    )  # type: ignore[return-value]


def _composite_weighted_thumbnail(images: list[Image.Image], weights: list[float]) -> Image.Image:
    """Horizontal strips proportional to weight, each center-cropped to fill
    its slot via ImageOps.fit (already in Pillow, no new dependency) — so a
    blended Essence's shelf thumbnail visually reads as "mostly this, a bit
    of that" instead of picking one arbitrary ingredient's image.
    """
    w, h = THUMBNAIL_SIZE
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    total = sum(weights) or 1.0

    # Ingredients with negligible weight don't claim canvas space — without
    # this, a zero-weight ingredient that happens to land last would still
    # visually fill the remainder (see below: the last visible strip
    # absorbs rounding remainder, which is correct for real ingredients but
    # wrong for one contributing nothing).
    visible = [(img, weight) for img, weight in zip(images, weights) if weight / total > 0.005]
    if not visible:
        visible = list(zip(images, weights))  # degenerate: draw something rather than an empty canvas

    x = 0
    last = len(visible) - 1
    for i, (img, weight) in enumerate(visible):
        remaining = w - x
        strip_w = remaining if i == last else round(w * weight / total)
        strip_w = max(0, min(strip_w, remaining))
        if strip_w <= 0:
            continue
        cropped = ImageOps.fit(img.convert("RGB"), (strip_w, h), method=Image.LANCZOS)
        canvas.paste(cropped, (x, 0))
        x += strip_w
    return canvas


def blend_essences(ingredients: list[dict], name: str | None = None) -> dict:
    """Distills one new Essence from a weighted mix of other Essences and/or
    fresh reference photos — the Cauldron. Both source kinds resolve to the
    same underlying thing (an IP-Adapter embedding tensor + a dominant
    color + a thumbnail-source image), so blending is just a weighted
    average across whichever mix of sources was given — the same operation
    whether it reads as "blend two essences" or "distill from several
    photos at once."

    Each ingredient dict: {"type": "image"|"essence",
    "image_path"|"essence_id": str, "weight": float}.

    No style_analysis/ run here — palette/texture/stroke/style_statistics
    all stay None on the result. Those analyzers need one single reference
    image; there isn't one for a blend, and fabricating one from a
    synthesized composite would misrepresent what was actually analyzed
    (see essence_models.py's EssenceMeta.blended_from docstring).
    """
    if not ingredients:
        raise ValueError("blend_essences needs at least one ingredient")

    pipe = pipeline_manager.get_pipeline_blocking()
    device = pipe._execution_device

    embed_tensors = []
    thumb_sources: list[Image.Image] = []
    colors: list[tuple[int, int, int]] = []
    weights: list[float] = []
    provenance: list[BlendIngredientInfo] = []

    for ing in ingredients:
        kind = ing.get("type")
        weight = float(ing.get("weight") or 1.0)

        if kind == "image":
            image_path = ing.get("image_path")
            if not image_path:
                raise ValueError("image ingredient missing image_path")
            src = Image.open(image_path)
            embed_tensors.append(_extract_purified_embedding(pipe, src))
            thumb = src.convert("RGB").copy()
            thumb.thumbnail(THUMBNAIL_SIZE)
            thumb_sources.append(thumb)
            colors.append(_dominant_color(src))
            label = Path(image_path).stem

        elif kind == "essence":
            essence_id = ing.get("essence_id")
            if not essence_id:
                raise ValueError("essence ingredient missing essence_id")
            essence_dir = paths.essences_dir() / essence_id
            meta_path = essence_dir / "meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(essence_id)
            source_meta = EssenceMeta.model_validate(json.loads(meta_path.read_text()))
            [tensor] = load_embedding(essence_id, device)
            embed_tensors.append(tensor)
            thumb_path = essence_dir / "thumbnail.png"
            thumb_sources.append(
                Image.open(thumb_path).convert("RGB")
                if thumb_path.exists()
                else Image.new("RGB", THUMBNAIL_SIZE, source_meta.color)
            )
            colors.append(source_meta.color)
            label = source_meta.name

        else:
            raise ValueError(f"unknown ingredient type: {kind!r}")

        weights.append(weight)
        provenance.append(BlendIngredientInfo(name=label, weight=weight, source=kind))

    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("ingredient weights must sum to a positive value")
    normalized_weights = [w / total_weight for w in weights]
    for info, w in zip(provenance, normalized_weights):
        info.weight = w

    blended_tensor = sum(t.to(device) * w for t, w in zip(embed_tensors, normalized_weights))
    blended_color = _blend_colors(colors, normalized_weights)
    blended_thumb = _composite_weighted_thumbnail(thumb_sources, normalized_weights)

    essence_id = uuid.uuid4().hex[:12]
    out_dir = paths.essences_dir() / essence_id
    out_dir.mkdir(parents=True, exist_ok=True)

    blended_thumb.save(out_dir / "thumbnail.png")
    save_file({"ip_adapter_embed": blended_tensor.contiguous().cpu()}, out_dir / "embedding.safetensors")
    (out_dir / "embedding.json").write_text(json.dumps({"dominant_color": list(blended_color)}, indent=2))

    default_name = " + ".join(p.name for p in provenance)
    meta = EssenceMeta(
        id=essence_id,
        name=name or default_name,
        technique=BLEND_TECHNIQUE,
        created_at=datetime.now(timezone.utc).isoformat(),
        color=blended_color,
        version=ESSENCE_SCHEMA_VERSION,
        blended_from=provenance,
    )
    (out_dir / "meta.json").write_text(meta.model_dump_json(indent=2))

    return _meta_response(meta, blended_thumb)


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
