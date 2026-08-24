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

Essence-on-disk schema (spec §7):
    essences/<id>/
        meta.json             {id, name, technique, created_at, color}
        thumbnail.png          source reference, resized
        embedding.safetensors  the real IP-Adapter image embedding
        embedding.json         UI-only color/tone stats (shelf badge, bottle
                                pour animation color) — NOT used for the
                                actual style transfer
"""
from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image
from safetensors.torch import load_file, save_file

import paths
import pipeline_manager

TECHNIQUE = "instantstyle-sdxl-controlnet-v1"
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


def extract_essence(reference_image_path: str, name: str | None = None) -> dict:
    src = Image.open(reference_image_path)
    dominant = _dominant_color(src)

    pipe = pipeline_manager.get_pipeline_blocking()
    # `_execution_device`, not `.device` — under enable_model_cpu_offload
    # (see pipeline_manager.py) components idle on CPU until their hook
    # moves them to GPU for their turn, so `.device` doesn't reliably name
    # the GPU. `_execution_device` is what the pipeline's own __call__ uses
    # internally for this exact reason.
    embeds = pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_image=[src.convert("RGB")],
        ip_adapter_image_embeds=None,
        device=pipe._execution_device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )

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

    meta = {
        "id": essence_id,
        "name": name or Path(reference_image_path).stem,
        "technique": TECHNIQUE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "color": list(dominant),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return {**meta, "thumbnail": _to_data_url(thumb)}


def list_essences() -> list[dict]:
    out = []
    root = paths.essences_dir()
    for d in root.iterdir():
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        thumb_path = d / "thumbnail.png"
        thumbnail = _to_data_url(Image.open(thumb_path)) if thumb_path.exists() else None
        out.append({**meta, "thumbnail": thumbnail})
    out.sort(key=lambda e: e["created_at"], reverse=True)
    return out


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


def apply_essence(
    essence_id: str,
    target_image_path: str,
    steps: int = DEFAULT_STEPS,
    strength: float | None = None,
    controlnet_scale: float | None = None,
) -> dict:
    """Runs the real SDXL img2img + InstantStyle IP-Adapter + Tile ControlNet
    pipeline: the target photo is used both as the img2img init image and as
    the ControlNet's control image (the Tile ControlNet's "Tile Var" /
    image-variation mode wants just the plain resized image, no edge/blur
    preprocessing — see pipeline_manager.py), restyled toward the essence's
    embedding. The ControlNet holds structure throughout denoising — this is
    what actually keeps content/layout intact; `strength` alone couldn't
    (see the module + pipeline_manager docstrings).

    Returns only the original and final frames (not a per-diffusion-step
    sequence) — the Main Stage's crossfade still runs over its own fixed
    duration regardless (spec §4.2.1's animation-generation decoupling), it
    just has one hop instead of several for now. True progressive previews
    (decoding intermediate latents during generation) are a natural
    follow-up, not yet implemented.
    """
    pipe = pipeline_manager.get_pipeline_blocking()
    embeds = _load_embedding(essence_id, pipe._execution_device)

    target = Image.open(target_image_path)
    working = _resize_working(target)

    result = pipe(
        prompt="",
        negative_prompt=NEGATIVE_PROMPT,
        image=working,
        control_image=working,
        controlnet_conditioning_scale=(
            controlnet_scale if controlnet_scale is not None else pipeline_manager.CONTROLNET_CONDITIONING_SCALE
        ),
        ip_adapter_image_embeds=embeds,
        strength=strength if strength is not None else DEFAULT_STRENGTH,
        guidance_scale=DEFAULT_GUIDANCE,
        num_inference_steps=steps,
    )
    final = result.images[0]

    return {"steps": [_to_data_url(working), _to_data_url(final)], "final": _to_data_url(final)}
