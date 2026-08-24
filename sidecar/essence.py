"""Essence extraction + application (spec §2.1, §2.2b, §7).

This module is a placeholder "style-extraction technique": instead of
InstantStyle/StyleShot it derives a color/tone signature from the reference
image (dominant palette, average saturation/brightness) and "applies" it as a
progressive tint+contrast blend. It exists so the real pipeline contracts —
Essence file/folder schema, the extract/apply API shape, and the
animation-generation step-list used for the Main Stage's decoupled clock
(spec §4.2.1) — can be built and exercised end-to-end before the actual
models land.

`StyleExtractor.extract(reference_image) -> embedding` and
`StyleExtractor.apply(embedding, target_image) -> conditioning` (spec §2.2b)
are the real interface this mimics. Swapping this module for a real
InstantStyle/StyleShot implementation should only require changes here —
app.py, paths.py, and the frontend all talk to the shapes defined below.

Essence-on-disk schema (resolves spec §7's open item, v1/mock):
    essences/<id>/
        meta.json        {id, name, technique, created_at, color}
        thumbnail.png     source reference, resized
        embedding.json    the "style embedding" (mock: palette + stats)
"""
from __future__ import annotations

import colorsys
import json
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps

import paths

TECHNIQUE = "mock-palette-v2"  # stand-in until InstantStyle/StyleShot land
THUMBNAIL_SIZE = (160, 160)
APPLY_MAX_DIM = 768  # keep step payloads a reasonable size over the local IPC/HTTP hop


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


def _dominant_color_and_stats(img: Image.Image) -> tuple[tuple[int, int, int], float, float]:
    """Very small stand-in for real style-feature extraction: quantize down
    to a handful of colors and pick one that's both common *and* vivid
    (weighting by count alone tends to just pick a background — sky, wall,
    a white page — which produces a washed-out, barely-visible result once
    applied), plus average saturation/brightness in HSV. Good enough to
    drive a visibly distinct tint per reference image; not a real style
    embedding.
    """
    small = img.convert("RGB").resize((64, 64))
    quantized = small.quantize(colors=8, method=Image.MEDIANCUT)
    palette = quantized.getpalette()
    color_counts = quantized.getcolors()  # [(count, index), ...]
    total = sum(count for count, _ in color_counts)

    def rgb_of(index: int) -> tuple[int, int, int]:
        return tuple(palette[index * 3: index * 3 + 3])

    # Rank purely by saturation among colors that cover a meaningful share of
    # the image (ignoring rarer noise), rather than by raw pixel count — a
    # weighting-by-count approach still lets a duller background (sky, wall,
    # a white page) beat a much more vivid accent color it outnumbers, since
    # even a modest saturation floor isn't enough to overcome a large count
    # gap. This can still land on a low-saturation color for a genuinely
    # monochrome reference — nothing to do about that with a color-based
    # technique — but no longer discards an available vivid color in favor
    # of whatever covers the most pixels.
    MIN_SHARE = 0.02
    candidates = [c for c in color_counts if c[0] / total >= MIN_SHARE] or color_counts
    best = max(candidates, key=lambda c: _saturation(rgb_of(c[1])))
    dominant = rgb_of(best[1])

    hsv = small.convert("HSV")
    pixels = list(hsv.getdata())
    avg_sat = sum(p[1] for p in pixels) / len(pixels) / 255.0
    avg_val = sum(p[2] for p in pixels) / len(pixels) / 255.0
    return dominant, round(avg_sat, 3), round(avg_val, 3)


def _scale_toward(color: tuple[int, int, int], target: int, factor: float) -> tuple[int, int, int]:
    return tuple(int(c + (target - c) * factor) for c in color)


def _vivid(color: tuple[int, int, int], sat_factor: float = 1.9) -> tuple[int, int, int]:
    """Push a color's saturation up (pulling lightness toward mid-tone so it
    doesn't just wash out) for use as the *apply* tone-map color. Keeps
    `dominant_color` in the stored embedding/shelf badge as the true
    measured color; only the tone-map actually used to restyle a photo needs
    to be punchy enough to read as an obvious style shift even when the
    reference itself was fairly neutral (a plain-background screenshot, an
    overcast photo, etc).
    """
    r, g, b = (c / 255 for c in color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s = min(1.0, s * sat_factor)
    l = 0.5 + (l - 0.5) * 0.65
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return tuple(int(max(0, min(255, c * 255))) for c in (r2, g2, b2))


def extract_essence(reference_image_path: str, name: str | None = None) -> dict:
    src = Image.open(reference_image_path)
    dominant, avg_sat, avg_val = _dominant_color_and_stats(src)

    essence_id = uuid.uuid4().hex[:12]
    out_dir = paths.essences_dir() / essence_id
    out_dir.mkdir(parents=True, exist_ok=True)

    thumb = src.convert("RGB").copy()
    thumb.thumbnail(THUMBNAIL_SIZE)
    thumb.save(out_dir / "thumbnail.png")

    embedding = {
        "technique": TECHNIQUE,
        "dominant_color": list(dominant),
        "avg_saturation": avg_sat,
        "avg_brightness": avg_val,
    }
    (out_dir / "embedding.json").write_text(json.dumps(embedding, indent=2))

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
    for d in sorted(root.iterdir(), reverse=True):  # newest-created dirs sort last alphabetically(ish); good enough for now
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        thumb_path = d / "thumbnail.png"
        thumbnail = _to_data_url(Image.open(thumb_path)) if thumb_path.exists() else None
        out.append({**meta, "thumbnail": thumbnail})
    out.sort(key=lambda e: e["created_at"], reverse=True)
    return out


def get_essence_embedding(essence_id: str) -> dict:
    embedding_path = paths.essences_dir() / essence_id / "embedding.json"
    return json.loads(embedding_path.read_text())


def apply_essence(essence_id: str, target_image_path: str, steps: int = 8) -> dict:
    """Returns a list of intermediate frames from the original target image to
    the "styled" result, plus the final frame — the shape the Main Stage's
    animation clock consumes (spec §4.2.1's animation-generation decoupling:
    the frontend blends between these on its own timer, independent of how
    long producing each one actually took).
    """
    embedding = get_essence_embedding(essence_id)
    dominant = tuple(embedding["dominant_color"])
    avg_sat = embedding["avg_saturation"]

    target = Image.open(target_image_path).convert("RGB")
    target.thumbnail((APPLY_MAX_DIM, APPLY_MAX_DIM))

    # A flat alpha-blended color wash (the previous approach) just looks like
    # haze/blur regardless of the reference — it erases luminance detail
    # instead of restyling it. A duotone tone-map instead keeps the target's
    # own luminance (so content/edges stay legible) and recolors shadows and
    # highlights using the essence's color, which reads as an actual style
    # shift rather than fog.
    vivid = _vivid(dominant)
    grayscale = ImageEnhance.Contrast(ImageOps.grayscale(target)).enhance(1.15)
    shadow = _scale_toward(vivid, 0, 0.75)  # vivid, darkened
    highlight = _scale_toward(vivid, 255, 0.6)  # vivid, lightened
    duotone = ImageOps.colorize(grayscale, black=shadow, white=highlight, mid=vivid)

    saturated_original = ImageEnhance.Color(target).enhance(0.5 + avg_sat)
    final = Image.blend(saturated_original, duotone, alpha=0.82)

    frames = []
    for i in range(steps + 1):
        t = i / steps
        frame = Image.blend(target, final, alpha=t)
        frames.append(_to_data_url(frame))

    return {"steps": frames, "final": frames[-1]}
