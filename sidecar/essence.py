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

import json
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageEnhance

import paths

TECHNIQUE = "mock-palette-v1"  # stand-in until InstantStyle/StyleShot land
THUMBNAIL_SIZE = (160, 160)
APPLY_MAX_DIM = 768  # keep step payloads a reasonable size over the local IPC/HTTP hop


def _to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    import base64

    buf = BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


def _dominant_color_and_stats(img: Image.Image) -> tuple[tuple[int, int, int], float, float]:
    """Very small stand-in for real style-feature extraction: quantize down
    to a handful of colors and take the most common one, plus average
    saturation/brightness in HSV. Good enough to drive a visibly distinct
    tint per reference image; not a real style embedding.
    """
    small = img.convert("RGB").resize((64, 64))
    quantized = small.quantize(colors=5, method=Image.MEDIANCUT)
    palette = quantized.getpalette()
    color_counts = quantized.getcolors()  # [(count, index), ...]
    color_counts.sort(reverse=True)
    top_index = color_counts[0][1]
    dominant = tuple(palette[top_index * 3: top_index * 3 + 3])

    hsv = small.convert("HSV")
    pixels = list(hsv.getdata())
    avg_sat = sum(p[1] for p in pixels) / len(pixels) / 255.0
    avg_val = sum(p[2] for p in pixels) / len(pixels) / 255.0
    return dominant, round(avg_sat, 3), round(avg_val, 3)


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

    tint_layer = Image.new("RGB", target.size, dominant)
    contrasted = ImageEnhance.Contrast(target).enhance(1.08)
    saturated = ImageEnhance.Color(contrasted).enhance(0.6 + avg_sat)
    final = Image.blend(saturated, tint_layer, alpha=0.32)

    frames = []
    for i in range(steps + 1):
        t = i / steps
        frame = Image.blend(target, final, alpha=t)
        frames.append(_to_data_url(frame))

    return {"steps": frames, "final": frames[-1]}
