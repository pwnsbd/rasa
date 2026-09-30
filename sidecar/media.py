"""Media Page archive (spec §4.2.3): every finished creation is saved here
automatically, regardless of export status. This module owns that on-disk
record; export/provenance-metadata embedding (spec §3) is a later addition
that reads from here rather than replacing it.

media/<id>/
    meta.json    {id, essence_id, essence_name, created_at}
    image.png
    depth.png    optional — the depth map generation.py's apply_essence
                 computed for this creation (see its compute_depth param),
                 used by the frontend's ParallaxImage for the Media Page's
                 hover parallax effect. Absent for creations made before
                 this existed, or with compute_depth=False — those just
                 render as plain (non-parallax) images, no migration needed.
    animated.gif optional — the sweeping-light relight loop (see relight.py),
                 generated on demand via POST /media/<id>/gif, not at apply
                 time. Absent until a user actually asks for one.
"""
from __future__ import annotations

import base64
import json
import shutil
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image

import paths


def save_creation(
    essence_id: str,
    essence_name: str,
    final_data_url: str,
    depth_map_data_url: str | None = None,
) -> dict:
    creation_id = uuid.uuid4().hex[:12]
    out_dir = paths.media_dir() / creation_id
    out_dir.mkdir(parents=True, exist_ok=True)

    header, b64 = final_data_url.split(",", 1)
    img = Image.open(BytesIO(base64.b64decode(b64)))
    img.save(out_dir / "image.png")

    if depth_map_data_url:
        depth_header, depth_b64 = depth_map_data_url.split(",", 1)
        depth_img = Image.open(BytesIO(base64.b64decode(depth_b64)))
        depth_img.save(out_dir / "depth.png")

    meta = {
        "id": creation_id,
        "essence_id": essence_id,
        "essence_name": essence_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def creation_dir(creation_id: str) -> Path:
    """Validated path to a creation's folder — shared by delete_creation,
    save_relight_gif, and app.py's GIF endpoint (which needs to read
    image.png/depth.png directly). creation_id ultimately comes from an
    HTTP path param — guard against a "../.." id escaping media_dir.
    """
    root = paths.media_dir().resolve()
    out_dir = (root / creation_id).resolve()
    if not out_dir.is_relative_to(root) or not out_dir.is_dir():
        raise FileNotFoundError(creation_id)
    return out_dir


def save_relight_gif(creation_id: str, gif_bytes: bytes) -> str:
    """Writes the relight GIF (see relight.py) alongside this creation's
    image.png/depth.png — same per-creation folder, no new storage concept.
    Returns the absolute path (as a str) so the caller can hand it to
    Electron's already-existing shell:showInFolder — see app.py.
    """
    gif_path = creation_dir(creation_id) / "animated.gif"
    gif_path.write_bytes(gif_bytes)
    return str(gif_path)


def _image_data_url(path) -> str:
    img = Image.open(path)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def list_creations() -> list[dict]:
    out = []
    root = paths.media_dir()
    for d in root.iterdir():
        meta_path = d / "meta.json"
        image_path = d / "image.png"
        if not meta_path.exists() or not image_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        depth_path = d / "depth.png"
        depth = _image_data_url(depth_path) if depth_path.exists() else None
        has_gif = (d / "animated.gif").exists()
        out.append({**meta, "image": _image_data_url(image_path), "depth": depth, "has_gif": has_gif})
    out.sort(key=lambda c: c["created_at"], reverse=True)
    return out


def delete_creation(creation_id: str) -> None:
    shutil.rmtree(creation_dir(creation_id))
