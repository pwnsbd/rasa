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
"""
from __future__ import annotations

import base64
import json
import shutil
import uuid
from datetime import datetime, timezone
from io import BytesIO

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
        out.append({**meta, "image": _image_data_url(image_path), "depth": depth})
    out.sort(key=lambda c: c["created_at"], reverse=True)
    return out


def delete_creation(creation_id: str) -> None:
    root = paths.media_dir().resolve()
    out_dir = (root / creation_id).resolve()
    # creation_id ultimately comes from an HTTP path param — guard against a
    # "../.." id escaping media_dir before it ever reaches rmtree.
    if not out_dir.is_relative_to(root) or not out_dir.is_dir():
        raise FileNotFoundError(creation_id)
    shutil.rmtree(out_dir)
