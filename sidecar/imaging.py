"""Tiny shared image-encoding helper.

Split out of essence.py during the essence_store/generation split — both
modules need to turn a PIL image into a data: URL for the frontend
(essence_store for thumbnails, generation for the working/final frames) and
neither should import the other just for this.
"""
from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image


def to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


def bytes_to_data_url(data: bytes, fmt: str) -> str:
    """Same idea as to_data_url, for callers that already have encoded
    bytes (e.g. relight.py's GIF output) rather than a PIL Image to encode."""
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"
