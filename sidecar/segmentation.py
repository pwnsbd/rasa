"""Subject/background segmentation + face detection, for the two-pass
subject-isolated strength blending in generation.py's apply_essence.

rembg (u2net model) does subject/background matting -> a soft, feathered
mask. OpenCV's bundled Haar cascade does face detection within the
segmented region. Both chosen for minimal added dependency weight over the
alternatives (SAM for segmentation, mediapipe for face detection) — see the
plan this was built from for the full reasoning. One correction made
against that plan during implementation: opencv-python-headless is pinned
to 4.10.0.84, not left unpinned — an unpinned install resolves to 5.0.0,
whose cv2/data/ directory ships with no Haar cascade XML files at all
(confirmed by direct inspection), a packaging gap not present in 4.x.

cv2/rembg imports are deferred into the functions that need them, not done
at module import time, mirroring pipeline_manager.py's handling of
torch/diffusers — keeps sidecar startup (health checks, etc.) fast even
when a request never touches this module.
"""
from __future__ import annotations

import os

from PIL import Image

import paths

# Route rembg's own model cache through the app's models dir, same pattern
# as HF_HOME in pipeline_manager.py, rather than the OS-default ~/.u2net.
# Must be set before rembg is imported anywhere, so it lives here at module
# level even though the import itself is deferred.
os.environ.setdefault("U2NET_HOME", str(paths.models_dir() / "rembg-cache"))

MIN_SUBJECT_SHARE = 0.02  # below this, treat as "no distinguishable subject" (e.g. a landscape)
MAX_SUBJECT_SHARE = 0.95  # above this, the "subject" is basically the whole frame -- no isolation needed
FACE_DETECT_PADDING = 0.15  # fraction of the subject bbox's own size, added on each side before detecting

_session = None  # lazy singleton -- avoid re-creating the onnxruntime session on every call
_face_cascade = None


def _rembg_session():
    global _session
    if _session is None:
        from rembg import new_session

        _session = new_session("u2net")
    return _session


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        import cv2

        _face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if _face_cascade.empty():
            raise RuntimeError("Failed to load OpenCV's bundled Haar cascade — opencv-python-headless install issue")
    return _face_cascade


def get_subject_mask(image: Image.Image) -> Image.Image | None:
    """Soft, feathered subject-vs-background mask (mode "L", same size as
    `image`) via rembg's alpha matting. Returns None if no meaningfully
    distinguishable subject was found, so callers can cleanly fall back to
    the single-pass path.
    """
    import numpy as np
    from rembg import remove

    mask = remove(
        image.convert("RGB"),
        session=_rembg_session(),
        only_mask=True,
        alpha_matting=True,
        post_process_mask=True,
    )
    share = float(np.asarray(mask, dtype=np.float32).mean()) / 255.0
    if share < MIN_SUBJECT_SHARE or share > MAX_SUBJECT_SHARE:
        return None
    return mask


FACE_DETECT_NORMALIZED_DIM = 640  # subject crop is upscaled/downscaled to this before detection


def detect_face(image: Image.Image, mask: Image.Image) -> bool:
    """Runs face detection cropped to the subject mask's bounding box (with
    padding), not the full frame — avoids picking up an incidental
    background face (a poster, someone walking by) that isn't the actual
    subject.

    The crop is normalized to a fixed size before detection. Without this,
    a fixed minSize=(40,40) pixel threshold means a face in a close-up
    selfie gets found but the same face in a distant full-body shot doesn't
    — the subject crop is smaller in absolute pixels, so the face within it
    is too, and it silently falls under the cascade's minimum. Confirmed
    with real photos: a close-up selfie correctly triggered the
    face-protected branch; a distant full-body street photo didn't (face
    fell below the threshold) and got the weaker no-face strength applied
    to a small face, visibly distorting it. Upscaling the crop first
    decouples "is a face detectable" from "how big is the subject in the
    original photo".
    """
    import cv2
    import numpy as np

    bbox = mask.getbbox()
    if bbox is None:
        return False
    x0, y0, x1, y1 = bbox
    pad_x = int((x1 - x0) * FACE_DETECT_PADDING)
    pad_y = int((y1 - y0) * FACE_DETECT_PADDING)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(image.width, x1 + pad_x)
    y1 = min(image.height, y1 + pad_y)

    crop = image.convert("L").crop((x0, y0, x1, y1))
    long_edge = max(crop.size)
    if long_edge == 0:
        return False
    scale = FACE_DETECT_NORMALIZED_DIM / long_edge
    normalized = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))))

    faces = _get_face_cascade().detectMultiScale(np.array(normalized), scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return len(faces) > 0


def suggest_subject_params(face_detected: bool) -> tuple[float, float]:
    """(strength, controlnet_scale) suggested for the subject-region pass —
    a starting point, always overridable per-request (see generation.py).
    Tighter (lower strength, higher controlnet_scale) when a face is
    present, since that's where small structural changes read as obviously
    wrong; looser for a faceless subject (an object, a body without a
    closeup face) where more restyling latitude is safe.
    """
    if face_detected:
        return 0.30, 0.97
    return 0.45, 0.85
