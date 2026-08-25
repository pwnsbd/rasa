"""Post-generation color preservation.

IP-Adapter's image embedding carries the essence reference's dominant
color/palette along with its texture — there's no separate "color channel"
vs. "texture channel" to scale independently via InstantStyle's block
separation (see pipeline_manager.py's INSTANT_STYLE_SCALE, which only
isolates style from layout, not color from texture). A strongly-colored
essence (e.g. a very purple painting) can end up tinting the whole target
photo toward its hue, which reads as "the photo turned purple" rather than
"the photo picked up this brushwork" — reported directly against a real run.

preserve_original_color fixes this as a cheap post-process, not a pipeline
change: convert both the stylized result and the original target to LAB,
keep the stylized image's L (luminance — carries the actual texture/
contrast/detail the essence contributed) and swap in the *original* photo's
a/b (chroma) channels. The photo's own colors come back; the visual texture
change survives because it's mostly a luminance-domain effect. This is the
standard "luminance-only style transfer" trick (the same idea behind most
"preserve color" toggles in other style-transfer tools), not a new
technique invented for this project.

Uses skimage.color for the LAB conversion (already a project dependency via
style_analysis/palette.py) rather than Pillow's own approximate "LAB" mode.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.color import lab2rgb, rgb2lab


def preserve_original_color(stylized: Image.Image, original: Image.Image) -> Image.Image:
    """Returns a copy of `stylized` with its color replaced by `original`'s.
    Both images must be the same size — true for every call site today
    (generation.py always calls this with the working-resolution target and
    the same-resolution generation output).
    """
    if stylized.size != original.size:
        raise ValueError(f"size mismatch: stylized={stylized.size} original={original.size}")

    stylized_arr = np.asarray(stylized.convert("RGB"), dtype=np.float64) / 255.0
    original_arr = np.asarray(original.convert("RGB"), dtype=np.float64) / 255.0

    stylized_lab = rgb2lab(stylized_arr)
    original_lab = rgb2lab(original_arr)

    combined_lab = np.stack(
        [stylized_lab[..., 0], original_lab[..., 1], original_lab[..., 2]], axis=-1
    )
    rgb = np.clip(lab2rgb(combined_lab), 0.0, 1.0)
    return Image.fromarray((rgb * 255).round().astype(np.uint8))
