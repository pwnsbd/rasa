"""Texture profile extraction (spec Phase 3). Traditional image processing,
not a neural model — the goal per the spec is a stable, relatively-ordered
descriptor ("flat < noisy"), not an academically precise texture
classification. Bounds used by normalize01() below are empirical, picked
from inspecting real values on the test images in sidecar/tests/, not
derived analytically — these are relative descriptors.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from essence_models import TextureProfile
from style_analysis._common import analysis_resize, normalize01, to_gray_array

LAPLACIAN_VAR_MAX = 0.06  # roughness — Laplacian response variance on a 0..1 image
GRADIENT_DETAIL_MAX = 0.35  # detail_density — fraction of pixels with above-threshold gradient magnitude
HF_ENERGY_MAX = 0.55  # high_frequency_energy — fraction of FFT spectrum energy outside the low-frequency core


def analyze_texture(image: Image.Image) -> TextureProfile:
    img = analysis_resize(image)
    gray = to_gray_array(img)  # 0..1 float32

    # Roughness: Laplacian response variance — a classic blur/sharpness
    # proxy (low variance = smooth/flat, high variance = rough/detailed).
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    roughness = normalize01(float(laplacian.var()), 0.0, LAPLACIAN_VAR_MAX)

    # Detail density: fraction of pixels with strong local gradient —
    # spatial "how much of the image has fine detail", distinct from
    # roughness's "how strong is that detail on average".
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.hypot(gx, gy)
    detail_density = normalize01(float((grad_mag > 0.15).mean()), 0.0, GRADIENT_DETAIL_MAX)

    # High-frequency energy: 2D FFT magnitude spectrum, fraction of total
    # energy outside a low-frequency core radius — a flat/smooth image
    # concentrates energy near DC (center); fine texture/noise spreads it
    # toward the edges.
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    h, w = spectrum.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    radius = np.hypot(yy - cy, xx - cx)
    core_radius = min(h, w) * 0.08
    total_energy = spectrum.sum()
    hf_energy_raw = float(spectrum[radius > core_radius].sum() / total_energy) if total_energy > 0 else 0.0
    high_frequency_energy = normalize01(hf_energy_raw, 0.0, HF_ENERGY_MAX)

    # Repetition: 2D autocorrelation via FFT (ifft(fft * conj(fft))) — a
    # repeating pattern (bricks, weave, hatching) produces a strong
    # secondary peak away from the zero-lag center; noise/natural texture
    # doesn't. Compares the strongest off-center peak to the zero-lag peak.
    f = np.fft.fft2(gray - gray.mean())
    autocorr = np.fft.fftshift(np.fft.ifft2(f * np.conj(f)).real)
    zero_lag = autocorr[h // 2, w // 2]
    # Mask out a small region around the zero-lag peak itself before
    # looking for the next-strongest peak.
    exclude_radius = max(2, min(h, w) // 20)
    masked = autocorr.copy()
    masked[int(cy - exclude_radius): int(cy + exclude_radius), int(cx - exclude_radius): int(cx + exclude_radius)] = -np.inf
    second_peak = float(masked.max()) if np.isfinite(masked.max()) else 0.0
    repetition = float(np.clip(second_peak / zero_lag, 0.0, 1.0)) if zero_lag > 0 else 0.0

    return TextureProfile(
        roughness=round(roughness, 4),
        detail_density=round(detail_density, 4),
        high_frequency_energy=round(high_frequency_energy, 4),
        repetition=round(repetition, 4),
    )
