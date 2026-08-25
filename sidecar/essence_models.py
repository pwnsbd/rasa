"""Structured Essence schema (versioned, backward-compatible).

Essence used to be just an IP-Adapter embedding plus a dominant color.
These models add a structured, inspectable representation of style —
palette/texture/stroke/statistics extracted from the reference at
Distillation time (see style_analysis/) — as a foundation for future
generation-time controls. None of this changes the actual style transfer
yet: apply_essence in generation.py is untouched by this module.

Backward compatibility is structural, not a migration. Every analysis
field is Optional with a default of None, and `version` defaults to 1 —
an old on-disk meta.json (written before this module existed, with none
of these keys) validates as-is via EssenceMeta.model_validate(...),
Pydantic fills the defaults, and the essence loads with no analysis data
rather than crashing or needing a rewrite. New essences write version=2.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ColorSwatch(BaseModel):
    hex: str
    rgb: tuple[int, int, int]
    weight: float  # relative share of the image, 0..1


class PaletteProfile(BaseModel):
    dominant_colors: list[ColorSwatch]
    mean_saturation: float
    mean_luminance: float
    temperature: float  # -1 (cool) .. 1 (warm)
    contrast: float


class TextureProfile(BaseModel):
    roughness: float
    detail_density: float
    high_frequency_energy: float
    repetition: float


class StrokeProfile(BaseModel):
    directionality: float  # 0 (no dominant direction) .. 1 (strongly directional)
    curvature: float
    coherence: float  # how consistent the local orientation is across the image
    density: float
    orientation_map_path: str | None = None  # relative filename of the debug visualization, if one was saved


class StyleStatistics(BaseModel):
    abstraction: float | None = None  # left unimplemented for now — no reliable measurable proxy yet
    edge_density: float
    local_contrast: float


class BlendIngredientInfo(BaseModel):
    """Provenance record for one ingredient of a blended Essence (see
    essence_store.py's blend_essences) — not itself a live reference (the
    source image/essence can be deleted or moved without affecting the
    blend, same spirit as delete_essence's note about Media Page creations).
    """
    name: str
    weight: float  # normalized share of this blend, 0..1
    source: Literal["image", "essence"]


class EssenceMeta(BaseModel):
    id: str
    name: str
    created_at: str
    technique: str
    color: tuple[int, int, int]  # kept: existing shelf-badge/bottle-pour animation consumer
    version: int = 1

    palette: PaletteProfile | None = None
    texture: TextureProfile | None = None
    stroke: StrokeProfile | None = None
    style_statistics: StyleStatistics | None = None

    # Set only for Essences created via the Cauldron (blend_essences) — None
    # for a normal single-reference Essence. No analysis fields above are
    # populated for a blend: there's no single reference image to run
    # style_analysis/ against, and Optional already exists to represent
    # "no analysis available" honestly rather than fabricating one.
    blended_from: list[BlendIngredientInfo] | None = None
