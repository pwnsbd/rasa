# Contract — Essence store & analysis (`sidecar/essence_store.py`, `essence_models.py`, `style_analysis/`)

**Purpose:** Extract, store, blend (Cauldron), list and delete Essences: purified IP-Adapter embedding + palette/texture/stroke/statistics analysis.
**Files:** `essence_store.py`, `essence_models.py`, `style_analysis/*.py`, `imaging.py`

## Inputs
One or more reference images (PNG/JPEG/WebP/HEIC/BMP/GIF/TIFF); for blend: weighted list of essence ids and/or fresh images.

## Outputs
Essence record (schema in `essence_models.py`) + files on disk.

## Errors
Unsupported/corrupt image → 4xx; weights ≤ 0 or empty blend → 4xx ?; delete of unknown id → 404.

## Perf budget
Extract < 20s on GPU ? (multi-crop purification included). Analyzers CPU-only, < 2s each ?.

## Out of scope
Applying style.

## Done-check
`test_essence_schema.py`, `test_blend.py`, `test_purification.py`, `test_palette.py`, `test_texture.py`, `test_stroke.py` green.
