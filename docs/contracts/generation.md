# Contract — apply / generation (`sidecar/generation.py`, `pipeline_manager.py`)

**Purpose:** Apply an Essence to a target photo via SDXL+IP-Adapter+Tile ControlNet (`mode: "restyle"`) or the non-diffusion texture engine (`mode: "texture_overlay"`).
**Files:** `generation.py`, `pipeline_manager.py`, `segmentation.py`, `depth.py`, `content_mask.py`, `color_transfer.py`, `stroke_texture.py`, `texture_overlay.py`

## Inputs
`ApplyRequest`: target image, essence id, `mode`, `blend_mode` (`subject`|`depth`|`none`, default `none`), `strength` (0.85), `controlnet_scale` (0.85), `color_preservation`, `content_aware_masking` (off), `stroke_amount` (0), seed ?.

## Outputs
A Creation (image + metadata, optional saved depth map) in the media store.

## Errors
Models not downloaded → clear "not ready" error, not a hang. OOM → reported error ?.

## Perf budget
Restyle on RTX 5070 at ~1024px: < 60s ? Texture overlay: < 5s ?, never loads SDXL. Depth estimated at most once per apply.

## Out of scope
Essence extraction, UI animation.

## Done-check
Synthetic-image tests pass (`test_texture_overlay.py`, `test_stroke_texture.py`, `test_content_mask.py`, `test_color_transfer.py`); visual approval by Pawan on a real photo for any new effect.
