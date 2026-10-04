# Contract — Media & living images (`sidecar/media.py`, `relight.py`)

**Purpose:** Store/list/delete Creations; produce relight GIFs; supply depth for UI parallax.
**Files:** `media.py`, `relight.py`, `ui/src/components/ParallaxImage.tsx`, `ui/src/screens/MediaPage.tsx`

## Inputs
Creation id; GIF params (frames, light path) ?.

## Outputs
Creation list JSON; GIF file; reused saved depth when present.

## Errors
Unknown id → 404; missing depth → compute once, not fail.

## Perf budget
GIF export < 15s ?; parallax 60fps on hover ?.

## Out of scope
Provenance metadata (PLAN #6, not built yet).

## Done-check
`test_relight.py`, `test_depth.py` green; parallax/relight look approved by Pawan in-app.
