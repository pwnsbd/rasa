# PLAN — Rasa

**Goal:** Ship a Windows release candidate that a non-developer can install and use end-to-end (extract → apply → export) on an NVIDIA GPU.

**Metric:** `docs/release-readiness.md` "Before public release" items closed (0/6 today) + sidecar test suite green (count ?) + one clean-machine extract/apply run under ? min.

## Tasks

| # | task | owner | files | done when | status |
|---|------|-------|-------|-----------|--------|
| 1 | ~~Depth-aware texture overlay ("Follow scene")~~ dropped — Pawan rejected the look; patch kept in session scratchpad. Kept: Media Page Save-as + restyle loading spinner | Conductor → Pawan | `MainStage.tsx`, `MediaPage.tsx`, `electron/main.js`, `preload.js` | build + tests green; Pawan sees spinner in app; committed | in review |
| 2 | Lock Python deps + torch version | builder | `sidecar/requirements.txt`, `scripts/setup-sidecar.js`, bootstrap in `electron/main.js` ? | pinned lockfile; fresh venv install reproduces same versions | todo |
| 3 | Disk-space preflight before model download | builder | `sidecar/model_downloads.py`, `sidecar/app.py`, UI download button | download refuses with clear error when free space < required; unit test with stubbed disk usage | todo |
| 4 | User-selectable storage directory | builder | `electron/main.js`, `sidecar/paths.py`, Settings screen | setting persists; models/essences/media land in chosen dir; restart keeps it | todo |
| 5 | Progressive per-step previews during apply | builder | `sidecar/generation.py`, `sidecar/app.py`, `MainStage.tsx` | crossfade uses ≥3 real intermediate frames | todo |
| 6 | Provenance metadata embed + export (spec §3) | builder | `sidecar/media.py`, Media Page | exported image carries essence id/params; test reads them back | todo |
| 7 | Clean-machine installer test (fresh account, interrupted setup, offline restart, uninstall/reinstall, non-ASCII path) | Pawan | — | checklist in `docs/release-readiness.md` ticked | todo |
| 8 | Code signing with trusted certificate | Pawan | `package.json` build config | installer signed by real publisher cert | blocked (needs cert) |

## Next: LinkedIn launch (after task 1)
- Public repo + installer on GitHub Releases + showcase page (before/after gallery, short demo video). No hosted web demo.

## Decisions log
- 2026-10-03 — Dropped "Follow scene" texture overlay: conversion quality not good enough.
- 2026-10-03 — Share path: installer + showcase only (no hosted demo); repo pwnsbd/rasa public.
- 2026-10-02 — Onboarded to Conductor workflow. Contracts in `docs/contracts/` are first drafts; Pawan to review before dispatch.
- (prior) SDXL + InstantStyle + Tile ControlNet chosen over spec's Flux (licensing/VRAM — see README).
- (prior) New `/apply` effects ship default-off until validated on real photos.
- (prior) Windows + NVIDIA only for first release; per-user NSIS install, embedded Python bootstrap on first run.
