# Rasa

Local-first, GPU-aware desktop app for extracting the visual style of a reference image (an **Essence**) and reapplying it to other photos. See [`docs/rasa-product-spec.md`](docs/rasa-product-spec.md) for the full product/technical spec.

Architecture follows the same Electron + Python-sidecar pattern as the sibling `pdfToAudio` project: a React/Vite/TypeScript renderer talks to a local-only FastAPI sidecar (spawned by Electron's main process) that owns all model inference and stays off any network beyond `127.0.0.1`.

## Layout

```
electron/       Electron main process + preload (window, IPC, sidecar lifecycle)
ui/             React + Vite + TypeScript renderer
sidecar/        Python FastAPI backend — device detection, (soon) the extraction/generation pipeline
scripts/        Dev tooling (sidecar venv + torch setup)
docs/           Spec and architecture notes
```

## Setup

```
npm install
npm run sidecar:setup   # creates sidecar/venv, installs deps + the right torch build
npm run dev              # Vite dev server + Electron, concurrently
```

`sidecar:setup` detects an NVIDIA GPU via `nvidia-smi` and installs a CUDA (cu128) torch build if found, CPU otherwise. cu128 specifically (not an older CUDA build) is required for current-gen "Blackwell" GPUs (RTX 50-series, compute capability sm_120, including this project's RTX 5070 dev target) — see the comment in [`scripts/setup-sidecar.js`](scripts/setup-sidecar.js).

Device status (GPU name, CUDA availability, compute capability, VRAM, torch version) is visible in-app on the Settings screen, and always available at `GET http://127.0.0.1:8843/health` while the sidecar is up.

## Status

All four zones (Main Stage, Distillation Room, Media Page, Settings) are wired up and functional, including the drag-a-bottle-onto-a-photo interaction, the pour/thread/crossfade animations, and the Essence-appears-on-the-shelf-only-after-leaving-the-room behavior. **The style extraction/application itself is a placeholder**, not InstantStyle/StyleShot/Flux — `sidecar/essence.py`'s `mock-palette-v1` technique derives a dominant color + saturation/brightness from the reference image and applies it as a progressive tint, so the real pipeline contracts (Essence file/folder schema, extract/apply API shape, the animation-generation-decoupling step-list) exist and are exercised end-to-end. Swapping in the real models means rewriting `sidecar/essence.py`'s internals only — `app.py`, `paths.py`, and the whole frontend already speak the real interface from spec §2.2b.

Not yet built: the real Flux/InstantStyle/StyleShot pipeline, provenance metadata embedding + export (spec §3), and the future style marketplace (deferred beyond v1 per spec §7).

### Verifying it runs

`npm run dev` opens the real window. For headless/agent verification (this shell sets `ELECTRON_RUN_AS_NODE=1`, which breaks a direct Electron launch — unset it):

```
npm run dev:ui                                              # start Vite first, separately
env -u ELECTRON_RUN_AS_NODE node scripts/verify-launch.mjs   # launch + screenshot the current screen
env -u ELECTRON_RUN_AS_NODE node scripts/verify-flow.mjs <ref-image> <target-image>   # full extract -> apply -> media flow, screenshot per step
```

Both need `playwright-core` (`npm install --no-save playwright-core`) and screenshot to `.tmp-shots/`.
