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

All four zones (Main Stage, Distillation Room, Media Page, Settings) are wired up and functional, including the drag-a-bottle-onto-a-photo interaction, the pour/thread/crossfade animations, and the Essence-appears-on-the-shelf-only-after-leaving-the-room behavior.

**Style extraction/application runs a real model**: SDXL + InstantStyle (IP-Adapter), not the spec's originally-suggested Flux — see [`sidecar/pipeline_manager.py`](sidecar/pipeline_manager.py) for why (the only available Flux IP-Adapter checkpoints are gated to FLUX.1-dev and their own authors say they aren't for fine-grained style transfer, whereas InstantStyle's actual style/layout block-separation is diffusers-native and proven on SDXL). Base model and style-extraction technique stay swappable behind `sidecar/essence.py`'s `extract_essence`/`apply_essence` per spec §2.2 — swapping in Flux later if a real InstantStyle-for-Flux combination matures is a contained change to `pipeline_manager.py`.

First run downloads ~9GB (SDXL base + IP-Adapter + its CLIP image encoder) into `<app-data>/models/hf-cache`, in the background from sidecar startup — `GET /models/status` and the in-app banner/Settings screen show progress. Extraction and application both fail fast with a 503 until it's ready, rather than hanging.

Not yet built: progressive per-diffusion-step previews (the Main Stage crossfade currently animates original → final rather than several real intermediate frames), provenance metadata embedding + export (spec §3), and the future style marketplace (deferred beyond v1 per spec §7).

### Verifying it runs

`npm run dev` opens the real window. For headless/agent verification (this shell sets `ELECTRON_RUN_AS_NODE=1`, which breaks a direct Electron launch — unset it):

```
npm run dev:ui                                              # start Vite first, separately
env -u ELECTRON_RUN_AS_NODE node scripts/verify-launch.mjs   # launch + screenshot the current screen
env -u ELECTRON_RUN_AS_NODE node scripts/verify-flow.mjs <ref-image> <target-image>   # full extract -> apply -> media flow, screenshot per step
```

Both need `playwright-core` (`npm install --no-save playwright-core`) and screenshot to `.tmp-shots/`.
