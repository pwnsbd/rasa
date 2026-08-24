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

Device status (GPU name, CUDA availability, compute capability, VRAM, torch version) is visible in-app once running — currently rendered as a bare `Settings` screen, and always available at `GET http://127.0.0.1:8843/health` while the sidecar is up.

## Status

Scaffold + GPU/device detection layer only. Not yet built: style extraction (InstantStyle/StyleShot), base generation (Flux), Essence storage schema, the alchemy-workbench UI, and the provenance/export system. See spec §7 for open decisions.
