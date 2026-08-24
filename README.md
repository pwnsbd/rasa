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

All app data — models (~13GB+ once the subject-isolation dependencies' own model downloads are included), Essences, Media, cache, db, and Electron's own internal caches — lives in `<project-root>/appdata/`, not wherever Electron's OS-default `userData` would otherwise put it (usually the C: drive on Windows, `%APPDATA%\rasa`). This is a deliberate `app.setPath('userData', ...)` in `electron/main.js`, not a default worth trusting an env var to redirect every launch — that was tried first, and a forgotten env var on one real run silently re-downloaded the entire model cache back onto C:. `RASA_MODELS_DIR` still exists as a further override if you want the model cache specifically somewhere other than `appdata/models` (everything else stays under `appdata/`).

Device status (GPU name, CUDA availability, compute capability, VRAM, torch version) is visible in-app on the Settings screen, and always available at `GET http://127.0.0.1:8843/health` while the sidecar is up.

## Status

All four zones (Main Stage, Distillation Room, Media Page, Settings) are wired up and functional, including the drag-a-bottle-onto-a-photo interaction, the pour/thread/crossfade animations, and the Essence-appears-on-the-shelf-only-after-leaving-the-room behavior.

**Style extraction/application runs a real model**: SDXL + InstantStyle (IP-Adapter) + a Tile ControlNet, not the spec's originally-suggested Flux — see [`sidecar/pipeline_manager.py`](sidecar/pipeline_manager.py) for why SDXL (the only available Flux IP-Adapter checkpoints are gated to FLUX.1-dev and their own authors say they aren't for fine-grained style transfer, whereas InstantStyle's actual style/layout block-separation is diffusers-native and proven on SDXL) and why ControlNet is needed alongside IP-Adapter (plain img2img `strength` alone let style-driven regeneration drift the target's own content/layout away — the same failure the InstantStyle authors' own follow-up paper, InstantStyle-Plus, fixes with a Tile ControlNet; verified via direct testing that content now stays intact even at `strength=0.85`). Base model and style-extraction technique stay swappable behind `sidecar/essence.py`'s `extract_essence`/`apply_essence` per spec §2.2 — swapping in Flux later if a real InstantStyle-for-Flux combination matures is a contained change to `pipeline_manager.py`.

`strength` and `controlnet_scale` are exposed as optional `/apply` overrides (`sidecar/app.py`'s `ApplyRequest`) for further tuning without a code change — defaults (0.85 / 0.85) favor a visible style shift given ControlNet's now doing the structural heavy lifting, but how much style actually shows through varies a lot by reference image; flat/simple references show less than a texture- or color-rich photo or painting.

First run downloads ~11.5GB (SDXL base + IP-Adapter + its CLIP image encoder + Tile ControlNet + the fp16-fix VAE) into `<models-dir>/hf-cache` (see the `RASA_MODELS_DIR` note above), in the background from sidecar startup — `GET /models/status` and the in-app banner/Settings screen show progress. Extraction and application both fail fast with a 503 until it's ready, rather than hanging.

**Image formats**: PNG, JPEG, WebP, HEIC/HEIF (iPhone photos, via `pillow-heif`), BMP, GIF, and TIFF are all accepted, both as reference/target images and for display. HEIC and TIFF specifically have no Chromium `<img>` codec at all, so previewing them (not just processing them) routes through a new `POST /utils/preview` sidecar endpoint that decodes via Pillow and re-encodes as a displayable PNG — see `electron/main.js`'s `image:readAsDataUrl` handler. That endpoint doesn't require the style model to be loaded, so previews work even during the first-run download.

**Subject-isolated strength blending**: on a target photo with a distinguishable subject, `/apply` now segments subject from background (`rembg`, `u2net` model, soft feathered mask) and runs generation twice — background at the usual strength/`controlnet_scale`, subject region at a lower, tighter strength suggested by whether a face was detected in that region (OpenCV's bundled Haar cascade, scoped to the subject's bounding box) — then composites the two through the mask. Both passes share one seeded generator so they only differ by strength, not grain/noise. Fixes over-distorted faces from applying one global strength across the whole photo. `isolate_subject` (default on), `subject_strength`, and `subject_controlnet_scale` are exposed as `/apply` overrides alongside the existing `strength`/`controlnet_scale`; the response reports what was detected and suggested. Falls back to the original single-pass behavior automatically when no distinguishable subject is found (a landscape, for instance) or when `isolate_subject=false`. See `sidecar/segmentation.py` for the library-choice reasoning (why `rembg`+Haar cascade over SAM/`mediapipe`).

Not yet built: progressive per-diffusion-step previews (the Main Stage crossfade currently animates original → final rather than several real intermediate frames), provenance metadata embedding + export (spec §3), and the future style marketplace (deferred beyond v1 per spec §7).

### Verifying it runs

`npm run dev` opens the real window. For headless/agent verification (this shell sets `ELECTRON_RUN_AS_NODE=1`, which breaks a direct Electron launch — unset it):

```
npm run dev:ui                                              # start Vite first, separately
env -u ELECTRON_RUN_AS_NODE node scripts/verify-launch.mjs   # launch + screenshot the current screen
env -u ELECTRON_RUN_AS_NODE node scripts/verify-flow.mjs <ref-image> <target-image>   # full extract -> apply -> media flow, screenshot per step
```

Both need `playwright-core` (`npm install --no-save playwright-core`) and screenshot to `.tmp-shots/`.
