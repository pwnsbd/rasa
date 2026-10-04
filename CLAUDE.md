# Rasa — project instructions

## Project
- **Name:** Rasa
- **Goal:** Local-first, GPU-aware Windows desktop app that extracts a reference image's visual style (an **Essence**) and reapplies it to other photos.
- **Stack:** Electron (`electron/`) + React/Vite/TypeScript renderer (`ui/`, npm workspace) + Python FastAPI sidecar (`sidecar/`, port 8843, 127.0.0.1 only). Models: SDXL + InstantStyle (IP-Adapter) + Tile ControlNet, rembg, Depth-Anything-V2-Small. Dev target: RTX 5070 (Blackwell, needs cu128 torch).
- Spec: `docs/rasa-product-spec.md`. Release checklist: `docs/release-readiness.md`. Plan: `PLAN.md`. Contracts: `docs/contracts/`.

## Commands
- Install: `npm install` then `npm run sidecar:setup` (creates `sidecar/venv`, picks CUDA/CPU torch)
- Run (dev): `npm run dev` (Vite + Electron)
- Build UI: `npm run build:ui`
- Typecheck UI: `npm run build:ui` (tsc + vite build)
- Sidecar tests: `sidecar/venv/Scripts/python -m pytest sidecar/tests -q` (run from repo root; `conftest.py` puts `sidecar/` on sys.path)
- Headless launch check: `npm run dev:ui` then `env -u ELECTRON_RUN_AS_NODE node scripts/verify-launch.mjs` (needs `npm install --no-save playwright-core`; screenshots to `.tmp-shots/`)
- Full flow check: `env -u ELECTRON_RUN_AS_NODE node scripts/verify-flow.mjs <ref-image> <target-image>`
- Package: `npm run dist` → `release/Rasa-Setup-<ver>-x64.exe`

## Project rules
- Sidecar never talks to any network beyond `127.0.0.1` (except model downloads in `model_downloads.py`).
- Every renderer→sidecar call goes through Electron's `sidecar:call` IPC proxy (`electron/main.js`), never a direct `fetch` from the UI. New endpoints need `ui/src/lib/api.ts` + `appBridge.d.ts` types.
- New `/apply` options default **off** unless validated on a real photo (pattern: `content_aware_masking`, `stroke_amount`, `depth_aware_texture`).
- Never commit `appdata/`, models, `resources/`, `release/`, `sidecar/venv/`, `.tmp-shots/`.
- Every model added must allow commercial distribution; record it in `THIRD_PARTY_LICENSES.md`.
- README is the living feature log — update it when a user-visible behaviour changes.

## Agent notes
- The shell sets `ELECTRON_RUN_AS_NODE=1`; unset it (`env -u ELECTRON_RUN_AS_NODE`) before launching Electron.
- Tests must not download models or need a GPU: use synthetic images / stubs (see existing `sidecar/tests/test_*.py`).
- First real run downloads ~11.5GB+ into `appdata/`; don't trigger it from a worktree.
- Worktrees don't have `sidecar/venv` or `node_modules`; use the main checkout's venv python by absolute path ?
- Visual quality (texture, stroke, relight look) is judged by Pawan in the running app, never by an agent.
- 2026-10-03 — "Follow scene" overlay was fully built + tested before Pawan saw it, then rejected on look → for any new visual effect, get Pawan to eyeball a quick prototype on a real photo before writing tests/UI polish.
- 2026-10-04 — resident-mode distill OOM'd (25GB): image encoder embedded under grad, autograd graph pinned its GPU copy after moving back to CPU; stub tests couldn't see it → any change to model device placement gets a Conductor GPU run of BOTH apply and essence extraction (watch torch.cuda.memory_allocated per stage) before telling Pawan it's done.
