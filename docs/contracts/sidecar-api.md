# Contract — sidecar API (`sidecar/app.py`)

**Purpose:** Local FastAPI surface the Electron proxy calls; validates requests and delegates to pipeline modules.
**Files:** `sidecar/app.py`, `sidecar/paths.py`, `sidecar/gpu.py`

## Inputs
HTTP on `127.0.0.1:8843`: `/health`, `/models/status|downloads|downloads/pause|downloads/resume`, `/utils/preview`, `/essences` (GET, `/extract`, `/blend`, DELETE `/{id}`), `/apply`, `/media` (GET, DELETE `/{id}`, POST `/{id}/gif`).

## Outputs
JSON responses; images written under the app-data dir (`RASA_MODELS_DIR` / appdata).

## Errors
4xx with a readable `detail` for bad input/missing essence/models not ready ?; 5xx only for genuine pipeline failures.

## Perf budget
Non-generation endpoints < 200ms ?. No endpoint loads SDXL unless it generates.

## Out of scope
Model math (see `generation.md`), UI.

## Done-check
`pytest sidecar/tests -q` green; `GET /health` returns device info with sidecar running.
