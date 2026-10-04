# Contract — model downloads (`sidecar/model_downloads.py`)

**Purpose:** Background, pausable download of all required models (~11.5GB+) into the models dir, with status for the UI.
**Files:** `model_downloads.py`, `/models/*` routes in `app.py`, download button in UI ?

## Inputs
Start (on sidecar start or button ?), pause, resume.

## Outputs
Per-model status/progress JSON; files in `<models-dir>/hf-cache`.

## Errors
Network failure → resumable, surfaced state; insufficient disk → refuse before downloading: state `insufficient_disk` with `required_bytes` (remaining bytes + 2 GiB margin), `free_bytes`, `path`; re-checked on every start/resume. Remaining bytes = Hub file sizes minus bytes already in hf-cache; 14 GiB static estimate if offline.

## Perf budget
Status endpoint < 50ms; download never blocks request handling.

## Out of scope
Python/runtime bootstrap (Electron first-run).

## Done-check
`test_model_downloads.py` green with network stubbed.
