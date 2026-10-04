# Contract — model downloads (`sidecar/model_downloads.py`)

**Purpose:** Background, pausable download of all required models (~11.5GB+) into the models dir, with status for the UI.
**Files:** `model_downloads.py`, `/models/*` routes in `app.py`, download button in UI ?

## Inputs
Start (on sidecar start or button ?), pause, resume.

## Outputs
Per-model status/progress JSON; files in `<models-dir>/hf-cache`.

## Errors
Network failure → resumable, surfaced state; insufficient disk → refuse with message (PLAN #3, not built).

## Perf budget
Status endpoint < 50ms; download never blocks request handling.

## Out of scope
Python/runtime bootstrap (Electron first-run).

## Done-check
`test_model_downloads.py` green with network stubbed.
