# Contract — Electron shell (`electron/main.js`, `preload.js`)

**Purpose:** Window, single-instance lock, sidecar lifecycle (spawn/retry/cleanup), first-run embedded-Python bootstrap, `sidecar:call` IPC proxy, appdata relocation, uninstall prompt.
**Files:** `electron/main.js`, `electron/preload.js`, `ui/src/lib/appBridge.d.ts`

## Inputs
App launch; renderer IPC `sidecar:call(method, path, body)` ?.

## Outputs
Proxied sidecar responses (40-min fetch timeout); startup status/errors to renderer.

## Errors
Sidecar fails to start → visible error + retry button; setup interrupted → recoverable on next launch ?.

## Perf budget
Window visible < 3s ?; sidecar ready (no model load) < 15s ?.

## Out of scope
Inference.

## Done-check
`verify-launch.mjs` screenshot shows Main Stage; `npm run dist` builds the installer.
