"""Local-only FastAPI sidecar. Electron's main process spawns this and talks to
it over 127.0.0.1 only — never bind 0.0.0.0, this must never be reachable off-device.

Run directly for dev: `python app.py` (reads SIDECAR_PORT / APP_*_DIR env vars set
by electron/main.js; falls back to sane local defaults when run standalone).
"""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import paths
from gpu import detect_gpu

app = FastAPI(title="rasa sidecar")

# Dev convenience only: lets the Vite dev server (a different origin) call the
# sidecar directly while iterating. Packaged builds proxy everything through
# the Electron main process instead, so this never matters in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "ok": True,
        "gpu": detect_gpu(),
        "dirs": {
            "models": str(paths.models_dir()),
            "essences": str(paths.essences_dir()),
            "media": str(paths.media_dir()),
            "cache": str(paths.cache_dir()),
            "db": str(paths.db_dir()),
        },
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("SIDECAR_PORT", "8843"))
    # 127.0.0.1 only, per the module docstring — this must not be reachable off-device.
    uvicorn.run(app, host="127.0.0.1", port=port)
