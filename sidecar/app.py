"""Local-only FastAPI sidecar. Electron's main process spawns this and talks to
it over 127.0.0.1 only — never bind 0.0.0.0, this must never be reachable off-device.

Run directly for dev: `python app.py` (reads SIDECAR_PORT / APP_*_DIR env vars set
by electron/main.js; falls back to sane local defaults when run standalone).
"""
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import essence
import media
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


class ExtractRequest(BaseModel):
    image_path: str
    name: str | None = None


@app.post("/essences/extract")
def extract_essence_endpoint(req: ExtractRequest):
    try:
        return essence.extract_essence(req.image_path, req.name)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"Image not found: {req.image_path}")
    except Exception as e:  # noqa: BLE001 — surface any Pillow/decoding failure as a 400, not a 500 stack trace
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/essences")
def list_essences_endpoint():
    return {"essences": essence.list_essences()}


@app.delete("/essences/{essence_id}")
def delete_essence_endpoint(essence_id: str):
    try:
        essence.delete_essence(essence_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Essence not found")
    return {"ok": True}


class ApplyRequest(BaseModel):
    essence_id: str
    image_path: str
    steps: int = 8


@app.post("/apply")
def apply_endpoint(req: ApplyRequest):
    try:
        result = essence.apply_essence(req.essence_id, req.image_path, req.steps)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Essence or target image not found")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))

    essences_by_id = {e["id"]: e for e in essence.list_essences()}
    essence_name = essences_by_id.get(req.essence_id, {}).get("name", "unknown")
    saved = media.save_creation(req.essence_id, essence_name, result["final"])
    return {**result, "media_id": saved["id"]}


@app.get("/media")
def list_media_endpoint():
    return {"media": media.list_creations()}


@app.delete("/media/{creation_id}")
def delete_media_endpoint(creation_id: str):
    try:
        media.delete_creation(creation_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Creation not found")
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("SIDECAR_PORT", "8843"))
    # 127.0.0.1 only, per the module docstring — this must not be reachable off-device.
    uvicorn.run(app, host="127.0.0.1", port=port)
