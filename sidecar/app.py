"""Local-only FastAPI sidecar. Electron's main process spawns this and talks to
it over 127.0.0.1 only — never bind 0.0.0.0, this must never be reachable off-device.

Run directly for dev: `python app.py` (reads SIDECAR_PORT / APP_*_DIR env vars set
by electron/main.js; falls back to sane local defaults when run standalone).
"""
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import depth
import essence_store
import generation
import media
import paths
import pipeline_manager
from gpu import detect_gpu


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Kick off the (large, multi-GB on first run) SDXL + InstantStyle
    # download/load in the background as soon as the sidecar starts, rather
    # than waiting for the first extract/apply request — see
    # pipeline_manager.py.
    pipeline_manager.ensure_loading_started()
    # Small/fast by comparison (~100MB, CPU-only — see depth.py), but same
    # treatment: start it now so a first depth-mode /apply doesn't stall on
    # an unexpected download.
    depth.ensure_loading_started()
    yield


app = FastAPI(title="rasa sidecar", lifespan=lifespan)

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
        "model": pipeline_manager.status(),
        "dirs": {
            "models": str(paths.models_dir()),
            "essences": str(paths.essences_dir()),
            "media": str(paths.media_dir()),
            "cache": str(paths.cache_dir()),
            "db": str(paths.db_dir()),
        },
    }


@app.get("/models/status")
def models_status():
    return pipeline_manager.status()


class PreviewRequest(BaseModel):
    image_path: str


@app.post("/utils/preview")
def preview_endpoint(req: PreviewRequest):
    # Pure decode/re-encode, no model involved — deliberately does not call
    # _require_model_ready() below, so HEIC/TIFF previews still work while
    # the style model is still downloading/loading.
    try:
        return {"data_url": essence_store.preview_data_url(req.image_path)}
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"Image not found: {req.image_path}")
    except Exception as e:  # noqa: BLE001 — surface any Pillow/decoding failure as a 400, not a 500 stack trace
        raise HTTPException(status_code=400, detail=str(e))


class ExtractRequest(BaseModel):
    image_path: str
    name: str | None = None


def _require_model_ready():
    st = pipeline_manager.status()
    if st["state"] != "ready":
        # Fail fast with a clear message rather than blocking the HTTP
        # request (and the renderer's fetch/IPC chain) for however many
        # minutes the first-run download takes — the frontend surfaces this
        # detail directly and /models/status lets it poll for readiness.
        raise HTTPException(status_code=503, detail=f"Style model not ready ({st['state']}): {st.get('detail') or '…'}")


@app.post("/essences/extract")
def extract_essence_endpoint(req: ExtractRequest):
    _require_model_ready()
    try:
        return essence_store.extract_essence(req.image_path, req.name)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"Image not found: {req.image_path}")
    except Exception as e:  # noqa: BLE001 — surface any Pillow/decoding failure as a 400, not a 500 stack trace
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/essences")
def list_essences_endpoint():
    return {"essences": essence_store.list_essences()}


class BlendIngredient(BaseModel):
    type: Literal["image", "essence"]
    image_path: str | None = None  # required when type == "image"
    essence_id: str | None = None  # required when type == "essence"
    weight: float = 1.0  # relative, not required to sum to 1 — blend_essences normalizes


class BlendRequest(BaseModel):
    name: str | None = None
    ingredients: list[BlendIngredient]


@app.post("/essences/blend")
def blend_essences_endpoint(req: BlendRequest):
    # Kept unconditional even though an all-existing-essence blend
    # technically wouldn't need the SDXL pipe at all — the model is already
    # loading in the background from startup regardless, so this simplicity
    # costs nothing in practice.
    _require_model_ready()
    try:
        return essence_store.blend_essences([i.model_dump() for i in req.ingredients], req.name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Essence not found: {e}")
    except Exception as e:  # noqa: BLE001 — surface any decoding/validation failure as a 400, not a 500 stack trace
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/essences/{essence_id}")
def delete_essence_endpoint(essence_id: str):
    try:
        essence_store.delete_essence(essence_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Essence not found")
    return {"ok": True}


class ApplyRequest(BaseModel):
    essence_id: str
    image_path: str
    steps: int = generation.DEFAULT_STEPS
    strength: float | None = None  # background-region override; None -> generation.DEFAULT_STRENGTH
    controlnet_scale: float | None = None  # background-region override; None -> pipeline_manager.CONTROLNET_CONDITIONING_SCALE
    blend_mode: str = "subject"  # "subject" (rembg+face two-pass, default) | "depth" (continuous depth two-pass) | "none" (flat single pass)
    subject_strength: float | None = None  # override for the suggested subject-region strength (see segmentation.py)
    subject_controlnet_scale: float | None = None  # override for the suggested subject-region controlnet_scale
    depth_near_strength: float | None = None  # override for depth mode's near-camera strength (see generation.DEPTH_NEAR_STRENGTH)
    depth_near_controlnet_scale: float | None = None  # override for depth mode's near-camera controlnet_scale
    depth_far_strength: float | None = None  # override for depth mode's far-camera strength (see generation.DEPTH_FAR_STRENGTH)
    depth_far_controlnet_scale: float | None = None  # override for depth mode's far-camera controlnet_scale
    preserve_color: bool = True  # restore the target's original color post-generation (see color_transfer.py); False lets the essence's own color through
    compute_depth: bool = False  # estimate + persist a depth map for the Media Page's parallax hover effect (see depth.py) — off by default: real, reported cost with no UI toggle to disable it
    content_aware_masking: bool = False  # apply-time target-aware content suppression, on top of distillation-time purification (see content_mask.py) — off by default, same reasoning


@app.post("/apply")
def apply_endpoint(req: ApplyRequest):
    _require_model_ready()
    try:
        result = generation.apply_essence(
            req.essence_id,
            req.image_path,
            steps=req.steps,
            strength=req.strength,
            controlnet_scale=req.controlnet_scale,
            blend_mode=req.blend_mode,
            subject_strength=req.subject_strength,
            subject_controlnet_scale=req.subject_controlnet_scale,
            depth_near_strength=req.depth_near_strength,
            depth_near_controlnet_scale=req.depth_near_controlnet_scale,
            depth_far_strength=req.depth_far_strength,
            depth_far_controlnet_scale=req.depth_far_controlnet_scale,
            preserve_color=req.preserve_color,
            compute_depth=req.compute_depth,
            content_aware_masking=req.content_aware_masking,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Essence or target image not found")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))

    essences_by_id = {e["id"]: e for e in essence_store.list_essences()}
    essence_name = essences_by_id.get(req.essence_id, {}).get("name", "unknown")
    saved = media.save_creation(req.essence_id, essence_name, result["final"], result.get("depth_map"))
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
