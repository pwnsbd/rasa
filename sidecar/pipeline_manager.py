"""Lazy, background-loaded singleton for the real SDXL + InstantStyle +
ControlNet pipeline (spec §2.2a/b) — replaces essence.py's earlier
mock-palette placeholder.

Base model: SDXL, not Flux. InstantStyle's block-separation technique (style
vs. layout blocks) is diffusers-native and proven on SDXL; the available
Flux IP-Adapter checkpoints are FLUX.1-dev only (gated, non-commercial
license) and their own authors say they aren't for fine-grained style
transfer — see the decision recorded in chat and BaseModel's interface in
spec §2.2a, which is exactly what makes swapping base models later (if a
Flux/InstantStyle combination matures) a contained change: this module is
the only place that knows which base model is loaded.

Also loads a Tile ControlNet alongside the IP-Adapter. Plain img2img
`strength` alone (the earlier version of this module) is a global noise-mix
knob, not a real structural constraint — any strength high enough to let
the style actually take hold also let content/layout drift, which is
exactly the failure the original InstantStyle team hit and published a
follow-up for: InstantStyle-Plus (arXiv:2407.00788) adds a Tile ControlNet
specifically to hold the source image's structure in place throughout
denoising while IP-Adapter drives style. See essence.py's apply_essence for
how the two conditioning signals combine.

Loading ~11.5GB of weights (SDXL base fp16 + IP-Adapter + its CLIP image
encoder + Tile ControlNet + the fp16-fix VAE) takes real time on first run —
several minutes to download, then ~10-20s to move onto the GPU. Runs in a
background thread from server startup (see app.py's lifespan) so /health and
/models/status can report progress instead of the first extract/apply
request just hanging with no feedback.
"""
from __future__ import annotations

import os
import threading
import time

import paths

# Must be set before diffusers/transformers/huggingface_hub are imported
# anywhere in the process — keeps downloaded weights inside Rasa's own
# app-data folder (paths.models_dir()) rather than the global
# ~/.cache/huggingface, consistent with the "local filesystem, no cloud
# dependency" philosophy (spec §1).
os.environ.setdefault("HF_HOME", str(paths.models_dir() / "hf-cache"))

_lock = threading.Lock()
_pipeline = None
_status: dict = {"state": "idle", "detail": None}  # idle | loading | ready | error

BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_WEIGHT = "ip-adapter_sdxl.bin"
CONTROLNET_ID = "xinsir/controlnet-tile-sdxl-1.0"  # InstantStyle-Plus's content-preservation fix, see module docstring
VAE_FIX_ID = "madebyollin/sdxl-vae-fp16-fix"  # avoids SDXL's known fp16 VAE instability without upcasting to fp32
# Model card default is 1.0; dropped to 0.85 after testing — content stayed
# perfectly intact at 1.0 too, but 0.85 left slightly more room for style to
# show (see essence.py's DEFAULT_STRENGTH comment for the fuller picture).
CONTROLNET_CONDITIONING_SCALE = 0.85

# InstantStyle (spec §2.2b): activate the IP-Adapter only in the
# style-carrying block (up_block_0) and hold it at 0 in the layout-carrying
# block (down_block_2) — isolates style from content/layout so the target
# photo's structure survives the restyle. See
# https://github.com/huggingface/diffusers/blob/main/docs/source/en/using-diffusers/ip_adapter.md
INSTANT_STYLE_SCALE = {"down": {"block_2": [0.0, 1.0]}, "up": {"block_0": [0.0, 1.0, 0.0]}}


def status() -> dict:
    return dict(_status)


def _load() -> None:
    global _pipeline
    try:
        import torch
        from diffusers import (
            AutoencoderKL,
            ControlNetModel,
            EulerAncestralDiscreteScheduler,
            StableDiffusionXLControlNetImg2ImgPipeline,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        if device == "cpu":
            _status.update(detail="No GPU detected — loading for CPU. This will be very slow.")

        _status.update(state="loading", detail="Downloading/loading Tile ControlNet (~2.5GB, first run only)…")
        controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=dtype)

        _status.update(detail="Downloading/loading fp16-fix VAE (~160MB, first run only)…")
        vae = AutoencoderKL.from_pretrained(VAE_FIX_ID, torch_dtype=dtype)

        _status.update(detail="Downloading/loading SDXL base model (~7GB, first run only)…")
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            BASE_MODEL_ID,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=dtype,
            variant="fp16" if device == "cuda" else None,
            use_safetensors=True,
        )
        # Recommended by the Tile ControlNet's model card for best results.
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

        _status.update(detail="Downloading/loading InstantStyle IP-Adapter (~2GB, first run only)…")
        pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder="sdxl_models", weight_name=IP_ADAPTER_WEIGHT)
        pipe.set_ip_adapter_scale(INSTANT_STYLE_SCALE)

        pipe.enable_vae_slicing()  # keeps VAE decode memory bounded, cheap to always have on

        if device == "cuda":
            # SDXL + IP-Adapter + its CLIP-H image encoder + dual text
            # encoders + Tile ControlNet resident all at once leaves almost
            # no headroom on a ~12GB card (measured ~11.3GB baseline on an
            # RTX 5070 Ti Laptop's 12227MiB even before adding the
            # ControlNet) — not enough for the UNet's own activation memory
            # during denoising, which manifested as ~35s/step (should be
            # ~1-3s/step) rather than an outright OOM. enable_model_cpu_offload
            # keeps only the actively-computing submodule on GPU, swapping
            # others to CPU RAM between stages — use this INSTEAD of
            # `pipe.to(device)` (offload manages device placement itself).
            # See essence.py's use of `_execution_device` rather than
            # `.device` for why direct pipeline calls need to account for
            # this too.
            _status.update(detail="Configuring GPU memory offload…")
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)

        _pipeline = pipe
        _status.update(state="ready", detail=None)
    except Exception as e:  # noqa: BLE001 — record the failure so status()/callers can surface it, not crash the sidecar
        _status.update(state="error", detail=str(e))


def ensure_loading_started() -> None:
    with _lock:
        if _status["state"] == "idle":
            _status.update(state="loading", detail="Starting…")
            threading.Thread(target=_load, daemon=True, name="pipeline-loader").start()


def get_pipeline_blocking(timeout_s: float = 1800.0):
    """Blocks the CALLING thread until the pipeline is ready, or raises.
    Only call this from a synchronous (`def`, not `async def`) FastAPI
    endpoint — those already run in Starlette's threadpool, so blocking here
    doesn't stall /health or other concurrent requests.
    """
    ensure_loading_started()
    deadline = time.time() + timeout_s
    while _status["state"] == "loading":
        if time.time() > deadline:
            raise TimeoutError("Style model is still loading after the timeout")
        time.sleep(0.5)
    if _status["state"] == "error":
        raise RuntimeError(f"Style model failed to load: {_status['detail']}")
    return _pipeline
