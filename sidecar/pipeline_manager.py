"""Lazy, background-loaded singleton for the real SDXL + InstantStyle +
ControlNet pipeline (spec §2.2a/b) — replaces essence_store.py's earlier
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
denoising while IP-Adapter drives style. See generation.py's apply_essence for
how the two conditioning signals combine.

Loading ~11.5GB of weights (SDXL base fp16 + IP-Adapter + its CLIP image
encoder + Tile ControlNet + the fp16-fix VAE) takes real time on first run —
several minutes to download, then ~10-20s to move onto the GPU. Runs in a
background thread from server startup (see app.py's lifespan) so /health and
/models/status can report progress instead of the first extract/apply
request just hanging with no feedback.
"""
from __future__ import annotations

import contextlib
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
_status: dict = {"state": "idle", "detail": None, "memory_plan": None}  # state: idle | loading | ready | error
_prompt_embeds: dict | None = None

NEGATIVE_PROMPT = "lowres, blurry, bad anatomy, worst quality, low quality, watermark, text"

BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_WEIGHT = "ip-adapter_sdxl.bin"
CONTROLNET_ID = "xinsir/controlnet-tile-sdxl-1.0"  # InstantStyle-Plus's content-preservation fix, see module docstring
VAE_FIX_ID = "madebyollin/sdxl-vae-fp16-fix"  # avoids SDXL's known fp16 VAE instability without upcasting to fp32
# Model card default is 1.0; dropped to 0.85 after testing — content stayed
# perfectly intact at 1.0 too, but 0.85 left slightly more room for style to
# show (see generation.py's DEFAULT_STRENGTH comment for the fuller picture).
CONTROLNET_CONDITIONING_SCALE = 0.85

# InstantStyle (spec §2.2b): activate the IP-Adapter only in the
# style-carrying block (up_block_0) and hold it at 0 in the layout-carrying
# block (down_block_2) — isolates style from content/layout so the target
# photo's structure survives the restyle. See
# https://github.com/huggingface/diffusers/blob/main/docs/source/en/using-diffusers/ip_adapter.md
INSTANT_STYLE_SCALE = {"down": {"block_2": [0.0, 1.0]}, "up": {"block_0": [0.0, 1.0, 0.0]}}


# Memory plan (see docs/contracts/restyle-performance.md). "resident" keeps
# UNet + ControlNet + VAE on the GPU for the whole session (no per-run
# RAM<->VRAM shuffling); text encoders and CLIP image encoder stay on CPU.
# ~9.0GB = UNet 5.1 + ControlNet 2.5 + VAE/IP-adapter ~0.4 + activation headroom.
RESIDENT_MIN_FREE_BYTES = int(9.0 * 1024**3)


def choose_memory_plan(free_vram_bytes: int, total_vram_bytes: int, device: str) -> str:
    """Pure placement decision: "resident" | "offload" | "cpu"."""
    if str(device) != "cuda":
        return "cpu"
    if free_vram_bytes >= RESIDENT_MIN_FREE_BYTES and total_vram_bytes >= RESIDENT_MIN_FREE_BYTES:
        return "resident"
    return "offload"


def status() -> dict:
    return dict(_status)


def compute_device():
    """Device the denoising runs on. Use this instead of
    `pipe._execution_device`, which can report CPU in resident mode where
    components are deliberately split across devices."""
    import torch

    if _status.get("memory_plan") == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cached_prompt_embeds() -> dict | None:
    """Pre-computed embeds for prompt="" + the constant negative prompt
    (kwargs for the pipeline call), or None if not cached."""
    return _prompt_embeds


_encoder_depth = 0  # nesting count for image_encoder_on_compute (one move per outermost use)


@contextlib.contextmanager
def image_encoder_on_compute(pipe):
    """In resident mode the CLIP image encoder (~3.4GB) lives on CPU; move it
    to the compute device only while embedding, then back. Reentrant: nested
    uses (e.g. per-crop calls inside a multi-crop extraction) move it once.

    Always runs under no_grad: an embedding computed with grad enabled keeps
    the autograd graph — and through it the GPU copy of the encoder weights —
    alive after the move back to CPU, leaking ~3.4GB per call."""
    global _encoder_depth
    import torch

    enc = getattr(pipe, "image_encoder", None)
    move = _status.get("memory_plan") == "resident" and enc is not None and _encoder_depth == 0
    if move:
        enc.to(compute_device())
    _encoder_depth += 1
    try:
        with torch.no_grad():
            yield
    finally:
        _encoder_depth -= 1
        if move:
            enc.to("cpu")
            torch.cuda.empty_cache()


def _cache_prompt_embeds(pipe, resident: bool) -> None:
    """Encode the fixed empty prompt once so the text encoders never run
    per-apply. In resident mode they are briefly moved to the GPU for it."""
    global _prompt_embeds
    import torch

    dev = compute_device()
    encoders = [getattr(pipe, n, None) for n in ("text_encoder", "text_encoder_2")]
    if resident:
        for e in encoders:
            if e is not None:
                e.to(dev)
    try:
        with torch.no_grad():
            pe, npe, ppe, nppe = pipe.encode_prompt(
                prompt="",
                prompt_2=None,
                device=dev,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=NEGATIVE_PROMPT,
            )
    finally:
        if resident:
            for e in encoders:
                if e is not None:
                    e.to("cpu")
            torch.cuda.empty_cache()
    edtype = torch.float16 if dev.type == "cuda" else torch.float32
    _prompt_embeds = {
        "prompt_embeds": pe.to(dev, edtype),
        "negative_prompt_embeds": npe.to(dev, edtype),
        "pooled_prompt_embeds": ppe.to(dev, edtype),
        "negative_pooled_prompt_embeds": nppe.to(dev, edtype),
    }


def _make_resident(pipe):
    """Put UNet/ControlNet/VAE on cuda; text/image encoders stay on CPU.
    Components are then split across devices, so pin _execution_device to
    cuda (the stock property can report the CPU-resident encoder's device)."""
    import torch

    pipe.unet.to("cuda")
    pipe.controlnet.to("cuda")
    pipe.vae.to("cuda")
    cls = type(pipe)
    pipe.__class__ = type(cls.__name__, (cls,), {"_execution_device": property(lambda self: torch.device("cuda"))})
    return pipe


def _cache_or_degrade(pipe, plan: str):
    """Cache prompt embeds. If that fails in resident mode, the string
    fallback in _run_generation can't work (text encoders are on CPU while
    the execution device is pinned to cuda), so undo resident placement and
    switch to offload. Other plans keep the string fallback. Returns (pipe, plan)."""
    try:
        _cache_prompt_embeds(pipe, resident=(plan == "resident"))
        return pipe, plan
    except Exception as e:  # noqa: BLE001 — optimisation only
        if plan != "resident":
            print(f"[pipeline] prompt-embed caching failed, using per-run encoding: {e}")
            return pipe, plan
        print(f"[pipeline] prompt-embed caching failed in resident mode, falling back to offload: {e}")
    import torch

    for m in (pipe.unet, pipe.controlnet, pipe.vae):
        m.to("cpu")
    if "_execution_device" in type(pipe).__dict__:
        pipe.__class__ = type(pipe).__mro__[1]
    torch.cuda.empty_cache()
    pipe.enable_model_cpu_offload()
    _status.update(memory_plan="offload")
    return pipe, "offload"


def _is_oom(e: Exception) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


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

        plan = "cpu"
        if device == "cuda":
            free, total = torch.cuda.mem_get_info()
            plan = choose_memory_plan(free, total, device)
            print(f"[pipeline] memory plan: {plan} (free VRAM {free / 1024**3:.1f}GB of {total / 1024**3:.1f}GB)")

        if plan == "resident":
            _status.update(detail="Moving models onto the GPU…")
            try:
                pipe = _make_resident(pipe)
            except Exception as e:  # noqa: BLE001
                if not _is_oom(e):
                    raise
                print(f"[pipeline] resident placement hit OOM, falling back to offload: {e}")
                for m in (pipe.unet, pipe.controlnet, pipe.vae):
                    m.to("cpu")
                torch.cuda.empty_cache()
                plan = "offload"

        if plan == "offload":
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
            # Callers use compute_device() rather than `.device` since
            # components idle on CPU under offload.
            _status.update(detail="Configuring GPU memory offload…")
            pipe.enable_model_cpu_offload()
        elif plan == "cpu":
            pipe = pipe.to(device)

        _status.update(memory_plan=plan)
        pipe, plan = _cache_or_degrade(pipe, plan)

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
