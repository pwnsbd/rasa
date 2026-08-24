"""GPU/device detection (spec §2.3).

Kept separate from app.py so /health stays cheap and doesn't hard-fail if
torch isn't installed yet (e.g. first sidecar boot before `sidecar:setup`
has run torch installation).

Generic by design: reads whatever CUDA device index 0 reports at runtime
rather than hard-coding any specific card. The RTX 5070 (CUDA 12.6,
compute capability sm_120 / "Blackwell") is the primary dev target, but
nothing here assumes it — see scripts/setup-sidecar.js for why sm_120
specifically needs a cu128+ torch build (older CUDA wheels report
cuda_available=True but have no compiled kernels for it and fail at first
real inference).
"""
from __future__ import annotations

# Below this, the app should still work but should show the CPU-fallback-style
# "expect this to be slow" warning even though a GPU was detected — a genuine
# low-VRAM card (e.g. a 4GB laptop GPU) can't hold Flux comfortably.
MIN_RECOMMENDED_VRAM_GB = 8.0


def detect_gpu() -> dict:
    try:
        import torch
    except ImportError:
        return {
            "torch_installed": False,
            "cuda_available": False,
            "device_name": None,
            "device_count": 0,
            "compute_capability": None,
            "vram_gb": None,
            "cuda_runtime_version": None,
            "torch_version": None,
            "warning": "PyTorch is not installed in the sidecar venv yet. Run `npm run sidecar:setup`.",
        }

    cuda_available = torch.cuda.is_available()
    device_name = None
    device_count = 0
    compute_capability = None
    vram_gb = None
    cuda_runtime_version = torch.version.cuda  # None on a CPU-only build

    if cuda_available:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        compute_capability = f"{props.major}.{props.minor}"
        vram_gb = round(props.total_memory / (1024 ** 3), 1)

    warning = None
    if not cuda_available:
        warning = "No compatible GPU detected — running on CPU. Generation will be significantly slower."
    elif vram_gb is not None and vram_gb < MIN_RECOMMENDED_VRAM_GB:
        warning = (
            f"Detected GPU has {vram_gb} GB VRAM, below the {MIN_RECOMMENDED_VRAM_GB:.0f} GB "
            "recommended for comfortable generation. Expect slower runs or the need for lower-memory settings."
        )

    return {
        "torch_installed": True,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "device_count": device_count,
        "compute_capability": compute_capability,
        "vram_gb": vram_gb,
        "cuda_runtime_version": cuda_runtime_version,
        "torch_version": torch.__version__,
        "warning": warning,
    }
