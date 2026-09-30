"""Background model downloads with byte-level progress (drives the UI's
download button).

Downloading is split from loading: this module fetches every weight file Rasa
needs into the Hugging Face cache (`<models-dir>/hf-cache`) on a background
thread, reports progress, and can be paused/resumed. pipeline_manager.py and
depth.py then load from that cache lazily on first use.

Each item downloads only the files the loaders actually read (e.g. SDXL's
fp16 UNet/text encoders, not the 7GB single-file checkpoint or the ONNX/
OpenVINO/Flax copies), so the total is ~13GB rather than ~60GB. The patterns
mirror what `from_pretrained` resolves in pipeline_manager.py/depth.py — if a
loader starts reading a new file, add it here or the load will fall back to
fetching it on demand.

Pausing works by raising from the progress bar's update(); partial files
stay in the cache as `.incomplete` and resume on the next attempt.
"""
from __future__ import annotations

import fnmatch
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import paths

# Must be set before huggingface_hub is imported — see pipeline_manager.py.
os.environ.setdefault("HF_HOME", str(paths.models_dir() / "hf-cache"))
# hf-xet downloads bypass the tqdm progress bar (no byte progress, no pause);
# the plain HTTP path reports both.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


@dataclass
class Item:
    id: str
    label: str
    repo: str
    allow: list[str]
    total: int = 0          # bytes, known once we've asked the Hub (0 until then)
    done_bytes: int = 0
    state: str = "pending"  # pending | downloading | done | error
    files: list[str] = field(default_factory=list)  # resolved file names, once known


def build_items(use_fp16: bool) -> list[Item]:
    suffix = ".fp16" if use_fp16 else ""
    return [
        Item("controlnet", "Tile ControlNet", "xinsir/controlnet-tile-sdxl-1.0",
             ["config.json", "diffusion_pytorch_model.safetensors"]),
        Item("vae", "fp16-fix VAE", "madebyollin/sdxl-vae-fp16-fix",
             ["config.json", "diffusion_pytorch_model.safetensors"]),
        Item("sdxl", "SDXL base model", "stabilityai/stable-diffusion-xl-base-1.0",
             ["model_index.json", "scheduler/*.json", "tokenizer*/*",
              "text_encoder/config.json", "text_encoder_2/config.json", "unet/config.json", "vae/config.json",
              f"text_encoder/model{suffix}.safetensors",
              f"text_encoder_2/model{suffix}.safetensors",
              f"unet/diffusion_pytorch_model{suffix}.safetensors"]),
        Item("ip_adapter", "InstantStyle IP-Adapter", "h94/IP-Adapter",
             ["sdxl_models/ip-adapter_sdxl.bin",
              "sdxl_models/image_encoder/config.json",
              "sdxl_models/image_encoder/model.safetensors"]),
        Item("depth", "Depth model", "depth-anything/Depth-Anything-V2-Small-hf",
             ["config.json", "preprocessor_config.json", "model.safetensors"]),
    ]


class _Paused(Exception):
    pass


_lock = threading.RLock()
_items: list[Item] = []
_state = "idle"  # idle | downloading | paused | ready | error
_error: str | None = None
_pause_requested = False
_thread: threading.Thread | None = None
_samples: deque = deque()  # (timestamp, total downloaded bytes) for speed


def _use_fp16() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001 — no torch => CPU loader path
        return False


def _bytes_done() -> int:
    return sum(i.total if i.state == "done" else i.done_bytes for i in _items)


def status() -> dict:
    with _lock:
        total = sum(i.total for i in _items)
        done = _bytes_done()
        now = time.time()
        while _samples and now - _samples[0][0] > 6:
            _samples.popleft()
        speed = 0.0
        if len(_samples) >= 2 and _state == "downloading":
            dt = _samples[-1][0] - _samples[0][0]
            if dt > 0:
                speed = max(0.0, (_samples[-1][1] - _samples[0][1]) / dt)
        eta = (total - done) / speed if speed > 1 and total else None
        return {
            "state": _state,
            "error": _error,
            "total_bytes": total,
            "downloaded_bytes": done,
            "speed_bps": speed,
            "eta_seconds": eta,
            "items": [
                {"id": i.id, "label": i.label, "total_bytes": i.total,
                 "downloaded_bytes": i.total if i.state == "done" else i.done_bytes,
                 "state": i.state}
                for i in _items
            ],
        }


def is_ready() -> bool:
    return _state == "ready"


def _record_sample() -> None:
    _samples.append((time.time(), _bytes_done()))


_current: Item | None = None
_bar_installed = False


def _install_progress_hook() -> None:
    """snapshot_download only applies `tqdm_class` to its outer file-count bar;
    the per-file byte bars are created inside huggingface_hub.utils.tqdm via
    its module-level `tqdm` name, so that is what we subclass and replace."""
    global _bar_installed
    if _bar_installed:
        return
    import sys
    import huggingface_hub  # noqa: F401 — ensures the submodule is loaded
    tqdm_mod = sys.modules["huggingface_hub.utils.tqdm"]
    Base = tqdm_mod.tqdm

    class Bar(Base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            item = _current
            if item is not None and getattr(self, "unit", None) == "B" and self.n:
                with _lock:
                    item.done_bytes += int(self.n)  # resumed partial file

        def update(self, n=1):
            item = _current
            if item is not None and getattr(self, "unit", None) == "B":
                with _lock:
                    item.done_bytes += int(n)
                    _record_sample()
                if _pause_requested:
                    raise _Paused()
            return super().update(n)

    tqdm_mod.tqdm = Bar
    _bar_installed = True


def _manifest_path(item: Item):
    return paths.models_dir() / "rasa-manifests" / f"{item.id}.json"


def _all_in_cache(item: Item, files: list[str]) -> bool:
    from huggingface_hub import try_to_load_from_cache
    return bool(files) and all(
        isinstance(try_to_load_from_cache(item.repo, f), str) for f in files
    )


def _partial_or_missing(item: Item) -> bool:
    """Offline fallback when the file list can't be fetched: a repo folder with
    no `.incomplete` blobs is taken as complete."""
    repo_dir = paths.models_dir() / "hf-cache" / "hub" / ("models--" + item.repo.replace("/", "--"))
    if not (repo_dir / "snapshots").exists():
        return True
    return any((repo_dir / "blobs").glob("*.incomplete"))


def _cached(item: Item) -> bool:
    """True only when every file this item needs is fully in the cache.
    (snapshot_download(local_files_only=True) is NOT usable for this — it
    succeeds as soon as any snapshot folder exists, even a half-downloaded one.)"""
    import json
    mp = _manifest_path(item)
    try:
        files = json.loads(mp.read_text())
        if _all_in_cache(item, files):
            return True
    except Exception:  # noqa: BLE001 — no/invalid manifest: fall through to verifying
        pass
    try:
        _measure(item)
    except Exception:  # noqa: BLE001 — offline
        return not _partial_or_missing(item)
    if _all_in_cache(item, item.files):
        _write_manifest(item)
        return True
    return False


def _write_manifest(item: Item) -> None:
    import json
    try:
        mp = _manifest_path(item)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(item.files))
    except Exception:  # noqa: BLE001 — manifest is an optimization; re-verify next launch
        pass


def _measure(item: Item) -> None:
    from huggingface_hub import HfApi
    info = HfApi().model_info(item.repo, files_metadata=True)
    matched = [s for s in info.siblings if any(fnmatch.fnmatch(s.rfilename, p) for p in item.allow)]
    item.files = [s.rfilename for s in matched]
    item.total = sum(s.size or 0 for s in matched)


def _download_item(item: Item) -> None:
    """Fetch one item. Module-level so tests can replace it."""
    global _current
    from huggingface_hub import snapshot_download
    _install_progress_hook()
    _current = item
    try:
        snapshot_download(item.repo, allow_patterns=item.allow)
    finally:
        _current = None


def _run() -> None:
    global _state, _error
    try:
        for item in _items:
            if item.state == "done":
                continue
            if _cached(item):
                with _lock:
                    if not item.total:
                        item.total = 1
                    item.state = "done"
                continue
            if not item.files:
                _measure(item)
            with _lock:
                item.state = "downloading"
                item.done_bytes = 0
            _download_item(item)
            _write_manifest(item)
            with _lock:
                item.state = "done"
                _record_sample()
        with _lock:
            _state = "ready"
            _error = None
    except _Paused:
        with _lock:
            _state = "paused"
            for i in _items:
                if i.state == "downloading":
                    i.state = "pending"
    except Exception as e:  # noqa: BLE001 — surface the failure to the UI with a retry
        with _lock:
            _state = "error"
            _error = str(e)
            for i in _items:
                if i.state == "downloading":
                    i.state = "error"


def _start_thread() -> None:
    global _thread, _state, _error, _pause_requested
    with _lock:
        if _thread is not None and _thread.is_alive():
            return
        _pause_requested = False
        _state = "downloading"
        _error = None
        for i in _items:
            if i.state == "error":
                i.state = "pending"
        _samples.clear()
        _record_sample()
        _thread = threading.Thread(target=_run, daemon=True, name="model-downloader")
        _thread.start()


def start() -> None:
    """Begin downloading in the background (idempotent). Called at sidecar startup."""
    with _lock:
        if not _items:
            _items.extend(build_items(_use_fp16()))
        if _state in ("idle", "error"):
            _start_thread()


def pause() -> None:
    global _pause_requested
    with _lock:
        if _state == "downloading":
            _pause_requested = True


def resume() -> None:
    with _lock:
        if _state not in ("paused", "error"):
            return
        t = _thread
    if t is not None:
        t.join(timeout=10)  # a pause in flight may not have unwound yet
    _start_thread()
