"""App-data directory access. Reads the env vars electron/main.js sets when it
spawns the sidecar; falls back to a local ./tmp-devdata for standalone runs
(e.g. `python app.py` or test scripts run outside Electron).
"""
import os
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).parent / "tmp-devdata"


def _dir(env_var: str, subdir: str) -> Path:
    raw = os.environ.get(env_var)
    p = Path(raw) if raw else _DEFAULT_ROOT / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    """Cached base-model weights (Flux) and style-extraction adapter weights."""
    return _dir("APP_MODELS_DIR", "models")


def essences_dir() -> Path:
    """Root folder holding one subfolder per saved Essence (embedding + metadata).
    Exact schema is still an open item (spec §7) — resolved when extraction lands.
    """
    return _dir("APP_ESSENCES_DIR", "essences")


def media_dir() -> Path:
    """Media Page archive — every finished creation, regardless of export status."""
    return _dir("APP_MEDIA_DIR", "media")


def cache_dir() -> Path:
    return _dir("APP_CACHE_DIR", "cache")


def db_dir() -> Path:
    return _dir("APP_DB_DIR", "db")
