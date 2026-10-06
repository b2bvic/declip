"""Resolve model names and announce hub downloads before fetching weights."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from declip.contracts import TranscriberUnavailable

MLX_MODELS = {
    "small": "mlx-community/whisper-small-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}
FASTER_MODELS = {
    "small": "Systran/faster-whisper-small",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}
# Compatibility with names in old configs and the draft rig-default table.
ALIASES = {
    "mlx-community/whisper-small": "small",
    "mlx-community/whisper-large-v3": "large-v3",
}
PATTERNS = {
    "mlx": ["config.json", "weights.npz", "weights.safetensors"],
    "faster": [
        "config.json",
        "model.bin",
        "tokenizer.json",
        "vocabulary.*",
        "preprocessor_config.json",
    ],
}


def model_repository(name: str, backend: str) -> str:
    short = ALIASES.get(name, name)
    for key, repo in MLX_MODELS.items():
        if short == repo:
            short = key
    return (MLX_MODELS if backend == "mlx" else FASTER_MODELS).get(short, short)


def ensure_model(name: str, *, backend: str) -> str:
    if backend not in PATTERNS:
        raise TranscriberUnavailable(f"Unknown model backend: {backend}")
    local = Path(name).expanduser()
    if local.is_dir():
        return str(local.resolve())
    if local.is_absolute() or name.startswith(("./", "../", "~")):
        raise TranscriberUnavailable(f"Local model directory does not exist: {name}")
    repository = model_repository(name, backend)
    try:
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.constants import HF_HUB_CACHE
    except (ImportError, OSError) as exc:
        raise TranscriberUnavailable(
            f"Install the {backend} extra to load {name}: {exc}"
        ) from exc
    kwargs = {"repo_id": repository, "allow_patterns": PATTERNS[backend]}
    try:
        cached = snapshot_download(**kwargs, local_files_only=True)
        # A snapshot directory can exist even when its weights were never fetched.
        root = Path(cached)
        weights = (
            ["model.bin"]
            if backend == "faster"
            else ["weights.npz", "weights.safetensors"]
        )
        if (root / "config.json").is_file() and any(
            (root / w).is_file() for w in weights
        ):
            return cached
    except Exception:
        pass
    if os.environ.get("HF_HUB_OFFLINE", "").upper() in {"1", "TRUE", "YES", "ON"}:
        raise TranscriberUnavailable(
            f"Model {repository} is not cached; HF_HUB_OFFLINE prevents download"
        )
    size = "size unknown"
    try:
        info = HfApi().model_info(repository, files_metadata=True)
        sizes = [s.size for s in info.siblings]
        if sizes and all(s is not None for s in sizes):
            size = f"{sum(sizes)} bytes"
    except Exception:
        pass
    print(
        f"Downloading model {repository} from huggingface.co to {HF_HUB_CACHE} ({size})",
        file=sys.stderr,
        flush=True,
    )
    try:
        return snapshot_download(**kwargs)
    except Exception as exc:
        raise TranscriberUnavailable(
            f"Cannot download model {repository}: {exc}"
        ) from exc
