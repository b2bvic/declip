"""Resolve user state paths at first use, with explicit test overrides."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import platformdirs


def _directory(kind: str) -> Path:
    override = os.environ.get(f"DECLIP_{kind.upper()}_DIR")
    if override:
        directory = Path(override).expanduser()
    elif sys.platform in ("darwin", "linux"):
        directory = Path.home() / f".{kind}" / "declip"
    else:
        provider = getattr(platformdirs, f"user_{kind}_dir")
        directory = Path(provider("declip", appauthor=False))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def config_dir() -> Path:
    return _directory("config")


def cache_dir() -> Path:
    return _directory("cache")
