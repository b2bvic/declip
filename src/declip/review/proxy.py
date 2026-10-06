"""Frozen cross-packet stubs. The owning packet supplies the implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from declip.contracts import (
    MediaInfo,
)


def proxy_args(media: MediaInfo) -> list[str]:
    raise NotImplementedError("Implementation belongs to a later packet")


def proxy_path(cache_dir: Path, source_sha256: str, media: MediaInfo) -> Path:
    raise NotImplementedError("Implementation belongs to a later packet")


def build_proxy(source: Path, proxy_path: Path, ffmpeg_args: Sequence[str]) -> Path:
    raise NotImplementedError("Implementation belongs to a later packet")
