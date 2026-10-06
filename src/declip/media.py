"""Frozen cross-packet stubs. The owning packet supplies the implementation."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from declip.contracts import (
    MediaInfo,
)


def require_tools() -> tuple[Path, Path]:
    raise NotImplementedError("Implementation belongs to a later packet")


def probe_media(path: Path) -> MediaInfo:
    raise NotImplementedError("Implementation belongs to a later packet")


def file_hash(path: Path, *, refresh: bool = False) -> str:
    raise NotImplementedError("Implementation belongs to a later packet")


def snap_fps(rate: Fraction) -> Fraction | None:
    raise NotImplementedError("Implementation belongs to a later packet")
