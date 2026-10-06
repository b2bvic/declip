"""Frozen cross-packet stubs. The owning packet supplies the implementation."""

from __future__ import annotations

from pathlib import Path

from declip.contracts import (
    ClipMeasurement,
)


def measure_clip(path: Path, *, seconds: float = 60.0) -> ClipMeasurement:
    raise NotImplementedError("Implementation belongs to a later packet")
