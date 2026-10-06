"""Frozen cross-packet stubs. The owning packet supplies the implementation."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Callable

from declip.contracts import (
    EditList,
    ExportFormat,
    ExportPlan,
    ExportResult,
    ResolvedOptions,
)


def export_paths(source: Path, *, out_dir: Path | None = None) -> dict[str, Path]:
    raise NotImplementedError("Implementation belongs to a later packet")


def frames_to_timecode(frames: int, fps: Fraction, *, drop_frame: bool) -> str:
    raise NotImplementedError("Implementation belongs to a later packet")


def build_export_plan(
    edit_list: EditList,
    source: Path,
    fmt: ExportFormat,
    options: ResolvedOptions,
    *,
    out_dir: Path | None = None,
    overwrite: bool = False,
) -> ExportPlan:
    raise NotImplementedError("Implementation belongs to a later packet")


def run_export_plan(
    plan: ExportPlan, *, render_audio: Callable[[Path], Path] | None = None
) -> ExportResult:
    raise NotImplementedError("Implementation belongs to a later packet")
