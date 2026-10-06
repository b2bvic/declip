"""Frozen cross-packet stubs. The owning packet supplies the implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from declip.contracts import (
    Capabilities,
    EditList,
    MediaInfo,
    RenderPlan,
    RenderResult,
    ResolvedOptions,
)


def default_output_path(source: Path, media: MediaInfo) -> Path:
    raise NotImplementedError("Implementation belongs to a later packet")


def build_render_plan(
    edit_list: EditList,
    source: Path,
    output: Path,
    options: ResolvedOptions,
    caps: Capabilities,
    *,
    temp_dir: Path,
    overwrite: bool = False,
) -> RenderPlan:
    raise NotImplementedError("Implementation belongs to a later packet")


def build_enhance_plan(
    source: Path,
    output: Path,
    media: MediaInfo,
    options: ResolvedOptions,
    caps: Capabilities,
    *,
    temp_dir: Path,
    overwrite: bool = False,
) -> RenderPlan:
    raise NotImplementedError("Implementation belongs to a later packet")


def run_render_plan(
    plan: RenderPlan,
    *,
    progress: Callable[[str, float], None] | None = None,
    verbose: bool = False,
) -> RenderResult:
    raise NotImplementedError("Implementation belongs to a later packet")


def render_processed_audio(
    edit_list: EditList,
    source: Path,
    output_wav: Path,
    options: ResolvedOptions,
    *,
    temp_dir: Path,
) -> Path:
    raise NotImplementedError("Implementation belongs to a later packet")
