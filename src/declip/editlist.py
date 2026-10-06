"""Frozen cross-packet stubs. The owning packet supplies the implementation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Sequence

from declip.contracts import (
    CutKind,
    CutProposal,
    CutStatus,
    EditList,
    EffectiveTimeline,
    Loudness,
    MediaInfo,
    OutputSpec,
    Processing,
    RigRef,
    Transcript,
)


def cut_id(kind: CutKind, start: float, end: float, label: str) -> str:
    raise NotImplementedError("Implementation belongs to a later packet")


def new_edit_list(
    source: Path,
    media: MediaInfo,
    sha256: str,
    *,
    transcript: Transcript | None,
    processing: Processing,
    output: OutputSpec,
    rig: RigRef,
) -> EditList:
    raise NotImplementedError("Implementation belongs to a later packet")


def load_edit_list(path: Path, *, source: Path | None = None) -> EditList:
    raise NotImplementedError("Implementation belongs to a later packet")


def save_edit_list(
    path: Path, edit_list: EditList, *, expected_revision: str | None = None
) -> str:
    raise NotImplementedError("Implementation belongs to a later packet")


def revision_of(path: Path) -> str:
    raise NotImplementedError("Implementation belongs to a later packet")


def migrate_schema2(
    path: Path,
    *,
    source: Path | None,
    resolve_preset: Callable[[str], tuple[str, Loudness | None]],
) -> EditList:
    raise NotImplementedError("Implementation belongs to a later packet")


def migration_destination(path: Path) -> Path:
    raise NotImplementedError("Implementation belongs to a later packet")


def replace_stage_cuts(
    edit_list: EditList, kind: CutKind, proposals: Sequence[CutProposal]
) -> EditList:
    raise NotImplementedError("Implementation belongs to a later packet")


def reset_decisions(edit_list: EditList) -> EditList:
    raise NotImplementedError("Implementation belongs to a later packet")


def apply_decisions(
    edit_list: EditList,
    *,
    decisions: Mapping[str, CutStatus],
    add_manual: Sequence[tuple[float, float, str]],
    remove_manual: Sequence[str],
) -> EditList:
    raise NotImplementedError("Implementation belongs to a later packet")


def review_content_hash(edit_list: EditList) -> str:
    raise NotImplementedError("Implementation belongs to a later packet")


def mark_review_passed(edit_list: EditList, *, now: datetime) -> EditList:
    raise NotImplementedError("Implementation belongs to a later packet")


def render_allowed(edit_list: EditList, source_sha256: str) -> bool:
    raise NotImplementedError("Implementation belongs to a later packet")


def require_current_review(edit_list: EditList, source: Path) -> None:
    raise NotImplementedError("Implementation belongs to a later packet")


def accepted_intervals(edit_list: EditList) -> list[tuple[float, float]]:
    raise NotImplementedError("Implementation belongs to a later packet")


def effective_timeline(edit_list: EditList) -> EffectiveTimeline:
    raise NotImplementedError("Implementation belongs to a later packet")


def keep_intervals(edit_list: EditList) -> list[tuple[float, float]]:
    raise NotImplementedError("Implementation belongs to a later packet")


def remap_time(seconds: float, timeline: EffectiveTimeline) -> float | None:
    raise NotImplementedError("Implementation belongs to a later packet")
