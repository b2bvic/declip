"""Frozen cross-packet stubs. The owning packet supplies the implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from declip.contracts import (
    CutProposal,
    FillerFile,
    Word,
)


def detect_fillers(
    words: Sequence[Word],
    fillers: FillerFile,
    *,
    min_confidence: float,
    margin_ms: float,
    duration: float,
) -> list[CutProposal]:
    raise NotImplementedError("Implementation belongs to a later packet")


def detect_retakes(
    words: Sequence[Word],
    *,
    min_confidence: float,
    similarity_threshold: float = 0.6,
    window_s: float = 15.0,
) -> list[CutProposal]:
    raise NotImplementedError("Implementation belongs to a later packet")


def detect_waveform_gaps(
    audio: Path,
    *,
    noise_db: float,
    max_gap_ms: float,
    min_silence_ms: float,
    duration: float,
    audio_index: int | None = None,
) -> list[CutProposal]:
    raise NotImplementedError("Implementation belongs to a later packet")


def legacy_words(transcript: Mapping[str, Any]) -> list[Word]:
    raise NotImplementedError("Implementation belongs to a later packet")
