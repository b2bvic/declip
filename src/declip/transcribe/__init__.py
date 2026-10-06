"""Frozen cross-packet stubs. The owning packet supplies the implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from declip.contracts import (
    BackendSelection,
    FillerFile,
    TranscribeOptions,
    Transcriber,
    Transcript,
)


def get(name: str) -> Transcriber:
    raise NotImplementedError("Implementation belongs to a later packet")


def select(device: str = "auto", *, backend: str = "auto") -> BackendSelection:
    raise NotImplementedError("Implementation belongs to a later packet")


def extract_transcription_audio(
    source: Path, out_wav: Path, *, audio_index: int | None
) -> Path:
    raise NotImplementedError("Implementation belongs to a later packet")


def cache_key(wav_sha256: str, backend: str, opts: TranscribeOptions) -> str:
    raise NotImplementedError("Implementation belongs to a later packet")


def transcribe_file(
    source: Path,
    opts: TranscribeOptions,
    *,
    selection: BackendSelection,
    cache_dir: Path,
    audio_index: int | None,
    fillers_for: Callable[[str], FillerFile | None],
) -> Transcript:
    raise NotImplementedError("Implementation belongs to a later packet")
