"""Frozen cross-packet stubs. The owning packet supplies the implementation."""

from __future__ import annotations

from pathlib import Path

from declip.contracts import (
    AudioTarget,
    Capabilities,
    Encoder,
    MediaInfo,
    VideoTarget,
)


def probe_capabilities(*, refresh: bool = False) -> Capabilities:
    raise NotImplementedError("Implementation belongs to a later packet")


def video_target_for(
    media: MediaInfo, *, codec: str, quality: str, allow_8bit: bool, container: str
) -> VideoTarget:
    raise NotImplementedError("Implementation belongs to a later packet")


def audio_target_for(
    media: MediaInfo, *, bitrate: int, container: str, pcm: bool = False
) -> AudioTarget:
    raise NotImplementedError("Implementation belongs to a later packet")


def select_encoder(
    target: VideoTarget, caps: Capabilities, *, prefer: str = "auto"
) -> Encoder:
    raise NotImplementedError("Implementation belongs to a later packet")


def hwaccel_args(source: Path, media: MediaInfo, caps: Capabilities) -> list[str]:
    raise NotImplementedError("Implementation belongs to a later packet")


def check_output(path: Path, target: VideoTarget) -> None:
    raise NotImplementedError("Implementation belongs to a later packet")
