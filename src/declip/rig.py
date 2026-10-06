"""Frozen cross-packet stubs. The owning packet supplies the implementation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from declip.contracts import (
    ClipMeasurement,
    HardwareInfo,
    Loudness,
    ResolvedOptions,
    RigAnswers,
    RigProfile,
)


def rig_path(name: str, config_dir: Path) -> Path:
    raise NotImplementedError("Implementation belongs to a later packet")


def load_rig(path: Path) -> tuple[RigProfile, list[str]]:
    raise NotImplementedError("Implementation belongs to a later packet")


def save_rig(path: Path, profile: RigProfile) -> None:
    raise NotImplementedError("Implementation belongs to a later packet")


def default_model_for(hardware: HardwareInfo) -> tuple[str, str, str]:
    raise NotImplementedError("Implementation belongs to a later packet")


def compute_profile(
    name: str,
    answers: RigAnswers,
    measured: ClipMeasurement | None,
    hardware: HardwareInfo,
    *,
    preset: tuple[str, Loudness | None] | None = None,
    now: datetime,
) -> RigProfile:
    raise NotImplementedError("Implementation belongs to a later packet")


def resolve_options(
    cli_values: Mapping[str, Any],
    profile: RigProfile | None,
    config_defaults: Mapping[str, Any],
) -> ResolvedOptions:
    raise NotImplementedError("Implementation belongs to a later packet")
