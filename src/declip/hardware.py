"""Frozen cross-packet stubs. The owning packet supplies the implementation."""

from __future__ import annotations

from declip.contracts import (
    HardwareInfo,
)


def detect() -> HardwareInfo:
    raise NotImplementedError("Implementation belongs to a later packet")
