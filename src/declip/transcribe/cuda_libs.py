"""Frozen cross-packet stubs. The owning packet supplies the implementation."""

from __future__ import annotations


def preload() -> tuple[bool, str]:
    raise NotImplementedError("Implementation belongs to a later packet")
