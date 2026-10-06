"""Frozen cross-packet stubs. The owning packet supplies the implementation."""

from __future__ import annotations

from declip.contracts import (
    Enhancer,
)


def select_enhancer(name: str = "auto") -> Enhancer:
    raise NotImplementedError("Implementation belongs to a later packet")


def list_enhancers() -> list[tuple[str, bool, str]]:
    raise NotImplementedError("Implementation belongs to a later packet")
