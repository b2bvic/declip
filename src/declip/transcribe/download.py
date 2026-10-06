"""Frozen cross-packet stubs. The owning packet supplies the implementation."""

from __future__ import annotations


def ensure_model(name: str, *, backend: str) -> str:
    raise NotImplementedError("Implementation belongs to a later packet")
