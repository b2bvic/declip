"""Frozen cross-packet stubs. The owning packet supplies the implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from declip.contracts import (
    ReviewResult,
)


def serve(
    edit_list_path: Path,
    proxy_path: Path,
    *,
    open_browser: bool = True,
    port: int = 0,
    on_ready: Callable[[str], None] | None = None,
) -> ReviewResult:
    raise NotImplementedError("Implementation belongs to a later packet")
