"""Neutral consoles. Diagnostics always use stderr; JSON uses stdout."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.theme import Theme

THEME = Theme(
    {
        "accent": "cyan",
        "done": "green",
        "warn": "yellow",
        "error": "red",
        "muted": "dim",
        "stat": "bold cyan",
    }
)
MARK, DONE = "◎", "●"
IS_TTY = sys.stderr.isatty()
console = Console(
    stderr=True, highlight=False, theme=THEME, no_color="NO_COLOR" in os.environ
)
stdout_console = Console(
    highlight=False, theme=THEME, no_color="NO_COLOR" in os.environ
)


def echo(message=None, **kwargs):
    kwargs.setdefault("err", True)
    click.echo(message, **kwargs)


class Spinner:
    def __init__(self, message: str):
        self.message = message
        self._progress = None

    def __enter__(self):
        if console.is_terminal:
            self._progress = Progress(
                SpinnerColumn(style="accent"),
                TextColumn("{task.description}"),
                console=console,
                transient=True,
            )
            self._progress.__enter__()
            self._task = self._progress.add_task(self.message, total=1)
        else:
            echo(f"  {self.message}")
        return self

    def __exit__(self, *args):
        if self._progress:
            self._progress.__exit__(*args)

    def update(self, message: str):
        self.message = message
        if self._progress:
            self._progress.update(self._task, description=message)


def progress_tick(i: int, n: int, label: str, milestones: set | None = None):
    pct = int((i + 1) / n * 100) if n else 100
    if milestones is not None and pct in milestones:
        milestones.discard(pct)
        echo(f"  {pct}%: {label} {i + 1}/{n}")


def ui_done(message: str, detail: str = ""):
    echo(f"  {DONE} {message}" + (f" ({detail})" if detail else ""))


def ui_warn(message: str):
    echo(f"Warning: {message}")


def ui_error(message: str):
    echo(f"Error: {message}")


def ui_file_info(path: Path, info: dict, duration: float):
    echo(f"  {path.name}: {duration:.1f}s")


def ui_result(
    path: Path,
    old_duration: float,
    new_duration: float,
    n_edits: float,
    time_removed: float,
    preset: str,
    enhanced: bool,
):
    echo(
        f"  {path.name}: {old_duration:.1f}s to {new_duration:.1f}s ({int(n_edits)} edits, {time_removed:.1f}s removed)"
    )
    echo(f"  Preset: {preset}; enhanced: {enhanced}")
