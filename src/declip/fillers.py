"""Filler files and their optional language prompt."""

from __future__ import annotations

from dataclasses import replace
from importlib.resources import files
from pathlib import Path

from .contracts import DeclipError, FillerFile

SHIPPED_LANGUAGES: tuple[str, ...] = ("en",)


def parse_filler_file(text: str, *, language: str) -> FillerFile:
    lines = text.splitlines()
    prompt = None
    if lines and lines[0].startswith("# prompt:"):
        prompt = lines[0].partition(":")[2].strip() or None
    single, double = set(), set()
    for line in lines:
        entry = " ".join(line.strip().lower().split())
        if not entry or entry.startswith("#"):
            continue
        count = len(entry.split())
        if count not in (1, 2):
            raise DeclipError(
                f"Filler for {language} must contain one or two words: {entry}"
            )
        (single if count == 1 else double).add(entry)
    return FillerFile(language, prompt, frozenset(single), frozenset(double))


def load_fillers(language: str, *, config_dir: Path) -> FillerFile | None:
    # Language codes must never escape the fillers directory.
    if not language or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in language
    ):
        return None
    custom = config_dir / "fillers" / f"{language}.txt"
    if custom.is_file():
        return parse_filler_file(custom.read_text(encoding="utf-8"), language=language)
    if language not in SHIPPED_LANGUAGES:
        return None
    shipped = parse_filler_file(
        files("declip")
        .joinpath("data", "fillers", f"{language}.txt")
        .read_text(encoding="utf-8"),
        language=language,
    )
    legacy = config_dir / "fillers.txt"
    if language == "en" and legacy.is_file():
        user = parse_filler_file(legacy.read_text(encoding="utf-8"), language=language)
        return replace(user, prompt=shipped.prompt)
    return shipped
