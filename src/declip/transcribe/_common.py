"""PCM chunking and timestamp normalization shared by real backends."""

from __future__ import annotations

import tempfile
import wave
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable, Mapping, Any

from declip.contracts import (
    Segment,
    Transcript,
    TranscribeOptions,
    TranscriberUnavailable,
    Word,
)

# Updated only from the P3 probe decision rule. Explicit modes remain available.
AUTO_PROMPT_MODES = {"mlx": "initial", "faster": "initial"}


def prompt_mode(backend: str, opts: TranscribeOptions) -> str:
    mode = (
        AUTO_PROMPT_MODES[backend] if opts.prompt_mode == "auto" else opts.prompt_mode
    )
    if mode not in {"initial", "chunked", "hotwords"}:
        raise TranscriberUnavailable(f"Unknown prompt mode: {mode}")
    if backend == "mlx" and mode == "hotwords":
        raise TranscriberUnavailable("mlx does not support hotwords; use chunked")
    return mode


def normalize(
    rows: Iterable[Mapping[str, Any]],
    *,
    backend: str,
    opts: TranscribeOptions,
    language: str,
) -> Transcript:
    words, segments = [], []
    for row in rows:
        si = len(segments)
        indices = []
        for raw in row.get("words") or ():
            wi = len(words)
            probability = raw.get("probability")
            words.append(
                Word(
                    wi,
                    float(raw["start"]),
                    float(raw["end"]),
                    str(raw["word"]).strip(),
                    1.0 if probability is None else float(probability),
                    si,
                )
            )
            indices.append(wi)
        segments.append(
            Segment(
                si,
                float(row["start"]),
                float(row["end"]),
                str(row["text"]).strip(),
                tuple(indices),
            )
        )
    return Transcript(
        backend,
        opts.model,
        language,
        opts.initial_prompt,
        tuple(words),
        tuple(segments),
    )


def copy_window(source: Path, target: Path, start: float, seconds: float) -> None:
    with wave.open(str(source), "rb") as src, wave.open(str(target), "wb") as dst:
        dst.setparams(src.getparams())
        src.setpos(min(src.getnframes(), round(start * src.getframerate())))
        dst.writeframes(src.readframes(round(seconds * src.getframerate())))


def chunked(
    wav: Path,
    opts: TranscribeOptions,
    run: Callable[[Path, TranscribeOptions], Transcript],
) -> Transcript:
    with wave.open(str(wav), "rb") as src:
        duration = src.getnframes() / src.getframerate()
    if duration <= 0:
        raise TranscriberUnavailable("Cannot transcribe an empty WAV")
    words, segments = [], []
    language = opts.language
    with tempfile.TemporaryDirectory(prefix="declip-chunks-", dir=wav.parent) as tmp:
        for index in range(int((duration + 29.999999) // 30)):
            offset = index * 30.0
            path = Path(tmp) / f"{index}.wav"
            copy_window(wav, path, offset, 30)
            result = run(path, replace(opts, prompt_mode="initial", language=language))
            language = result.language
            word_base, segment_base = len(words), len(segments)
            words.extend(
                replace(
                    w,
                    i=w.i + word_base,
                    start=w.start + offset,
                    end=w.end + offset,
                    segment=w.segment + segment_base,
                )
                for w in result.words
            )
            segments.extend(
                replace(
                    s,
                    i=s.i + segment_base,
                    start=s.start + offset,
                    end=s.end + offset,
                    word_indices=tuple(i + word_base for i in s.word_indices),
                )
                for s in result.segments
            )
    return Transcript(
        result.backend,
        opts.model,
        language or "und",
        opts.initial_prompt,
        tuple(words),
        tuple(segments),
    )
