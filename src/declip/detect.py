"""Propose cuts from flat transcript words and measured waveform silence."""

from __future__ import annotations

import difflib
import math
import re
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from declip.contracts import (
    CutKind,
    CutProposal,
    DeclipError,
    FillerFile,
    ToolMissing,
    Word,
)


def detect_fillers(
    words: Sequence[Word],
    fillers: FillerFile,
    *,
    min_confidence: float,
    margin_ms: float,
    duration: float,
) -> list[CutProposal]:
    """Apply context and midpoint limits without making review decisions."""
    _nonnegative("margin_ms", margin_ms)
    _nonnegative("duration", duration)
    margin = margin_ms / 1000.0
    # Track positions separately from public word indices for safe merging.
    candidates: list[tuple[int, int, CutProposal]] = []
    i = 0
    while i < len(words):
        word = words[i]
        token = _token(word.text)
        last = i
        label = token
        if not token or word.end <= word.start:
            i += 1
            continue
        pair = f"{token} {_token(words[i + 1].text)}" if i + 1 < len(words) else ""
        if (
            pair in fillers.double
            and words[i + 1].end > words[i + 1].start
            and (_sentence_start(words, i) or _ends(words[i - 1], ","))
            and (_following_pause(words, i + 1) or _ends(words[i + 1], ",.!?"))
        ):
            last = i + 1
            label = pair
        elif token not in fillers.single or not _single_context(words, i, token):
            i += 1
            continue

        start = max(0.0, word.start - margin)
        end = min(duration, words[last].end + margin)
        if i > 0:
            start = max(start, (words[i - 1].end + word.start) / 2.0)
        if last + 1 < len(words):
            end = min(end, (words[last].end + words[last + 1].start) / 2.0)
        confidence = min(w.p for w in words[i : last + 1])
        if start < end:
            candidates.append(
                (
                    i,
                    last,
                    CutProposal(
                        CutKind.FILLER,
                        start,
                        end,
                        label,
                        confidence,
                        (word.i, words[last].i),
                        confidence < min_confidence,
                    ),
                )
            )
        i = last + 1

    merged: list[tuple[int, int, CutProposal]] = []
    for first, last, cut in candidates:
        if merged and first == merged[-1][1] + 1 and cut.start <= merged[-1][2].end:
            prev_first, _, prev = merged[-1]
            confidence = min(prev.confidence, cut.confidence)
            merged[-1] = (
                prev_first,
                last,
                replace(
                    prev,
                    end=max(prev.end, cut.end),
                    label=f"{prev.label}, {cut.label}",
                    confidence=confidence,
                    word_indices=(words[prev_first].i, words[last].i),
                    low_confidence=confidence < min_confidence,
                ),
            )
        else:
            merged.append((first, last, cut))
    return sorted((cut for _, _, cut in merged), key=lambda cut: cut.start)


def _token(text: str) -> str:
    return re.sub(r"[^\w\s']", "", text.strip().lower())


def _ends(word: Word, punctuation: str) -> bool:
    return word.text.rstrip().endswith(tuple(punctuation))


def _gap_over(start: float, end: float, threshold: float) -> bool:
    # Timestamp subtraction must not turn exactly 200/300 ms into a pause.
    return round(start - end, 9) > threshold


def _sentence_start(words: Sequence[Word], i: int) -> bool:
    return (
        i == 0
        or _gap_over(words[i].start, words[i - 1].end, 0.3)
        or _ends(words[i - 1], ".!?")
    )


def _following_pause(words: Sequence[Word], i: int) -> bool:
    return i == len(words) - 1 or _gap_over(words[i + 1].start, words[i].end, 0.2)


def _single_context(words: Sequence[Word], i: int, token: str) -> bool:
    if token == "like":
        return _sentence_start(words, i) or _following_pause(words, i)
    if token in ("so", "well"):
        return _sentence_start(words, i)
    if token in ("right", "actually"):
        return _following_pause(words, i)
    return True


def detect_retakes(
    words: Sequence[Word],
    *,
    min_confidence: float,
    similarity_threshold: float = 0.6,
    window_s: float = 15.0,
) -> list[CutProposal]:
    """Compare sentence token lists and propose removal of the earlier take."""
    _nonnegative("window_s", window_s)
    chunks: list[tuple[Word, ...]] = []
    current: list[Word] = []
    for i, word in enumerate(words):
        if not _token(word.text):
            continue
        current.append(word)
        if (
            _ends(word, ".!?")
            or i == len(words) - 1
            or _gap_over(words[i + 1].start, word.end, 1.0)
        ):
            # Keep the baseline's three-word minimum to avoid repeated fragments.
            if len(current) >= 3:
                chunks.append(tuple(current))
            current = []
    if len(current) >= 3:
        chunks.append(tuple(current))

    tokens = [tuple(_token(word.text) for word in chunk) for chunk in chunks]
    cuts: list[CutProposal] = []
    for i, earlier in enumerate(chunks):
        for j in range(i + 1, len(chunks)):
            later = chunks[j]
            if _gap_over(later[0].start, earlier[-1].end, window_s):
                break
            ratio = difflib.SequenceMatcher(None, tokens[i], tokens[j]).ratio()
            if ratio >= similarity_threshold:
                cuts.append(
                    CutProposal(
                        CutKind.RETAKE,
                        earlier[0].start,
                        earlier[-1].end,
                        f"retake ({ratio:.0%} match)",
                        ratio,
                        (earlier[0].i, earlier[-1].i),
                        ratio < min_confidence,
                    )
                )
                break
    return sorted(cuts, key=lambda cut: cut.start)


def detect_waveform_gaps(
    audio: Path,
    *,
    noise_db: float,
    max_gap_ms: float,
    min_silence_ms: float,
    duration: float,
    audio_index: int | None = None,
) -> list[CutProposal]:
    """Measure the selected audio stream, including leading and EOF silence."""
    _nonnegative("max_gap_ms", max_gap_ms)
    _nonnegative("min_silence_ms", min_silence_ms)
    _nonnegative("duration", duration)
    if not math.isfinite(noise_db):
        raise DeclipError("noise_db must be finite")
    if audio_index is not None and (type(audio_index) is not int or audio_index < 0):
        raise DeclipError("audio_index must be a nonnegative stream index")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-i",
        str(audio),
        "-map",
        f"0:{audio_index}" if audio_index is not None else "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-af",
        f"asetpts=PTS-STARTPTS,silencedetect=noise={noise_db}dB:d={min_silence_ms / 1000.0}",
        "-f",
        "null",
        "-",
    ]
    try:
        process = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ToolMissing(
            "ffmpeg is missing; install ffmpeg to detect waveform gaps"
        ) from exc
    except OSError as exc:
        raise DeclipError(
            f"Could not run ffmpeg for waveform silence detection: {exc}"
        ) from exc
    if process.returncode:
        tail = "\n".join(process.stderr.splitlines()[-40:])
        raise DeclipError(f"Waveform silence detection failed (ffmpeg):\n{tail}")

    silence_start: float | None = None
    silences: list[tuple[float, float]] = []
    for match in re.finditer(
        r"silence_(start|end):\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
        process.stderr,
    ):
        event, timestamp = match.groups()
        time = min(duration, max(0.0, float(timestamp)))
        if event == "start":
            silence_start = time
        elif silence_start is not None:
            silences.append((silence_start, time))
            silence_start = None
    if silence_start is not None:
        silences.append((silence_start, duration))

    preserve = max_gap_ms / 1000.0
    cuts: list[CutProposal] = []
    for start, end in silences:
        cut_start = start if start == 0.0 else start + preserve / 2.0
        cut_end = end if end == duration else end - preserve / 2.0
        if cut_start < cut_end:
            retained = (end - start) - (cut_end - cut_start)
            cuts.append(
                CutProposal(
                    CutKind.GAP,
                    cut_start,
                    cut_end,
                    f"waveform silence ({end - start:.2f}s→{retained:.2f}s at {noise_db:.0f}dB)",
                    1.0,
                    None,
                    False,
                )
            )
    return sorted(cuts, key=lambda cut: cut.start)


def legacy_words(transcript: Mapping[str, Any]) -> list[Word]:
    """Flatten legacy segments, keeping indices contiguous across segments."""
    words: list[Word] = []
    for segment_i, segment in enumerate(transcript.get("segments", [])):
        for word in segment.get("words", []):
            probability = word.get("probability")
            words.append(
                Word(
                    len(words),
                    float(word.get("start", 0.0)),
                    float(word.get("end", 0.0)),
                    word.get("word", "").strip(),
                    1.0 if probability is None else float(probability),
                    segment_i,
                )
            )
    return words


def _nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise DeclipError(f"{name} must be finite and nonnegative")
