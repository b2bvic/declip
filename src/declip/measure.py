"""Measure a setup sample with ffprobe and ffmpeg, without numeric libraries."""

from __future__ import annotations

import json
import math
import re
import shutil
import statistics
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from declip.contracts import (
    STANDARD_RATES,
    AudioStream,
    ClipMeasurement,
    DeclipError,
    ToolMissing,
    VideoSummary,
)


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, check=False, timeout=180
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DeclipError(f"sample measurement failed ({argv[0]}): {exc}") from exc
    if result.returncode:
        tail = "\n".join(result.stderr.splitlines()[-20:])
        raise DeclipError(f"sample measurement failed ({argv[0]}): {tail}")
    return result


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number > 0 else None


def _json(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except ValueError:
        pass
    raise DeclipError("sample measurement failed: invalid ffprobe/filter JSON")


def _rate(value: Any) -> Fraction | None:
    try:
        rate = Fraction(str(value))
        return rate if rate > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def _video_summary(ffprobe: str, path: Path, stream: dict[str, Any]) -> VideoSummary:
    average = _rate(stream.get("avg_frame_rate"))
    base = _rate(stream.get("r_frame_rate"))
    rate = average or base
    vfr = rate is None or bool(
        average and base and abs(base - average) / average > Fraction(1, 1000)
    )
    packets = _json(
        _run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                str(stream["index"]),
                "-read_intervals",
                "%+#300",
                "-show_entries",
                "packet=pts_time",
                "-of",
                "json",
                str(path),
            ]
        ).stdout
    ).get("packets", [])
    times = sorted(
        value
        for packet in packets
        if (value := _number(packet.get("pts_time"))) is not None
    )
    deltas = [right - left for left, right in zip(times, times[1:])]
    if deltas:
        median = statistics.median(deltas)
        if (
            median <= 0
            or sum(abs(d - median) > median * 0.25 for d in deltas) / len(deltas) > 0.01
        ):
            vfr = True
    if vfr:
        snapped = (
            min(STANDARD_RATES, key=lambda r: abs(r - average))
            if average
            else Fraction(30000, 1001)
        )
    elif rate is not None:
        nearest = min(STANDARD_RATES, key=lambda r: abs(r - rate))
        snapped = (
            nearest if abs(nearest - rate) / nearest <= Fraction(1, 1000) else rate
        )
    else:
        snapped = None
    pix_fmt = stream.get("pix_fmt")
    # Pixel-format precision wins over inconsistent raw-sample metadata.
    match = re.search(
        r"(?:p0?(10|12|16)|(?:p|gray|gbrp|rgb|bgr)(9|10|12|14|16))(?:le|be)?$",
        pix_fmt or "",
    )
    depth = (
        int(next(group for group in match.groups() if group))
        if match
        else _integer(stream.get("bits_per_raw_sample")) or 8
    )
    return VideoSummary(
        stream.get("codec_name"),
        pix_fmt,
        depth,
        stream.get("color_transfer"),
        f"{snapped.numerator}/{snapped.denominator}" if snapped else None,
        vfr,
    )


def _audio_stream(stream: dict[str, Any]) -> AudioStream:
    sample_format = (stream.get("sample_fmt") or "").removesuffix("p")
    depth = _integer(stream.get("bits_per_raw_sample")) or {
        "u8": 8,
        "s16": 16,
        "s32": 32,
        "s64": 64,
        "flt": 32,
        "dbl": 64,
    }.get(sample_format)
    return AudioStream(
        int(stream["index"]),
        stream.get("codec_name"),
        _integer(stream.get("sample_rate")),
        _integer(stream.get("channels")),
        stream.get("channel_layout"),
        depth,
        stream.get("tags", {}).get("language"),
    )


def _audio_measurements(
    ffmpeg: str,
    path: Path,
    audio: AudioStream,
    start: float,
    duration: float,
) -> tuple[float | None, float | None, float | None, float | None, bool, int]:
    prefix = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-ss",
        str(start),
        "-i",
        str(path),
        "-t",
        str(duration),
        "-map",
        f"0:{audio.index}",
        "-vn",
        "-sn",
        "-dn",
    ]
    # Trim in the filter graph too, so statistics exclude frames beyond -t.
    trim = f"atrim=duration={duration},asetpts=PTS-STARTPTS,"
    stats = _run(
        prefix
        + [
            "-af",
            trim
            + "astats=measure_perchannel=none:measure_overall=Peak_level+Peak_count,loudnorm=print_format=json",
            "-f",
            "null",
            "-",
        ]
    ).stderr
    reports = re.findall(r'\{\s*"input_i".*?\}', stats, re.DOTALL)
    if not reports:
        raise DeclipError("sample measurement failed: loudnorm returned no measurement")
    loudness = _json(reports[-1])
    peak_levels = re.findall(r"\] Peak level dB:\s*(\S+)", stats)
    peak_counts = re.findall(r"\] Peak count:\s*(\S+)", stats)
    peak_level = _number(peak_levels[-1]) if peak_levels else None
    count = _number(peak_counts[-1]) if peak_counts else None
    if not peak_levels or count is None:
        raise DeclipError(
            "sample measurement failed: astats returned no peak measurement"
        )
    peak_count = int(count)
    # Use complete 100 ms mono blocks only. Never pad a partial block with zeros.
    samples = math.floor(duration * 10 + 1e-8) * 4800
    blocks = _run(
        prefix
        + [
            "-af",
            trim + "aformat=channel_layouts=mono,aresample=48000,"
            f"atrim=end_sample={samples},asetnsamples=n=4800:p=0,"
            "astats=metadata=1:reset=1:measure_perchannel=none:measure_overall=RMS_level,"
            "ametadata=print:key=lavfi.astats.Overall.RMS_level",
            "-f",
            "null",
            "-",
        ]
    ).stderr
    finite = sorted(
        number
        for value in re.findall(r"lavfi\.astats\.Overall\.RMS_level=(\S+)", blocks)
        if (number := _number(value)) is not None
    )
    floor = finite[math.ceil(len(finite) * 0.1) - 1] if len(finite) >= 10 else None
    return (
        floor,
        _number(loudness.get("input_i")),
        _number(loudness.get("input_tp")),
        _number(loudness.get("input_lra")),
        peak_level is not None and peak_level >= -0.1 and peak_count >= 3,
        peak_count,
    )


def measure_clip(path: Path, *, seconds: float = 60.0) -> ClipMeasurement:
    """Measure the centered window, keeping source audio and video metadata."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise DeclipError("measurement seconds must be finite and greater than zero")
    ffprobe, ffmpeg = shutil.which("ffprobe"), shutil.which("ffmpeg")
    if not ffprobe or not ffmpeg:
        raise ToolMissing(
            "sample measurement requires ffmpeg and ffprobe; install ffmpeg with brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
        )
    path = Path(path).resolve()
    data = _json(
        _run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(path),
            ]
        ).stdout
    )
    streams = data.get("streams", [])
    duration = _number(data.get("format", {}).get("duration"))
    if duration is None:
        lengths = [_number(stream.get("duration")) for stream in streams]
        duration = max(
            (length for length in lengths if length is not None), default=None
        )
    if duration is None:
        raise DeclipError("sample measurement failed: duration is unavailable")
    if duration < 10:
        raise DeclipError("sample too short (minimum 10 s)")
    window = min(duration, seconds)
    start = max(0.0, (duration - window) / 2)
    audio_data = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"), None
    )
    video_data = next(
        (
            stream
            for stream in streams
            if stream.get("codec_type") == "video"
            and not stream.get("disposition", {}).get("attached_pic")
        ),
        None,
    )
    audio = _audio_stream(audio_data) if audio_data else None
    video = _video_summary(ffprobe, path, video_data) if video_data else None
    values = (
        _audio_measurements(ffmpeg, path, audio, start, window)
        if audio
        else (None, None, None, None, False, 0)
    )
    return ClipMeasurement(
        path.name, start, window, audio is not None, *values, audio, video
    )
