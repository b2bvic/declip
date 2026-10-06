"""Media metadata, rational frame rates, and full-content source hashes."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import threading
from fractions import Fraction
from pathlib import Path

from declip.contracts import (
    STANDARD_RATES,
    AudioStream,
    ColorTags,
    DeclipError,
    MediaInfo,
    ToolMissing,
)
from declip.fsutil import atomic_write_json
from declip.paths import cache_dir

_HASH_LOCK = threading.Lock()


def require_tools() -> tuple[Path, Path]:
    tools = tuple(shutil.which(name) for name in ("ffmpeg", "ffprobe"))
    missing = [name for name, value in zip(("ffmpeg", "ffprobe"), tools) if not value]
    if missing:
        hint = {
            "darwin": "Install with: brew install ffmpeg",
            "linux": "Install with your package manager, e.g.: sudo apt install ffmpeg",
            "win32": "Install FFmpeg and add its bin directory to PATH",
        }.get(sys.platform, "Install FFmpeg and add ffmpeg and ffprobe to PATH")
        raise ToolMissing(f"Missing {', '.join(missing)} on PATH. {hint}")
    return Path(tools[0]), Path(tools[1])


def _rate(value) -> Fraction | None:
    try:
        rate = Fraction(value)
        return rate if rate > 0 else None
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None


def snap_fps(rate: Fraction) -> Fraction | None:
    parsed = _rate(rate)
    if parsed is None:
        return None
    nearest = min(STANDARD_RATES, key=lambda candidate: abs(candidate - parsed))
    return nearest if abs(parsed - nearest) / nearest <= Fraction(1, 1000) else parsed


def _number(value, *, integer=False):
    try:
        number = float(value)
        if not math.isfinite(number):
            return None
        return int(number) if integer else number
    except (ValueError, TypeError, OverflowError):
        return None


def _positive_int(value):
    number = _number(value, integer=True)
    return number if number is not None and number > 0 else None


def _probe(ffprobe: Path, args: list[str]) -> dict:
    try:
        result = subprocess.run(
            [str(ffprobe), "-v", "error", *args, "-of", "json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
    except OSError as exc:
        raise ToolMissing(
            f"Cannot run ffprobe: {exc}. Install FFmpeg and check PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise DeclipError("ffprobe timed out after 60 seconds") from exc
    if result.returncode:
        raise DeclipError(f"ffprobe failed: {result.stderr[-4000:].strip()}")
    try:
        data = json.loads(result.stdout)
        if not isinstance(data, dict):
            raise ValueError("expected an object")
        return data
    except (ValueError, TypeError) as exc:
        raise DeclipError(f"Invalid ffprobe JSON: {exc}") from exc


def _packet_vfr(packets: list[dict]) -> bool:
    timestamps = sorted(
        time
        for packet in packets
        if (time := _number(packet.get("pts_time"))) is not None
    )
    deltas = [b - a for a, b in zip(timestamps, timestamps[1:])]
    if not deltas:
        return False
    median = statistics.median(deltas)
    if median <= 0:
        return True
    unusual = sum(abs(delta - median) > median * 0.25 for delta in deltas)
    return unusual / len(deltas) > 0.01


def _video_depth(stream: dict) -> int:
    fmt = stream.get("pix_fmt", "")
    if fmt.startswith("p010"):
        return 10
    match = re.search(r"(9|10|12|14|16)(?:le|be)$", fmt)
    if match:
        return int(match.group(1))
    return _positive_int(stream.get("bits_per_raw_sample")) or 8


def _audio_depth(stream: dict) -> int | None:
    raw = _positive_int(stream.get("bits_per_raw_sample"))
    if raw:
        return raw
    return {"u8": 8, "s16": 16, "s32": 32, "s64": 64, "flt": 32, "dbl": 64}.get(
        stream.get("sample_fmt", "").removesuffix("p")
    )


def _rotation(stream: dict) -> int:
    angle = stream.get("tags", {}).get("rotate", 0)
    for item in stream.get("side_data_list", []):
        if "rotation" in item:
            angle = item["rotation"]
            break
    number = _number(angle) or 0
    return (math.floor(number / 90 + 0.5) * 90) % 360


def probe_media(path: Path) -> MediaInfo:
    _, ffprobe = require_tools()
    path = path.resolve()
    if not path.is_file():
        raise DeclipError(f"Media file not found: {path}")
    data = _probe(ffprobe, ["-show_format", "-show_streams", str(path)])
    streams = data.get("streams", [])
    container = data.get("format", {})
    video = next(
        (
            s
            for s in streams
            if s.get("codec_type") == "video"
            and not s.get("disposition", {}).get("attached_pic")
        ),
        None,
    )
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    selected = next(
        (s for s in audio if s.get("disposition", {}).get("default")),
        audio[0] if audio else None,
    )
    duration = _number(container.get("duration"))
    if duration is None:
        duration = max((_number(s.get("duration")) or 0 for s in streams), default=0)
    if duration <= 0:
        raise DeclipError(f"Media has no positive duration: {path}")
    fps, vfr = None, False
    if video is not None:
        avg = _rate(video.get("avg_frame_rate"))
        nominal = _rate(video.get("r_frame_rate"))
        vfr = avg is None and nominal is None
        if avg is not None and nominal is not None:
            vfr = abs(avg - nominal) / avg > Fraction(1, 1000)
        packet_data = _probe(
            ffprobe,
            [
                "-select_streams",
                str(video["index"]),
                "-read_intervals",
                "%+#300",
                "-show_packets",
                "-show_entries",
                "packet=pts_time",
                str(path),
            ],
        )
        vfr = vfr or _packet_vfr(packet_data.get("packets", []))
        if vfr:
            fps = (
                min(STANDARD_RATES, key=lambda r: abs(r - avg))
                if avg
                else Fraction(30000, 1001)
            )
        else:
            fps = snap_fps(avg or nominal)
    video_data = video or {}
    tags = video_data.get("tags", {})
    return MediaInfo(
        duration=duration,
        start_time=_number(container.get("start_time")) or 0.0,
        container=container.get("format_name", ""),
        has_video=video is not None,
        video_index=video_data.get("index"),
        audio_index=selected["index"] if selected else None,
        fps=fps,
        vfr=vfr,
        width=_positive_int(video_data.get("width")),
        height=_positive_int(video_data.get("height")),
        rotation=_rotation(video_data),
        vcodec=video_data.get("codec_name"),
        pix_fmt=video_data.get("pix_fmt"),
        bit_depth=_video_depth(video_data) if video else None,
        video_bitrate=_positive_int(video_data.get("bit_rate")),
        color=ColorTags(
            *(
                video_data.get(key)
                for key in (
                    "color_primaries",
                    "color_transfer",
                    "color_space",
                    "color_range",
                )
            )
        ),
        timecode=tags.get("timecode")
        or container.get("tags", {}).get("timecode")
        or next(
            (
                s.get("tags", {}).get("timecode")
                for s in streams
                if s.get("tags", {}).get("timecode")
            ),
            None,
        ),
        audio_streams=tuple(
            AudioStream(
                s["index"],
                s.get("codec_name"),
                _positive_int(s.get("sample_rate")),
                _positive_int(s.get("channels")),
                s.get("channel_layout"),
                _audio_depth(s),
                s.get("tags", {}).get("language"),
            )
            for s in audio
        ),
        dropped_streams=tuple(
            s["index"]
            for s in streams
            if s.get("codec_type") in {"subtitle", "data", "attachment"}
            or s.get("disposition", {}).get("attached_pic")
        ),
    )


def file_hash(path: Path, *, refresh: bool = False) -> str:
    path = path.resolve()
    with _HASH_LOCK:
        try:
            stat = path.stat()
            store = cache_dir() / "hashes.json"
            try:
                cache = json.loads(store.read_text(encoding="utf-8"))
                if not isinstance(cache, dict):
                    cache = {}
            except (OSError, ValueError):
                cache = {}
            key = json.dumps(
                [str(path), stat.st_size, stat.st_mtime_ns], separators=(",", ":")
            )
            cached = cache.get(key)
            if (
                not refresh
                and isinstance(cached, str)
                and re.fullmatch(r"[0-9a-f]{64}", cached)
            ):
                return cached
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            after = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise DeclipError(f"Source changed while hashing: {path}")
            result = digest.hexdigest()
            cache[key] = result
            try:
                atomic_write_json(store, cache)
            except OSError:
                pass  # A read-only cache must not prevent source verification.
            return result
        except OSError as exc:
            raise DeclipError(f"Cannot hash source {path}: {exc}") from exc
