"""Source-preserving encoding, measured capabilities, and output verification.

Only contracts cross packet boundaries. Cache I/O and ffprobe use small private
helpers so this module does not depend on the concurrent media packet.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from declip.contracts import (
    AudioTarget,
    Capabilities,
    ColorTags,
    Encoder,
    EncoderUnavailable,
    MediaInfo,
    RenderError,
    ToolMissing,
    VideoTarget,
)

# Quality values are data, shared by probes and the final argument builders.
_CRF = {"match": (18, 20), "high": (16, 18), "small": (22, 24)}
_CQ = {"match": (19, 21), "high": (17, 19), "small": (23, 25)}
_VIDEO_ENCODERS = {
    "videotoolbox": {"h264": "h264_videotoolbox", "hevc": "hevc_videotoolbox"},
    "nvenc": {"h264": "h264_nvenc", "hevc": "hevc_nvenc"},
    "software": {"h264": "libx264", "hevc": "libx265"},
}
_COLOR_FLAGS = (
    ("primaries", "-color_primaries"),
    ("trc", "-color_trc"),
    ("matrix", "-colorspace"),
    ("range", "-color_range"),
)


def _cache_root() -> Path:
    override = os.environ.get("DECLIP_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform in {"darwin", "linux"}:
        return Path.home() / ".cache" / "declip"
    # Windows is deferred; keep the path portable without another package import.
    return (
        Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        / "declip"
        / "Cache"
    )


def _read_cache(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        # A read-only cache must not prevent an encode or verification.
        pass
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        hint = (
            "brew install ffmpeg"
            if sys.platform == "darwin"
            else "install ffmpeg with your package manager"
        )
        raise ToolMissing(f"{name} is missing; {hint}")
    return path


def _run(
    argv: list[str], *, timeout: float = 30
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _video_stream(path: Path) -> dict[str, Any]:
    result = _run(
        [_tool("ffprobe"), "-v", "error", "-show_streams", "-of", "json", str(path)]
    )
    if result is None or result.returncode != 0:
        tail = (
            result.stderr[-2000:]
            if result is not None
            else "probe could not run or timed out"
        )
        raise RenderError(f"output guard: cannot probe {path}: {tail}")
    try:
        streams = json.loads(result.stdout)["streams"]
        return next(
            s
            for s in streams
            if s.get("codec_type") == "video"
            and not s.get("disposition", {}).get("attached_pic", 0)
        )
    except (ValueError, KeyError, TypeError, AttributeError, StopIteration) as exc:
        raise RenderError(f"output guard: no readable video stream in {path}") from exc


def _pixel_depth(pix_fmt: str | None) -> int | None:
    if not pix_fmt:
        return None
    if pix_fmt.startswith(("p010", "p210", "p410")):
        return 10
    match = re.search(r"(?:p|gray|gbrap?|yuva?\d*p)(9|10|12|14|16)(?:le|be)$", pix_fmt)
    if match:
        return int(match.group(1))
    if pix_fmt in {
        "yuv420p",
        "yuv422p",
        "yuv444p",
        "yuvj420p",
        "yuvj422p",
        "yuvj444p",
        "nv12",
        "nv21",
        "rgb24",
        "bgr24",
        "rgba",
        "bgra",
        "gray",
    }:
        return 8
    return None


def _verify_stream(stream: dict[str, Any], target: VideoTarget) -> None:
    codec = stream.get("codec_name")
    if codec != target.codec:
        raise RenderError(f"output guard: expected codec {target.codec}, got {codec!r}")
    pix_fmt = stream.get("pix_fmt")
    depth = _pixel_depth(pix_fmt)
    if depth != target.bit_depth:
        raise RenderError(
            f"output guard: expected {target.bit_depth}-bit pixel format, got {pix_fmt!r}"
        )
    if (
        target.bit_depth == 10
        and target.codec == "hevc"
        and stream.get("profile") != "Main 10"
    ):
        raise RenderError(
            f"output guard: expected HEVC profile Main 10, got {stream.get('profile')!r}"
        )


def check_output(path: Path, target: VideoTarget) -> None:
    """Refuse a codec, pixel-depth, or Main 10 mismatch after encoding."""
    _verify_stream(_video_stream(path), target)


def video_target_for(
    media: MediaInfo, *, codec: str, quality: str, allow_8bit: bool, container: str
) -> VideoTarget:
    if codec not in {"match", "h264", "hevc"} or quality not in _CRF:
        raise EncoderUnavailable(f"invalid codec or quality: {codec}, {quality}")
    if container not in {"mp4", "mov"}:
        raise EncoderUnavailable(f"unsupported video container: {container}")
    if (
        not media.has_video
        or not media.width
        or not media.height
        or not media.fps
        or media.fps <= 0
    ):
        raise EncoderUnavailable(
            "video target requires video dimensions and a valid frame rate"
        )
    depth = max(media.bit_depth or 8, _pixel_depth(media.pix_fmt) or 8)
    needs_10bit = depth >= 10 or media.vcodec == "prores"
    resolved = codec
    if codec == "match":
        resolved = "hevc" if needs_10bit or media.vcodec == "hevc" else "h264"
    if resolved == "h264" and needs_10bit and not allow_8bit:
        raise EncoderUnavailable(
            "H.264 would reduce this source to 8-bit; use HEVC or --allow-8bit"
        )
    bit_depth = 10 if resolved == "hevc" and needs_10bit else 8
    return VideoTarget(
        resolved,
        bit_depth,
        "yuv420p10le" if bit_depth == 10 else "yuv420p",
        media.color,
        quality,
        media.video_bitrate,
        media.width,
        media.height,
        media.fps,
        container,
    )


def audio_target_for(
    media: MediaInfo, *, bitrate: int, container: str, pcm: bool = False
) -> AudioTarget:
    stream = next(
        (s for s in media.audio_streams if s.index == media.audio_index), None
    )
    if stream is None or not stream.sample_rate or not stream.channels:
        raise EncoderUnavailable(
            "audio target requires a selected audio stream with rate and channels"
        )
    is_pcm = pcm or container == "wav"
    if container not in {"mp4", "mov", "m4a", "wav"} or (is_pcm and container != "wav"):
        raise EncoderUnavailable(f"unsupported audio container: {container}")
    if not is_pcm and bitrate <= 0:
        raise EncoderUnavailable("AAC bitrate must be positive")
    return AudioTarget(
        "pcm_s24le" if is_pcm else "aac",
        None if is_pcm else bitrate,
        stream.sample_rate,
        stream.channels,
        container,
    )


def _bitrate(target: VideoTarget) -> int:
    small_frame = target.width * target.height <= 1920 * 1080
    floor, ceiling, default = (
        (8_000_000, 50_000_000, 20_000_000)
        if small_frame
        else (35_000_000, 150_000_000, 80_000_000)
    )
    if target.quality == "high":
        return ceiling
    if target.quality == "small":
        return floor
    rate = (
        target.source_bitrate
        if target.source_bitrate and target.source_bitrate > 0
        else default
    )
    return max(floor, min(ceiling, rate))


@dataclass(frozen=True)
class _Encoder:
    name: str
    aac_codec: str = "aac"

    def supports(self, target: VideoTarget, caps: Capabilities) -> bool:
        if target.codec not in {"h264", "hevc"} or target.bit_depth not in {8, 10}:
            return False
        if target.codec == "h264" and target.bit_depth != 8:
            return False
        if target.quality not in _CRF or target.container not in {"mp4", "mov"}:
            return False
        if _pixel_depth(target.pix_fmt) != target.bit_depth:
            return False
        encoder = _VIDEO_ENCODERS[self.name][target.codec]
        if encoder not in caps.encoders:
            return False
        if self.name == "software":
            return True
        if self.name == "videotoolbox" and sys.platform != "darwin":
            return False
        # HEVC 8-bit uses the same encoder validated by the required Main 10 probe.
        probe_depth = 10 if target.codec == "hevc" else 8
        return f"{encoder}:{probe_depth}" in caps.hw_encode_ok

    def video_args(self, target: VideoTarget) -> list[str]:
        index = 0 if target.codec == "h264" else 1
        encoder = _VIDEO_ENCODERS[self.name][target.codec]
        args = ["-c:v", encoder]
        if self.name == "videotoolbox":
            if target.bit_depth == 10:
                args += ["-profile:v", "main10"]
            args += [
                "-b:v",
                str(_bitrate(target)),
                "-pix_fmt",
                "p010le" if target.bit_depth == 10 else "yuv420p",
            ]
        elif self.name == "nvenc":
            if target.bit_depth == 10:
                args += ["-profile:v", "main10"]
            args += [
                "-preset",
                "p5",
                "-rc",
                "vbr",
                "-cq",
                str(_CQ[target.quality][index]),
                "-b:v",
                "0",
                "-pix_fmt",
                "p010le" if target.bit_depth == 10 else "yuv420p",
            ]
        else:
            args += [
                "-crf",
                str(_CRF[target.quality][index]),
                "-preset",
                "medium",
                "-pix_fmt",
                "yuv420p10le" if target.bit_depth == 10 else "yuv420p",
            ]
        if target.codec == "hevc" and target.container in {"mp4", "mov"}:
            args += ["-tag:v", "hvc1"]
        for field, flag in _COLOR_FLAGS:
            value = getattr(target.color, field)
            if value is not None:
                args += [flag, value]
        return args

    def audio_args(self, target: AudioTarget) -> list[str]:
        if target.codec == "pcm_s24le":
            args = ["-c:a", "pcm_s24le"]
        elif target.codec == "aac":
            args = ["-c:a", self.aac_codec, "-b:a", str(target.bitrate)]
        else:
            raise EncoderUnavailable(f"unsupported audio codec: {target.codec}")
        return args + ["-ar", str(target.sample_rate), "-ac", str(target.channels)]


def select_encoder(
    target: VideoTarget, caps: Capabilities, *, prefer: str = "auto"
) -> Encoder:
    if prefer not in {"auto", "software"}:
        raise EncoderUnavailable(f"unknown encoder preference: {prefer}")
    names = (
        ("software",) if prefer == "software" else ("videotoolbox", "nvenc", "software")
    )
    for name in names:
        audio_codec = (
            "aac_at" if name == "videotoolbox" and "aac_at" in caps.encoders else "aac"
        )
        candidate = _Encoder(name, audio_codec)
        if candidate.supports(target, caps):
            return candidate
    raise EncoderUnavailable(
        f"no encoder supports {target.codec} {target.bit_depth}-bit ({prefer}); install an ffmpeg build with the required encoder"
    )


def _driver_version() -> str | None:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    result = _run(
        [binary, "--query-gpu=driver_version", "--format=csv,noheader"], timeout=5
    )
    if result is None or result.returncode != 0:
        return None
    return ",".join(sorted(set(result.stdout.split()))) or None


def probe_capabilities(*, refresh: bool = False) -> Capabilities:
    ffmpeg = _tool("ffmpeg")
    version_result = _run([ffmpeg, "-version"], timeout=5)
    if (
        version_result is None
        or version_result.returncode
        or not version_result.stdout.strip()
    ):
        raise EncoderUnavailable("cannot read ffmpeg version")
    version = version_result.stdout.splitlines()[0]
    driver = _driver_version()
    path = _cache_root() / "capabilities.json"
    key = hashlib.sha256(json.dumps([version, driver]).encode()).hexdigest()
    cached = _read_cache(path)
    entry = cached.get(key)
    if not refresh and isinstance(entry, dict):
        try:
            listed, passed = entry["encoders"], entry["hw_encode_ok"]
            if not all(
                isinstance(v, list) and all(isinstance(s, str) for s in v)
                for v in (listed, passed)
            ):
                raise ValueError("invalid cache")
            return Capabilities(version, frozenset(listed), frozenset(passed), driver)
        except (KeyError, ValueError, TypeError):
            pass
    result = _run([ffmpeg, "-hide_banner", "-encoders"], timeout=5)
    if result is None or result.returncode != 0:
        raise EncoderUnavailable("cannot list ffmpeg encoders")
    listed = frozenset(
        match.group(1)
        for line in result.stdout.splitlines()
        if (match := re.match(r"\s*[VAS][A-Z.]{5}\s+(\w+)\s", line))
    )
    passed: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="declip-encode-probe-") as directory:
        for name in ("videotoolbox", "nvenc"):
            for codec, depth in (("h264", 8), ("hevc", 10)):
                encoder = _VIDEO_ENCODERS[name][codec]
                if encoder not in listed:
                    continue
                target = VideoTarget(
                    codec,
                    depth,
                    "yuv420p10le" if depth == 10 else "yuv420p",
                    ColorTags(None, None, None, None),
                    "match",
                    None,
                    128,
                    128,
                    Fraction(30),
                    "mp4",
                )
                output = Path(directory) / f"{encoder}.mp4"
                argv = [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=128x128:rate=30",
                    "-frames:v",
                    "1",
                    "-an",
                    *_Encoder(name).video_args(target),
                    str(output),
                ]
                encoded = _run(argv)
                if encoded is None or encoded.returncode != 0:
                    continue
                try:
                    check_output(output, target)
                except (RenderError, ToolMissing):
                    continue
                passed.add(f"{encoder}:{depth}")
    cached[key] = {"encoders": sorted(listed), "hw_encode_ok": sorted(passed)}
    _write_cache(path, cached)
    return Capabilities(version, listed, frozenset(passed), driver)


def hwaccel_args(source: Path, media: MediaInfo, caps: Capabilities) -> list[str]:
    """Use hardware decoding only after this source passes a software-filter probe."""
    if not media.has_video or media.video_index is None:
        return []
    candidates = []
    if sys.platform == "darwin" and any(
        "videotoolbox:" in entry for entry in caps.hw_encode_ok
    ):
        candidates.append("videotoolbox")
    if any("nvenc:" in entry for entry in caps.hw_encode_ok):
        candidates.append("cuda")
    if not candidates:
        return []
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return []
    path = _cache_root() / "decoders.json"
    cached = _read_cache(path)
    for name in candidates:
        key = hashlib.sha256(
            json.dumps(
                [
                    digest.hexdigest(),
                    caps.ffmpeg_version,
                    caps.gpu_driver,
                    media.video_index,
                    name,
                ]
            ).encode()
        ).hexdigest()
        success = cached.get(key)
        if not isinstance(success, bool):
            result = _run(
                [
                    _tool("ffmpeg"),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-hwaccel",
                    name,
                    "-i",
                    str(source),
                    "-map",
                    f"0:{media.video_index}",
                    "-t",
                    "1",
                    "-vf",
                    "scale=128:128,format=yuv420p",
                    "-an",
                    "-sn",
                    "-dn",
                    "-progress",
                    "pipe:1",
                    "-nostats",
                    "-f",
                    "null",
                    "-",
                ]
            )
            # FFmpeg may return success with no decoded frames for an empty stream.
            success = (
                result is not None
                and result.returncode == 0
                and any(
                    int(value) > 0
                    for value in re.findall(r"^frame=(\d+)$", result.stdout, re.M)
                )
            )
            cached[key] = success
            _write_cache(path, cached)
        if success:
            return ["-hwaccel", name]
    return []
