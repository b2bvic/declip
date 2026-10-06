"""Small browser previews. Rendering always uses the original source."""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

from declip.contracts import PROXY_SETTINGS_VERSION, DeclipError, MediaInfo
from declip.media import require_tools


def proxy_args(media: MediaInfo) -> list[str]:
    args = []
    if media.has_video:
        if not media.width or not media.height or media.video_index is None:
            raise DeclipError("Preview needs video dimensions and a stream index")
        width, height = media.width, media.height
        if media.rotation in (90, 270):
            width, height = height, width
        limit_w, limit_h = (540, 960) if height > width else (960, 540)
        scale = min(1.0, limit_w / width, limit_h / height)
        out_w = math.floor(width * scale / 2) * 2
        out_h = math.floor(height * scale / 2) * 2
        if min(out_w, out_h) < 2:
            raise DeclipError("Preview dimensions must be at least two pixels")
        args += [
            "-map",
            f"0:{media.video_index}",
            "-vf",
            f"scale={out_w}:{out_h},setsar=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            "1000000",
            "-metadata:s:v:0",
            "rotate=0",
        ]
    else:
        args += ["-vn"]
    if media.audio_index is not None:
        args += ["-map", f"0:{media.audio_index}", "-c:a", "aac", "-b:a", "128000"]
        if media.has_video:
            args += ["-ac", "2"]
    else:
        if not media.has_video:
            raise DeclipError("Preview needs an audio or video stream")
        args += ["-an"]
    return args + ["-map_metadata", "-1", "-movflags", "+faststart"]


def proxy_path(cache_dir: Path, source_sha256: str, media: MediaInfo) -> Path:
    key = hashlib.sha256(
        f"{source_sha256}:{PROXY_SETTINGS_VERSION}".encode("utf-8")
    ).hexdigest()
    return cache_dir / "proxies" / (key + (".mp4" if media.has_video else ".m4a"))


def build_proxy(source: Path, proxy_path: Path, ffmpeg_args: Sequence[str]) -> Path:
    if source.resolve() == proxy_path.resolve():
        raise DeclipError("Preview cannot replace its source")
    if proxy_path.is_file():
        return proxy_path
    ffmpeg, _ = require_tools()
    proxy_path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=".preview-", suffix=proxy_path.suffix, dir=proxy_path.parent
    )
    os.close(fd)
    temporary = Path(name)
    try:
        result = subprocess.run(
            [
                str(ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(source),
                *ffmpeg_args,
                str(temporary),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise DeclipError(
                "Preview failed: " + "\n".join(result.stderr.splitlines()[-40:])
            )
        os.replace(temporary, proxy_path)
    except OSError as exc:
        raise DeclipError(f"Cannot build preview: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return proxy_path
