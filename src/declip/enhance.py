"""PCM enhancement backends. No model library or automatic download is required."""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .config import load_config
from .contracts import Enhancer, EnhancerUnavailable, RenderError

DEEPFILTER_RELEASE_URL = "https://github.com/Rikorose/DeepFilterNet/releases/tag/v0.5.6"
# Verified release assets. Availability never queries the network.
DEEPFILTER_ASSETS = {
    ("darwin", "arm64"): "deep-filter-0.5.6-aarch64-apple-darwin",
    ("darwin", "x86_64"): "deep-filter-0.5.6-x86_64-apple-darwin",
    ("linux", "aarch64"): "deep-filter-0.5.6-aarch64-unknown-linux-gnu",
    ("linux", "arm64"): "deep-filter-0.5.6-aarch64-unknown-linux-gnu",
    ("linux", "armv7l"): "deep-filter-0.5.6-armv7-unknown-linux-gnueabihf",
    ("linux", "x86_64"): "deep-filter-0.5.6-x86_64-unknown-linux-musl",
}


def _run(argv: list[str], *, probe: bool = False) -> str:
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, check=False,
            timeout=10 if probe else None,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RenderError(f"{Path(argv[0]).name}: {exc}") from exc
    if result.returncode:
        tail = "\n".join(result.stderr.splitlines()[-40:])
        raise RenderError(f"{Path(argv[0]).name} exited {result.returncode}: {tail}")
    return result.stdout


def _strength(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RenderError("enhancer strength must be a finite number between 0 and 1")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise RenderError("enhancer strength must be a finite number between 0 and 1")
    return float(value)


def _check_paths(wav_in: Path, wav_out: Path) -> None:
    if wav_in.resolve() == wav_out.resolve() or (
        wav_out.exists() and os.path.samefile(wav_in, wav_out)
    ):
        raise RenderError("enhancer output must differ from its input")


@dataclass(frozen=True)
class _Audio:
    rate: int
    channels: int
    layout: str | None
    samples: int


def _audio(path: Path) -> _Audio:
    raw = _run([
        "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_streams",
        "-of", "json", str(path.resolve()),
    ], probe=True)
    try:
        stream = json.loads(raw)["streams"][0]
        rate, channels = int(stream["sample_rate"]), int(stream["channels"])
        # WAV duration_ts is exact; decimal durations can lose a sample.
        duration = Fraction(str(stream["duration_ts"])) * Fraction(stream["time_base"])
        samples = round(duration * rate)
        if rate <= 0 or channels <= 0 or samples <= 0:
            raise ValueError("empty or invalid audio stream")
        return _Audio(rate, channels, stream.get("channel_layout"), samples)
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise RenderError(f"Cannot read PCM WAV audio in {path}: {exc}") from exc


def _output_args(audio: _Audio) -> list[str]:
    args = ["-c:a", "pcm_s24le", "-ar", str(audio.rate), "-ac", str(audio.channels)]
    if audio.layout:
        args += ["-channel_layout", audio.layout]
    return args


def _check_audio(path: Path, expected: _Audio) -> None:
    actual = _audio(path)
    if (actual.rate, actual.channels, actual.samples) != (
        expected.rate, expected.channels, expected.samples
    ) or (expected.layout and actual.layout != expected.layout):
        raise RenderError(f"Enhancer changed sample rate, channels, layout, or duration: {actual}")


class NoneEnhancer:
    name = "none"

    def available(self) -> tuple[bool, str]:
        return True, "exact PCM copy"

    def enhance(self, wav_in: Path, wav_out: Path, strength: float) -> None:
        _strength(strength)
        _check_paths(wav_in, wav_out)
        try:
            shutil.copyfile(wav_in, wav_out)
        except OSError as exc:
            raise RenderError(f"Cannot copy PCM audio: {exc}") from exc


class AfftdnEnhancer:
    name = "afftdn"

    def available(self) -> tuple[bool, str]:
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            return False, "unavailable: ffmpeg and ffprobe are required"
        try:
            filters = _run(["ffmpeg", "-hide_banner", "-filters"], probe=True)
        except RenderError as exc:
            return False, f"unavailable: {exc}"
        if not any(len(row.split()) > 1 and row.split()[1] == "afftdn"
                   for row in filters.splitlines()):
            return False, "unavailable: ffmpeg has no afftdn filter"
        return True, "ffmpeg FFT denoiser"

    def enhance(self, wav_in: Path, wav_out: Path, strength: float) -> None:
        strength = _strength(strength)
        _check_paths(wav_in, wav_out)
        _require(self)
        audio = _audio(wav_in)
        try:
            with tempfile.TemporaryDirectory(prefix="declip-afftdn-", dir=wav_out.parent) as raw:
                output = Path(raw) / "enhanced.wav"
                _run([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(wav_in.resolve()), "-map", "0:a:0", "-af",
                    f"afftdn=nr={6 + 18 * strength:g}:nf=-50:tn=1",
                    *_output_args(audio), str(output),
                ])
                _check_audio(output, audio)
                os.replace(output, wav_out)
        except OSError as exc:
            raise RenderError(f"afftdn output failed: {exc}") from exc


class DeepFilterEnhancer:
    name = "deepfilter"

    def _binary(self) -> str | None:
        configured = load_config().get("deepfilter_path")
        if configured is not None:
            if not isinstance(configured, str) or not configured.strip():
                return None
            return shutil.which(str(Path(configured).expanduser()))
        return shutil.which("deep-filter")

    def available(self) -> tuple[bool, str]:
        asset = DEEPFILTER_ASSETS.get((sys.platform, platform.machine().lower()))
        if not asset:
            return False, f"unavailable: no supported release asset for this platform; {DEEPFILTER_RELEASE_URL}"
        binary = self._binary()
        if not binary:
            return False, f"unavailable: install {asset} as deep-filter or set deepfilter_path; {DEEPFILTER_RELEASE_URL}"
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            return False, "unavailable: ffmpeg and ffprobe are required"
        try:
            help_text = _run([binary, "--help"], probe=True)
        except RenderError as exc:
            return False, f"unavailable: {exc}; {DEEPFILTER_RELEASE_URL}"
        required = ("--atten-lim-db", "--output-dir", "--compensate-delay")
        if not all(flag in help_text for flag in required):
            return False, f"unavailable: incompatible deep-filter CLI; {DEEPFILTER_RELEASE_URL}"
        return True, f"{binary}; --atten-lim-db; {DEEPFILTER_RELEASE_URL}"

    def enhance(self, wav_in: Path, wav_out: Path, strength: float) -> None:
        strength = _strength(strength)
        _check_paths(wav_in, wav_out)
        _require(self)
        binary = self._binary()
        if binary is None:
            raise EnhancerUnavailable("deep-filter disappeared after its availability probe")
        audio = _audio(wav_in)
        try:
            with tempfile.TemporaryDirectory(prefix="declip-deepfilter-", dir=wav_out.parent) as raw:
                temp = Path(raw)
                enhanced = []
                # Isolate channels even for binaries that accept multichannel input.
                # This avoids shared masks and keeps index order on every binary.
                for index in range(audio.channels):
                    mono = temp / f"channel-{index}.wav"
                    _run([
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(wav_in.resolve()), "-map", "0:a:0", "-af",
                        f"pan=mono|c0=c{index},aresample=48000,apad=pad_dur=1",
                        "-c:a", "pcm_f32le", str(mono),
                    ])
                    # Float input avoids the binary's i16-only integer WAV reader.
                    # Tail padding flushes its lookahead before delay compensation.
                    out_dir = temp / f"output-{index}"
                    _run([
                        binary, "--atten-lim-db", f"{100 * strength:g}",
                        "--compensate-delay", "--output-dir", str(out_dir), str(mono),
                    ])
                    result = out_dir / mono.name
                    if not result.is_file():
                        raise RenderError(f"deep-filter produced no WAV: {result.name}")
                    result_audio = _audio(result)
                    if result_audio.channels != 1 or result_audio.rate != 48000:
                        raise RenderError("deep-filter returned invalid mono 48 kHz audio")
                    if Fraction(result_audio.samples, 48000) < Fraction(audio.samples, audio.rate):
                        raise RenderError("deep-filter returned truncated audio")
                    enhanced.append(result)
                inputs = [arg for path in enhanced for arg in ("-i", str(path))]
                labels = "".join(f"[{index}:a:0]" for index in range(audio.channels))
                merge = f"amerge=inputs={audio.channels}," if audio.channels > 1 else ""
                graph = f"{labels}{merge}aresample={audio.rate},apad,atrim=end_sample={audio.samples}[out]"
                output = temp / "enhanced.wav"
                _run([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs,
                    "-filter_complex", graph, "-map", "[out]",
                    *_output_args(audio), str(output),
                ])
                _check_audio(output, audio)
                os.replace(output, wav_out)
        except OSError as exc:
            raise RenderError(f"deepfilter output failed: {exc}") from exc


def _require(enhancer: Enhancer) -> None:
    available, reason = enhancer.available()
    if not available:
        raise EnhancerUnavailable(f"{enhancer.name}: {reason}")


def _registry() -> dict[str, Enhancer]:
    return {backend.name: backend for backend in (
        NoneEnhancer(), AfftdnEnhancer(), DeepFilterEnhancer(),
    )}


def select_enhancer(name: str = "auto") -> Enhancer:
    backends = _registry()
    if name == "auto":
        if backends["deepfilter"].available()[0]:
            return backends["deepfilter"]
        name = "afftdn"
    if name not in backends:
        raise EnhancerUnavailable(f"Unknown enhancer {name!r}; choose none, afftdn, deepfilter, or auto")
    _require(backends[name])
    return backends[name]


def list_enhancers() -> list[tuple[str, bool, str]]:
    return [(name, *backend.available()) for name, backend in _registry().items()]
