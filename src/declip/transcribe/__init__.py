"""Backend registry, explicit device selection, extraction, and transcript cache."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Callable

from declip.contracts import (
    SCHEMA_VERSION,
    BackendSelection,
    FillerFile,
    ToolMissing,
    TranscribeOptions,
    Transcriber,
    Transcript,
    TranscriberUnavailable,
)
from declip.fsutil import atomic_write_json
from . import cuda_libs
from ._common import AUTO_PROMPT_MODES, copy_window


def get(name: str) -> Transcriber:
    from .fake import FakeTranscriber
    from .faster import FasterTranscriber
    from .mlx import MlxTranscriber

    factories = {
        "mlx": MlxTranscriber,
        "faster": FasterTranscriber,
        "fake": FakeTranscriber,
    }
    if name not in factories:
        raise TranscriberUnavailable(f"Unknown transcriber: {name}")
    if name == "fake" and os.environ.get("DECLIP_TRANSCRIBER") != "fake":
        raise TranscriberUnavailable("fake requires DECLIP_TRANSCRIBER=fake")
    return factories[name]()


def _cuda_available() -> tuple[bool, str]:
    ok, reason = cuda_libs.preload()
    if not ok:
        return False, reason
    try:
        count = importlib.import_module("ctranslate2").get_cuda_device_count()
        return count > 0, f"CTranslate2 CUDA device count: {count}"
    except Exception as exc:
        return False, f"CUDA unavailable: {exc}"


def select(device: str = "auto", *, backend: str = "auto") -> BackendSelection:
    if device not in {"auto", "metal", "cuda", "cpu"}:
        raise TranscriberUnavailable(f"Unknown device: {device}")
    if backend not in {"auto", "mlx", "faster", "fake"}:
        raise TranscriberUnavailable(f"Unknown transcriber: {backend}")
    environment = os.environ.get("DECLIP_TRANSCRIBER")
    if environment:
        if environment not in {"mlx", "faster", "fake"}:
            raise TranscriberUnavailable(f"Unknown DECLIP_TRANSCRIBER: {environment}")
        if backend != "auto" and backend != environment:
            raise TranscriberUnavailable(
                f"Backend {backend} conflicts with DECLIP_TRANSCRIBER={environment}"
            )
        backend = environment
    if backend == "fake":
        if device not in {"auto", "cpu"}:
            raise TranscriberUnavailable(f"fake cannot use {device}")
        return BackendSelection(get("fake"), "cpu", "int8")
    reasons = []
    if backend in {"auto", "mlx"} and device in {"auto", "metal"}:
        mlx = get("mlx")
        ok, reason = mlx.available()
        if ok:
            return BackendSelection(mlx, "metal", "auto")
        reasons.append(reason)
    if backend == "mlx" or device == "metal":
        raise TranscriberUnavailable(
            "; ".join(reasons) or f"{backend} cannot use {device}"
        )
    faster = get("faster")
    # preload must run before faster imports CTranslate2, even on the CPU fallback.
    ok, reason = faster.available()
    if ok:
        if device in {"auto", "cuda"}:
            cuda_ok, cuda_reason = _cuda_available()
            if cuda_ok:
                return BackendSelection(faster, "cuda", "float16")
            if device == "cuda":
                raise TranscriberUnavailable(cuda_reason)
        return BackendSelection(faster, "cpu", "int8")
    reasons.append(reason)
    raise TranscriberUnavailable("; ".join(reasons))


def extract_transcription_audio(
    source: Path, out_wav: Path, *, audio_index: int | None
) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ToolMissing(
            "ffmpeg is missing; install ffmpeg with brew or your Linux package manager"
        )
    if source.resolve() == out_wav.resolve():
        raise TranscriberUnavailable("Transcription WAV must differ from the source")
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-map",
        f"0:{audio_index}" if audio_index is not None else "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ]
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode:
        out_wav.unlink(missing_ok=True)
        raise TranscriberUnavailable(
            f"Transcription audio extraction failed: {result.stderr[-4000:]}"
        )
    return out_wav


def cache_key(wav_sha256: str, backend: str, opts: TranscribeOptions) -> str:
    payload = {
        "wav_sha256": wav_sha256,
        "backend": backend,
        "model": opts.model,
        "language": opts.language,
        "prompt": opts.initial_prompt,
        "prompt_mode": opts.prompt_mode,
        "condition_on_previous_text": opts.condition_on_previous_text,
        "compute_type": opts.compute_type,
        "schema_version": SCHEMA_VERSION,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _full_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def transcribe_file(
    source: Path,
    opts: TranscribeOptions,
    *,
    selection: BackendSelection,
    cache_dir: Path,
    audio_index: int | None,
    fillers_for: Callable[[str], FillerFile | None],
) -> Transcript:
    transcriber = selection.transcriber
    resolved = replace(
        opts,
        device=selection.device,
        compute_type=selection.compute_type
        if opts.compute_type == "auto"
        else opts.compute_type,
        prompt_mode=AUTO_PROMPT_MODES.get(transcriber.name, "initial")
        if opts.prompt_mode == "auto"
        else opts.prompt_mode,
    )
    if transcriber.name == "fake":
        return transcriber.transcribe(source, resolved)
    folder = cache_dir / "transcripts"
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="declip-transcribe-", dir=folder) as tmp:
        wav = extract_transcription_audio(
            source, Path(tmp) / "audio.wav", audio_index=audio_index
        )
        digest = _full_hash(wav)
        if resolved.language is None:
            detect_opts = replace(resolved, initial_prompt=None, prompt_mode="initial")
            language_path = folder / (
                cache_key(digest, transcriber.name, detect_opts) + ".language.json"
            )
            try:
                language = json.loads(language_path.read_text(encoding="utf-8"))[
                    "language"
                ]
                if not isinstance(language, str) or not language:
                    raise ValueError("Invalid cached language")
            except (OSError, ValueError, KeyError, TypeError):
                first = Path(tmp) / "first.wav"
                copy_window(wav, first, 0, 30)
                language = transcriber.transcribe(first, detect_opts).language
                atomic_write_json(language_path, {"language": language})
            resolved = replace(resolved, language=language)
        fillers = fillers_for(resolved.language)
        prompt = (
            opts.initial_prompt
            if opts.initial_prompt is not None
            else (fillers.prompt if fillers else None)
        )
        resolved = replace(resolved, initial_prompt=prompt)
        path = folder / (cache_key(digest, transcriber.name, resolved) + ".json")
        try:
            return Transcript.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError):
            pass
        result = transcriber.transcribe(wav, resolved)
        atomic_write_json(path, result.to_dict())
        return result
