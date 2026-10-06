"""Lazy CTranslate2 Whisper backend for CPU and CUDA."""

from __future__ import annotations

import importlib
import wave
from pathlib import Path

from declip.contracts import TranscribeOptions, Transcript, TranscriberUnavailable
from . import cuda_libs
from ._common import chunked, normalize, prompt_mode
from .download import ensure_model


class FasterTranscriber:
    name = "faster"

    def __init__(self):
        self._models = {}

    def available(self) -> tuple[bool, str]:
        try:
            cuda_libs.preload()
            importlib.import_module("faster_whisper")
            return True, "faster-whisper available"
        except Exception as exc:
            return False, f"Install declip[cpu] or declip[cuda]: {exc}"

    def transcribe(self, wav: Path, opts: TranscribeOptions) -> Transcript:
        mode = prompt_mode(self.name, opts)
        if opts.device == "metal":
            raise TranscriberUnavailable("faster cannot use metal")
        if mode == "chunked":
            return chunked(wav, opts, self.transcribe)
        # Direct protocol callers receive the same explicit-device checks as the CLI.
        from . import select

        selection = select(opts.device, backend="faster")
        device = selection.device
        compute = (
            selection.compute_type if opts.compute_type == "auto" else opts.compute_type
        )
        try:
            model_path = ensure_model(opts.model, backend=self.name)
            key = (model_path, device, compute)
            if key not in self._models:
                self._models[key] = importlib.import_module(
                    "faster_whisper"
                ).WhisperModel(
                    model_path,
                    device=device,
                    compute_type=compute,
                    local_files_only=True,
                )
            kwargs = {
                "language": opts.language,
                "word_timestamps": True,
                "condition_on_previous_text": opts.condition_on_previous_text,
                "initial_prompt": opts.initial_prompt if mode == "initial" else None,
                "vad_filter": False,
            }
            if mode == "hotwords":
                kwargs["hotwords"] = opts.initial_prompt
            # Extraction already gives us 16 kHz mono PCM. Read it directly so
            # upstream PyAV API changes cannot affect the model input.
            import numpy as np

            with wave.open(str(wav), "rb") as audio:
                if (
                    audio.getframerate(),
                    audio.getnchannels(),
                    audio.getsampwidth(),
                ) != (16000, 1, 2):
                    raise TranscriberUnavailable(
                        "faster requires 16 kHz mono PCM 16-bit WAV"
                    )
                samples = (
                    np.frombuffer(
                        audio.readframes(audio.getnframes()), dtype="<i2"
                    ).astype(np.float32)
                    / 32768.0
                )
            rows, info = self._models[key].transcribe(samples, **kwargs)
            normalized = [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "words": [
                        {
                            "start": w.start,
                            "end": w.end,
                            "word": w.word,
                            "probability": getattr(w, "probability", None),
                        }
                        for w in (s.words or ())
                    ],
                }
                for s in rows
            ]
            return normalize(
                normalized,
                backend=self.name,
                opts=opts,
                language=info.language or opts.language or "und",
            )
        except TranscriberUnavailable:
            raise
        except Exception as exc:
            raise TranscriberUnavailable(
                f"faster transcription failed on {device}: {exc}"
            ) from exc
