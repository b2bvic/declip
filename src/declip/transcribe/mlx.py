"""Lazy Apple silicon Whisper backend."""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

from declip.contracts import TranscribeOptions, Transcript, TranscriberUnavailable
from ._common import chunked, normalize, prompt_mode
from .download import ensure_model


class MlxTranscriber:
    name = "mlx"

    def available(self) -> tuple[bool, str]:
        if sys.platform != "darwin" or platform.machine() != "arm64":
            return False, "mlx requires macOS arm64"
        try:
            importlib.import_module("mlx_whisper")
            return True, "mlx-whisper available"
        except Exception as exc:
            return False, f"Install declip[mac]: {exc}"

    def transcribe(self, wav: Path, opts: TranscribeOptions) -> Transcript:
        mode = prompt_mode(self.name, opts)
        if opts.device not in {"auto", "metal"}:
            raise TranscriberUnavailable(f"mlx cannot use device {opts.device}")
        if mode == "chunked":
            return chunked(wav, opts, self.transcribe)
        try:
            model = ensure_model(opts.model, backend=self.name)
            result = importlib.import_module("mlx_whisper").transcribe(
                str(wav),
                path_or_hf_repo=model,
                language=opts.language,
                initial_prompt=opts.initial_prompt,
                word_timestamps=True,
                condition_on_previous_text=opts.condition_on_previous_text,
                verbose=False,
            )
            return normalize(
                result.get("segments", ()),
                backend=self.name,
                opts=opts,
                language=result.get("language") or opts.language or "und",
            )
        except TranscriberUnavailable:
            raise
        except Exception as exc:
            raise TranscriberUnavailable(f"mlx transcription failed: {exc}") from exc
