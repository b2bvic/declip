"""Fixture backend, enabled only by the explicit testing environment."""

from __future__ import annotations

import json
import os
from pathlib import Path

from declip.contracts import TranscribeOptions, Transcript, TranscriberUnavailable
from ._common import normalize


class FakeTranscriber:
    name = "fake"

    def available(self) -> tuple[bool, str]:
        enabled = os.environ.get("DECLIP_TRANSCRIBER") == "fake"
        return (
            enabled,
            "fixture backend enabled"
            if enabled
            else "Set DECLIP_TRANSCRIBER=fake for fixtures",
        )

    def transcribe(self, wav: Path, opts: TranscribeOptions) -> Transcript:
        if not self.available()[0]:
            raise TranscriberUnavailable(self.available()[1])
        directory = os.environ.get("DECLIP_FAKE_TRANSCRIPT_DIR")
        if not directory:
            raise TranscriberUnavailable("Set DECLIP_FAKE_TRANSCRIPT_DIR for fixtures")
        path = Path(directory) / f"{wav.stem}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if "words" in data:
                return Transcript.from_dict(
                    {
                        **data,
                        "backend": "fake",
                        "model": opts.model,
                        "prompt": opts.initial_prompt,
                    }
                )
            return normalize(
                data["segments"],
                backend=self.name,
                opts=opts,
                language=data.get("language") or opts.language or "en",
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise TranscriberUnavailable(f"Cannot load fixture {path}: {exc}") from exc
