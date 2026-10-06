"""Generate public synthetic speech and measure prompt persistence without saving transcripts.

Run with the mac or cpu extra. Model downloads use the normal Hugging Face cache.
Real-footage receipts contain counts and window times only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import stat
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import replace
from pathlib import Path
from typing import Literal

from declip.contracts import DeclipError, TranscribeOptions, Transcript
from declip.fillers import load_fillers
from declip.fsutil import atomic_write_json
from declip.paths import config_dir
from declip.transcribe import extract_transcription_audio, select

SENTENCES = (
    "The light crosses the table while the room stays quiet.",
    "I place the blue notebook beside the empty glass.",
    "Today we describe a simple way to arrange the shelves.",
    "Each page contains a sentence about an ordinary afternoon.",
    "The small wooden box has a smooth lid and four corners.",
    "We can count the steps from the door to the window.",
)
PHRASES = tuple(
    p for i in range(6) for p in ("um" if i % 2 == 0 else "uh", "The room is quiet.")
) + tuple(
    p
    for i in range(12)
    for p in (SENTENCES[i % len(SENTENCES)], "uh" if i % 2 == 0 else "um")
)
LICENSE = "synthetic, generated locally with macOS `say`"


def make_clip(folder: Path, voice: str) -> dict:
    if sys.platform != "darwin":
        raise DeclipError("--make-clip requires macOS say")
    folder.mkdir(parents=True, exist_ok=True)
    frames, phrases, labels = bytearray(), [], []
    with tempfile.TemporaryDirectory(prefix="phrases-", dir=folder) as tmp:
        script = list(PHRASES)
        index = 0
        late_boundary = 12
        while index < len(script) or len(frames) / 96000 < 120:
            if index == late_boundary and len(frames) / 96000 < 30:
                script.insert(index, SENTENCES[index % len(SENTENCES)])
                late_boundary += 1
            if index == len(script):
                script.append(SENTENCES[index % len(SENTENCES)])
            phrase = script[index]
            aiff, wav = Path(tmp) / "phrase.aiff", Path(tmp) / "phrase.wav"
            subprocess.run(
                ["say", "-v", voice, "-r", "185", "-o", str(aiff), phrase], check=True
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(aiff),
                    "-map",
                    "0:a:0",
                    "-ar",
                    "48000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(wav),
                ],
                check=True,
            )
            with wave.open(str(wav), "rb") as source:
                audio = source.readframes(source.getnframes())
            start, end = len(frames) / 96000, (len(frames) + len(audio)) / 96000
            phrases.append({"text": phrase, "start": start, "end": end})
            if phrase in {"um", "uh"}:
                labels.append({"label": phrase, "start": start, "end": end})
            frames.extend(audio)
            frames.extend(b"\0\0" * 19200)  # 400 ms, measured on the sample grid.
            index += 1
    clip = folder / "clip.wav"
    with wave.open(str(clip), "wb") as target:
        target.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
        target.writeframes(frames)
    manifest = {
        "clip_sha256": file_hash(clip),
        "duration": len(frames) / 96000,
        "voice": voice,
        "phrases": phrases,
        "fillers": labels,
        "license": LICENSE,
    }
    early = sum(label["start"] < 30 for label in labels)
    if early < 6 or len(labels) - early < 12:
        raise DeclipError(
            f"Voice timing violates fixture requirements: {early} early, {len(labels) - early} late"
        )
    atomic_write_json(folder / "manifest.json", manifest)
    return {
        "clip": str(clip),
        "manifest": str(folder / "manifest.json"),
        "duration": manifest["duration"],
        "early_labels": early,
        "late_labels": len(labels) - early,
        "clip_sha256": manifest["clip_sha256"],
    }


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def filler_words(transcript: Transcript):
    return [
        w
        for w in transcript.words
        if w.text.lower().strip(".,!?;:\"'()") in {"um", "uh"}
    ]


def recall(transcript: Transcript, manifest: dict) -> dict:
    candidates = filler_words(transcript)
    used = set()
    result = {}
    for name, early in (("first_30", True), ("after_30", False)):
        labels = [f for f in manifest["fillers"] if (f["start"] < 30) == early]
        hits = 0
        for label in labels:
            for i, word in enumerate(candidates):
                if i in used or word.text.lower().strip(".,!?;:") != label["label"]:
                    continue
                if (
                    word.start <= label["end"] + 0.3
                    and word.end >= label["start"] - 0.3
                ):
                    hits += 1
                    used.add(i)
                    break
        result[name] = {
            "labels": len(labels),
            "hits": hits,
            "recall": hits / len(labels) if labels else 0,
        }
    result["false_hits"] = len(candidates) - len(used)
    return result


def window_counts(transcript: Transcript, duration: float) -> list[dict]:
    words = filler_words(transcript)
    return [
        {
            "start": start,
            "end": min(start + 30, duration),
            "count": sum(start <= word.start < start + 30 for word in words),
        }
        for start in range(0, math.ceil(duration), 30)
    ]


def real_pass(
    prompt: list[dict], baseline: list[dict]
) -> Literal["passed", "failed", "inconclusive"]:
    total = sum(w["count"] for w in prompt)
    no_prompt = sum(w["count"] for w in baseline)
    if no_prompt == 0:
        return "inconclusive"
    # Compare mean counts per 30-second window. Normalize a partial final window.
    late_seconds = sum(w["end"] - w["start"] for w in prompt[1:])
    late_mean = (
        sum(w["count"] for w in prompt[1:]) * 30 / late_seconds if late_seconds else 0
    )
    return (
        "passed"
        if total >= no_prompt and late_mean >= prompt[0]["count"] * 0.5
        else "failed"
    )


def is_dataless(path: Path) -> bool:
    info = path.stat()
    return bool(getattr(info, "st_flags", 0) & getattr(stat, "SF_DATALESS", 0)) or (
        sys.platform == "darwin"
        and info.st_size > 0
        and getattr(info, "st_blocks", 1) == 0
    )


def probe(
    clip: Path,
    manifest_path: Path,
    backend: str,
    device: str,
    model: str | None,
    real: Path | None,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if file_hash(clip) != manifest["clip_sha256"]:
        raise DeclipError("clip SHA-256 differs from manifest")
    selection = select(device, backend=backend)
    if selection.transcriber.name == "fake":
        raise DeclipError("Probe requires a real backend")
    model = model or (
        "mlx-community/whisper-large-v3-turbo" if backend == "mlx" else "large-v3-turbo"
    )
    fillers = load_fillers("en", config_dir=config_dir())
    prompt = fillers.prompt if fillers else None
    opts = TranscribeOptions(
        model,
        language="en",
        initial_prompt=prompt,
        device=selection.device,
        compute_type=selection.compute_type,
    )
    report = {
        "backend": backend,
        "device": selection.device,
        "compute_type": selection.compute_type,
        "model": model,
        "clip_sha256": manifest["clip_sha256"],
        "duration": manifest["duration"],
        "synthetic": {},
        "real": {"status": "not requested"},
    }
    fallback = "chunked" if backend == "mlx" else "hotwords"
    with tempfile.TemporaryDirectory(prefix="declip-probe-", dir=clip.parent) as tmp:
        wav = extract_transcription_audio(
            clip, Path(tmp) / "clip.wav", audio_index=None
        )

        def run(path, mode, use_prompt=True):
            start = time.monotonic()
            result = selection.transcriber.transcribe(
                path,
                replace(
                    opts,
                    prompt_mode=mode,
                    initial_prompt=prompt if use_prompt else None,
                ),
            )
            print(
                f"{backend} {mode if use_prompt else 'no_prompt'} finished in {time.monotonic() - start:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            return result

        for name, use_prompt in (("initial", True), ("no_prompt", False)):
            report["synthetic"][name] = recall(
                run(wav, "initial", use_prompt), manifest
            )
        with_prompt, without = (
            report["synthetic"]["initial"],
            report["synthetic"]["no_prompt"],
        )
        inconclusive = all(
            without[k]["recall"] >= 0.9 for k in ("first_30", "after_30")
        )
        persists = (
            with_prompt["first_30"]["recall"] - with_prompt["after_30"]["recall"] <= 0.2
        )
        mode = "initial" if inconclusive or persists else fallback
        report["synthetic_status"] = (
            "inconclusive"
            if inconclusive
            else "persists"
            if persists
            else "does not persist"
        )
        if mode == fallback:
            report["synthetic"][fallback] = recall(run(wav, fallback), manifest)
        if real is not None:
            if not real.is_file():
                report["real"] = {"status": "file missing"}
            elif is_dataless(real):
                report["real"] = {
                    "status": "unavailable",
                    "reason": "iCloud dataless file; local bytes unavailable",
                }
            else:
                # Never copy real media or retain its transcript. Only transient mono PCM.
                try:
                    real_wav = extract_transcription_audio(
                        real, Path(tmp) / "real.wav", audio_index=None
                    )
                    with wave.open(str(real_wav), "rb") as audio:
                        duration = audio.getnframes() / audio.getframerate()
                    prompted = window_counts(run(real_wav, "initial"), duration)
                    no_prompt = window_counts(run(real_wav, "initial", False), duration)
                    status = real_pass(prompted, no_prompt)
                    report["real"] = {
                        "status": "completed",
                        "duration": duration,
                        "initial": prompted,
                        "no_prompt": no_prompt,
                        "initial_status": status,
                    }
                    # Only conclusive real-speech evidence overrides the synthetic result.
                    if status == "passed":
                        mode = "initial"
                    elif status == "failed":
                        mode = fallback
                        report["real"][fallback] = window_counts(
                            run(real_wav, fallback), duration
                        )
                        report["real"][fallback + "_status"] = real_pass(
                            report["real"][fallback], no_prompt
                        )
                        if fallback not in report["synthetic"]:
                            report["synthetic"][fallback] = recall(
                                run(wav, fallback), manifest
                            )
                except (DeclipError, OSError, ValueError):
                    report["real"] = {
                        "status": "unavailable",
                        "reason": "real-speech probe failed; synthetic results retained",
                    }
        report["auto_prompt_mode"] = mode
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make-clip", type=Path)
    parser.add_argument("--voice", default="Samantha")
    parser.add_argument("--clip", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--backend", choices=("mlx", "faster"))
    parser.add_argument(
        "--device", default="auto", choices=("auto", "metal", "cuda", "cpu")
    )
    parser.add_argument("--model")
    parser.add_argument("--real", type=Path)
    parser.add_argument("--out", type=Path, help="Write a JSON count/recall receipt")
    args = parser.parse_args()
    try:
        if args.make_clip:
            result = make_clip(args.make_clip, args.voice)
        else:
            if not all((args.clip, args.manifest, args.backend)):
                parser.error("provide --clip, --manifest, and --backend")
            result = probe(
                args.clip,
                args.manifest,
                args.backend,
                args.device,
                args.model,
                args.real,
            )
        if args.out:
            atomic_write_json(args.out, result)
        print(json.dumps(result, indent=2))
    except (DeclipError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"prompt probe failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
