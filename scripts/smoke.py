"""Measure real transcription and rendering; retain receipts, never generated media.

Run from the worktree root with isolated DECLIP_CONFIG_DIR and DECLIP_CACHE_DIR.
The CPU column forces software encoding. Synthetic sync fixtures use the same
reviewed-plan helper as render tests; this is not a user-media review shortcut.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import wave
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from declip import __version__, editlist, encoders, hardware, media, render, rig
from declip.contracts import (
    Cut,
    CutKind,
    CutOrigin,
    CutStatus,
    DeclipError,
    OutputMode,
    OutputSpec,
    Processing,
    RigRef,
    TranscribeOptions,
)
from declip.fillers import load_fillers
from declip.fsutil import atomic_write_json
from declip.paths import config_dir
from declip.transcribe import extract_transcription_audio, select
from prompt_probe import recall

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "declip_version",
    "git_commit",
    "os",
    "arch",
    "chip",
    "gpu",
    "driver",
    "ffmpeg_version",
    "backend",
    "device",
    "compute_type",
    "model",
    "ctranslate2_cuda_device_count",
    "nvidia_smi_during_run",
    "clip_sha256",
    "filler_recall",
    "filler_false_hits",
    "real_time_factor",
    "encoder",
    "render_8bit_probe",
    "render_10bit_probe",
    "av_sync_max_offset_frames",
    "passed",
    "requested_device",
    "filler_measurement",
    "av_sync_probe",
    "errors",
}


def run(argv: list[str], *, timeout: float = 300) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise DeclipError(f"{Path(argv[0]).name} failed: {result.stderr[-4000:]}")
    return result.stdout


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def finite(value, lower=0, upper=math.inf) -> bool:
    return (
        type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper
    )


def validation_errors(receipt: dict) -> list[str]:
    """Recompute rules from measurements, without trusting the passed flag."""
    if not isinstance(receipt, dict):
        return ["receipt must be an object"]
    missing = sorted(REQUIRED - receipt.keys())
    if missing:
        return ["missing fields: " + ", ".join(missing)]
    errors = []
    for key in ("git_commit", "clip_sha256"):
        if not isinstance(receipt[key], str) or not re.fullmatch(
            r"[0-9a-f]{40}" if key == "git_commit" else r"[0-9a-f]{64}", receipt[key]
        ):
            errors.append(f"invalid {key}")
    for key in ("declip_version", "os", "arch", "chip", "ffmpeg_version", "model"):
        if not isinstance(receipt[key], str) or not receipt[key].strip():
            errors.append(f"missing {key} evidence")
    requested, device, backend = (
        receipt["requested_device"],
        receipt["device"],
        receipt["backend"],
    )
    expected = {"metal": "mlx", "cpu": "faster", "cuda": "faster"}
    if requested not in {"auto", *expected} or device not in expected:
        errors.append("invalid requested or selected device")
    elif (requested != "auto" and requested != device) or backend != expected[device]:
        errors.append("backend and device do not match the request")
    if backend not in {"mlx", "faster"}:
        errors.append("a real backend is required")
    if device == "metal" and (receipt["os"], receipt["arch"]) != ("darwin", "arm64"):
        errors.append("Metal requires macOS arm64")
    if device == "cpu" and receipt["compute_type"] != "int8":
        errors.append("CPU requires int8")
    if device == "metal" and receipt["compute_type"] != "auto":
        errors.append("Metal requires auto compute type")
    if backend == "faster":
        count = receipt["ctranslate2_cuda_device_count"]
        if type(count) is not int or count < 0 or (device == "cuda" and count < 1):
            errors.append("invalid CTranslate2 CUDA device count")
    if device == "cuda":
        sample = receipt["nvidia_smi_during_run"]
        if (
            not isinstance(sample, dict)
            or sample.get("exit_code") != 0
            or not sample.get("output")
        ):
            errors.append("missing NVIDIA sample during transcription")
        if receipt["compute_type"] not in {"float16", "int8_float16"}:
            errors.append("invalid CUDA compute type")
    if not finite(receipt["filler_recall"], 0.5, 1):
        errors.append("filler recall must be at least 0.5")
    if (
        type(receipt["filler_false_hits"]) is not int
        or receipt["filler_false_hits"] < 0
    ):
        errors.append("invalid filler false-hit count")
    measurement = receipt["filler_measurement"]
    if not isinstance(measurement, dict):
        errors.append("missing manifest label measurements")
    else:
        labels, hits = measurement.get("labels"), measurement.get("hits")
        if (
            type(labels) is not int
            or labels < 18
            or type(hits) is not int
            or not 0 <= hits <= labels
        ):
            errors.append("invalid manifest label measurements")
        elif not finite(receipt["filler_recall"], 0, 1) or not math.isclose(
            hits / labels, receipt["filler_recall"], abs_tol=1e-9
        ):
            errors.append("recall disagrees with label measurements")
        if measurement.get("tolerance_seconds") != 0.3:
            errors.append("filler tolerance must be 300 ms")
    if not finite(receipt["real_time_factor"]) or receipt["real_time_factor"] <= 0:
        errors.append("invalid real-time factor")
    expected_encoder = {
        "metal": "videotoolbox",
        "cuda": "nvenc",
        "cpu": "software",
    }.get(device)
    if receipt["encoder"] != expected_encoder:
        errors.append("encoder does not match the verification column")
    for key, codec, depth in (
        ("render_8bit_probe", "h264", 8),
        ("render_10bit_probe", "hevc", 10),
    ):
        probe = receipt[key]
        if not isinstance(probe, dict):
            errors.append(f"missing {key}")
            continue
        pixel = probe.get("pix_fmt")
        valid_pixel = (
            pixel == "yuv420p"
            if depth == 8
            else pixel in {"yuv420p10le", "p010le", "yuv422p10le", "yuv444p10le"}
        )
        if (
            not valid_pixel
            or probe.get("codec") != codec
            or probe.get("bit_depth") != depth
        ):
            errors.append(f"{key} codec or pixel depth failed")
        if depth == 10 and probe.get("profile") != "Main 10":
            errors.append("10-bit probe must have Main 10 profile")
        if probe.get("encoder") != expected_encoder:
            errors.append(f"{key} used the wrong encoder")
        if not finite(probe.get("duration_error_frames"), 0, 1):
            errors.append(f"{key} output duration differs by more than one frame")
    if not finite(receipt["av_sync_max_offset_frames"], 0, 1):
        errors.append("A/V sync exceeds one frame or was not measured")
    sync = receipt["av_sync_probe"]
    if not isinstance(sync, dict) or any(
        sync.get(k) != 250 for k in ("keeps", "flashes", "beeps")
    ):
        errors.append("sync probe requires 250 keeps, flashes, and beeps")
    else:
        if sync.get("encoder") != expected_encoder:
            errors.append("sync probe used the wrong encoder")
        if not finite(sync.get("duration_error_frames"), 0, 1):
            errors.append("sync output duration differs by more than one frame")
        if sync.get("max_offset_frames") != receipt["av_sync_max_offset_frames"]:
            errors.append("sync offset measurements disagree")
    if receipt["errors"] != []:
        errors.append("run recorded errors")
    return errors


def options_for(device: str):
    return rig.resolve_options(
        {
            "preset": "none",
            "crossfade_ms": 0,
            "encoder": "software" if device == "cpu" else "auto",
        },
        None,
        {},
    )


def render_probe(folder, depth, options, caps):
    source = folder / f"source-{depth}.mp4"
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=30000/1001",
            "-f",
            "lavfi",
            "-i",
            "sine=sample_rate=48000",
            "-t",
            "3",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx265" if depth == 10 else "libx264",
            "-pix_fmt",
            "yuv420p10le" if depth == 10 else "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ]
    )
    info = media.probe_media(source)
    plan = render.build_enhance_plan(
        source,
        folder / f"output-{depth}.mp4",
        info,
        options,
        caps,
        temp_dir=folder / f"render-{depth}",
    )
    result = render.run_render_plan(plan)
    return {
        "codec": result.video_codec,
        "pix_fmt": result.pix_fmt,
        "profile": result.profile,
        "bit_depth": result.bit_depth,
        "encoder": result.encoder,
        "duration": result.duration,
        "duration_error_frames": abs(result.duration - plan.timeline.duration_out)
        * float(info.fps),
    }


def sync_probe(folder, options, caps):
    source = folder / "flash-beep.mp4"
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=black:size=320x240:rate=30000/1001,geq=lum='if(lt(abs(T-round(T)),0.016684),235,16)':cb=128:cr=128",
            "-f",
            "lavfi",
            "-i",
            "aevalsrc=if(lt(mod(t\\,1)\\,0.02)\\,0.5*sin(2*PI*1000*t)\\,0):s=48000",
            "-t",
            "251",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            str(source),
        ]
    )
    info = media.probe_media(source)
    keeps = [(n - 0.213, n + 0.287) for n in range(1, 251)]
    removed = [
        (0, keeps[0][0]),
        *[(a[1], b[0]) for a, b in zip(keeps, keeps[1:])],
        (keeps[-1][1], info.duration),
    ]
    edit = editlist.new_edit_list(
        source,
        info,
        media.file_hash(source),
        transcript=None,
        processing=Processing("none", 0.5, "", None, 0),
        output=OutputSpec(OutputMode.RENDER, None, False),
        rig=RigRef(None, None, {}),
    )
    cuts = tuple(
        Cut(
            editlist.cut_id(CutKind.MANUAL, a, b, "synthetic fixture"),
            CutKind.MANUAL,
            a,
            b,
            "synthetic fixture",
            1.0,
            False,
            None,
            CutOrigin.MANUAL,
            CutStatus.ACCEPTED,
        )
        for a, b in removed
    )
    edit = editlist.mark_review_passed(
        replace(edit, cuts=cuts), now=datetime.now(timezone.utc)
    )
    plan = render.build_render_plan(
        edit,
        source,
        folder / "sync-output.mp4",
        options,
        caps,
        temp_dir=folder / "sync-render",
    )
    result = render.run_render_plan(plan)

    # Pipe measurements rather than writing decoded media to disk.
    def report(stream, flag, filters):
        output = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-i",
                str(result.output),
                "-map",
                stream,
                flag,
                filters,
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if output.returncode:
            raise DeclipError(output.stderr[-4000:])
        return output.stderr

    flashes, current = [], None
    for line in report("0:v:0", "-vf", "signalstats,metadata=mode=print").splitlines():
        match = re.search(r"pts_time:([\d.]+)", line)
        if match:
            current = float(match[1])
        match = re.search(r"lavfi.signalstats.YAVG=([\d.]+)", line)
        if match and float(match[1]) > 200 and current is not None:
            flashes.append(current)
    beeps = [
        float(v)
        for v in re.findall(
            r"silence_end:\s*([\d.]+)",
            report("0:a:0", "-af", "silencedetect=noise=-25dB:d=0.05"),
        )
        if float(v) < result.duration - float(1 / info.fps)
    ]
    offsets = [abs(a - b) * float(info.fps) for a, b in zip(flashes, beeps)]
    return {
        "keeps": len(plan.timeline.keeps),
        "flashes": len(flashes),
        "beeps": len(beeps),
        "max_offset_frames": max(offsets) if offsets else None,
        "batch_border_offsets_frames": {
            "40/41": offsets[39:41],
            "80/81": offsets[79:81],
        },
        "duration_error_frames": abs(result.duration - plan.timeline.duration_out)
        * float(info.fps),
        "encoder": result.encoder,
        "fps": str(info.fps),
        "width": 320,
        "height": 240,
    }


def smoke(clip: Path, manifest_path: Path, requested: str) -> dict:
    info = hardware.detect()
    receipt = dict.fromkeys(sorted(REQUIRED))
    receipt.update(
        declip_version=__version__,
        git_commit=run(["git", "rev-parse", "HEAD"]).strip(),
        os=info.os,
        arch=info.arch,
        chip=info.chip,
        gpu=[g.name for g in info.gpus],
        driver=[g.driver for g in info.gpus],
        requested_device=requested,
        clip_sha256=digest(clip),
        passed=False,
        errors=[],
        checked_at=datetime.now(timezone.utc).isoformat(),
        scope="synthetic speech and generated render fixtures; no user-media decisions",
    )
    if info.os == "darwin":
        try:
            displays = json.loads(
                run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=5)
            )["SPDisplaysDataType"]
            receipt["gpu"] = [
                g["sppci_model"] for g in displays if g.get("sppci_model")
            ]
            receipt["metal_gpu_family"] = [
                g["spdisplays_mtlgpufamilysupport"]
                for g in displays
                if g.get("spdisplays_mtlgpufamilysupport")
            ]
            receipt["driver_note"] = (
                "Apple GPU driver supplied by macOS; no separate driver version reported"
            )
        except (DeclipError, OSError, ValueError, KeyError, subprocess.SubprocessError):
            receipt["driver_note"] = (
                "Apple GPU metadata unavailable; NVIDIA driver list is empty"
            )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if receipt["clip_sha256"] != manifest["clip_sha256"]:
        raise DeclipError("clip SHA-256 differs from manifest")
    with wave.open(str(clip), "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()
    labels = manifest.get("fillers", [])
    if duration < 120 or not math.isclose(
        duration, manifest["duration"], abs_tol=1 / 48000
    ):
        raise DeclipError(
            "manifest duration does not match the 120-second speech fixture"
        )
    if (
        sum(f["start"] < 30 for f in labels) < 6
        or sum(f["start"] >= 30 for f in labels) < 12
    ):
        raise DeclipError("manifest needs six early and twelve late filler labels")
    with tempfile.TemporaryDirectory(prefix="declip-smoke-") as tmp:
        folder = Path(tmp)
        try:
            selection = select(requested)
            device, backend = selection.device, selection.transcriber.name
            receipt.update(
                device=device, backend=backend, compute_type=selection.compute_type
            )
            if backend not in {"mlx", "faster"}:
                raise DeclipError("Smoke requires a real backend")
            model_hardware = (
                replace(info, apple_silicon=False, gpus=()) if device == "cpu" else info
            )
            _, model, _ = rig.default_model_for(model_hardware)
            receipt["model"] = model
            if backend == "faster":
                import ctranslate2

                receipt["ctranslate2_cuda_device_count"] = (
                    ctranslate2.get_cuda_device_count()
                )
            fillers = load_fillers("en", config_dir=config_dir())
            opts = TranscribeOptions(
                model,
                language="en",
                initial_prompt=fillers.prompt if fillers else None,
                device=device,
                compute_type=selection.compute_type,
            )
            wav = extract_transcription_audio(
                clip, folder / "speech.wav", audio_index=None
            )
            # One sample while the real model call runs, never an after-run sample.
            done = threading.Event()

            def sample_nvidia():
                if not done.wait(0.5):
                    try:
                        sample = subprocess.run(
                            [
                                "nvidia-smi",
                                "--query-gpu=name,driver_version,utilization.gpu,memory.used",
                                "--format=csv,noheader,nounits",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        receipt["nvidia_smi_during_run"] = {
                            "exit_code": sample.returncode,
                            "output": sample.stdout.strip(),
                            "elapsed_seconds": time.monotonic() - start,
                        }
                    except (OSError, subprocess.TimeoutExpired) as exc:
                        receipt["nvidia_smi_during_run"] = {"error": str(exc)}

            start = time.monotonic()
            thread = (
                threading.Thread(target=sample_nvidia) if device == "cuda" else None
            )
            if thread:
                thread.start()
            try:
                transcript = selection.transcriber.transcribe(wav, opts)
            finally:
                elapsed = time.monotonic() - start
                done.set()
                if thread:
                    thread.join()
            counts = recall(transcript, manifest)
            hits = sum(counts[k]["hits"] for k in ("first_30", "after_30"))
            receipt.update(
                filler_recall=hits / len(labels),
                filler_false_hits=counts["false_hits"],
                real_time_factor=elapsed / duration,
                filler_measurement={
                    "labels": len(labels),
                    "hits": hits,
                    "tolerance_seconds": 0.3,
                    **counts,
                },
                transcription_seconds=elapsed,
                clip_duration=duration,
            )
            print(
                f"{backend}/{device}: recall={hits}/{len(labels)}, RTF={elapsed / duration:.4f}",
                file=sys.stderr,
                flush=True,
            )
        except (DeclipError, OSError, ValueError, subprocess.SubprocessError) as exc:
            receipt["errors"].append(f"transcription: {exc}")
        try:
            caps = encoders.probe_capabilities(refresh=True)
            receipt["ffmpeg_version"] = caps.ffmpeg_version
            options = options_for(receipt["device"] or requested)
            for depth in (8, 10):
                try:
                    receipt[f"render_{depth}bit_probe"] = render_probe(
                        folder, depth, options, caps
                    )
                except (
                    DeclipError,
                    OSError,
                    ValueError,
                    subprocess.SubprocessError,
                ) as exc:
                    receipt["errors"].append(f"render {depth}-bit: {exc}")
            probe = receipt["render_8bit_probe"]
            receipt["encoder"] = probe["encoder"] if probe else None
            try:
                receipt["av_sync_probe"] = sync_probe(folder, options, caps)
                receipt["av_sync_max_offset_frames"] = receipt["av_sync_probe"][
                    "max_offset_frames"
                ]
            except (
                DeclipError,
                OSError,
                ValueError,
                subprocess.SubprocessError,
            ) as exc:
                receipt["errors"].append(f"sync: {exc}")
        except (DeclipError, OSError, ValueError, subprocess.SubprocessError) as exc:
            receipt["errors"].append(f"capabilities: {exc}")
    receipt["validation_errors"] = validation_errors(receipt)
    receipt["passed"] = not receipt["validation_errors"]
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "metal", "cpu"), default="auto"
    )
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    if not all(
        os.environ.get(key) for key in ("DECLIP_CONFIG_DIR", "DECLIP_CACHE_DIR")
    ):
        parser.error("set isolated DECLIP_CONFIG_DIR and DECLIP_CACHE_DIR")
    try:
        if args.validate:
            receipt = json.loads(args.validate.read_text(encoding="utf-8"))
            errors = validation_errors(receipt)
            if isinstance(receipt, dict) and receipt.get("passed") is not True:
                errors.append("receipt is not marked passed")
            print(
                json.dumps(
                    {
                        "receipt": str(args.validate),
                        "passed": not errors,
                        "errors": errors,
                    },
                    indent=2,
                )
            )
            raise SystemExit(bool(errors))
        if not args.clip or not args.manifest:
            parser.error("provide --clip and --manifest, or --validate")
        if Path.cwd().resolve() != ROOT:
            parser.error("run from the worktree root")
        receipt = smoke(args.clip, args.manifest, args.device)
        platform = {"darwin": "macos", "linux": "linux", "windows": "windows"}[
            receipt["os"]
        ]
        path = ROOT / "docs" / "receipts" / f"{platform}-{args.device}.json"
        atomic_write_json(path, receipt)
        print(
            json.dumps(
                {
                    "receipt": str(path.relative_to(ROOT)),
                    "passed": receipt["passed"],
                    "errors": receipt["validation_errors"],
                },
                indent=2,
            )
        )
        raise SystemExit(not receipt["passed"])
    except (
        DeclipError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
