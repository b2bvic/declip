"""Backend boundary tests use synthetic text and mocked model libraries only."""

from __future__ import annotations

import hashlib
import json
import sys
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from declip.contracts import (
    BackendSelection,
    FillerFile,
    TranscribeOptions,
    Transcript,
    TranscriberUnavailable,
)
from declip import transcribe
from declip.transcribe import cuda_libs, download
from declip.transcribe.faster import FasterTranscriber
from declip.transcribe.mlx import MlxTranscriber


def wav_file(path, seconds=61, middle=0):
    frames = bytearray(int(16000 * seconds) * 2)
    if middle:
        frames[len(frames) // 2] = middle
    with wave.open(str(path), "wb") as out:
        out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        out.writeframes(frames)
    return path


@pytest.fixture
def libraries(monkeypatch):
    calls, loads = [], []

    class Samples:
        def __init__(self, data):
            self.data = data

        def astype(self, dtype):
            assert dtype == "float32"
            return self

        def __truediv__(self, divisor):
            assert divisor == 32768.0
            return self

    monkeypatch.setitem(
        sys.modules,
        "numpy",
        NS(frombuffer=lambda data, dtype: Samples(data), float32="float32"),
    )

    def mlx_run(path, **kwargs):
        calls.append((path, kwargs))
        return {
            "language": "en",
            "segments": [
                {
                    "id": 7,
                    "start": 1,
                    "end": 2,
                    "text": " Um. ",
                    "words": [
                        {"start": 1, "end": 2, "word": " Um. ", "probability": 0.8}
                    ],
                }
            ],
        }

    class Model:
        def __init__(self, path, **kwargs):
            loads.append((path, kwargs))

        def transcribe(self, path, **kwargs):
            calls.append((path, kwargs))
            return iter(
                [
                    NS(
                        start=1,
                        end=2,
                        text=" uh ",
                        words=[NS(start=1, end=2, word=" uh ", probability=None)],
                    )
                ]
            ), NS(language="en")

    monkeypatch.setitem(sys.modules, "mlx_whisper", NS(transcribe=mlx_run))
    monkeypatch.setitem(sys.modules, "faster_whisper", NS(WhisperModel=Model))
    monkeypatch.setitem(sys.modules, "ctranslate2", NS(get_cuda_device_count=lambda: 0))
    monkeypatch.setattr(cuda_libs, "preload", lambda: (True, "ok"))
    monkeypatch.setattr("declip.transcribe.mlx.ensure_model", lambda name, **k: "local")
    monkeypatch.setattr(
        "declip.transcribe.faster.ensure_model", lambda name, **k: "local"
    )
    monkeypatch.setattr("declip.transcribe.mlx.platform.machine", lambda: "arm64")
    monkeypatch.setattr("declip.transcribe.mlx.sys.platform", "darwin")
    return calls, loads


@pytest.mark.contract
@pytest.mark.parametrize("backend", ["mlx", "faster"])
def test_real_normalization_and_kwargs(backend, libraries, tmp_path):
    calls, loads = libraries
    opts = TranscribeOptions(
        "small",
        language="en",
        initial_prompt="Um, uh.",
        prompt_mode="initial",
        device="metal" if backend == "mlx" else "cpu",
    )
    result = transcribe.get(backend).transcribe(wav_file(tmp_path / "clip.wav"), opts)
    assert result.backend == backend
    assert result.words[0].i == result.words[0].segment == 0
    assert result.words[0].text == ("Um." if backend == "mlx" else "uh")
    assert result.words[0].p == (0.8 if backend == "mlx" else 1.0)
    assert result.segments[0].word_indices == (0,)
    assert calls[0][1]["initial_prompt"] == "Um, uh."
    assert calls[0][1]["word_timestamps"] is True
    assert calls[0][1]["condition_on_previous_text"] is False
    assert calls[0][1]["language"] == "en"
    if backend == "faster":
        assert not isinstance(calls[0][0], str)  # PCM array bypasses the PyAV decoder.
        assert isinstance(calls[0][0].data, bytes)
        assert loads[0][1] == {
            "device": "cpu",
            "compute_type": "int8",
            "local_files_only": True,
        }
        assert calls[0][1]["vad_filter"] is False


@pytest.mark.parametrize("backend", ["mlx", "faster"])
def test_chunked_prompt_every_window_and_offsets(backend, libraries, tmp_path):
    calls, _ = libraries
    result = transcribe.get(backend).transcribe(
        wav_file(tmp_path / "clip.wav"),
        TranscribeOptions(
            "small",
            language="en",
            initial_prompt="um, uh",
            prompt_mode="chunked",
            device="metal" if backend == "mlx" else "cpu",
        ),
    )
    assert len(calls) == 3
    assert all(k["initial_prompt"] == "um, uh" for _, k in calls)
    assert [w.start for w in result.words] == [1, 31, 61]
    assert [w.i for w in result.words] == [0, 1, 2]
    assert [w.segment for w in result.words] == [0, 1, 2]
    assert [s.word_indices for s in result.segments] == [(0,), (1,), (2,)]
    assert not list(tmp_path.glob("declip-chunks-*"))


def test_faster_hotwords(libraries, tmp_path):
    calls, _ = libraries
    FasterTranscriber().transcribe(
        wav_file(tmp_path / "clip.wav"),
        TranscribeOptions(
            "small",
            language="en",
            initial_prompt="um, uh",
            prompt_mode="hotwords",
            device="cpu",
        ),
    )
    assert calls[0][1]["hotwords"] == "um, uh"
    assert calls[0][1]["initial_prompt"] is None


def test_select_priority_and_no_silent_explicit_fallback(libraries, monkeypatch):
    assert transcribe.select().transcriber.name == "mlx"
    assert transcribe.select("cpu").compute_type == "int8"
    with pytest.raises(TranscriberUnavailable, match="CUDA device count"):
        transcribe.select("cuda")
    with pytest.raises(TranscriberUnavailable):
        transcribe.select("cpu", backend="mlx")
    with pytest.raises(TranscriberUnavailable):
        transcribe.select("metal", backend="faster")
    monkeypatch.setitem(sys.modules, "ctranslate2", NS(get_cuda_device_count=lambda: 1))
    assert transcribe.select("auto", backend="faster").device == "cuda"
    monkeypatch.setattr(
        MlxTranscriber, "available", lambda self: (False, "missing mlx")
    )
    assert transcribe.select().device == "cuda"
    with pytest.raises(TranscriberUnavailable, match="missing mlx"):
        transcribe.select(backend="mlx")
    monkeypatch.setattr(cuda_libs, "preload", lambda: (False, "missing CUDA library"))
    assert transcribe.select().device == "cpu"
    with pytest.raises(TranscriberUnavailable, match="missing CUDA library"):
        transcribe.select("cuda")


def test_unknown_and_fake_env_only(tmp_path, monkeypatch):
    for name in ("unknown", "fake"):
        with pytest.raises(TranscriberUnavailable):
            transcribe.get(name)
    monkeypatch.setattr(MlxTranscriber, "available", lambda self: (False, "missing"))
    monkeypatch.setattr(FasterTranscriber, "available", lambda self: (False, "missing"))
    with pytest.raises(TranscriberUnavailable):
        transcribe.select()
    monkeypatch.setenv("DECLIP_TRANSCRIBER", "fake")
    monkeypatch.setenv("DECLIP_FAKE_TRANSCRIPT_DIR", str(tmp_path))
    (tmp_path / "clip.json").write_text(
        json.dumps(
            {
                "language": "en",
                "segments": [
                    {
                        "start": 0,
                        "end": 1,
                        "text": "um",
                        "words": [{"start": 0, "end": 1, "word": "um"}],
                    }
                ],
            }
        )
    )
    selection = transcribe.select()
    result = transcribe.transcribe_file(
        tmp_path / "clip.mp4",
        TranscribeOptions("test"),
        selection=selection,
        cache_dir=tmp_path,
        audio_index=None,
        fillers_for=lambda _: None,
    )
    assert result.backend == "fake"
    assert result.words[0].p == 1
    monkeypatch.delenv("DECLIP_TRANSCRIBER")
    with pytest.raises(TranscriberUnavailable):
        selection.transcriber.transcribe(Path("clip.wav"), TranscribeOptions("test"))


@pytest.mark.contract
def test_cache_key_all_fields_and_full_middle(tmp_path):
    opts = TranscribeOptions("small")
    paths = [wav_file(tmp_path / f"{i}.wav", 10, i) for i in (0, 1)]
    blobs = [p.read_bytes() for p in paths]
    assert blobs[0][:65536] == blobs[1][:65536]
    assert blobs[0][-65536:] == blobs[1][-65536:]
    digests = [hashlib.sha256(b).hexdigest() for b in blobs]
    assert transcribe.cache_key(digests[0], "mlx", opts) != transcribe.cache_key(
        digests[1], "mlx", opts
    )
    baseline = transcribe.cache_key(digests[0], "mlx", opts)
    for field, value in {
        "model": "large-v3",
        "language": "en",
        "initial_prompt": "um",
        "prompt_mode": "chunked",
        "condition_on_previous_text": True,
        "compute_type": "int8",
    }.items():
        assert (
            transcribe.cache_key(digests[0], "mlx", replace(opts, **{field: value}))
            != baseline
        )
    assert transcribe.cache_key(digests[0], "faster", opts) != baseline
    monkey_schema = transcribe.SCHEMA_VERSION
    try:
        transcribe.SCHEMA_VERSION = 999
        assert transcribe.cache_key(digests[0], "mlx", opts) != baseline
    finally:
        transcribe.SCHEMA_VERSION = monkey_schema


@pytest.mark.parametrize("language", ["en", "de"])
def test_language_two_pass_and_prompt_cache(tmp_path, monkeypatch, language):
    src = wav_file(tmp_path / "source.wav")
    calls = []

    class Backend:
        name = "mlx"

        def transcribe(self, path, opts):
            with wave.open(str(path), "rb") as w:
                calls.append((w.getnframes() / w.getframerate(), opts))
            return Transcript("mlx", opts.model, language, opts.initial_prompt, (), ())

    monkeypatch.setattr(
        transcribe,
        "extract_transcription_audio",
        lambda source, out_wav, **k: Path(
            __import__("shutil").copyfile(source, out_wav)
        ),
    )
    filler = FillerFile("en", "um, uh", frozenset({"um"}), frozenset())

    def run():
        return transcribe.transcribe_file(
            src,
            TranscribeOptions("small"),
            selection=BackendSelection(Backend(), "metal", "auto"),
            cache_dir=tmp_path / "cache",
            audio_index=None,
            fillers_for=lambda lang: filler if lang == "en" else None,
        )

    assert run().language == language
    assert calls[0][0] == 30
    assert calls[0][1].language is None and calls[0][1].initial_prompt is None
    assert calls[1][0] == 61 and calls[1][1].language == language
    assert calls[1][1].initial_prompt == ("um, uh" if language == "en" else None)
    run()
    assert len(calls) == 2
    src.write_bytes(src.read_bytes()[:100000] + b"\0\1" + src.read_bytes()[100002:])
    run()
    assert len(calls) == 4


@pytest.fixture
def hub(monkeypatch, tmp_path):
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise OSError("not cached")
        return str(tmp_path / "model")

    def model_info(repo, **kwargs):
        calls.append("metadata")
        return NS(siblings=[NS(size=100), NS(size=200)])

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        NS(snapshot_download=snapshot, HfApi=lambda: NS(model_info=model_info)),
    )
    monkeypatch.setitem(
        sys.modules, "huggingface_hub.constants", NS(HF_HUB_CACHE=str(tmp_path / "hub"))
    )
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return calls, snapshot


def test_download_notice_precedes_download(hub, monkeypatch, capsys):
    calls, snapshot = hub

    def checked(**kwargs):
        if not kwargs.get("local_files_only"):
            notice = capsys.readouterr().err
            assert (
                "300 bytes" in notice and "huggingface.co" in notice and "hub" in notice
            )
            assert "mobiuslabsgmbh/faster-whisper-large-v3-turbo" in notice
        return snapshot(**kwargs)

    monkeypatch.setattr(sys.modules["huggingface_hub"], "snapshot_download", checked)
    download.ensure_model("mlx-community/whisper-large-v3-turbo", backend="faster")
    assert calls[0]["local_files_only"] is True
    assert calls[1] == "metadata"


def test_offline_no_network_and_cached_weights(hub, monkeypatch, tmp_path):
    calls, _ = hub
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(TranscriberUnavailable, match="large-v3-turbo.*HF_HUB_OFFLINE"):
        download.ensure_model("large-v3-turbo", backend="faster")
    assert len(calls) == 1 and calls[0]["local_files_only"] is True
    root = tmp_path / "cached"
    root.mkdir()
    monkeypatch.setattr(
        sys.modules["huggingface_hub"], "snapshot_download", lambda **k: str(root)
    )
    with pytest.raises(TranscriberUnavailable, match="not cached"):
        download.ensure_model("small", backend="mlx")
    (root / "config.json").write_text("{}")
    (root / "weights.npz").write_bytes(b"weights")
    assert download.ensure_model("small", backend="mlx") == str(root)


def test_local_model_bypasses_hub(hub, tmp_path):
    calls, _ = hub
    assert download.ensure_model(str(tmp_path), backend="faster") == str(tmp_path)
    assert calls == []


def test_download_metadata_failure_still_notices(hub, monkeypatch, capsys):
    monkeypatch.setattr(
        sys.modules["huggingface_hub"],
        "HfApi",
        lambda: (_ for _ in ()).throw(OSError()),
    )
    download.ensure_model("small", backend="mlx")
    assert "size unknown" in capsys.readouterr().err


@pytest.mark.extra("cpu")
def test_cpu_extra_available():
    ok, reason = transcribe.get("faster").available()
    if not ok:
        pytest.skip(reason)
    assert ok


@pytest.mark.ffmpeg
def test_extraction_real_pcm_mapping(tmp_path, make_media):
    src = make_media(
        tmp_path,
        {
            "name": "stereo.wav",
            "inputs": ["sine=frequency=500:sample_rate=48000"],
            "duration": 1,
            "args": ["-ac", "2", "-c:a", "pcm_s24le"],
        },
    )
    path = transcribe.extract_transcription_audio(
        src, tmp_path / "transcribe.wav", audio_index=0
    )
    with wave.open(str(path), "rb") as w:
        assert (
            w.getnchannels(),
            w.getframerate(),
            w.getsampwidth(),
            w.getnframes(),
        ) == (1, 16000, 2, 16000)


def test_cuda_preload_order_and_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(cuda_libs.sys, "platform", "linux")
    monkeypatch.setattr(cuda_libs.sys, "path", [str(tmp_path)])
    paths = []
    for name in (
        "libcublasLt.so.12",
        "libcublas.so.12",
        "libcudnn.so.9",
        "libcudnn_ops.so.9",
    ):
        p = tmp_path / "nvidia" / "test" / "lib" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    monkeypatch.setattr(cuda_libs, "_LOADED", set())
    monkeypatch.setattr(cuda_libs, "_HANDLES", [])
    monkeypatch.setattr(
        cuda_libs.ctypes, "CDLL", lambda path, **k: paths.append(Path(path).name)
    )
    assert cuda_libs.preload()[0]
    assert paths == [
        "libcublasLt.so.12",
        "libcublas.so.12",
        "libcudnn.so.9",
        "libcudnn_ops.so.9",
    ]
    assert cuda_libs.preload()[0] and len(paths) == 4
    monkeypatch.setattr(cuda_libs, "_LOADED", set())
    monkeypatch.setattr(
        cuda_libs.ctypes,
        "CDLL",
        lambda *a, **k: (_ for _ in ()).throw(OSError("broken")),
    )
    assert cuda_libs.preload() == (False, "CUDA library preload failed: broken")


def test_prompt_probe_recall_one_to_one_and_real_count_rule():
    import importlib.util
    from declip.contracts import Word

    spec = importlib.util.spec_from_file_location(
        "prompt_probe", Path(__file__).parents[1] / "scripts" / "prompt_probe.py"
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    transcript = Transcript(
        "mlx",
        "test",
        "en",
        None,
        (
            Word(0, 1, 1.4, "um,", 1, 0),
            Word(1, 31, 31.4, "Uh.", 1, 0),
            Word(2, 40, 40.2, "um", 1, 0),
        ),
        (),
    )
    labels = {
        "fillers": [
            {"label": "um", "start": 0.8, "end": 1.1},
            {"label": "um", "start": 0.9, "end": 1.2},
            {"label": "uh", "start": 31, "end": 31.5},
        ]
    }
    scores = probe.recall(transcript, labels)
    assert scores["first_30"]["recall"] == 0.5
    assert scores["after_30"]["recall"] == 1
    assert scores["false_hits"] == 1
    windows = probe.window_counts(transcript, 60)
    assert windows == [
        {"start": 0, "end": 30, "count": 1},
        {"start": 30, "end": 60, "count": 2},
    ]
    assert probe.real_pass(windows, windows) == "passed"
    assert (
        probe.real_pass([{**windows[0], "count": 10}, windows[1]], windows) == "failed"
    )


def test_missing_ffmpeg_and_source_overwrite(tmp_path, monkeypatch):
    from declip.contracts import ToolMissing

    source = wav_file(tmp_path / "source.wav", 1)
    with pytest.raises(TranscriberUnavailable, match="differ"):
        transcribe.extract_transcription_audio(source, source, audio_index=None)
    monkeypatch.setattr(transcribe.shutil, "which", lambda name: None)
    with pytest.raises(ToolMissing, match="ffmpeg"):
        transcribe.extract_transcription_audio(
            source, tmp_path / "output.wav", audio_index=None
        )


@pytest.fixture
def prompt_probe():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "prompt_probe", Path(__file__).parents[1] / "scripts" / "prompt_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "prompt,baseline,expected",
    [
        ([0, 0], [0, 0], "inconclusive"),
        ([1, 1], [0, 0], "inconclusive"),
        ([0, 0], [1, 1], "failed"),
        ([1, 1], [1, 1], "passed"),
        ([1, 1], [2, 2], "failed"),
        ([4, 1], [1, 1], "failed"),
    ],
)
def test_real_speech_status(prompt_probe, prompt, baseline, expected):
    def windows(counts):
        return [
            {"start": i * 30, "end": (i + 1) * 30, "count": count}
            for i, count in enumerate(counts)
        ]

    assert prompt_probe.real_pass(windows(prompt), windows(baseline)) == expected


def test_real_speech_partial_window_normalization(prompt_probe):
    windows = [
        {"start": 0, "end": 30, "count": 4},
        {"start": 30, "end": 45, "count": 1},
    ]
    assert prompt_probe.real_pass(windows, windows) == "passed"


@pytest.mark.parametrize("backend", ["mlx", "faster"])
@pytest.mark.parametrize("synthetic_persists", [False, True])
@pytest.mark.parametrize(
    "prompt_counts,baseline_counts,status",
    [
        ([0, 0], [0, 0], "inconclusive"),
        ([1, 1], [0, 0], "inconclusive"),
        ([1, 1], [1, 1], "passed"),
        ([1, 0], [1, 1], "failed"),
    ],
)
def test_probe_real_receipt_and_verdict(
    prompt_probe,
    tmp_path,
    monkeypatch,
    backend,
    synthetic_persists,
    prompt_counts,
    baseline_counts,
    status,
):
    from declip.contracts import Word

    clip = wav_file(tmp_path / "clip.wav", 60)
    real = wav_file(tmp_path / "real.wav", 60)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "clip_sha256": prompt_probe.file_hash(clip),
                "duration": 60,
                "fillers": [
                    {"label": "um", "start": 1, "end": 1.2},
                    {"label": "um", "start": 31, "end": 31.2},
                ],
            }
        )
    )
    calls = []

    def run(path, opts):
        calls.append((path.name, opts.prompt_mode, bool(opts.initial_prompt)))
        if path.name == "real.wav":
            counts = prompt_counts if opts.initial_prompt else baseline_counts
        elif not opts.initial_prompt:
            counts = [0, 0]
        else:
            counts = [1, int(synthetic_persists or opts.prompt_mode != "initial")]
        words = tuple(
            Word(i, start, start + 0.2, "um", 1, 0)
            for i, start in enumerate(
                window * 30 + 1 + j
                for window, count in enumerate(counts)
                for j in range(count)
            )
        )
        return Transcript(backend, "test", "en", None, words, ())

    monkeypatch.setattr(
        prompt_probe,
        "select",
        lambda *a, **k: BackendSelection(
            NS(name=backend, transcribe=run), "cpu", "int8"
        ),
    )
    monkeypatch.setattr(
        prompt_probe, "extract_transcription_audio", lambda path, *a, **k: path
    )
    report = prompt_probe.probe(clip, manifest, backend, "cpu", "test", real)
    # Check the serialized receipt, not just the result helper.
    receipt = json.loads(json.dumps(report))
    assert receipt["real"]["status"] == "completed"
    assert receipt["real"]["initial_status"] == status
    assert "initial_passed" not in receipt["real"]
    fallback = "chunked" if backend == "mlx" else "hotwords"
    expected_mode = (
        "initial"
        if status == "passed" or (status == "inconclusive" and synthetic_persists)
        else fallback
    )
    assert receipt["auto_prompt_mode"] == expected_mode
    if status == "failed":
        assert receipt["real"][fallback + "_status"] == "failed"
        assert fallback in receipt["synthetic"]
    else:
        assert not any(
            name == "real.wav" and mode == fallback for name, mode, _ in calls
        )


def test_probe_make_clip_timing_with_mocked_say(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "prompt_probe_make", Path(__file__).parents[1] / "scripts" / "prompt_probe.py"
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    phrase = [""]

    def run(argv, **kwargs):
        if argv[0] == "say":
            phrase[0] = argv[-1]
        else:
            duration = 0.3 if phrase[0] in {"um", "uh"} else 1.5
            with wave.open(argv[-1], "wb") as output:
                output.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
                output.writeframes(b"\0\0" * int(48000 * duration))

    monkeypatch.setattr(probe.sys, "platform", "darwin")
    monkeypatch.setattr(probe.subprocess, "run", run)
    report = probe.make_clip(tmp_path / "fixture", "test")
    assert report["early_labels"] >= 6 and report["late_labels"] >= 12
    assert report["duration"] >= 120
    manifest = json.loads((tmp_path / "fixture" / "manifest.json").read_text())
    assert manifest["license"] == "synthetic, generated locally with macOS `say`"
    assert manifest["clip_sha256"] == probe.file_hash(tmp_path / "fixture" / "clip.wav")
    assert all(
        abs(b["start"] - a["end"] - 0.4) < 1e-6
        for a, b in zip(manifest["phrases"], manifest["phrases"][1:])
    )


def test_probe_dataless_detection():
    import importlib.util
    import stat

    spec = importlib.util.spec_from_file_location(
        "prompt_probe_dataless",
        Path(__file__).parents[1] / "scripts" / "prompt_probe.py",
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    placeholder = NS(
        stat=lambda: NS(
            st_flags=getattr(stat, "SF_DATALESS", 1 << 30), st_size=100, st_blocks=0
        )
    )
    # SF_DATALESS exists on Darwin. The zero-block fallback covers flag transitions.
    if sys.platform == "darwin":
        assert probe.is_dataless(placeholder)
        assert probe.is_dataless(
            NS(stat=lambda: NS(st_flags=0, st_size=100, st_blocks=0))
        )
    assert not probe.is_dataless(
        NS(stat=lambda: NS(st_flags=0, st_size=100, st_blocks=8))
    )
