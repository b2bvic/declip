"""Synthetic signal measurements and ffprobe parsing boundaries."""

import json
import math
import subprocess
from pathlib import Path

import pytest

from declip import measure
from declip.contracts import AudioStream, ClipMeasurement, DeclipError, ToolMissing


def wav(make_media, tmp_path, signal, duration=12, codec="pcm_s24le"):
    return make_media(
        tmp_path,
        {
            "name": "sample.wav",
            "inputs": [signal],
            "duration": duration,
            "args": ["-c:a", codec],
        },
    )


@pytest.mark.ffmpeg
@pytest.mark.contract
def test_sine_loudness_within_one_lu(make_media, tmp_path):
    # A 1 kHz sine with peak amplitude 0.1 has RMS -23.01 dBFS.
    # The mono K-weighted reference is approximately -23.05 LUFS.
    path = wav(make_media, tmp_path, "aevalsrc=0.1*sin(2*PI*1000*t):s=48000")
    result = measure.measure_clip(path)
    assert isinstance(result, ClipMeasurement)
    assert result.sample == "sample.wav"
    assert result.window_start == 0 and result.window_seconds == 12
    assert result.has_audio and result.video is None
    assert result.audio == AudioStream(0, "pcm_s24le", 48000, 1, "mono", 24, None)
    assert result.integrated_lufs == pytest.approx(-23.05, abs=1)
    assert result.true_peak_db == pytest.approx(-20, abs=0.1)
    assert result.noise_floor_db == pytest.approx(-23.0103, abs=0.05)
    assert result.lra == pytest.approx(0, abs=0.1)
    assert not result.clipping


@pytest.mark.ffmpeg
def test_known_noise_rms(make_media, tmp_path):
    # Uniform white noise RMS is amplitude / sqrt(3).
    amplitude = math.sqrt(3) * 0.01
    path = wav(
        make_media,
        tmp_path,
        f"anoisesrc=color=white:amplitude={amplitude}:sample_rate=48000:seed=1234",
    )
    result = measure.measure_clip(path)
    assert result.noise_floor_db == pytest.approx(-40, abs=0.3)
    assert not result.clipping


@pytest.mark.ffmpeg
def test_clipped_sine(make_media, tmp_path):
    path = wav(
        make_media,
        tmp_path,
        "aevalsrc=clip(2*sin(2*PI*1000*t)\\,-1\\,1):s=48000",
        codec="pcm_s16le",
    )
    result = measure.measure_clip(path)
    assert result.clipping and result.peak_count >= 3
    assert result.audio.bit_depth == 16


@pytest.mark.ffmpeg
def test_five_second_sample_refused(make_media, tmp_path):
    path = wav(
        make_media, tmp_path, "sine=frequency=1000:sample_rate=48000", duration=5
    )
    with pytest.raises(DeclipError, match=r"sample too short \(minimum 10 s\)"):
        measure.measure_clip(path)


@pytest.mark.ffmpeg
@pytest.mark.contract
def test_no_audio_video(make_media, tmp_path):
    path = make_media(
        tmp_path,
        {
            "name": "silent-video.mp4",
            "inputs": [
                "color=size=32x32:rate=30000/1001,setparams=color_trc=bt709:color_primaries=bt709:colorspace=bt709"
            ],
            "duration": 10,
            "args": [
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-an",
            ],
        },
    )
    result = measure.measure_clip(path)
    assert not result.has_audio and result.audio is None
    assert result.noise_floor_db is None and result.integrated_lufs is None
    assert result.true_peak_db is None and result.lra is None
    assert not result.clipping and result.peak_count == 0
    assert result.video.codec == "h264" and result.video.bit_depth == 8
    assert result.video.fps == "30000/1001" and not result.video.vfr
    assert result.video.trc == "bt709"


@pytest.mark.ffmpeg
def test_middle_sixty_seconds_only(make_media, tmp_path):
    # Loud ends must not contaminate the quiet middle window.
    path = wav(
        make_media,
        tmp_path,
        "aevalsrc=if(between(t\\,5\\,65)\\,0.01\\,0.5)*sin(2*PI*1000*t):s=48000",
        duration=70,
    )
    result = measure.measure_clip(path)
    assert result.window_start == 5 and result.window_seconds == 60
    assert result.true_peak_db == pytest.approx(-40, abs=0.1)
    assert result.integrated_lufs == pytest.approx(-43.05, abs=1)


@pytest.mark.ffmpeg
def test_custom_centered_window(make_media, tmp_path):
    path = wav(
        make_media, tmp_path, "sine=frequency=1000:sample_rate=48000", duration=14
    )
    result = measure.measure_clip(path, seconds=4)
    assert result.window_start == 5 and result.window_seconds == 4


@pytest.mark.ffmpeg
def test_silence_nonfinite_measurements_are_none(make_media, tmp_path):
    path = wav(make_media, tmp_path, "anullsrc=r=48000:cl=mono", duration=10)
    result = measure.measure_clip(path)
    assert result.noise_floor_db is None
    assert result.integrated_lufs is None and result.true_peak_db is None
    assert not result.clipping


@pytest.mark.ffmpeg
def test_fewer_than_ten_finite_blocks(make_media, tmp_path):
    path = wav(
        make_media,
        tmp_path,
        "aevalsrc=if(lt(t\\,0.5)\\,0.1*sin(2*PI*1000*t)\\,0):s=48000",
        duration=10,
    )
    assert measure.measure_clip(path).noise_floor_db is None


@pytest.mark.ffmpeg
def test_nearest_rank_ignores_silent_blocks(make_media, tmp_path):
    # 10 quiet blocks and 80 loud blocks remain after the 10 silent blocks.
    path = wav(
        make_media,
        tmp_path,
        "aevalsrc=if(lt(t\\,1)\\,0\\,if(lt(t\\,2)\\,0.01\\,0.1))*sin(2*PI*1000*t):s=48000",
        duration=10,
    )
    assert measure.measure_clip(path).noise_floor_db == pytest.approx(
        -43.0103, abs=0.05
    )


@pytest.mark.parametrize("seconds", [0, -1, math.inf, math.nan])
def test_invalid_window(seconds):
    with pytest.raises(DeclipError, match="measurement seconds"):
        measure.measure_clip(Path("unused"), seconds=seconds)


def test_missing_tools(monkeypatch):
    monkeypatch.setattr(measure.shutil, "which", lambda name: None)
    with pytest.raises(ToolMissing, match="install ffmpeg"):
        measure.measure_clip(Path("unused"))


@pytest.mark.ffmpeg
def test_invalid_media_error_has_stderr(tmp_path):
    path = tmp_path / "bad.mp4"
    path.write_text("invalid media", encoding="utf-8")
    with pytest.raises(DeclipError, match="sample measurement failed") as error:
        measure.measure_clip(path)
    assert "ffprobe" in str(error.value)


def packet_probe(monkeypatch, packets):
    monkeypatch.setattr(
        measure,
        "_run",
        lambda argv: subprocess.CompletedProcess(
            argv, 0, json.dumps({"packets": packets}), ""
        ),
    )


@pytest.mark.parametrize(
    "fmt,depth",
    [("yuv420p10le", 10), ("p010le", 10), ("yuv422p12le", 12), ("yuv420p", 8)],
)
def test_video_precision_and_snap(monkeypatch, fmt, depth):
    packet_probe(monkeypatch, [{"pts_time": str(i / 30)} for i in range(30)])
    result = measure._video_summary(
        "ffprobe",
        Path("unused"),
        {
            "index": 2,
            "codec_name": "hevc",
            "pix_fmt": fmt,
            "avg_frame_rate": "29.999",
            "r_frame_rate": "30/1",
            "color_transfer": "arib-std-b67",
        },
    )
    assert result.bit_depth == depth and result.fps == "30/1"
    assert not result.vfr and result.trc == "arib-std-b67"


@pytest.mark.parametrize(
    "average,base,packets,expected",
    [
        ("30/1", "60/1", [], True),
        ("0/0", "0/0", [], True),
        ("0/0", "25/1", [], False),
        ("30/1", "30/1", [{"pts_time": str(t)} for t in [0, 0.033, 0.1, 0.133]], True),
    ],
)
def test_vfr_rules(monkeypatch, average, base, packets, expected):
    packet_probe(monkeypatch, packets)
    result = measure._video_summary(
        "ffprobe",
        Path("unused"),
        {
            "index": 0,
            "avg_frame_rate": average,
            "r_frame_rate": base,
        },
    )
    assert result.vfr is expected
    assert result.fps == (
        "30000/1001"
        if average == base == "0/0"
        else "25/1"
        if average == "0/0"
        else "30/1"
    )


@pytest.mark.parametrize(
    "fmt,raw,depth",
    [
        ("s16p", "0", 16),
        ("s32", "24", 24),
        ("fltp", None, 32),
        ("dblp", None, 64),
        ("unknown", None, None),
    ],
)
def test_audio_precision_fallback(fmt, raw, depth):
    result = measure._audio_stream(
        {"index": 1, "sample_fmt": fmt, "bits_per_raw_sample": raw}
    )
    assert result.bit_depth == depth


def test_first_audio_stream_and_cover_art_ignored(monkeypatch, tmp_path):
    monkeypatch.setattr(measure.shutil, "which", lambda name: name)
    packet_probe(monkeypatch, [])
    monkeypatch.setattr(
        measure,
        "_run",
        lambda argv: subprocess.CompletedProcess(
            argv,
            0,
            json.dumps(
                {
                    "format": {"duration": "10"},
                    "streams": [
                        {
                            "index": 0,
                            "codec_type": "video",
                            "disposition": {"attached_pic": 1},
                        },
                        {
                            "index": 1,
                            "codec_type": "audio",
                            "codec_name": "pcm_s16le",
                            "sample_fmt": "s16",
                        },
                        {
                            "index": 2,
                            "codec_type": "audio",
                            "disposition": {"default": 1},
                        },
                    ],
                }
            ),
            "",
        ),
    )
    monkeypatch.setattr(
        measure,
        "_audio_measurements",
        lambda ffmpeg, path, audio, start, duration: (
            None,
            None,
            None,
            None,
            False,
            audio.index,
        ),
    )
    result = measure.measure_clip(tmp_path / "multi.wav")
    assert result.audio.index == 1 and result.video is None


@pytest.mark.ffmpeg
def test_stereo_clipping_measured_before_mono_mix(make_media, tmp_path):
    path = wav(
        make_media,
        tmp_path,
        "aevalsrc=clip(2*sin(2*PI*1000*t)\\,-1\\,1)|-clip(2*sin(2*PI*1000*t)\\,-1\\,1):s=48000",
        duration=10,
    )
    result = measure.measure_clip(path)
    assert result.audio.channels == 2 and result.audio.channel_layout == "stereo"
    assert result.clipping and result.peak_count >= 3
    # Opposite phases cancel to the quantization floor in integer PCM.
    assert result.noise_floor_db is None or result.noise_floor_db < -80


@pytest.mark.ffmpeg
def test_first_audio_even_when_second_is_default(make_media, tmp_path):
    path = make_media(
        tmp_path,
        {
            "name": "tracks.m4a",
            "duration": 10,
            "inputs": [
                "aevalsrc=0.1*sin(2*PI*1000*t):s=48000",
                "aevalsrc=0.5*sin(2*PI*1000*t):s=48000",
            ],
            "args": [
                "-map",
                "0:a",
                "-map",
                "1:a",
                "-c:a",
                "aac",
                "-disposition:a:0",
                "0",
                "-disposition:a:1",
                "default",
            ],
        },
    )
    result = measure.measure_clip(path)
    assert result.audio.index == 0
    assert result.integrated_lufs == pytest.approx(-23.05, abs=1)


@pytest.mark.parametrize("bad_count,expected", [(1, False), (2, True)])
def test_packet_vfr_one_percent_boundary(monkeypatch, bad_count, expected):
    # Exactly 100 deltas, with one or two 50% deviations from the median.
    times = [0.0]
    for i in range(100):
        times.append(times[-1] + (0.06 if i < bad_count else 0.04))
    packet_probe(monkeypatch, [{"pts_time": str(t)} for t in reversed(times)])
    result = measure._video_summary(
        "ffprobe",
        Path("unused"),
        {
            "index": 0,
            "avg_frame_rate": "25/1",
            "r_frame_rate": "25/1",
        },
    )
    assert result.vfr is expected


@pytest.mark.parametrize(
    "peak,count,expected", [(-0.1, 3, True), (-0.10001, 3, False), (0, 2, False)]
)
def test_clipping_boundaries(monkeypatch, peak, count, expected):
    reports = iter(
        [
            f"[astats @ x] Overall\n[astats @ x] Peak level dB: {peak}\n[astats @ x] Peak count: {count}\n"
            '{"input_i": "-inf", "input_tp": "-inf", "input_lra": "-inf"}',
            "",
        ]
    )
    monkeypatch.setattr(
        measure,
        "_run",
        lambda argv: subprocess.CompletedProcess(argv, 0, "", next(reports)),
    )
    values = measure._audio_measurements(
        "ffmpeg",
        Path("unused"),
        AudioStream(1, None, 48000, 1, None, None, None),
        0,
        10,
    )
    assert values == (None, None, None, None, expected, count)


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError("tool disappeared"), subprocess.TimeoutExpired("ffmpeg", 180)],
)
def test_execution_errors_are_declip_errors(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(measure.subprocess, "run", fail)
    with pytest.raises(DeclipError, match="sample measurement failed"):
        measure._run(["ffmpeg"])


def test_missing_loudnorm_report_is_an_error(monkeypatch):
    monkeypatch.setattr(
        measure, "_run", lambda argv: subprocess.CompletedProcess(argv, 0, "", "")
    )
    with pytest.raises(DeclipError, match="loudnorm returned no measurement"):
        measure._audio_measurements(
            "ffmpeg",
            Path("unused"),
            AudioStream(1, None, 48000, 1, None, None, None),
            0,
            10,
        )
