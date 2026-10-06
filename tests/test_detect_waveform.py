"""Waveform acceptance tests use generated PCM, never transcript timestamps."""

import hashlib
import subprocess
from pathlib import Path

import pytest

from declip.contracts import CutKind, CutProposal, DeclipError, ToolMissing
from declip.detect import detect_waveform_gaps


@pytest.fixture
def waveform(make_media, tmp_path):
    def generate(expression, *, duration, name="silence.wav", args=None):
        return make_media(
            tmp_path,
            {
                "name": name,
                "duration": duration,
                "inputs": [f"aevalsrc={expression}:s=48000:d={duration}"],
                "args": args or ["-c:a", "pcm_s16le"],
            },
        )

    return generate


def gaps(audio, duration, **overrides):
    options = dict(noise_db=-55, max_gap_ms=300, min_silence_ms=450, duration=duration)
    options.update(overrides)
    return detect_waveform_gaps(audio, **options)


def intervals(cuts):
    return [(cut.start, cut.end) for cut in cuts]


@pytest.mark.ffmpeg
@pytest.mark.contract
def test_waveform_gap_detection_uses_audio_silence(waveform):
    audio = waveform(
        r"if(between(t\,1\,2)+between(t\,3\,4)\,0.2*sin(2*PI*440*t)\,0)",
        duration=5,
    )
    original = audio.read_bytes()
    cuts = gaps(audio, 5)
    assert len(cuts) == 3
    for result, expected in zip(intervals(cuts), [(0, 0.85), (2.15, 2.85), (4.15, 5)]):
        assert result == pytest.approx(expected, abs=0.001)
    assert all(isinstance(cut, CutProposal) and cut.kind is CutKind.GAP for cut in cuts)
    assert all(
        cut.confidence == 1 and cut.low_confidence is False and cut.word_indices is None
        for cut in cuts
    )
    assert all(cut.label.startswith("waveform silence") for cut in cuts)
    assert audio.read_bytes() == original
    # Removing these intervals leaves 150 ms before/after sound and 300 ms inside.
    assert cuts[0].end == pytest.approx(0.85, abs=0.001)
    assert cuts[1].end - cuts[1].start == pytest.approx(0.7, abs=0.001)
    assert cuts[2].start == pytest.approx(4.15, abs=0.001)


@pytest.mark.ffmpeg
def test_voiced_pause_is_never_a_waveform_gap(waveform):
    audio = waveform(
        r"if(between(t\,1\,2)\,0.02\,0.2)*sin(2*PI*440*t)",
        duration=3,
    )
    assert gaps(audio, 3) == []


@pytest.mark.ffmpeg
def test_noise_threshold_controls_quiet_waveform_detection(waveform):
    audio = waveform(
        r"if(between(t\,1\,2)\,0.0005\,0.2)*sin(2*PI*440*t)",
        duration=3,
    )
    assert gaps(audio, 3, noise_db=-70) == []
    cuts = gaps(audio, 3, noise_db=-55)
    assert len(cuts) == 1
    assert intervals(cuts)[0] == pytest.approx((1.15, 1.85), abs=0.001)


@pytest.mark.ffmpeg
@pytest.mark.parametrize("leading", [True, False])
def test_edge_silence_keeps_half_max_gap_even_below_full_max_gap(waveform, leading):
    silence = r"lt(t\,0.2)" if leading else r"gt(t\,1.8)"
    audio = waveform(rf"if({silence}\,0\,0.2*sin(2*PI*440*t))", duration=2)
    cuts = gaps(audio, 2, min_silence_ms=100)
    assert len(cuts) == 1
    expected = (0, 0.05) if leading else (1.95, 2)
    assert intervals(cuts)[0] == pytest.approx(expected, abs=0.001)


@pytest.mark.ffmpeg
@pytest.mark.parametrize("silence,found", [(0.449, False), (0.451, True)])
def test_min_silence_threshold(waveform, silence, found):
    audio = waveform(
        rf"if(between(t\,1\,{1 + silence})\,0\,0.2*sin(2*PI*440*t))",
        duration=3,
    )
    cuts = gaps(audio, 3)
    assert bool(cuts) == found
    if found:
        assert intervals(cuts)[0] == pytest.approx(
            (1.15, 1 + silence - 0.15), abs=0.001
        )


@pytest.mark.ffmpeg
@pytest.mark.parametrize("silence,found", [(0.299, False), (0.301, True)])
def test_max_gap_threshold(waveform, silence, found):
    audio = waveform(
        rf"if(between(t\,1\,{1 + silence})\,0\,0.2*sin(2*PI*440*t))",
        duration=3,
    )
    cuts = gaps(audio, 3, min_silence_ms=100)
    assert bool(cuts) == found


@pytest.mark.ffmpeg
def test_one_silent_channel_does_not_cut_sound_in_other_channel(waveform):
    audio = waveform(r"0|0.2*sin(2*PI*440*t)", duration=2)
    assert gaps(audio, 2) == []


@pytest.mark.ffmpeg
def test_explicit_absolute_audio_stream_index(make_media, tmp_path):
    audio = make_media(
        tmp_path,
        {
            "name": "tracks.mka",
            "duration": 3,
            "inputs": [
                "sine=frequency=440:sample_rate=48000",
                r"aevalsrc=if(between(t\,1\,2)\,0\,0.2*sin(2*PI*440*t)):s=48000:d=3",
            ],
            "args": ["-map", "0:a:0", "-map", "1:a:0", "-c:a", "pcm_s16le"],
        },
    )
    assert gaps(audio, 3) == []
    cuts = gaps(audio, 3, audio_index=1)
    assert len(cuts) == 1
    assert intervals(cuts)[0] == pytest.approx((1.15, 1.85), abs=0.001)


@pytest.mark.ffmpeg
def test_container_offset_is_subtracted_from_gap_times(waveform):
    audio = waveform(
        r"if(between(t\,1\,2)\,0\,0.2*sin(2*PI*440*t))",
        duration=5,
        name="offset.mka",
        args=["-af", "asetpts=PTS+2/TB", "-c:a", "pcm_s16le"],
    )
    # Output timestamps stop at 5, so the decoded content covers 3 seconds.
    cuts = gaps(audio, 3)
    assert len(cuts) == 1
    assert intervals(cuts)[0] == pytest.approx((1.15, 1.85), abs=0.001)


@pytest.mark.ffmpeg
def test_all_silence_can_propose_entire_source_removal(waveform):
    audio = waveform("0", duration=2)
    cut = gaps(audio, 2)[0]
    assert (cut.start, cut.end) == (0, 2)
    assert cut.word_indices is None


@pytest.mark.ffmpeg
def test_real_ffmpeg_failure_includes_stderr_tail(tmp_path):
    with pytest.raises(DeclipError, match="Waveform silence detection failed") as error:
        gaps(tmp_path / "missing-input.wav", 2)
    assert "missing-input.wav" in str(error.value)
    assert "No such file or directory" in str(error.value)


@pytest.mark.contract
def test_unterminated_trailing_silence_is_closed_at_duration(monkeypatch):
    def run(argv, **kwargs):
        assert argv[argv.index("-map") + 1] == "0:2"
        assert "asetpts=PTS-STARTPTS" in argv[argv.index("-af") + 1]
        return subprocess.CompletedProcess(
            argv, 0, stderr="[silencedetect] silence_start: 2e0\n"
        )

    monkeypatch.setattr(subprocess, "run", run)
    cuts = gaps(Path("source.wav"), 5, audio_index=2)
    assert cuts[0].start == 2.15 and cuts[0].end == 5


def test_stderr_failure_is_limited_to_last_40_lines(monkeypatch):
    def run(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv, 1, stderr="\n".join(f"diagnostic {i}" for i in range(100))
        )

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(DeclipError) as error:
        gaps(Path("source.wav"), 5)
    assert str(error.value).splitlines()[1:] == [
        f"diagnostic {i}" for i in range(60, 100)
    ]


def test_missing_ffmpeg_raises_tool_error(monkeypatch):
    def run(*args, **kwargs):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ToolMissing, match="install ffmpeg"):
        gaps(Path("source.wav"), 5)


@pytest.mark.parametrize(
    "option,value",
    [
        ("noise_db", float("nan")),
        ("max_gap_ms", -1),
        ("min_silence_ms", float("inf")),
        ("duration", -1),
        ("audio_index", -1),
    ],
)
def test_invalid_waveform_options_fail_before_process(monkeypatch, option, value):
    def run(*args, **kwargs):
        pytest.fail("invalid options must not launch ffmpeg")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(DeclipError, match=option):
        if option == "duration":
            gaps(Path("source.wav"), value)
        else:
            gaps(Path("source.wav"), 5, **{option: value})


@pytest.mark.ffmpeg
def test_waveform_source_hash_is_unchanged(waveform):
    audio = waveform(r"if(between(t\,1\,2)\,0\,0.2*sin(2*PI*440*t))", duration=3)
    before = hashlib.sha256(audio.read_bytes()).hexdigest()
    gaps(audio, 3)
    assert hashlib.sha256(audio.read_bytes()).hexdigest() == before
