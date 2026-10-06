"""Compatibility assertions that remain after the CLI module migration."""

import json

import pytest
from click.testing import CliRunner

from declip import cli, media, transcribe
from declip.contracts import AudioStream, ColorTags, MediaInfo, Transcript
from declip.ui import Spinner, echo, progress_tick, ui_done, ui_warn


@pytest.mark.parametrize(
    "arguments",
    [
        ["--json-output", "process"],
        ["--json-output", "detect"],
        ["transcribe"],
    ],
)
def test_cli_json_stdout_has_no_diagnostics(arguments, tmp_path, monkeypatch):
    source = tmp_path / "clip.wav"
    source.touch()
    info = MediaInfo(
        duration=5,
        start_time=0,
        container="wav",
        has_video=False,
        video_index=None,
        audio_index=0,
        fps=None,
        vfr=False,
        width=None,
        height=None,
        rotation=0,
        vcodec=None,
        pix_fmt=None,
        bit_depth=None,
        video_bitrate=None,
        color=ColorTags(None, None, None, None),
        timecode=None,
        audio_streams=(AudioStream(0, "pcm_s16le", 48000, 1, "mono", 16, None),),
        dropped_streams=(),
    )
    monkeypatch.setattr(media, "probe_media", lambda path: info)
    monkeypatch.setenv("DECLIP_TRANSCRIBER", "fake")
    monkeypatch.setattr(cli.detect, "detect_waveform_gaps", lambda *args, **kwargs: [])

    def transcript(*args, **kwargs):
        echo("transcription progress")
        return Transcript("fake", "fixture", "en", None, (), ())

    monkeypatch.setattr(transcribe, "transcribe_file", transcript)
    result = CliRunner().invoke(cli.cli, [*arguments, str(source)])
    assert result.exit_code == 0, result.output
    assert isinstance(json.loads(result.stdout), dict)
    assert "transcription progress" in result.stderr
    assert "\x1b" not in result.stdout + result.stderr


def test_ui_plain_stderr_and_no_color(capsys, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    with Spinner("processing"):
        ui_done("done")
        ui_warn("check")
        progress_tick(0, 1, "work", {100})
    captured = capsys.readouterr()
    assert not captured.out
    assert "processing" in captured.err and "Warning: check" in captured.err
    assert "\x1b" not in captured.err
