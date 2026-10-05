from pathlib import Path


def test_cut_merge_and_inverse_preserve_unedited_time(tool):
    cuts = tool.merge_cuts([tool.CutRegion(1, 2, "um", 1), tool.CutRegion(1.5, 3, "uh", 1)], 0)
    assert [(cut.start, cut.end) for cut in cuts] == [(1, 3)]
    keeps = tool.invert_cuts(cuts, 5)
    assert [(keep.start, keep.end) for keep in keeps] == [(0, 1), (3, 5)]


def test_synthetic_filler_detection(tool):
    transcript = {"segments": [{"words": [{"word": "um", "start": 1, "end": 1.2, "probability": 1}, {"word": "hello", "start": 1.3, "end": 2}]}]}
    cuts = tool.detect_fillers(transcript, {"um"}, margin_ms=0)
    assert len(cuts) == 1
    assert cuts[0].word == "um"


def test_cli_help(tool):
    from click.testing import CliRunner
    result = CliRunner().invoke(tool.cli, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output
