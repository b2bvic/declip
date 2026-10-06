"""Hand-authored transcript expectations for context and speech preservation."""

import copy
import json
import os
from dataclasses import fields
from pathlib import Path

import pytest

from declip.contracts import CutKind, CutProposal, DeclipError, FillerFile, Word
from declip.detect import detect_fillers, detect_retakes, legacy_words
from declip.fillers import SHIPPED_LANGUAGES, load_fillers, parse_filler_file

FIXTURES = Path(__file__).parent / "fixtures" / "transcripts"
GOLDENS = Path(__file__).parent / "golden"


def word(i, text, start, end, p=1.0, segment=0):
    return Word(i, start, end, text, p, segment)


def english():
    return load_fillers("en", config_dir=Path(os.environ["DECLIP_CONFIG_DIR"]))


def filler_cuts(words, *, margin_ms=0, min_confidence=0.5, duration=60):
    return detect_fillers(
        words,
        english(),
        margin_ms=margin_ms,
        min_confidence=min_confidence,
        duration=duration,
    )


@pytest.mark.contract
@pytest.mark.parametrize(
    "name",
    [
        "fillers_basic",
        "context_like",
        "two_word",
        "segment_boundary",
        "short_word_between",
        "low_confidence",
        "retake",
        "de_unsupported",
    ],
)
def test_golden_cut_lists(name, request):
    fixture = json.loads((FIXTURES / f"{name}.json").read_text())
    words = legacy_words(fixture)
    before = copy.deepcopy(words)
    if name == "retake":
        cuts = detect_retakes(words, min_confidence=0.5)
    else:
        fillers = load_fillers(
            fixture["language"],
            config_dir=Path(os.environ["DECLIP_CONFIG_DIR"]),
        )
        cuts = (
            []
            if fillers is None
            else detect_fillers(
                words,
                fillers,
                min_confidence=0.5,
                margin_ms=120,
                duration=fixture["duration"],
            )
        )
    actual = [
        dict(
            kind=cut.kind.value,
            start=cut.start,
            end=cut.end,
            label=cut.label,
            low_confidence=cut.low_confidence,
        )
        for cut in cuts
    ]
    expected = json.loads((GOLDENS / f"{name}.cuts.json").read_text())
    try:
        assert len(actual) == len(expected)
        for result, golden in zip(actual, expected):
            assert result["start"] == pytest.approx(golden["start"], abs=0.001)
            assert result["end"] == pytest.approx(golden["end"], abs=0.001)
            assert {k: v for k, v in result.items() if k not in ("start", "end")} == {
                k: v for k, v in golden.items() if k not in ("start", "end")
            }
    except AssertionError:
        if request.config.getoption("--update-golden"):
            (GOLDENS / f"{name}.cuts.json.actual").write_text(
                json.dumps(actual, indent=2) + "\n",
                encoding="utf-8",
            )
        raise
    assert words == before
    assert all(isinstance(cut, CutProposal) for cut in cuts)
    assert "status" not in {field.name for field in fields(CutProposal)}


@pytest.mark.parametrize("token", ["so", "well", "like"])
@pytest.mark.parametrize("gap,expected", [(0.299, False), (0.3, False), (0.301, True)])
def test_sentence_start_threshold_is_strict_and_ignores_segments(token, gap, expected):
    words = [
        word(0, "continues", 0.0, 1.0),
        word(1, token, 1.0 + gap, 1.5, segment=1),
        word(2, "today", 1.55, 1.8, segment=1),
    ]
    assert bool(filler_cuts(words)) == expected


@pytest.mark.parametrize("token", ["like", "right", "actually"])
@pytest.mark.parametrize("gap,expected", [(0.199, False), (0.2, False), (0.201, True)])
def test_following_pause_threshold_is_strict(token, gap, expected):
    words = [
        word(0, "continues", 0.0, 0.3),
        word(1, token, 0.35, 1.0),
        word(2, "today", 1.0 + gap, 1.5),
    ]
    assert bool(filler_cuts(words)) == expected


@pytest.mark.parametrize("before", [0.299, 0.3, 0.301])
@pytest.mark.parametrize("after", [0.199, 0.2, 0.201])
@pytest.mark.parametrize("phrase", ["you know", "kind of", "i mean", "sort of"])
def test_two_word_pause_conditions_both_required(before, after, phrase):
    first, second = phrase.split()
    words = [
        word(0, "continues", 0.0, 1.0),
        word(1, first, 1.0 + before, 1.5, p=0.9),
        word(2, second, 1.55, 2.0, p=0.4),
        word(3, "today", 2.0 + after, 2.5),
    ]
    cuts = filler_cuts(words)
    assert bool(cuts) == (before > 0.3 and after > 0.2)
    if cuts:
        assert cuts == [
            CutProposal(CutKind.FILLER, 1.0 + before, 2.0, phrase, 0.4, (1, 2), True)
        ]


@pytest.mark.parametrize("previous", [None, "before.", "before!", "before?", "before,"])
@pytest.mark.parametrize("ending", [",", ".", "!", "?"])
def test_two_word_punctuation_and_first_word(previous, ending):
    words = [] if previous is None else [word(0, previous, 0, 0.2)]
    i = len(words)
    words += [
        word(i, "YOU", 0.25, 0.4),
        word(i + 1, f"know{ending}", 0.45, 0.6),
        word(i + 2, "today", 0.65, 0.8),
    ]
    assert filler_cuts(words)[0].label == "you know"


def test_two_word_last_word_and_cross_segment_match():
    words = [
        word(0, "ends.", 0, 0.3),
        word(1, "you", 0.35, 0.5),
        word(2, "know", 0.55, 0.7, segment=1),
    ]
    assert filler_cuts(words)[0].word_indices == (1, 2)


@pytest.mark.parametrize("probability,low", [(0.2, True), (0.5, False), (0.9, False)])
def test_confidence_is_a_flag_and_never_filters(probability, low):
    cuts = filler_cuts([word(0, "um", 0.5, 0.8, probability)])
    assert cuts == [
        CutProposal(CutKind.FILLER, 0.5, 0.8, "um", probability, (0, 0), low)
    ]


def test_merged_confidence_is_minimum_and_indices_are_inclusive():
    cuts = filler_cuts(
        [word(0, "UM,", 0.5, 0.7, 0.9), word(1, "uh", 0.75, 0.9, 0.2)], margin_ms=120
    )
    assert cuts == [
        CutProposal(CutKind.FILLER, 0.38, 1.02, "um, uh", 0.2, (0, 1), True)
    ]


def test_margins_clamp_to_neighbor_midpoints_and_duration():
    words = [
        word(0, "before", 0, 0.3),
        word(1, "um", 0.4, 0.6),
        word(2, "after", 0.7, 1.0),
    ]
    cut = filler_cuts(words, margin_ms=500)[0]
    assert (cut.start, cut.end) == pytest.approx((0.35, 0.65))
    boundary = filler_cuts([word(0, "uh", 0.05, 0.95)], margin_ms=500, duration=1)[0]
    assert (boundary.start, boundary.end) == (0, 1)


def test_overlapping_timestamp_words_are_protected_by_midpoints():
    words = [
        word(0, "before", 0, 0.5),
        word(1, "um", 0.4, 0.8),
        word(2, "after", 0.7, 1),
    ]
    cut = filler_cuts(words, margin_ms=500)[0]
    assert (cut.start, cut.end) == pytest.approx((0.45, 0.75))


def test_no_merge_across_context_kept_filler_word():
    words = [
        word(0, "um", 0.2, 0.4),
        word(1, "like", 0.45, 0.6),
        word(2, "uh", 0.65, 0.8),
    ]
    cuts = filler_cuts(words, margin_ms=500)
    assert len(cuts) == 2
    assert cuts[0].word_indices == (0, 0)
    assert cuts[1].word_indices == (2, 2)
    assert cuts[0].end <= 0.45 and cuts[1].start >= 0.6


def test_empty_and_invalid_duration_words():
    assert filler_cuts([]) == []
    assert filler_cuts([word(0, "", 0, 1), word(1, "um", 1, 1)]) == []
    assert detect_retakes([], min_confidence=0.5) == []
    assert legacy_words({}) == []


@pytest.mark.contract
def test_legacy_adapter_preserves_probability_and_segment_position():
    transcript = {
        "segments": [
            {
                "id": 20,
                "words": [
                    {"word": " Um, ", "start": 0.5, "end": 0.7, "probability": 0.0}
                ],
            },
            {
                "words": [
                    {"word": " speech ", "start": 0.8, "end": 1.0},
                    {"word": "uh", "start": 1.5, "end": 1.7, "probability": None},
                ]
            },
        ]
    }
    before = copy.deepcopy(transcript)
    assert legacy_words(transcript) == [
        word(0, "Um,", 0.5, 0.7, 0),
        word(1, "speech", 0.8, 1.0, segment=1),
        word(2, "uh", 1.5, 1.7, segment=1),
    ]
    assert transcript == before


def test_only_english_is_shipped_and_custom_language_uses_its_own_file(tmp_path):
    assert SHIPPED_LANGUAGES == ("en",)
    assert load_fillers("de", config_dir=tmp_path) is None
    custom = tmp_path / "fillers"
    custom.mkdir()
    (custom / "de.txt").write_text("# prompt: äh\näh\n", encoding="utf-8")
    fillers = load_fillers("de", config_dir=tmp_path)
    assert fillers == FillerFile("de", "äh", frozenset({"äh"}), frozenset())
    assert (
        detect_fillers(
            [word(0, "Äh,", 0.5, 0.7)],
            fillers,
            min_confidence=0.5,
            margin_ms=0,
            duration=2,
        )[0].label
        == "äh"
    )


def test_custom_filler_file_keeps_format():
    fillers = parse_filler_file("# prompt: pause\nhesitation\noh dear\n", language="en")
    assert (
        detect_fillers(
            [word(0, "hesitation", 0.5, 0.7)],
            fillers,
            min_confidence=0.5,
            margin_ms=0,
            duration=2,
        )[0].label
        == "hesitation"
    )


def repeated_words(gap=0.5, punctuation=True):
    first = ["We", "can", "begin." if punctuation else "begin"]
    return [word(i, text, i * 0.2, i * 0.2 + 0.15) for i, text in enumerate(first)] + [
        word(i + 3, text, 0.55 + gap + i * 0.2, 0.7 + gap + i * 0.2, segment=1)
        for i, text in enumerate(["we", "can", "BEGIN!"])
    ]


@pytest.mark.contract
def test_retakes_cut_earlier_take_with_similarity_confidence():
    words = repeated_words()
    before = copy.deepcopy(words)
    assert detect_retakes(words, min_confidence=0.5) == [
        CutProposal(
            CutKind.RETAKE, 0.0, 0.55, "retake (100% match)", 1.0, (0, 2), False
        ),
    ]
    assert words == before


@pytest.mark.parametrize("gap,found", [(14.999, True), (15.0, True), (15.001, False)])
def test_retake_window_uses_end_of_earlier_take(gap, found):
    assert bool(detect_retakes(repeated_words(gap), min_confidence=0.5)) == found


@pytest.mark.parametrize("gap,found", [(0.999, False), (1.0, False), (1.001, True)])
def test_retake_chunk_pause_threshold_and_segment_boundary(gap, found):
    assert bool(detect_retakes(repeated_words(gap, False), min_confidence=0.5)) == found


def test_retakes_compare_word_tokens_not_characters():
    words = [
        word(i, text, i * 0.2, i * 0.2 + 0.15)
        for i, text in enumerate(
            [
                "the",
                "cat",
                "sits.",
                "the",
                "bat",
                "sits!",
            ]
        )
    ]
    cuts = detect_retakes(words, min_confidence=0.8, similarity_threshold=0.6)
    assert len(cuts) == 1
    assert cuts[0].confidence == pytest.approx(2 / 3)
    assert cuts[0].low_confidence is True
    assert detect_retakes(words, min_confidence=0.5, similarity_threshold=0.7) == []


def test_repeated_three_takes_cut_first_two_once_each():
    words = [
        word(i, text, i * 0.2, i * 0.2 + 0.15)
        for i, text in enumerate(
            [
                "we",
                "can",
                "begin.",
                "we",
                "can",
                "begin!",
                "we",
                "can",
                "begin?",
            ]
        )
    ]
    cuts = detect_retakes(words, min_confidence=0.5)
    assert [cut.word_indices for cut in cuts] == [(0, 2), (3, 5)]


def test_retakes_keep_short_fragments_and_long_sentences_are_not_split_at_15_words():
    assert (
        detect_retakes(
            [word(i, "yes.", i, i + 0.2) for i in range(3)], min_confidence=0.5
        )
        == []
    )
    text = ["a"] * 19 + ["sentence."]
    words = [
        word(i, token, i * 0.1, i * 0.1 + 0.05) for i, token in enumerate(text + text)
    ]
    cuts = detect_retakes(words, min_confidence=0.5)
    assert len(cuts) == 1 and cuts[0].word_indices == (0, 19)


@pytest.mark.parametrize(
    "option,value",
    [("margin_ms", -1), ("margin_ms", float("nan")), ("duration", float("inf"))],
)
def test_invalid_filler_options_fail(option, value):
    options = {"margin_ms": 120, "duration": 2, "min_confidence": 0.5}
    options[option] = value
    with pytest.raises(DeclipError, match=option):
        detect_fillers([], english(), **options)
