import json

import pytest

from declip import audiochain, fillers, paths
from declip.contracts import DeclipError, FilterChainRejected, Loudness, Tier


@pytest.mark.parametrize(
    ("chain", "body", "target"),
    [
        ("", "", None),
        ("highpass=f=80", "highpass=f=80", None),
        ("loudnorm", "", Loudness(-16, -1.5, 11)),
        (
            "highpass=f=80,loudnorm=I=-23:TP=-2.5:LRA=5",
            "highpass=f=80",
            Loudness(-23, -2.5, 5),
        ),
        ("highpass=f=80,loudnorm=i=-14", "highpass=f=80", Loudness(-14, -1.5, 11)),
        (
            "highpass=f=80; loudnorm@final=I=-16",
            "highpass=f=80",
            Loudness(-16, -1.5, 11),
        ),
    ],
)
def test_split_loudnorm(chain, body, target):
    assert audiochain.split_loudnorm(chain) == (body, target)


@pytest.mark.parametrize(
    "chain",
    ["loudnorm,loudnorm", "loudnorm,highpass=f=80", "loudnorm=I=nan", "loudnorm=I=bad"],
)
def test_invalid_loudnorm_rejected(chain):
    with pytest.raises(FilterChainRejected):
        audiochain.split_loudnorm(chain)


@pytest.mark.parametrize(
    "name", ["movie", "amovie", "sendcmd", "asendcmd", "ladspa", "lv2"]
)
@pytest.mark.parametrize(
    "form",
    [
        "{name}=file=x",
        "[in]{name}@named=file=x[out]",
        "highpass=f=80; [a][b]{name}=file=x",
    ],
)
def test_denied_filter_names(name, form):
    with pytest.raises(FilterChainRejected, match=name):
        audiochain.validate_filter_chain(form.format(name=name))


@pytest.mark.parametrize("name", ["metadata", "ametadata"])
def test_metadata_file_option_rejected(name):
    for chain in (
        f"{name}=mode=print:file=/tmp/data",
        f"[in]{name}@log=file='/tmp/data'",
        f"{name}=mode=print:file='part,one;two'",
    ):
        with pytest.raises(FilterChainRejected, match=name):
            audiochain.validate_filter_chain(chain)


@pytest.mark.parametrize(
    "chain",
    [
        "",
        "highpass=f=80",
        "metadata=mode=print",
        "ametadata=mode=add:key=title:value='movie,amovie;lv2'",
        r"ametadata=mode=add:key=title:value=movie\,amovie",
    ],
)
def test_safe_chain_values_are_not_filter_names(chain):
    audiochain.validate_filter_chain(chain)


def test_shipped_audio_table():
    presets = audiochain.load_presets(paths.config_dir())
    assert set(presets) == {"voice", "natural", "podcast", "raw", "none"}
    assert presets["voice"].chain == audiochain.VOICE_PRESET
    assert presets["voice"].loudness == Loudness(-16, -1.5, 11)
    assert (
        presets["natural"].chain
        == "highpass=f=60:poles=2,equalizer=f=250:width_type=o:width=0.8:g=-2,equalizer=f=500:width_type=o:width=0.5:g=-1,equalizer=f=2500:width_type=o:width=1.0:g=2,equalizer=f=4000:width_type=o:width=0.7:g=1.5"
    )
    assert presets["natural"].loudness == Loudness(-19, -1.5, 14)
    assert (
        presets["podcast"].chain
        == "highpass=f=80:poles=2,bass=gain=4:frequency=100:width_type=s:width=0.7,equalizer=f=250:width_type=o:width=0.8:g=2,equalizer=f=500:width_type=o:width=0.6:g=-2,equalizer=f=2500:width_type=o:width=1.0:g=2,acompressor=threshold=0.08:ratio=3:attack=8:release=120:makeup=2:knee=4"
    )
    assert presets["podcast"].loudness == Loudness(-16, -2, 8)
    assert presets["raw"].chain == "" and presets["raw"].loudness == Loudness(
        -16, -1.5, 11
    )
    assert presets["none"].chain == "" and presets["none"].loudness is None
    assert all("loudnorm" not in preset.chain for preset in presets.values())
    assert audiochain.MIC_HIGHPASS_HZ == {
        "built-in": 100,
        "usb": 80,
        "lavalier": 70,
        "shotgun": 90,
        "xlr-interface": 60,
    }
    assert audiochain.TIER_AAC_BITRATE == {
        Tier.BASIC: 192000,
        Tier.CREATOR: 256000,
        Tier.PRO: 320000,
    }


def test_invalid_preset_reports_name_and_unknown_lists_names():
    directory = paths.config_dir()
    with pytest.raises(
        DeclipError, match="Available: natural, none, podcast, raw, voice"
    ):
        audiochain.resolve_preset("missing", directory)
    (directory / "presets.json").write_text(
        json.dumps({"desk": {"chain": "loudnorm,highpass=f=80"}}), encoding="utf-8"
    )
    with pytest.raises(FilterChainRejected, match="Preset 'desk'"):
        audiochain.load_presets(directory)


@pytest.mark.parametrize("target", [-14, -16, -23, None])
def test_rig_loudness_answers(target):
    assert audiochain.loudness_for_target(target) == (
        Loudness(target, -1.5, 11) if target is not None else None
    )


def test_shipped_fillers_and_prompt():
    data = fillers.load_fillers("en", config_dir=paths.config_dir())
    assert data.prompt == "Um, uh, hmm, like, you know, I mean. So, uh, yeah."
    assert data.single == frozenset(
        {
            "um",
            "uh",
            "uhh",
            "umm",
            "ummm",
            "hmm",
            "hm",
            "like",
            "basically",
            "actually",
            "literally",
            "right",
            "so",
            "well",
        }
    )
    assert data.double == frozenset({"you know", "i mean", "sort of", "kind of"})
    assert fillers.SHIPPED_LANGUAGES == ("en",)
    assert fillers.load_fillers("de", config_dir=paths.config_dir()) is None


def test_filler_file_override_priority_and_legacy_prompt():
    directory = paths.config_dir()
    (directory / "fillers.txt").write_text(
        "# prompt: ignored\nUH\nYou know\n", encoding="utf-8"
    )
    legacy = fillers.load_fillers("en", config_dir=directory)
    assert legacy.single == frozenset({"uh"})
    assert legacy.prompt == "Um, uh, hmm, like, you know, I mean. So, uh, yeah."
    (directory / "fillers").mkdir()
    (directory / "fillers" / "en.txt").write_text(
        "# prompt: custom\nHmm\n# ignored\nI mean\n", encoding="utf-8"
    )
    custom = fillers.load_fillers("en", config_dir=directory)
    assert custom.prompt == "custom"
    assert custom.single == frozenset({"hmm"}) and custom.double == frozenset(
        {"i mean"}
    )
    assert (
        fillers.parse_filler_file(
            "# comment\nuh\n# prompt: later comment\n", language="en"
        ).prompt
        is None
    )


def test_user_language_file_and_path_traversal():
    directory = paths.config_dir()
    (directory / "fillers").mkdir()
    (directory / "fillers" / "de.txt").write_text("äh\n", encoding="utf-8")
    assert fillers.load_fillers("de", config_dir=directory).single == frozenset({"äh"})
    assert fillers.load_fillers("../outside", config_dir=directory) is None


@pytest.mark.parametrize(
    "chain",
    [
        "'movie'=filename=x",
        r"mo\vie=filename=x",
        "[in] 'amovie'@source=filename=x",
        r"ametadata=mode=print:fi\le=x",
    ],
)
def test_escaped_or_quoted_denied_filters(chain):
    with pytest.raises(FilterChainRejected):
        audiochain.validate_filter_chain(chain)
