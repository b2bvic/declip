"""Hand-authored expectations for setup, profiles, and option precedence."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from declip import audiochain, rig
from declip.contracts import (
    AudioStream,
    ClipMeasurement,
    DeclipError,
    FilterChainRejected,
    GpuInfo,
    HardwareInfo,
    Loudness,
    RigAnswers,
    RigProfile,
    Tier,
    VideoSummary,
)

NOW = datetime(2026, 10, 6, 16, tzinfo=timezone.utc)
CPU = HardwareInfo("linux", "x86_64", "Synthetic CPU", 16, 8, False, ())


def measurement(
    *,
    floor=-60.0,
    video_depth=8,
    codec="h264",
    audio_depth=16,
    has_audio=True,
    clipping=False,
    vfr=False,
):
    return ClipMeasurement(
        "sample.mp4",
        0.0,
        60.0,
        has_audio,
        floor if has_audio else None,
        -20.0 if has_audio else None,
        -2.0 if has_audio else None,
        7.0 if has_audio else None,
        clipping,
        3 if clipping else 0,
        AudioStream(1, "pcm_s24le", 48000, 2, "stereo", audio_depth, None)
        if has_audio
        else None,
        VideoSummary(codec, "yuv420p", video_depth, "bt709", "30000/1001", vfr),
    )


def profile(*, measured=None, **answers):
    return rig.compute_profile(
        "desk", rig.setup_answers(answers), measured, CPU, now=NOW
    )


@pytest.mark.contract
def test_profile_round_trip_and_atomic_file(tmp_path):
    original = profile(measured=measurement(video_depth=10), destination="nle")
    path = rig.rig_path("desk", tmp_path)
    rig.save_rig(path, original)
    loaded, warnings = rig.load_rig(path)
    assert loaded == original == RigProfile.from_dict(original.to_dict())
    assert warnings == []
    assert list(path.parent.iterdir()) == [path]
    assert original.created == "2026-10-06T16:00:00Z"


@pytest.mark.parametrize(
    "name", ["", " ", ".", "..", "../desk", "a/b", "a\\b", "/absolute", "a\0b"]
)
def test_rig_path_rejects_unsafe_names(name, tmp_path):
    with pytest.raises(DeclipError):
        rig.rig_path(name, tmp_path)


def test_rig_path_accepts_unicode(tmp_path):
    assert (
        rig.rig_path("café studio", tmp_path) == tmp_path / "rigs" / "café studio.json"
    )


@pytest.mark.parametrize(
    "hardware, expected",
    [
        (
            replace(CPU, apple_silicon=True, ram_gb=16),
            ("mlx", "mlx-community/whisper-large-v3-turbo", "auto"),
        ),
        (
            replace(CPU, apple_silicon=True, ram_gb=15.9),
            ("mlx", "mlx-community/whisper-small", "auto"),
        ),
        (
            replace(CPU, apple_silicon=True, ram_gb=None),
            ("mlx", "mlx-community/whisper-small", "auto"),
        ),
        (
            replace(CPU, gpus=(GpuInfo("NVIDIA", 6, None),)),
            ("faster", "large-v3-turbo", "float16"),
        ),
        (
            replace(CPU, gpus=(GpuInfo("NVIDIA", 5.9, None),)),
            ("faster", "large-v3-turbo", "int8_float16"),
        ),
        (
            replace(CPU, gpus=(GpuInfo("NVIDIA", None, None),)),
            ("faster", "large-v3-turbo", "int8_float16"),
        ),
        (
            replace(CPU, gpus=(GpuInfo("NVIDIA", 2, None), GpuInfo("NVIDIA", 8, None))),
            ("faster", "large-v3-turbo", "float16"),
        ),
        (CPU, ("faster", "large-v3-turbo", "int8")),
        (replace(CPU, cpu_cores=7), ("faster", "small", "int8")),
    ],
)
@pytest.mark.contract
def test_hardware_model_table(hardware, expected):
    assert rig.default_model_for(hardware) == expected


@pytest.mark.parametrize(
    "measured, answers, tier",
    [
        (None, {}, Tier.BASIC),
        (None, {"log_profile": True}, Tier.PRO),
        (measurement(video_depth=10), {}, Tier.PRO),
        (measurement(video_depth=12), {}, Tier.PRO),
        (measurement(codec="prores"), {}, Tier.PRO),
        (measurement(audio_depth=24), {"mic": "xlr-interface"}, Tier.PRO),
        (measurement(audio_depth=24), {"mic": "usb"}, Tier.CREATOR),
        (measurement(floor=-55), {}, Tier.BASIC),
        (measurement(floor=-55.001), {}, Tier.CREATOR),
        (measurement(floor=None), {}, Tier.BASIC),
        (measurement(floor=-60), {"mic": "built-in"}, Tier.BASIC),
        (measurement(has_audio=False), {}, Tier.BASIC),
        (
            measurement(video_depth=None, codec=None, audio_depth=None, floor=None),
            {},
            Tier.BASIC,
        ),
    ],
)
def test_tier_conditions(measured, answers, tier):
    result = profile(measured=measured, **answers)
    assert result.tier == tier
    assert result.tier_reasons
    assert result.review == {"required": True}


@pytest.mark.parametrize(
    "mic, hz", [("usb", 80), ("lavalier", 70), ("shotgun", 90), ("xlr-interface", 60)]
)
def test_creator_mic_highpass(mic, hz):
    result = profile(measured=measurement(), mic=mic)
    assert result.tier == Tier.CREATOR
    assert result.audio["eq_chain"] == f"highpass=f={hz}:poles=2"
    assert result.audio["enhancer"] == "none"


@pytest.mark.parametrize(
    "tier, sample, expected_chain, enhancer, codec",
    [
        (Tier.BASIC, None, audiochain.VOICE_PRESET, "auto", "h264"),
        (Tier.CREATOR, measurement(), "highpass=f=80:poles=2", "none", "match"),
        (Tier.PRO, measurement(video_depth=10), "", "none", "match"),
    ],
)
@pytest.mark.parametrize("destination", ["publish", "nle"])
@pytest.mark.parametrize("nle_format", ["edl", "fcpxml"])
def test_tier_output_and_audio_table(
    tier, sample, expected_chain, enhancer, codec, destination, nle_format
):
    result = profile(measured=sample, destination=destination, nle_format=nle_format)
    assert result.tier == tier
    assert result.video["codec"] == codec
    assert result.output == {
        "mode": "nle" if destination == "nle" else "render",
        "nle_format": nle_format if destination == "nle" else None,
        "sidecar_audio": destination == "nle" and tier != Tier.PRO,
    }
    assert result.audio["eq_chain"] == expected_chain
    assert result.audio["enhancer"] == enhancer
    assert result.audio["loudness"] == (
        None
        if tier == Tier.PRO and destination == "nle"
        else {"i": -16.0, "tp": -1.5, "lra": 11.0}
    )


@pytest.mark.parametrize("room", ["normal", "treated", "echo", "outdoor"])
def test_creator_room_enhancement(room):
    result = profile(measured=measurement(), room=room)
    assert result.audio["enhancer"] == (
        "auto" if room in {"echo", "outdoor"} else "none"
    )


@pytest.mark.parametrize(
    "floor, expected", [(None, -55), (-90, -70), (-60, -54), (-20, -30)]
)
def test_gap_threshold(floor, expected):
    assert profile(measured=measurement(floor=floor)).detect["gap_noise_db"] == expected


@pytest.mark.parametrize("target", [-14, -16, -23, "none"])
def test_setup_loudness_targets(target):
    result = profile(loudness=target)
    assert result.audio["loudness"] == (
        None if target == "none" else {"i": float(target), "tp": -1.5, "lra": 11.0}
    )


def test_non_interactive_defaults_and_unanswered_pro_destination():
    assert rig.setup_answers({}) == RigAnswers(
        "usb", "normal", -16, "publish", "edl", False
    )
    assert rig.setup_answers({"mic": None, "loudness": None}) == rig.setup_answers({})
    unanswered = replace(rig.setup_answers({}), destination="", log_profile=True)
    result = rig.compute_profile("desk", unanswered, None, CPU, now=NOW)
    assert result.output == {"mode": "nle", "nle_format": "edl", "sidecar_audio": False}
    assert result.audio["loudness"] is None


def test_setup_preset_copy_and_pro_nle_no_processing():
    preset = ("highpass=f=65", Loudness(-19, -2.5, 5))
    result = rig.compute_profile(
        "desk", rig.setup_answers({}), None, CPU, preset=preset, now=NOW
    )
    assert result.audio["eq_chain"] == preset[0]
    assert result.audio["loudness"] == {"i": -19, "tp": -2.5, "lra": 5}
    pro = rig.compute_profile(
        "desk",
        rig.setup_answers({"destination": "nle", "log_profile": True}),
        None,
        CPU,
        preset=preset,
        now=NOW,
    )
    assert pro.audio["eq_chain"] == ""
    assert pro.audio["loudness"] is None


def test_review_required_false_is_ignored(tmp_path):
    import json

    original = profile()
    data = original.to_dict()
    data["review"] = {"required": False}
    path = tmp_path / "rig.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    before = path.read_bytes()
    loaded, warnings = rig.load_rig(path)
    assert loaded.review == {"required": True}
    assert warnings == [
        f"review is required for all tiers; review.required=false in {path} is ignored"
    ]
    assert path.read_bytes() == before
    assert original.review == {"required": True}


def test_load_splits_trailing_loudnorm(tmp_path):
    import json

    data = profile().to_dict()
    data["audio"]["eq_chain"] = "highpass=f=65,loudnorm=I=-19:TP=-2.5:LRA=5"
    path = tmp_path / "rig.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded, _ = rig.load_rig(path)
    assert loaded.audio["eq_chain"] == "highpass=f=65"
    assert loaded.audio["loudness"] == {"i": -19, "tp": -2.5, "lra": 5}


@pytest.mark.parametrize(
    "chain", ["amovie=/tmp/input", "loudnorm,highpass=f=80", "loudnorm,loudnorm"]
)
def test_profile_chain_rejections(chain, tmp_path):
    import json

    data = profile().to_dict()
    data["audio"]["eq_chain"] = chain
    path = tmp_path / "rig.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(FilterChainRejected):
        rig.load_rig(path)
    with pytest.raises(FilterChainRejected):
        rig.save_rig(path, replace(profile(), audio=data["audio"]))


@pytest.mark.parametrize(
    "sample",
    [
        measurement(clipping=True, floor=-40),
        measurement(clipping=True),
        measurement(clipping=True, video_depth=10),
    ],
)
def test_clipping_warning_all_tiers(sample, tmp_path):
    path = tmp_path / "rig.json"
    rig.save_rig(path, profile(measured=sample))
    loaded, warnings = rig.load_rig(path)
    assert warnings and "Clipping" in warnings[0]
    assert ("lower the input gain" in warnings[0]) == (loaded.tier == Tier.PRO)


@pytest.mark.parametrize("contents", ["{", "[]", "{}", '{"schema_version":2}'])
def test_malformed_rig_names_file(contents, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(DeclipError, match="bad.json"):
        rig.load_rig(path)


def test_vfr_profile_preserves_measurement():
    sample = measurement(vfr=True)
    assert profile(measured=sample).measured.video.vfr is True


@pytest.mark.contract
def test_builtin_options_and_origins():
    from dataclasses import fields
    from declip.contracts import OutputMode, TranscribeOptions

    result = rig.resolve_options({}, None, {})
    assert result.transcribe == TranscribeOptions("large-v3-turbo")
    assert result.rig_name is None and result.tier is None
    assert result.eq_chain == "" and result.loudness == Loudness(-16, -1.5, 11)
    assert result.enhancer == "none" and result.audio_bitrate == 192000
    assert result.video_codec == result.quality == "match"
    assert result.output_mode == OutputMode.RENDER
    assert result.nle_format is None and result.sidecar_audio is False
    assert result.allow_8bit is False
    expected = {field.name for field in fields(result)} - {"origins", "transcribe"}
    expected.update(field.name for field in fields(result.transcribe))
    assert set(result.origins) == expected
    assert set(result.origins.values()) == {"builtin"}


@pytest.mark.parametrize(
    "cli, use_rig, config, expected, origin",
    [
        ({"margin_ms": 23}, True, {"margin_ms": 45}, 23, "cli"),
        ({"margin_ms": None}, True, {"margin_ms": 45}, 120, "rig"),
        ({}, False, {"margin_ms": 45}, 45, "config"),
        ({}, False, {}, 120, "builtin"),
    ],
)
def test_four_precedence_levels(cli, use_rig, config, expected, origin):
    result = rig.resolve_options(cli, profile() if use_rig else None, config)
    assert result.margin_ms == expected
    assert result.origins["margin_ms"] == origin


@pytest.mark.parametrize(
    "sample, bitrate",
    [(None, 192000), (measurement(), 256000), (measurement(video_depth=10), 320000)],
)
def test_tier_bitrate_and_origin(sample, bitrate):
    result = rig.resolve_options(
        {}, profile(measured=sample), {"audio_bitrate": 123000}
    )
    assert result.audio_bitrate == bitrate
    assert result.origins["audio_bitrate"] == "rig"
    assert (
        rig.resolve_options(
            {"audio_bitrate": 128000}, profile(measured=sample), {}
        ).audio_bitrate
        == 128000
    )


@pytest.mark.parametrize(
    "field, value, nested",
    [
        ("model", "custom-model", True),
        ("language", "de", True),
        ("device", "cpu", True),
        ("compute_type", "int8", True),
        ("prompt_mode", "chunked", True),
        ("initial_prompt", "Um uh", True),
        ("condition_on_previous_text", True, True),
        ("backend", "faster", False),
        ("enhancer", "afftdn", False),
        ("enhancer_strength", 0.1, False),
        ("crossfade_ms", 5, False),
        ("video_codec", "hevc", False),
        ("quality", "small", False),
        ("encoder", "software", False),
        ("allow_8bit", True, False),
        ("audio_bitrate", 128000, False),
        ("max_gap_ms", 800, False),
        ("min_silence_ms", 900, False),
        ("min_confidence", 0.8, False),
        ("gap_noise_db", -60, False),
        ("retakes", False, False),
    ],
)
def test_each_cli_scalar_and_origin(field, value, nested):
    result = rig.resolve_options({field: value}, profile(), {})
    assert getattr(result.transcribe if nested else result, field) == value
    assert result.origins[field] == "cli"


def test_rig_nulls_override_config_and_false_is_not_missing():
    original = profile(measured=measurement(video_depth=10), destination="nle")
    result = rig.resolve_options(
        {},
        original,
        {
            "loudness": -14,
            "language": "en",
            "retakes": False,
            "sidecar_audio": True,
            "enhancer": "auto",
            "preset": "podcast",
        },
    )
    assert result.loudness is None
    assert result.transcribe.language is None
    assert result.retakes is True
    assert result.sidecar_audio is False
    assert result.enhancer == "none"
    assert result.eq_chain == ""
    assert result.origins["loudness"] == result.origins["language"] == "rig"
    changed = rig.resolve_options({"retakes": False, "crossfade_ms": 0}, original, {})
    assert changed.retakes is False and changed.crossfade_ms == 0


@pytest.mark.parametrize(
    "cli, target, chain, target_origin",
    [
        ({"preset": "natural"}, Loudness(-19, -1.5, 14), None, "cli"),
        (
            {"eq": "highpass=f=65,loudnorm=I=-23:TP=-2:LRA=5"},
            Loudness(-23, -2, 5),
            "highpass=f=65",
            "cli",
        ),
        ({"eq": "highpass=f=65"}, Loudness(-16, -1.5, 11), "highpass=f=65", "rig"),
        ({"preset": "natural", "loudness": -14}, Loudness(-14, -1.5, 11), None, "cli"),
        (
            {"eq": "highpass=f=65,loudnorm=I=-23", "loudness": -14},
            Loudness(-14, -1.5, 11),
            "highpass=f=65",
            "cli",
        ),
        ({"loudness": "none"}, None, None, "cli"),
        ({"preset": "none"}, None, "", "cli"),
    ],
)
def test_loudness_precedence(cli, target, chain, target_origin):
    result = rig.resolve_options(cli, profile(), {"loudness": -23})
    assert result.loudness == target
    assert result.origins["loudness"] == target_origin
    if chain is not None:
        assert result.eq_chain == chain
    assert "loudnorm" not in result.eq_chain


def test_preset_copy_from_user_file(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("DECLIP_CONFIG_DIR", str(tmp_path))
    (tmp_path / "presets.json").write_text(
        json.dumps(
            {
                "custom": {
                    "description": "Custom chain",
                    "chain": "highpass=f=65,loudnorm=I=-19:TP=-2.5:LRA=5",
                }
            }
        ),
        encoding="utf-8",
    )
    result = rig.resolve_options({"preset": "custom"}, profile(), {})
    assert result.eq_chain == "highpass=f=65"
    assert result.loudness == Loudness(-19, -2.5, 5)
    assert result.origins["eq_chain"] == result.origins["loudness"] == "cli"


def test_config_preset_origin_unknown_warning_and_explicit_unknown_failure(capsys):
    result = rig.resolve_options({}, None, {"preset": "natural"})
    assert result.loudness == Loudness(-19, -1.5, 14)
    assert result.origins["eq_chain"] == result.origins["loudness"] == "config"
    result = rig.resolve_options({}, None, {"preset": "missing-profile"})
    assert result.eq_chain == "" and result.loudness == Loudness(-16, -1.5, 11)
    captured = capsys.readouterr()
    assert captured.out == "" and "config.json" in captured.err
    assert captured.err.count("missing-profile") == 1
    with pytest.raises(DeclipError, match="Unknown preset"):
        rig.resolve_options({"preset": "missing-profile"}, None, {})


@pytest.mark.parametrize(
    "cli", [{"eq": "amovie=/tmp/input"}, {"eq": "loudnorm,loudnorm"}]
)
def test_resolve_rejects_unsafe_or_duplicate_filters(cli):
    with pytest.raises(FilterChainRejected):
        rig.resolve_options(cli, None, {})


def test_resolved_preset_pair_and_input_immutability():
    from copy import deepcopy

    original = profile(measured=measurement())
    cli = {"preset": ("highpass=f=65", Loudness(-19, -2.5, 5))}
    config = {"margin_ms": 50, "loudness": {"i": -23, "tp": -1.5, "lra": 11}}
    snapshots = deepcopy((cli, original, config))
    result = rig.resolve_options(cli, original, config)
    assert result.eq_chain == "highpass=f=65"
    assert result.loudness == Loudness(-19, -2.5, 5)
    assert (cli, original, config) == snapshots


def test_legacy_aliases_and_output_defaults():
    from declip.contracts import OutputMode, NleFormat

    result = rig.resolve_options(
        {"codec": "hevc", "enhance": True},
        None,
        {
            "margin": 50,
            "max_gap": 200,
            "remove_retakes": False,
            "crossfade": 10,
            "mode": "nle",
            "sidecar_audio": True,
        },
    )
    assert result.video_codec == "hevc" and result.enhancer == "auto"
    assert result.margin_ms == 50 and result.max_gap_ms == 200
    assert result.crossfade_ms == 10 and result.retakes is False
    assert result.output_mode == OutputMode.NLE and result.nle_format == NleFormat.EDL
    assert result.sidecar_audio is True
    assert result.origins["nle_format"] == "config"


@pytest.mark.parametrize(
    "field, value",
    [
        ("quality", "invalid"),
        ("device", "invalid"),
        ("enhancer", "invalid"),
        ("min_confidence", float("nan")),
        ("enhancer_strength", 1.1),
        ("margin_ms", -1),
        ("retakes", "false"),
        ("audio_bitrate", 0),
        ("model", ""),
        ("initial_prompt", 1),
    ],
)
def test_invalid_option_refusal(field, value):
    with pytest.raises(DeclipError):
        rig.resolve_options({field: value}, None, {})


def test_omitted_profile_options_fall_through_to_config():
    original = replace(
        profile(), transcribe={}, audio={}, video={}, output={}, detect={}
    )
    result = rig.resolve_options(
        {},
        original,
        {
            "eq_chain": "highpass=f=75",
            "loudness": -23,
            "model": "small",
            "quality": "high",
        },
    )
    assert result.eq_chain == "highpass=f=75"
    assert result.loudness == Loudness(-23, -1.5, 11)
    assert result.transcribe.model == "small" and result.quality == "high"
    assert all(
        result.origins[key] == "config"
        for key in ("eq_chain", "loudness", "model", "quality")
    )


@pytest.mark.parametrize(
    "target",
    [True, -16.5, "bad", {"i": -16}, {"i": float("nan"), "tp": -1.5, "lra": 11}],
)
def test_invalid_loudness_target(target):
    with pytest.raises(DeclipError):
        rig.resolve_options({"loudness": target}, None, {})


@pytest.mark.parametrize("sample", [None, measurement()])
def test_unanswered_basic_and_creator_destination_renders(sample):
    answers = replace(rig.setup_answers({}), destination="")
    result = rig.compute_profile("desk", answers, sample, CPU, now=NOW)
    assert result.output == {
        "mode": "render",
        "nle_format": None,
        "sidecar_audio": False,
    }


def test_save_normalizes_review_without_mutating_profile(tmp_path):
    original = replace(profile(), review={"required": False})
    path = tmp_path / "rig.json"
    rig.save_rig(path, original)
    loaded, warnings = rig.load_rig(path)
    assert warnings == [] and loaded.review == {"required": True}
    assert original.review == {"required": False}


def test_full_unsupported_schema_refusal(tmp_path):
    import json

    data = profile().to_dict()
    data["schema_version"] = 2
    path = tmp_path / "rig.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DeclipError, match="Unsupported rig schema: 2"):
        rig.load_rig(path)


def test_timestamp_utc_and_schema_example():
    local_now = datetime.fromisoformat("2026-10-06T12:00:00-04:00")
    answers = rig.setup_answers({"mic": "lavalier", "destination": "nle"})
    result = rig.compute_profile(
        "desk",
        answers,
        measurement(video_depth=10, floor=-58.2),
        replace(CPU, os="darwin", arch="arm64", apple_silicon=True, ram_gb=24),
        now=local_now,
    )
    assert result.created == "2026-10-06T16:00:00Z"
    assert result.tier_reasons == ("video bit depth is 10",)
    assert result.transcribe == {
        "backend": "auto",
        "model": "mlx-community/whisper-large-v3-turbo",
        "language": None,
        "compute_type": "auto",
        "device": "auto",
    }
    assert result.detect["gap_noise_db"] == pytest.approx(-52.2)
