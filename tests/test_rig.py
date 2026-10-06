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
