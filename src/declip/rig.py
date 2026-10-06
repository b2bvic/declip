"""Rig profiles, equipment tiers, and deterministic option resolution."""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from declip import audiochain, fsutil
from declip.contracts import (
    BUILTIN_DEFAULTS,
    RIG_SCHEMA_VERSION,
    ClipMeasurement,
    DeclipError,
    HardwareInfo,
    Loudness,
    NleFormat,
    OutputMode,
    ResolvedOptions,
    RigAnswers,
    RigProfile,
    Tier,
    TranscribeOptions,
)


def rig_path(name: str, config_dir: Path) -> Path:
    if not isinstance(name, str) or not name.strip() or name in {".", ".."}:
        raise DeclipError("Rig name must be non-empty")
    if any(character in name for character in ("/", "\\", "\0")):
        raise DeclipError("Rig name must not contain path separators or NUL")
    return config_dir / "rigs" / f"{name}.json"


def _loudness(value: Any) -> Loudness | None:
    if value is None or value == "none":
        return None
    try:
        if isinstance(value, Loudness):
            target = value
        elif isinstance(value, Mapping):
            target = Loudness(**value)
        else:
            if isinstance(value, bool) or isinstance(value, float):
                raise ValueError("expected an integer loudness target")
            target = audiochain.loudness_for_target(int(value))
        if target is not None and not all(
            not isinstance(v, bool) and math.isfinite(v)
            for v in (target.i, target.tp, target.lra)
        ):
            raise ValueError("loudness targets must be finite numbers")
    except (TypeError, ValueError, OverflowError) as exc:
        raise DeclipError(f"Invalid loudness target: {exc}") from exc
    return target


def _normalize_profile(profile: RigProfile) -> RigProfile:
    if profile.schema_version != RIG_SCHEMA_VERSION:
        raise DeclipError(f"Unsupported rig schema: {profile.schema_version}")
    rig_path(profile.name, Path("."))
    _check_answers(profile.answers)
    audio = dict(profile.audio)
    chain = audio.get("eq_chain", "")
    if not isinstance(chain, str):
        raise DeclipError("audio.eq_chain must be a string")
    audiochain.validate_filter_chain(chain)
    chain, trailing = audiochain.split_loudnorm(chain)
    if "eq_chain" in audio:
        audio["eq_chain"] = chain
    if trailing is not None or "loudness" in audio:
        target = trailing if trailing is not None else _loudness(audio["loudness"])
        audio["loudness"] = (
            None
            if target is None
            else {"i": target.i, "tp": target.tp, "lra": target.lra}
        )
    return replace(profile, audio=audio, review={"required": True})


def load_rig(path: Path) -> tuple[RigProfile, list[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("expected a rig object")
        stored_review = data.get("review", {})
        if not isinstance(stored_review, dict):
            raise ValueError("review must be an object")
        warnings = []
        if stored_review.get("required") is False:
            warnings.append(
                f"review is required for all tiers; review.required=false in {path} is ignored"
            )
        data["review"] = {"required": True}
        profile = _normalize_profile(RigProfile.from_dict(data))
        if profile.measured is not None and profile.measured.clipping:
            warning = "Clipping found in the sample"
            if profile.tier == Tier.PRO:
                warning += "; lower the input gain"
            warnings.append(warning)
        return profile, warnings
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise DeclipError(f"Cannot read rig {path}: {exc}") from exc


def save_rig(path: Path, profile: RigProfile) -> None:
    fsutil.atomic_write_json(path, _normalize_profile(profile).to_dict())


def default_model_for(hardware: HardwareInfo) -> tuple[str, str, str]:
    if hardware.apple_silicon:
        model = (
            "large-v3-turbo"
            if hardware.ram_gb is not None and hardware.ram_gb >= 16
            else "small"
        )
        return "mlx", f"mlx-community/whisper-{model}", "auto"
    if hardware.gpus:
        vram = max((gpu.vram_gb or 0 for gpu in hardware.gpus), default=0)
        return "faster", "large-v3-turbo", "float16" if vram >= 6 else "int8_float16"
    return "faster", "large-v3-turbo" if hardware.cpu_cores >= 8 else "small", "int8"


def setup_answers(values: Mapping[str, Any]) -> RigAnswers:
    """Apply setup --yes defaults; explicit 'none' disables loudness."""

    def answer(key: str, default: Any) -> Any:
        value = values.get(key)
        return default if value is None else value

    result = RigAnswers(
        answer("mic", "usb"),
        answer("room", "normal"),
        None if answer("loudness", -16) == "none" else int(answer("loudness", -16)),
        answer("destination", "publish"),
        answer("nle_format", "edl"),
        answer("log_profile", False),
    )
    _check_answers(result)
    return result


def _check_answers(answers: RigAnswers) -> None:
    if answers.mic not in audiochain.MIC_HIGHPASS_HZ:
        raise DeclipError(f"Unknown microphone type: {answers.mic}")
    if answers.room not in {"treated", "normal", "echo", "outdoor"}:
        raise DeclipError(f"Unknown room: {answers.room}")
    if answers.destination not in {"", "publish", "nle"}:
        raise DeclipError(f"Unknown destination: {answers.destination}")
    if answers.nle_format not in {None, "edl", "fcpxml"}:
        raise DeclipError(f"Unknown NLE format: {answers.nle_format}")
    if type(answers.log_profile) is not bool:
        raise DeclipError("log_profile must be a Boolean")
    audiochain.loudness_for_target(answers.loudness)


def compute_profile(
    name: str,
    answers: RigAnswers,
    measured: ClipMeasurement | None,
    hardware: HardwareInfo,
    *,
    preset: tuple[str, Loudness | None] | None = None,
    now: datetime,
) -> RigProfile:
    rig_path(name, Path("."))
    _check_answers(answers)
    video = measured.video if measured is not None else None
    audio = measured.audio if measured is not None and measured.has_audio else None
    floor = (
        measured.noise_floor_db if measured is not None and measured.has_audio else None
    )
    if floor is not None and not math.isfinite(floor):
        floor = None
    reasons = []
    if video is not None:
        if video.bit_depth is not None and video.bit_depth >= 10:
            reasons.append(f"video bit depth is {video.bit_depth}")
        if video.codec is not None and video.codec.lower().startswith("prores"):
            reasons.append("video codec is ProRes")
    if answers.log_profile:
        reasons.append("log profile confirmed")
    if (
        audio is not None
        and audio.bit_depth is not None
        and audio.bit_depth >= 24
        and answers.mic == "xlr-interface"
    ):
        reasons.append(f"audio bit depth is {audio.bit_depth} with an XLR interface")
    if reasons:
        tier = Tier.PRO
    elif answers.mic != "built-in" and floor is not None and floor < -55:
        tier = Tier.CREATOR
        reasons = [f"{answers.mic} microphone and noise floor below -55 dBFS"]
    else:
        tier = Tier.BASIC
        reasons = ["higher-tier equipment and measurement conditions are not met"]
    nle = answers.destination == "nle" or (
        answers.destination == "" and tier == Tier.PRO
    )
    enhancer = "none"
    chain = ""
    if tier == Tier.BASIC:
        enhancer, chain = "auto", audiochain.VOICE_PRESET
    elif tier == Tier.CREATOR:
        chain = f"highpass=f={audiochain.MIC_HIGHPASS_HZ[answers.mic]}:poles=2"
        if (floor is not None and floor > -50) or answers.room in {"echo", "outdoor"}:
            enhancer = "auto"
    target = audiochain.loudness_for_target(answers.loudness)
    if preset is not None:
        chain, target = preset
    if tier == Tier.PRO and nle:
        enhancer, chain, target = "none", "", None
    _, model, compute_type = default_model_for(hardware)
    detect = {
        key: BUILTIN_DEFAULTS[key]
        for key in (
            "margin_ms",
            "min_confidence",
            "gap_noise_db",
            "max_gap_ms",
            "min_silence_ms",
            "retakes",
        )
    }
    detect["gap_noise_db"] = (
        max(-70.0, min(-30.0, floor + 6)) if floor is not None else -55.0
    )
    timestamp = (
        now.replace(tzinfo=timezone.utc)
        if now.tzinfo is None
        else now.astimezone(timezone.utc)
    )
    return _normalize_profile(
        RigProfile(
            RIG_SCHEMA_VERSION,
            name,
            timestamp.isoformat().replace("+00:00", "Z"),
            answers,
            measured,
            hardware,
            tier,
            tuple(reasons),
            {
                "backend": "auto",
                "model": model,
                "language": None,
                "compute_type": compute_type,
                "device": "auto",
            },
            {
                "enhancer": enhancer,
                "enhancer_strength": BUILTIN_DEFAULTS["enhancer_strength"],
                "eq_chain": chain,
                "loudness": None
                if target is None
                else {"i": target.i, "tp": target.tp, "lra": target.lra},
            },
            {
                "encoder": "auto",
                "codec": "h264" if tier == Tier.BASIC else "match",
                "quality": "match",
            },
            {
                "mode": "nle" if nle else "render",
                "nle_format": (answers.nle_format or "edl") if nle else None,
                "sidecar_audio": nle and tier != Tier.PRO,
            },
            detect,
            {"required": True},
        )
    )


def resolve_options(
    cli_values: Mapping[str, Any],
    profile: RigProfile | None,
    config_defaults: Mapping[str, Any],
) -> ResolvedOptions:
    """Resolve explicit CLI values over rig, config, and built-in defaults.

    CLI None means an unset flag. Use 'none' to disable loudness explicitly.
    Origins use scalar option names, including the TranscribeOptions fields.
    Presets accept a name or an already resolved (chain, loudness) pair.
    """
    builtin = dict(BUILTIN_DEFAULTS)
    builtin.pop("preset")
    builtin.update(
        model="large-v3-turbo",
        compute_type="auto",
        initial_prompt=None,
        prompt_mode="auto",
        condition_on_previous_text=False,
        eq_chain="",
        loudness=audiochain.loudness_for_target(-16),
    )
    config = _option_layer(config_defaults, origin="config")
    rig_values = {}
    if profile is not None:
        profile = _normalize_profile(profile)
        rig_values.update(profile.transcribe)
        rig_values.update(profile.audio)
        rig_values.update(profile.detect)
        rig_values.update(profile.video)
        rig_values.update(profile.output)
        rig_values = _aliases(rig_values)
        rig_values.setdefault(
            "audio_bitrate", audiochain.TIER_AAC_BITRATE[profile.tier]
        )
    cli = _option_layer(
        {key: value for key, value in cli_values.items() if value is not None},
        origin="cli",
    )
    values, origins = {}, {}
    for origin, layer in (
        ("builtin", builtin),
        ("config", config),
        ("rig", rig_values),
        ("cli", cli),
    ):
        for key, value in layer.items():
            if key in builtin:
                values[key], origins[key] = value, origin
    values["loudness"] = _loudness(values["loudness"])
    audiochain.validate_filter_chain(values["eq_chain"])
    values["eq_chain"], trailing = audiochain.split_loudnorm(values["eq_chain"])
    if trailing is not None:
        values["loudness"] = trailing
        origins["loudness"] = origins["eq_chain"]
    _validate_options(values)
    mode = OutputMode(values["output_mode"])
    fmt = NleFormat(values["nle_format"]) if values["nle_format"] is not None else None
    if mode == OutputMode.RENDER:
        fmt = None
        values["sidecar_audio"] = False
        origins["nle_format"] = origins["sidecar_audio"] = origins["output_mode"]
    elif fmt is None:
        fmt = NleFormat.EDL
        origins["nle_format"] = origins["output_mode"]
    origins["rig_name"] = origins["tier"] = "rig" if profile is not None else "builtin"
    transcribe_fields = (
        "model",
        "language",
        "initial_prompt",
        "prompt_mode",
        "condition_on_previous_text",
        "device",
        "compute_type",
    )
    transcribe = TranscribeOptions(**{key: values[key] for key in transcribe_fields})
    scalar = {
        key: value for key, value in values.items() if key not in transcribe_fields
    }
    scalar.update(output_mode=mode, nle_format=fmt)
    return ResolvedOptions(
        rig_name=profile.name if profile is not None else None,
        tier=profile.tier if profile is not None else None,
        transcribe=transcribe,
        origins=origins,
        **scalar,
    )


def _aliases(values: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(values)
    for alias, key in {
        "eq": "eq_chain",
        "codec": "video_codec",
        "mode": "output_mode",
        "margin": "margin_ms",
        "max_gap": "max_gap_ms",
        "crossfade": "crossfade_ms",
        "remove_retakes": "retakes",
    }.items():
        if alias in result:
            result.setdefault(key, result.pop(alias))
    if "enhance" in result:
        result.setdefault("enhancer", "auto" if result.pop("enhance") else "none")
    return result


def _option_layer(values: Mapping[str, Any], *, origin: str) -> dict[str, Any]:
    result = _aliases(values)
    # Read-only preset loading stays in audiochain. P10 may supply a resolved pair.
    preset_dir = Path(
        os.environ.get("DECLIP_CONFIG_DIR", str(Path.home() / ".config" / "declip"))
    )
    if "preset" in result:
        preset = result.pop("preset")
        if isinstance(preset, (tuple, list)) and len(preset) == 2:
            chain, target = preset
        else:
            try:
                chain, target = audiochain.resolve_preset(preset, preset_dir)
            except DeclipError as exc:
                # Filter rejection and corrupt preset files must never fall back.
                if origin != "config" or not str(exc).startswith("Unknown preset '"):
                    raise
                print(
                    f"Unknown preset '{preset}' in {preset_dir / 'config.json'}; using raw",
                    file=sys.stderr,
                )
                chain, target = audiochain.resolve_preset("raw", preset_dir)
        result.setdefault("eq_chain", chain)
        result.setdefault("loudness", target)
    if "eq_chain" in result:
        chain = result["eq_chain"]
        if not isinstance(chain, str):
            raise DeclipError("eq_chain must be a string")
        audiochain.validate_filter_chain(chain)
        result["eq_chain"], trailing = audiochain.split_loudnorm(chain)
        if trailing is not None and "loudness" not in values:
            result["loudness"] = trailing
    return result


def _validate_options(values: Mapping[str, Any]) -> None:
    choices = {
        "backend": {"auto", "mlx", "faster", "fake"},
        "device": {"auto", "metal", "cuda", "cpu"},
        "compute_type": {"auto", "float16", "int8_float16", "int8"},
        "prompt_mode": {"auto", "initial", "hotwords", "chunked"},
        "enhancer": {"none", "auto", "afftdn", "deepfilter"},
        "video_codec": {"match", "h264", "hevc"},
        "quality": {"match", "high", "small"},
        "encoder": {"auto", "software"},
        "output_mode": {"render", "nle"},
        "nle_format": {None, "edl", "fcpxml"},
    }
    for key, allowed in choices.items():
        if (values[key] is not None and not isinstance(values[key], str)) or values[
            key
        ] not in allowed:
            raise DeclipError(f"Invalid {key}: {values[key]}")
    for key in ("allow_8bit", "sidecar_audio", "retakes", "condition_on_previous_text"):
        if type(values[key]) is not bool:
            raise DeclipError(f"{key} must be a Boolean")
    for key in (
        "enhancer_strength",
        "crossfade_ms",
        "audio_bitrate",
        "margin_ms",
        "min_confidence",
        "gap_noise_db",
        "max_gap_ms",
        "min_silence_ms",
    ):
        value = values[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise DeclipError(f"{key} must be a finite number")
        if key != "gap_noise_db" and value < 0:
            raise DeclipError(f"{key} must not be negative")
    for key in ("min_confidence", "enhancer_strength"):
        if values[key] > 1:
            raise DeclipError(f"{key} must be between 0 and 1")
    if type(values["audio_bitrate"]) is not int or values["audio_bitrate"] <= 0:
        raise DeclipError("audio_bitrate must be a positive integer")
    if not isinstance(values["model"], str) or not values["model"]:
        raise DeclipError("model must be a non-empty string")
    for key in ("language", "initial_prompt"):
        if values[key] is not None and not isinstance(values[key], str):
            raise DeclipError(f"{key} must be a string or null")
