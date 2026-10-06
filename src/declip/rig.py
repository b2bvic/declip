"""Rig profiles, equipment tiers, and deterministic option resolution."""

from __future__ import annotations

import json
import math
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
    ResolvedOptions,
    RigAnswers,
    RigProfile,
    Tier,
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
    if isinstance(value, Loudness):
        target = value
    elif isinstance(value, Mapping):
        target = Loudness(**value)
    else:
        target = audiochain.loudness_for_target(int(value))
    if target is not None and not all(
        math.isfinite(v) for v in (target.i, target.tp, target.lra)
    ):
        raise DeclipError("Loudness targets must be finite")
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
    audio["eq_chain"] = chain
    target = trailing if trailing is not None else _loudness(audio.get("loudness"))
    audio["loudness"] = (
        None if target is None else {"i": target.i, "tp": target.tp, "lra": target.lra}
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
    raise NotImplementedError(
        "Option resolution is implemented in the next logical change"
    )
