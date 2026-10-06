"""Shared audio values, preset compatibility, and filter-chain validation."""

from __future__ import annotations

import json
import math
import re
from importlib.resources import files
from pathlib import Path
from typing import Mapping

from .contracts import DeclipError, FilterChainRejected, Loudness, Preset, Tier

MIC_HIGHPASS_HZ: Mapping[str, int] = {
    "built-in": 100,
    "usb": 80,
    "lavalier": 70,
    "shotgun": 90,
    "xlr-interface": 60,
}
TIER_AAC_BITRATE: Mapping[Tier, int] = {
    Tier.BASIC: 192000,
    Tier.CREATOR: 256000,
    Tier.PRO: 320000,
}
VOICE_PRESET: str = "highpass=f=80:poles=2,equalizer=f=250:width_type=o:width=1.0:g=-2,equalizer=f=3000:width_type=o:width=1.0:g=2,acompressor=threshold=0.1:ratio=3:attack=10:release=150:makeup=1.5:knee=3"
_DENIED = frozenset({"movie", "amovie", "sendcmd", "asendcmd", "ladspa", "lv2"})


def _split_filters(chain: str) -> list[str]:
    """Honor escaped and quoted delimiters in ffmpeg's filter syntax."""
    elements, current = [], []
    quoted = escaped = False
    for character in chain:
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\":
            current.append(character)
            escaped = True
        elif character == "'":
            current.append(character)
            quoted = not quoted
        elif character in ",;" and not quoted:
            elements.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    elements.append("".join(current).strip())
    if quoted or escaped:
        raise FilterChainRejected("Unterminated quote or escape in filter chain")
    return elements


def _filter_name(element: str) -> str:
    name = (
        re.sub(r"^(?:\s*\[[^\]]*\]\s*)+", "", element)
        .split("=", 1)[0]
        .split("@", 1)[0]
        .strip()
        .lower()
    )
    return re.sub(r"\\(.)", r"\1", name).replace("'", "")


def validate_filter_chain(chain: str) -> None:
    for element in _split_filters(chain):
        name = _filter_name(element)
        if name in _DENIED or (
            name in {"metadata", "ametadata"}
            and re.search(
                r"(?:=|:)\s*['\"]?file\s*=",
                re.sub(r"\\(.)", r"\1", element),
                re.IGNORECASE,
            )
        ):
            raise FilterChainRejected(f"Denied filter: {name}")


def split_loudnorm(chain: str) -> tuple[str, Loudness | None]:
    elements = _split_filters(chain)
    indices = [i for i, e in enumerate(elements) if _filter_name(e) == "loudnorm"]
    if not indices:
        return chain, None
    if len(indices) > 1 or indices[0] != len(elements) - 1:
        raise FilterChainRejected("loudnorm must appear once, as the last filter")
    options = elements[-1].partition("=")[2]
    values = {"i": -16.0, "tp": -1.5, "lra": 11.0}
    try:
        for item in options.split(":"):
            key, _, value = item.partition("=")
            if key.strip().lower() in values:
                values[key.strip().lower()] = float(value)
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("non-finite loudness target")
    except ValueError as exc:
        raise FilterChainRejected(f"Invalid loudnorm target: {exc}") from exc
    # Removing a terminal element must not change the rest of the filter graph.
    body = chain.rsplit(elements[-1], 1)[0].rstrip().rstrip(",;").rstrip()
    return body, Loudness(**values)


def load_presets(config_dir: Path) -> dict[str, Preset]:
    shipped = files("declip").joinpath("data", "presets.json")
    data = json.loads(shipped.read_text(encoding="utf-8"))
    user = config_dir / "presets.json"
    try:
        if user.is_file():
            overrides = json.loads(user.read_text(encoding="utf-8"))
            if not isinstance(overrides, dict):
                raise ValueError("expected a preset object")
            data.update(overrides)
        result = {}
        for name, entry in data.items():
            try:
                if not isinstance(entry, dict) or not isinstance(
                    entry.get("chain", ""), str
                ):
                    raise ValueError("expected a preset with a string chain")
                validate_filter_chain(entry.get("chain", ""))
                chain, trailing = split_loudnorm(entry.get("chain", ""))
                loudness = entry.get("loudness")
                target = (
                    trailing
                    if trailing is not None
                    else (Loudness(**loudness) if loudness is not None else None)
                )
                if target is not None and not all(
                    math.isfinite(v) for v in (target.i, target.tp, target.lra)
                ):
                    raise ValueError("non-finite loudness target")
                result[name] = Preset(name, entry.get("description", ""), chain, target)
            except FilterChainRejected as exc:
                raise FilterChainRejected(f"Preset '{name}': {exc}") from exc
        return result
    except (OSError, ValueError, TypeError) as exc:
        raise DeclipError(f"Invalid presets in {user}: {exc}") from exc


def resolve_preset(name: str, config_dir: Path) -> tuple[str, Loudness | None]:
    presets = load_presets(config_dir)
    if name not in presets:
        raise DeclipError(
            f"Unknown preset '{name}'. Available: {', '.join(sorted(presets))}"
        )
    preset = presets[name]
    return preset.chain, preset.loudness


def loudness_for_target(target: int | None) -> Loudness | None:
    if target is None:
        return None
    if target not in (-14, -16, -23):
        raise DeclipError(
            f"Unsupported loudness target: {target}; choose -14, -16, -23, or none"
        )
    return Loudness(float(target), -1.5, 11.0)
