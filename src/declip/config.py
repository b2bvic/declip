"""User configuration, without writes during fallback."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping

from .contracts import BUILTIN_DEFAULTS
from .fsutil import atomic_write_json
from .paths import config_dir
from .ui import ui_warn

CONFIG_KEY_ALIASES = {"margin_ms": "margin", "max_gap_ms": "max_gap"}
DEFAULT_CONFIG = {
    "version": 2,
    "defaults": {
        **BUILTIN_DEFAULTS,
        "margin": 120,
        "max_gap": 300,
        "enhance": False,
        "cpu": False,
        "remove_retakes": True,
    },
}


def default_config() -> dict[str, Any]:
    return deepcopy(DEFAULT_CONFIG)


def normalize_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = default_config()
    if config is None:
        return normalized
    if not isinstance(config, Mapping) or not isinstance(
        config.get("defaults", {}), Mapping
    ):
        raise ValueError("expected a configuration object with a defaults object")
    defaults = dict(config.get("defaults", {}))
    for old, new in CONFIG_KEY_ALIASES.items():
        if old in defaults and new not in defaults:
            defaults[new] = defaults.pop(old)
    normalized.update(
        {key: value for key, value in config.items() if key != "defaults"}
    )
    normalized["version"] = max(
        int(config.get("version", 0) or 0), DEFAULT_CONFIG["version"]
    )
    normalized["defaults"].update(defaults)
    return normalized


def load_config() -> dict[str, Any]:
    path = config_dir() / "config.json"
    if not path.exists():
        return default_config()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("expected a configuration object")
        return normalize_config(data)
    except (OSError, ValueError, TypeError) as exc:
        ui_warn(f"Cannot read {path}: {exc}. Using defaults for this run.")
        return default_config()


def save_config(config: Mapping[str, Any]) -> None:
    atomic_write_json(config_dir() / "config.json", normalize_config(config))
