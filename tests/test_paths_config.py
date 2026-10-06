import json
from pathlib import Path

import pytest

from declip import audiochain, config, paths


@pytest.mark.parametrize("kind", ["config", "cache"])
def test_environment_override_created_on_use(tmp_path, monkeypatch, kind):
    target = tmp_path / "override" / kind
    monkeypatch.setenv(f"DECLIP_{kind.upper()}_DIR", str(target))
    assert not target.exists()
    assert getattr(paths, f"{kind}_dir")() == target
    assert target.is_dir()


@pytest.mark.parametrize("platform", ["darwin", "linux"])
@pytest.mark.parametrize("kind", ["config", "cache"])
def test_unix_paths_keep_existing_locations(tmp_path, monkeypatch, platform, kind):
    monkeypatch.delenv(f"DECLIP_{kind.upper()}_DIR")
    monkeypatch.setattr(paths.sys, "platform", platform)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert getattr(paths, f"{kind}_dir")() == tmp_path / f".{kind}" / "declip"


@pytest.mark.parametrize("kind", ["config", "cache"])
def test_windows_uses_platformdirs(tmp_path, monkeypatch, kind):
    monkeypatch.delenv(f"DECLIP_{kind.upper()}_DIR")
    monkeypatch.setattr(paths.sys, "platform", "win32")

    def directory(app, appauthor):
        assert app == "declip" and appauthor is False
        return str(tmp_path / kind)

    monkeypatch.setattr(paths.platformdirs, f"user_{kind}_dir", directory)
    assert getattr(paths, f"{kind}_dir")() == tmp_path / kind


@pytest.mark.parametrize(
    "content", ["{broken", "null", "[]", '{"defaults":[]}', '{"version":"bad"}']
)
def test_corrupt_config_warns_and_preserves_bytes(content, capsys):
    path = paths.config_dir() / "config.json"
    path.write_text(content, encoding="utf-8")
    assert config.load_config() == config.default_config()
    captured = capsys.readouterr()
    assert str(path) in captured.err
    assert not captured.out
    assert path.read_text(encoding="utf-8") == content


def test_config_defaults_are_fresh_and_preserve_unrelated_fields():
    first = config.default_config()
    first["defaults"]["preset"] = "other"
    assert config.default_config()["defaults"]["preset"] == "raw"
    config.save_config({"defaults": {"margin_ms": 20}, "default_rig": "desk"})
    assert config.load_config()["defaults"]["margin"] == 20
    assert config.load_config()["default_rig"] == "desk"


def test_user_presets_override_by_name_and_keep_shipped():
    directory = paths.config_dir()
    (directory / "presets.json").write_text(
        json.dumps(
            {
                "natural": {"chain": "highpass=f=90"},
                "desk": {"chain": "loudnorm=I=-23:TP=-2.5:LRA=5"},
            }
        ),
        encoding="utf-8",
    )
    presets = audiochain.load_presets(directory)
    assert set(presets) == {"voice", "natural", "podcast", "raw", "none", "desk"}
    assert presets["natural"].chain == "highpass=f=90"
    assert presets["desk"].chain == ""
    assert presets["desk"].loudness.i == -23
    assert presets["desk"].loudness.tp == -2.5
