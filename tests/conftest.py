ENTRY = 'declip'

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest


@pytest.fixture
def tool(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    path = Path(__file__).resolve().parents[1] / ENTRY
    loader = SourceFileLoader("tool_under_test", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module
