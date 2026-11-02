"""The sim scripts must refuse to run without DATAGOLF_API_KEY.

The old hardcoded fallback is gone. These checks load only the guard
function, so they do not execute the sim or print a key.
"""
import ast
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SOURCES = ("new_sim.py", "archive/new_sim_v1.py")


def _load(relative):
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "require_datagolf_api_key"
    )
    scope = {"os": os}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope["require_datagolf_api_key"]


@pytest.mark.parametrize("relative", SOURCES)
def test_missing_key_exits_with_a_clear_error(relative, monkeypatch):
    monkeypatch.delenv("DATAGOLF_API_KEY", raising=False)
    require = _load(relative)
    with pytest.raises(SystemExit) as caught:
        require()
    message = str(caught.value)
    assert "DATAGOLF_API_KEY is not set" in message
    assert not re.search(r"[0-9a-f]{16,}", message)


@pytest.mark.parametrize("relative", SOURCES)
def test_present_key_is_returned_stripped(relative, monkeypatch):
    monkeypatch.setenv("DATAGOLF_API_KEY", "  test-key  ")
    assert _load(relative)() == "test-key"


@pytest.mark.parametrize("relative", SOURCES)
def test_source_has_no_hardcoded_datagolf_default(relative):
    text = (ROOT / relative).read_text(encoding="utf-8")
    assert "require_datagolf_api_key()" in text
    assert not re.search(r"DATAGOLF_API_KEY['\"]\s*,", text)
