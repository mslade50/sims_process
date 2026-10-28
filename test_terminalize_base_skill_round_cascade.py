import json
from pathlib import Path

import pandas as pd
import pytest

from archive.terminalize_base_skill_round_cascade import (
    _fingerprint,
    _records_current,
    dataframe_to_markdown,
    local_markdown_formatter,
)


def test_local_markdown_formatter_does_not_require_tabulate():
    frame = pd.DataFrame({"name": ["a|b", "c"], "value": [1.23456, None]})
    rendered = dataframe_to_markdown(frame, index=False, floatfmt=".3f")
    assert "a\\|b" in rendered
    assert "1.235" in rendered
    assert rendered.splitlines()[1] == "| --- | --- |"


def test_context_patches_and_restores_dataframe_to_markdown():
    original = pd.DataFrame.to_markdown
    frame = pd.DataFrame({"x": [1.25]})
    with local_markdown_formatter():
        assert pd.DataFrame.to_markdown is not original
        assert "1.25" in frame.to_markdown(index=False, floatfmt=".2f")
    assert pd.DataFrame.to_markdown is original


def test_records_current_fails_closed_on_changed_input(tmp_path: Path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"value": 1}), encoding="utf-8")
    record = _fingerprint(path)
    assert _records_current([record], "fixture") == 1
    path.write_text(json.dumps({"value": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="changed after interrupted run"):
        _records_current([record], "fixture")


def test_records_current_rejects_missing_input(tmp_path: Path):
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        _records_current(
            [
                {
                    "path": str(missing),
                    "sha256": "0" * 64,
                    "bytes": 0,
                    "mtime_ns": 0,
                }
            ],
            "fixture",
        )
