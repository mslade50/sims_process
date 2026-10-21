"""Reprice runs on an Eastern weekday gate and fails a bad odds download."""
import json
import re
import subprocess
import sys
from pathlib import Path

WORKFLOW = Path(".github/workflows/reprice.yml")


def _section(text, start, end):
    return text.split(start, 1)[1].split(end, 1)[0]


def test_weekday_gate_uses_eastern_time_and_includes_thursday():
    text = WORKFLOW.read_text(encoding="utf-8")
    window = _section(
        text,
        "- name: Skip if outside tournament window",
        "- name: Checkout repository",
    )
    assert "TZ=America/New_York date +%u" in window
    assert "date -u" not in window
    assert '[ "$DOW" -le 3 ]' in window
    assert "Thu-Sun" in window


def test_odds_download_fails_on_http_error_or_empty_body(tmp_path):
    text = WORKFLOW.read_text(encoding="utf-8")
    step = _section(
        text,
        "- name: Fetch fresh round-matchup odds from golf_scraping",
        "- name: Run reprice",
    )
    assert "|| true" not in step
    assert "curl -fsSL" in step
    assert 'test -s "$out"' in step
    command = re.search(r"python -c '([^']+)'", step).group(1)

    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    failed = subprocess.run(
        [sys.executable, "-c", command, str(empty)],
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert "empty" in (failed.stderr + failed.stdout)

    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps([{"player": "a"}]), encoding="utf-8")
    ok = subprocess.run(
        [sys.executable, "-c", command, str(rows)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "round matchup odds ok (1)" in ok.stdout


def test_eastern_date_command_is_available():
    subprocess.run(
        ["date", "+%u"],
        check=True,
        env={"TZ": "America/New_York", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
