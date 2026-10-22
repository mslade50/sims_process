import re
from pathlib import Path

import pytest

from round_matchup_coverage import (
    DEFAULT_REQUIRED_BOOKS,
    MANUAL_PINNACLE_BOOKS,
    resolve_required_matchup_books,
)

_WORKFLOWS = Path(__file__).resolve().parent / ".github" / "workflows"


def test_default_pair_is_strict_for_repository_dispatch():
    assert resolve_required_matchup_books(
        {"GITHUB_EVENT_NAME": "repository_dispatch"}
    ) == DEFAULT_REQUIRED_BOOKS


def test_manual_pinnacle_pair_is_allowed_for_workflow_dispatch():
    assert resolve_required_matchup_books(
        {
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "ROUND_MATCHUP_REQUIRED_BOOK_PAIR": "betonline_pinnacle",
        }
    ) == MANUAL_PINNACLE_BOOKS


def test_repository_dispatch_cannot_enable_manual_pinnacle_pair():
    with pytest.raises(ValueError, match="only during an explicit workflow_dispatch"):
        resolve_required_matchup_books(
            {
                "GITHUB_EVENT_NAME": "repository_dispatch",
                "ROUND_MATCHUP_REQUIRED_BOOK_PAIR": "betonline_pinnacle",
            }
        )


def test_unknown_pair_mode_fails_closed():
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_required_matchup_books(
            {"ROUND_MATCHUP_REQUIRED_BOOK_PAIR": "any_two_books"}
        )


def test_betonline_only_maps_to_betonline_on_workflow_dispatch():
    assert resolve_required_matchup_books(
        {
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "ROUND_MATCHUP_REQUIRED_BOOK_PAIR": "betonline_only",
        }
    ) == ("betonline",)


@pytest.mark.parametrize("mode", ["betonline_only", "betonline_pinnacle"])
@pytest.mark.parametrize(
    "event_name",
    ["schedule", "repository_dispatch", "workflow_call", "push", None],
)
def test_manual_book_modes_share_the_dispatch_only_gate(mode, event_name):
    env = {"ROUND_MATCHUP_REQUIRED_BOOK_PAIR": mode}
    if event_name is not None:
        env["GITHUB_EVENT_NAME"] = event_name
    with pytest.raises(ValueError, match="only during an explicit workflow_dispatch"):
        resolve_required_matchup_books(env)


def test_conflicting_book_mode_token_fails_closed():
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_required_matchup_books(
            {
                "GITHUB_EVENT_NAME": "workflow_dispatch",
                "ROUND_MATCHUP_REQUIRED_BOOK_PAIR": "conflicting_book_modes",
            }
        )


def test_schedule_without_a_repair_mode_stays_on_the_production_pair():
    assert resolve_required_matchup_books(
        {"GITHUB_EVENT_NAME": "schedule"}
    ) == DEFAULT_REQUIRED_BOOKS


def _active_schedule(text: str) -> bool:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped == "schedule:" or stripped.startswith("schedule:"):
            return True
    return False


def _split_github_ops(text: str, op: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    quote = False
    i = 0
    while i < len(text):
        if text[i] == "'":
            quote = not quote
            buf.append(text[i])
            i += 1
            continue
        if not quote and text.startswith(op, i):
            parts.append("".join(buf))
            buf = []
            i += len(op)
            continue
        buf.append(text[i])
        i += 1
    parts.append("".join(buf))
    return parts


def _github_atom(text: str, env: dict):
    text = text.strip()
    if len(text) >= 2 and text[0] == "'" and text[-1] == "'":
        return text[1:-1]
    if text == "true":
        return True
    if text == "false":
        return False
    if text not in env:
        raise AssertionError(f"unbound expression name {text!r}")
    return env[text]


def _github_and(text: str, env: dict):
    value = True
    for part in _split_github_ops(text, "&&"):
        piece = part.strip()
        if "==" in piece:
            left, right = _split_github_ops(piece, "==")
            value = _github_atom(left, env) == _github_atom(right, env)
        else:
            value = _github_atom(piece, env)
        if not value:
            return value
    return value


def _github_expr(expr: str, env: dict):
    """JavaScript-style && / || evaluation used by GitHub Actions expressions."""
    value = False
    for part in _split_github_ops(expr, "||"):
        value = _github_and(part, env)
        if value:
            return value
    return value


def _run_sim_book_mode_expression() -> str:
    workflow = (_WORKFLOWS / "run-sim.yml").read_text(encoding="utf-8")
    match = re.search(
        r"ROUND_MATCHUP_REQUIRED_BOOK_PAIR:\s*\$\{\{\s*(.+?)\s*\}\}",
        workflow,
    )
    assert match, "run-sim.yml does not set ROUND_MATCHUP_REQUIRED_BOOK_PAIR"
    return match.group(1)


def test_run_sim_dispatch_input_maps_betonline_only_and_is_exclusive():
    workflow = (_WORKFLOWS / "run-sim.yml").read_text(encoding="utf-8")
    inputs = workflow.split("workflow_dispatch:", 1)[1].split("permissions:", 1)[0]
    betonline_input = inputs.split("use_betonline_only:", 1)[1]
    assert "type: boolean" in betonline_input
    assert "default: false" in betonline_input
    assert "use_pinnacle_for_betcris:" in inputs
    assert not _active_schedule(workflow)

    guard = workflow.split("- name: Reject conflicting manual book modes", 1)[1]
    guard = guard.split("- name: Checkout repository", 1)[0]
    assert "inputs.use_betonline_only && inputs.use_pinnacle_for_betcris" in guard
    assert "mutually exclusive" in guard

    expr = _run_sim_book_mode_expression()
    assert "github.event_name == 'workflow_dispatch'" in expr
    cases = [
        ("workflow_dispatch", False, False, "betcris_betonline"),
        ("workflow_dispatch", True, False, "betonline_only"),
        ("workflow_dispatch", False, True, "betonline_pinnacle"),
        ("workflow_dispatch", True, True, "conflicting_book_modes"),
        ("schedule", True, False, "betcris_betonline"),
        ("schedule", False, True, "betcris_betonline"),
        ("schedule", True, True, "betcris_betonline"),
        ("repository_dispatch", True, False, "betcris_betonline"),
    ]
    for event_name, betonline_only, pinnacle, expected in cases:
        assert _github_expr(
            expr,
            {
                "github.event_name": event_name,
                "inputs.use_betonline_only": betonline_only,
                "inputs.use_pinnacle_for_betcris": pinnacle,
            },
        ) == expected


def test_scheduled_workflows_do_not_enable_betonline_only():
    for path in sorted(_WORKFLOWS.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        if not _active_schedule(text):
            continue
        assert "betonline_only" not in text, path.name
        assert "use_betonline_only" not in text, path.name


def test_betonline_only_does_not_change_the_min_matchup_floor():
    source = Path(__file__).resolve().parent.joinpath("round_sim.py").read_text(
        encoding="utf-8"
    )
    assert 'os.getenv("ROUND_SIM_MIN_BOOK_MATCHUPS", "5")' in source
    workflow = (_WORKFLOWS / "run-sim.yml").read_text(encoding="utf-8")
    assert "ROUND_SIM_MIN_BOOK_MATCHUPS" not in workflow
