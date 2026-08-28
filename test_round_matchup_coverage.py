import pytest

from round_matchup_coverage import (
    DEFAULT_REQUIRED_BOOKS,
    MANUAL_PINNACLE_BOOKS,
    resolve_required_matchup_books,
)


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
