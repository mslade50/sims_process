"""Resolve the required sharp-book pair for automated round simulations.

Normal runs require BetCRIS and BetOnline.  A manually dispatched repair may
substitute Pinnacle for BetCRIS, but scraper-triggered automation cannot enable
that fallback through repository-dispatch payloads or inherited environment.
"""

from __future__ import annotations

from collections.abc import Mapping
import os


DEFAULT_REQUIRED_BOOKS = ("betcris", "betonline")
MANUAL_PINNACLE_BOOKS = ("betonline", "pinnacle")

REQUIRED_BOOK_PAIR_ENV = "ROUND_MATCHUP_REQUIRED_BOOK_PAIR"
DEFAULT_BOOK_PAIR_MODE = "betcris_betonline"
MANUAL_PINNACLE_MODE = "betonline_pinnacle"

_BOOKS_BY_MODE = {
    DEFAULT_BOOK_PAIR_MODE: DEFAULT_REQUIRED_BOOKS,
    MANUAL_PINNACLE_MODE: MANUAL_PINNACLE_BOOKS,
}


def resolve_required_matchup_books(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return the validated book pair for this process.

    The Pinnacle fallback is intentionally available only to GitHub's explicit
    ``workflow_dispatch`` event.  All absent, scheduled, and scraper-dispatched
    modes remain on the production BetCRIS + BetOnline requirement.
    """
    env = os.environ if environ is None else environ
    mode = str(env.get(REQUIRED_BOOK_PAIR_ENV) or DEFAULT_BOOK_PAIR_MODE).strip().lower()
    if mode not in _BOOKS_BY_MODE:
        allowed = ", ".join(sorted(_BOOKS_BY_MODE))
        raise ValueError(
            f"Unsupported {REQUIRED_BOOK_PAIR_ENV}={mode!r}; expected one of {allowed}"
        )
    if (
        mode == MANUAL_PINNACLE_MODE
        and str(env.get("GITHUB_EVENT_NAME") or "").strip().lower()
        != "workflow_dispatch"
    ):
        raise ValueError(
            "Pinnacle may replace BetCRIS only during an explicit "
            "workflow_dispatch repair run"
        )
    return _BOOKS_BY_MODE[mode]
