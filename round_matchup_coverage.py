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


def is_actionable_matchup_price(book: str, price) -> bool:
    """Whether a nested book quote has known, supported settlement metadata.

    BetCRIS historically omitted its handicap column, so those quotes are not
    allowed to satisfy an automated coverage gate. Other books' legacy straight
    schema remains supported. Explicit BetCRIS contracts must be straight or a
    reciprocal +/-0.5 pair.
    """
    if not isinstance(price, dict):
        return False
    book_key = str(book or "").strip().lower()
    is_betcris = "betcris" in book_key or "bookmaker" in book_key
    has_line_keys = "p1_line" in price or "p2_line" in price
    if not has_line_keys:
        return not is_betcris
    verified = (
        price.get("line_verified") is True
        or str(price.get("line_verified") or "").strip().lower()
        in ("1", "true", "yes")
    )
    if is_betcris and not verified:
        return False
    if "p1_line" not in price or "p2_line" not in price:
        return False

    def missing_line_value(value):
        return value is None or str(value).strip().lower() in ("", "nan", "none")

    p1_missing = missing_line_value(price.get("p1_line"))
    p2_missing = missing_line_value(price.get("p2_line"))
    if p1_missing != p2_missing:
        return False
    if p1_missing and p2_missing:
        return True

    def line_value(value):
        try:
            return float(value)
        except (TypeError, ValueError, OverflowError):
            return None

    p1_line = line_value(price.get("p1_line"))
    p2_line = line_value(price.get("p2_line"))
    if p1_line is None or p2_line is None:
        return False
    if abs(p1_line) <= 1e-9 and abs(p2_line) <= 1e-9:
        return True
    return (
        verified
        and
        abs(abs(p1_line) - 0.5) <= 1e-9
        and abs(abs(p2_line) - 0.5) <= 1e-9
        and abs(p1_line + p2_line) <= 1e-9
    )


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
