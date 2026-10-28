"""Resolve the required sharp books for automated round simulations.

Normal runs require BetCRIS and BetOnline.  An explicit workflow_dispatch
repair may substitute Pinnacle for BetCRIS, or require BetOnline alone.
Scheduled runs, repository_dispatch, and every other event stay on the
production BetCRIS + BetOnline pair.
"""

from __future__ import annotations

from collections.abc import Mapping
import os


DEFAULT_REQUIRED_BOOKS = ("betcris", "betonline")
MANUAL_PINNACLE_BOOKS = ("betonline", "pinnacle")
MANUAL_BETONLINE_ONLY_BOOKS = ("betonline",)

REQUIRED_BOOK_PAIR_ENV = "ROUND_MATCHUP_REQUIRED_BOOK_PAIR"
DEFAULT_BOOK_PAIR_MODE = "betcris_betonline"
MANUAL_PINNACLE_MODE = "betonline_pinnacle"
MANUAL_BETONLINE_ONLY_MODE = "betonline_only"

_BOOKS_BY_MODE = {
    DEFAULT_BOOK_PAIR_MODE: DEFAULT_REQUIRED_BOOKS,
    MANUAL_PINNACLE_MODE: MANUAL_PINNACLE_BOOKS,
    MANUAL_BETONLINE_ONLY_MODE: MANUAL_BETONLINE_ONLY_BOOKS,
}

# These modes exist only for an operator who starts the workflow by hand.
# The same event check applies to every entry.
_DISPATCH_ONLY_MODE_ERRORS = {
    MANUAL_PINNACLE_MODE: (
        "Pinnacle may replace BetCRIS only during an explicit "
        "workflow_dispatch repair run"
    ),
    MANUAL_BETONLINE_ONLY_MODE: (
        "BetOnline-only coverage is allowed only during an explicit "
        "workflow_dispatch repair run"
    ),
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


def _is_workflow_dispatch(env: Mapping[str, str]) -> bool:
    return (
        str(env.get("GITHUB_EVENT_NAME") or "").strip().lower()
        == "workflow_dispatch"
    )


def resolve_required_matchup_books(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return the validated books for this process.

    ``betonline_pinnacle`` and ``betonline_only`` are available only when
    ``GITHUB_EVENT_NAME`` is ``workflow_dispatch``.  Absent, scheduled, and
    scraper-dispatched modes stay on BetCRIS + BetOnline.
    """
    env = os.environ if environ is None else environ
    mode = str(env.get(REQUIRED_BOOK_PAIR_ENV) or DEFAULT_BOOK_PAIR_MODE).strip().lower()
    if mode not in _BOOKS_BY_MODE:
        allowed = ", ".join(sorted(_BOOKS_BY_MODE))
        raise ValueError(
            f"Unsupported {REQUIRED_BOOK_PAIR_ENV}={mode!r}; expected one of {allowed}"
        )
    dispatch_error = _DISPATCH_ONLY_MODE_ERRORS.get(mode)
    if dispatch_error and not _is_workflow_dispatch(env):
        raise ValueError(dispatch_error)
    return _BOOKS_BY_MODE[mode]
