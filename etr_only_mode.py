"""TEMPORARY legacy ETR-only mode switch (owner decision 2026-10-19).

Until the ETR golf sim is refactored after the season, the legacy weekly chain
runs only to produce ETR input files.  With ``ETR_ONLY_MODE=1`` every path that
could touch production (emails, Telegram, bets, odds board, sim_fairs, dashboard
data, exchange orders, git pushes) raises via :func:`refuse`.

Stdlib only.  With the variable unset every function here is a no-op.
"""
import os

BANNER = (
    "TEMPORARY LEGACY ETR-ONLY MODE (until post-season ETR refactor) -- "
    "owner decision 2026-10-19 -- produces ETR inputs only; NOT production"
)


def etr_only() -> bool:
    return (os.environ.get("ETR_ONLY_MODE") or "").strip().lower() in ("1", "true", "yes", "on")


def refuse(what: str) -> None:
    """Raise when ETR-only mode is active; otherwise do nothing."""
    if etr_only():
        raise RuntimeError(
            f"ETR-only mode: {what} is disabled (temporary legacy ETR mode, owner 2026-10-19)"
        )


def print_banner() -> None:
    if etr_only():
        print(f"*** {BANNER} ***", flush=True)
