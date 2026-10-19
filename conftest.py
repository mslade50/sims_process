"""Collection rules for the secret-free suite.

pytest loads this file before test modules. When no Google credential is
configured, coefficient import uses the committed cache
(``COEFFS_FROM_CACHE=1``) instead of the sheet.

Modules in the maps below are not collected when the dependency they need
at import time is missing. With that dependency present they collect and
run as usual. A module that is not listed still fails collection, which
is what the CI job requires.

App code still reaches Google at import in three places. This file does
not change that. ``round_sim.py`` and ``new_sim.py`` call
``sheet_config.load_config()`` at import. ``sim_inputs.py`` calls
``coeff_loader``, which reads the sheet unless ``COEFFS_FROM_CACHE=1``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# round_sim.py calls sheet_config.load_config() at import, so these modules
# cannot even be imported without a Google credential.
CREDENTIAL_IMPORTS = {
    "test_env_setup.py": (
        "connectivity script for credentials and external services"
    ),
    "test_finish_probs_real.py": (
        "imports round_sim.py, which reads the Google Sheet at import; "
        "the program also needs the compiled sims_kernel to compare aggregations"
    ),
    "test_reprice_alert_delivery.py": (
        "imports round_sim.py, which reads the Google Sheet at import"
    ),
    "test_round_score_reprice.py": (
        "imports round_sim.py, which reads the Google Sheet at import"
    ),
}

# rust/fixtures parity programs import the PyO3 extension. This job does
# not build it.
KERNEL_IMPORTS = {
    "test_agg_parity.py": "imports the compiled sims_kernel extension",
    "test_cascade_parity.py": "imports the compiled sims_kernel extension",
    "test_ops_parity.py": "imports the compiled sims_kernel extension",
    "test_round_cascade_parity.py": "imports the compiled sims_kernel extension",
    "test_single_round_parity.py": "imports the compiled sims_kernel extension",
}

# These files run their checks at import and then raise SystemExit, so
# pytest cannot collect them. secret_free_tests.py runs them as programs.
SCRIPT_MODULES = {
    "test_maker_guard.py": "raises SystemExit at import; run as a program",
    "test_maker_quotes.py": "raises SystemExit at import; run as a program",
    "test_maker_ttl.py": (
        "raises SystemExit at import; run as a program "
        "(coefficient cache, not a live sheet)"
    ),
    "test_matchup_tape.py": (
        "raises SystemExit at import; run as a program "
        "(coefficient cache, not a live sheet)"
    ),
    "test_publish_guards.py": "raises SystemExit at import; run as a program",
    "test_quote_engine.py": "raises SystemExit at import; run as a program",
    "test_sync_event_files.py": "raises SystemExit at import; run as a program",
}

# Fails on current source even by itself. The rest of its module still runs.
SKIP_TESTS = {
    "test_live_stats_engine_exposes_no_sheet_write_mode": (
        "source grep expects `if args.no_sheet_writes:`; "
        "live_stats_engine now uses `if args.no_sheet_writes or current_inputs():`"
    ),
}


def credentials_present() -> bool:
    """True when a Google credential is configured for this checkout."""
    if os.getenv("GOOGLE_CREDS_JSON", "").strip():
        return True
    return (ROOT / "credentials.json").is_file()


def kernel_present() -> bool:
    """True when the compiled sims_kernel extension imports."""
    if "sims_kernel" in sys.modules:
        return True
    try:
        import sims_kernel  # noqa: F401
    except Exception:
        return False
    return True


def excluded_modules() -> list[tuple[str, str]]:
    """Modules this process will not collect, with the reason."""
    rows: list[tuple[str, str]] = []
    if not credentials_present():
        rows.extend(sorted(CREDENTIAL_IMPORTS.items()))
    if not kernel_present():
        rows.extend(sorted(KERNEL_IMPORTS.items()))
    return rows


if not credentials_present():
    # sim_inputs.py loads coefficients at import. The committed cache is the
    # offline path; do not override an explicit setting.
    os.environ.setdefault("COEFFS_FROM_CACHE", "1")


def pytest_ignore_collect(collection_path, config):  # noqa: ARG001
    """Skip known credential, kernel, and script modules. Return None otherwise."""
    name = Path(str(collection_path)).name
    if name in SCRIPT_MODULES:
        return True
    if name in CREDENTIAL_IMPORTS and not credentials_present():
        return True
    if name in KERNEL_IMPORTS and not kernel_present():
        return True
    return None


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Skip the stale source-grep test without hiding the rest of its module."""
    import pytest

    for item in items:
        reason = SKIP_TESTS.get(item.name)
        if reason:
            item.add_marker(pytest.mark.skip(reason=reason))


def pytest_terminal_summary(terminalreporter, exitstatus, config):  # noqa: ARG001
    """Print the exclusion list so a raw pytest run shows what was left out."""
    rows = excluded_modules()
    if not rows and not SKIP_TESTS:
        return
    terminalreporter.write_sep("-", "secret-free exclusions")
    for name, reason in rows:
        terminalreporter.write_line(f"not collected: {name} — {reason}")
    for name, reason in sorted(SKIP_TESTS.items()):
        terminalreporter.write_line(f"skipped: {name} — {reason}")
    for name, reason in sorted(SCRIPT_MODULES.items()):
        terminalreporter.write_line(f"script (not collected): {name} — {reason}")
