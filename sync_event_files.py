"""
Pull-time event guard for stale prediction files.

Background: the root-level pred files the sim reads (r*_live_model.csv,
model_predictions_r*.csv, ...) are gitignored locally and only their copies in
dashboard_data/ are pushed during the event. So a `git pull` never refreshes the
working files, and a previous event's files can linger — which once caused
round_sim.py to merge last week's known-rounds data with this week's field.

This script fixes that. When the current event (golf_sims sheet, cell B20) differs
from the last-synced event recorded in .synced_event, the local root pred files are
from a PREVIOUS event: it deletes them and re-syncs the current event's versions
from the freshly-pulled dashboard_data/ copies. On the very first run it just
records the current event (assumes existing local files belong to it) so it never
wipes a generating machine's fresh work.

Invoked automatically by .githooks/post-merge after every `git pull`/merge. It is
designed to NEVER block a pull: any failure (no network, no credentials, sheet
unreachable) is caught and the process exits 0 without touching files.

The managed file set mirrors ROOT_FILES in push_dashboard_data.py — the same files
pushed to dashboard_data/ during the event — so the two stay in lockstep.

The field-filter 90% match guard in round_sim.py is the backstop: if a stale file
ever slips through, the sim aborts rather than storing garbage.
"""
import os
import sys
import shutil

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MARKER = os.path.join(PROJECT_ROOT, ".synced_event")
DASHBOARD_DATA = os.path.join(PROJECT_ROOT, "dashboard_data")

# Fallback list if push_dashboard_data can't be imported. Kept minimal — just the
# files whose staleness corrupts the sim (load_known_rounds + predictions).
_FALLBACK_FILES = [
    "model_predictions_r1.csv", "model_predictions_r2.csv",
    "model_predictions_r3.csv", "model_predictions_r4.csv",
    "r1_live_model.csv", "r2_live_model.csv",
    "r3_live_model.csv", "r4_live_model.csv",
    "simulated_probs_live.csv",
]


def _current_event():
    """Current tournament slug from the golf_sims sheet, cell B20 (lowercased)."""
    from sheet_config import _connect_sheet
    ws = _connect_sheet()
    val = ws.acell("B20").value
    return (val or "").strip().lower()


def _read_marker():
    try:
        with open(MARKER) as f:
            return f.read().strip().lower()
    except FileNotFoundError:
        return ""


def _write_marker(event):
    with open(MARKER, "w") as f:
        f.write(event + "\n")


def _managed_files():
    """Event-specific root pred files — mirror push_dashboard_data.ROOT_FILES."""
    try:
        from push_dashboard_data import ROOT_FILES
        return list(ROOT_FILES)
    except Exception:
        return list(_FALLBACK_FILES)


def sync():
    current = _current_event()
    if not current:
        print("[sync-event] current event unreadable (B20 blank) — skipping")
        return

    previous = _read_marker()

    if previous == "":
        # First run after installing the hook: assume the local files belong to the
        # current event and just record it. Never wipe on bootstrap.
        _write_marker(current)
        print(f"[sync-event] initialized marker to '{current}' (no cleanup on first run)")
        return

    if previous == current:
        return  # same event — never touch fresh mid-event files

    print(f"[sync-event] event changed: '{previous}' -> '{current}'")
    removed = resynced = 0
    for name in _managed_files():
        root_path = os.path.join(PROJECT_ROOT, name)
        if os.path.exists(root_path):
            os.remove(root_path)
            removed += 1
        src = os.path.join(DASHBOARD_DATA, name)
        if os.path.exists(src):
            shutil.copy2(src, root_path)
            resynced += 1
    print(f"[sync-event] cleared {removed} stale pred file(s); "
          f"resynced {resynced} from dashboard_data/ for '{current}'")
    _write_marker(current)


if __name__ == "__main__":
    try:
        sync()
    except Exception as e:  # never block a pull on this guard
        print(f"[sync-event] skipped (non-fatal): {e}")
    sys.exit(0)
