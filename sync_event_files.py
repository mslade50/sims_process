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

Every resync (both the event-change and mid-event paths) is gated by
dashboard_data/.sync_manifest.json, written by push_dashboard_data.py: a file is
only copied to root when the manifest attributes that exact file to the current
event. dashboard_data/ holds mixed vintages mid-week — files the new event hasn't
produced yet are still the previous event's — and the pre-manifest hook once
reinstalled the very files it had just cleared (2026-08-10: a stale wyndham
model_predictions_r1.csv silently aborted the sim-fairs publish). With no
manifest (or one for another event/year) the hook clears and resyncs nothing;
the weekly pipeline regenerates and the next push restamps.

Invoked automatically by .githooks/post-merge after every `git pull`/merge. It is
designed to NEVER block a pull: any failure (no network, no credentials, sheet
unreachable) is caught and the process exits 0 without touching files.

The managed file set mirrors ROOT_FILES in push_dashboard_data.py — the same files
pushed to dashboard_data/ during the event — so the two stay in lockstep.

The field-filter 90% match guard in round_sim.py is the backstop: if a stale file
ever slips through, the sim aborts rather than storing garbage.
"""
import json
import os
import sys
import shutil
from datetime import date

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MARKER = os.path.join(PROJECT_ROOT, ".synced_event")
DASHBOARD_DATA = os.path.join(PROJECT_ROOT, "dashboard_data")
SYNC_MANIFEST = os.path.join(DASHBOARD_DATA, ".sync_manifest.json")

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


def _manifest_allowed(current):
    """Files the dashboard_data/ manifest attributes to the CURRENT event.

    Matches on event slug + calendar year (slugs recur annually). Missing,
    unreadable, or other-event manifest -> empty set: resync nothing."""
    try:
        with open(SYNC_MANIFEST) as f:
            manifest = json.load(f)
        if str(manifest.get("event", "")).strip().lower() != current:
            return set()
        year = manifest.get("year")
        if year is not None and int(year) != date.today().year:
            return set()
        return {str(name) for name in manifest.get("files", [])}
    except Exception:
        return set()


def _resync_newer(allowed):
    """Mid-event refresh: copy any manifest-verified dashboard_data/ pred file
    that is newer than (or missing in) its root counterpart.

    The original event-change-only sync missed the case where a SECOND push of the
    *same* event lands in dashboard_data/ after the marker already flipped — a
    consuming machine then keeps reading last-push (or last-event) root files. This
    closes that gap.

    Freshness is per-file by mtime, and we only copy when dashboard_data/ is strictly
    newer. That preserves the original "never clobber a generating machine's fresh
    work" guarantee: on a generating machine the just-written root file is newer than
    its pushed dashboard_data/ copy, so it is left untouched.
    """
    resynced = 0
    for name in _managed_files():
        if name not in allowed:
            continue
        src = os.path.join(DASHBOARD_DATA, name)
        if not os.path.exists(src):
            continue
        root_path = os.path.join(PROJECT_ROOT, name)
        if os.path.exists(root_path) and os.path.getmtime(root_path) >= os.path.getmtime(src):
            continue  # root is as-new-or-newer — leave it (don't clobber local work)
        shutil.copy2(src, root_path)
        resynced += 1
    if resynced:
        print(f"[sync-event] mid-event refresh: resynced {resynced} newer file(s) "
              f"from dashboard_data/")
    return resynced


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

    allowed = _manifest_allowed(current)

    if previous == current:
        # Same event — never wipe, but DO pull in any newer same-event copies that a
        # later push deposited in dashboard_data/ (the gap that left R3 reading a
        # previous event's root files).
        _resync_newer(allowed)
        return

    print(f"[sync-event] event changed: '{previous}' -> '{current}'")
    removed = resynced = 0
    for name in _managed_files():
        root_path = os.path.join(PROJECT_ROOT, name)
        if os.path.exists(root_path):
            os.remove(root_path)
            removed += 1
        if name not in allowed:
            continue
        src = os.path.join(DASHBOARD_DATA, name)
        if os.path.exists(src):
            shutil.copy2(src, root_path)
            resynced += 1
    if resynced:
        print(f"[sync-event] cleared {removed} stale pred file(s); "
              f"resynced {resynced} manifest-verified file(s) for '{current}'")
    else:
        print(f"[sync-event] cleared {removed} stale pred file(s); no manifest-verified "
              f"files for '{current}' in dashboard_data/ — resynced nothing "
              f"(pipeline regenerates)")
    _write_marker(current)


if __name__ == "__main__":
    try:
        sync()
    except Exception as e:  # never block a pull on this guard
        print(f"[sync-event] skipped (non-fatal): {e}")
    sys.exit(0)
