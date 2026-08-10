"""Tests for sync_event_files manifest gating (2026-08-10: the unconditional
resync reinstalled wyndham's model_predictions_r1.csv on the Monday event flip,
silently aborting the sim-fairs publish). Runs against a throwaway temp dir —
the sheet lookup and push_dashboard_data import are monkeypatched.
Run: python test_sync_event_files.py"""
import json
import os
import shutil
import tempfile
import time

import sync_event_files as sef

_p = _f = 0


def eq(name, got, want):
    global _p, _f
    if got == want:
        _p += 1
    else:
        _f += 1
        print(f"FAIL {name}: got {got!r}, want {want!r}")


MANAGED = ["a.csv", "b.csv"]


def setup(tmp, current="st_jude", marker=None, manifest=None,
          root=(), dashboard=()):
    sef.PROJECT_ROOT = tmp
    sef.DASHBOARD_DATA = os.path.join(tmp, "dashboard_data")
    sef.MARKER = os.path.join(tmp, ".synced_event")
    sef.SYNC_MANIFEST = os.path.join(sef.DASHBOARD_DATA, ".sync_manifest.json")
    sef._current_event = lambda: current
    sef._managed_files = lambda: list(MANAGED)
    os.makedirs(sef.DASHBOARD_DATA)
    if marker is not None:
        with open(sef.MARKER, "w") as f:
            f.write(marker + "\n")
    if manifest is not None:
        with open(sef.SYNC_MANIFEST, "w") as f:
            json.dump(manifest, f)
    now = time.time()
    for name in root:
        path = os.path.join(tmp, name)
        with open(path, "w") as f:
            f.write(f"root {name}")
        os.utime(path, (now - 3600, now - 3600))
    for name in dashboard:
        path = os.path.join(sef.DASHBOARD_DATA, name)
        with open(path, "w") as f:
            f.write(f"dashboard {name}")
        os.utime(path, (now, now))


def run_case(name, check, **kwargs):
    tmp = tempfile.mkdtemp(prefix="sync_event_test_")
    try:
        setup(tmp, **kwargs)
        sef.sync()
        check(name, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


THIS_YEAR = time.localtime().tm_year


# 1. Event change, manifest still stamped for the PREVIOUS event (the Monday
#    state): clear-only — nothing resynced, marker updated.
def check_clear_only(name, tmp):
    eq(f"{name}: root a.csv removed", os.path.exists(os.path.join(tmp, "a.csv")), False)
    eq(f"{name}: root b.csv removed", os.path.exists(os.path.join(tmp, "b.csv")), False)
    with open(sef.MARKER) as f:
        eq(f"{name}: marker flipped", f.read().strip(), "st_jude")

run_case(
    "event change / stale manifest", check_clear_only,
    marker="wyndham",
    manifest={"event": "wyndham", "year": THIS_YEAR, "files": MANAGED},
    root=MANAGED, dashboard=MANAGED,
)

# 2. Event change, manifest stamped for the CURRENT event but only listing
#    a.csv: a.csv resynced, b.csv (stale vintage) cleared and NOT resynced.
def check_partial_resync(name, tmp):
    eq(f"{name}: a.csv resynced", os.path.exists(os.path.join(tmp, "a.csv")), True)
    eq(f"{name}: b.csv not resynced", os.path.exists(os.path.join(tmp, "b.csv")), False)

run_case(
    "event change / partial manifest", check_partial_resync,
    marker="wyndham",
    manifest={"event": "st_jude", "year": THIS_YEAR, "files": ["a.csv"]},
    root=MANAGED, dashboard=MANAGED,
)

# 3. Event change, no manifest at all: clear-only.
run_case(
    "event change / no manifest", check_clear_only,
    marker="wyndham", root=MANAGED, dashboard=MANAGED,
)

# 4. Same event: a.csv missing in root and manifest-verified -> copied;
#    b.csv missing in root but NOT in manifest -> left absent.
def check_same_event_missing(name, tmp):
    eq(f"{name}: a.csv restored", os.path.exists(os.path.join(tmp, "a.csv")), True)
    eq(f"{name}: b.csv left absent", os.path.exists(os.path.join(tmp, "b.csv")), False)

run_case(
    "same event / missing root files", check_same_event_missing,
    marker="st_jude",
    manifest={"event": "st_jude", "year": THIS_YEAR, "files": ["a.csv"]},
    root=(), dashboard=MANAGED,
)

# 5. Same event: root file NEWER than dashboard copy (generating machine) —
#    never clobbered, even when manifest-verified.
def check_no_clobber(name, tmp):
    with open(os.path.join(tmp, "a.csv")) as f:
        eq(f"{name}: newer root kept", f.read(), "root a.csv")

def setup_no_clobber(tmp, **kwargs):
    setup(tmp, **kwargs)
    now = time.time()
    os.utime(os.path.join(tmp, "a.csv"), (now + 3600, now + 3600))

_tmp = tempfile.mkdtemp(prefix="sync_event_test_")
try:
    setup_no_clobber(
        _tmp, marker="st_jude",
        manifest={"event": "st_jude", "year": THIS_YEAR, "files": ["a.csv"]},
        root=["a.csv"], dashboard=["a.csv"],
    )
    sef.sync()
    check_no_clobber("same event / newer root", _tmp)
finally:
    shutil.rmtree(_tmp, ignore_errors=True)

# 6. Manifest year mismatch (recurring slug a year later): treated as stale.
run_case(
    "event change / year mismatch", check_clear_only,
    marker="wyndham",
    manifest={"event": "st_jude", "year": THIS_YEAR - 1, "files": MANAGED},
    root=MANAGED, dashboard=MANAGED,
)

print(f"\n{_p} passed, {_f} failed")
raise SystemExit(1 if _f else 0)
