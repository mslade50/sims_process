"""
monday_grading.py - Scheduled Monday Grading Pipeline

Orchestrates the post-tournament grading pipeline:
1. Detect last completed event(s) from DataGolf. On opposite-field /
   co-sanctioned multi-event weeks (e.g. ISCO Championship + Genesis Scottish
   Open) every same-week event is detected by shared DataGolf date.
2. Check which event(s) actually have ungraded bets (idempotent for retry)
3. Run grade_bets.py for the event(s) with bets — passing --event-id on
   multi-event weeks so we grade the event we bet on, not whichever event
   DataGolf happens to list first
4. Verify grading succeeded
5. Run sg_diagnostic.py --no-email (graceful skip if prediction CSV missing)
6. Run push_dashboard_data.py (graceful skip if no files to push)

Exit codes:
    0 = grading complete (just now or already done)
    1 = data not available yet (triggers retry at next scheduled run)
    2 = unexpected error

Usage:
    python monday_grading.py              # Auto-detect event, grade, deploy
    python monday_grading.py --dry-run    # Preview without running sub-scripts

Scheduled via .github/workflows/monday-grading.yml:
    Monday 9 AM EST (14:00 UTC) with 10 AM EST retry (15:00 UTC)
"""

import os
import sys
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers — reuse grade_bets functions without triggering argparse
# ---------------------------------------------------------------------------

def _setup_env():
    """Ensure project root is on sys.path and .env is loaded."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from dotenv import load_dotenv
    load_dotenv()
    return project_root


def _diagnostic_tourney_slug(project_root, event_id, event_name):
    """Resolve the artifact slug for a completed event without using sim_inputs.

    push_dashboard_data.py archives finish distributions under
    ``permanent_data/historical_dists/{event_id}_{tourney}`` during the event,
    making that directory the authoritative bridge between DataGolf's event
    name and the short artifact slug (for example 525 -> ``3m_open``).
    """
    archive_root = Path(project_root) / "permanent_data" / "historical_dists"
    matches = sorted(
        path for path in archive_root.glob(f"{event_id}_*") if path.is_dir()
    )
    if len(matches) == 1:
        return matches[0].name.split("_", 1)[1]
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        print(
            f"  [warn] Multiple diagnostic archive slugs for event {event_id}: "
            f"{names}; skipping diagnostic rather than guessing."
        )
        return None

    fallback = str(event_name or "").lower().strip().replace(" ", "_")
    source = (
        Path(project_root)
        / "permanent_data"
        / f"avg_expected_cat_sg_{fallback}.csv"
    )
    if fallback and source.exists():
        return fallback

    print(
        f"  [warn] No diagnostic artifact slug/source found for "
        f"{event_name} (ID {event_id}); skipping."
    )
    return None


def _detect_week():
    """
    Detect the most recently completed event and every other current-year event
    that finished the SAME week (shares the DataGolf `date`).

    A normal week returns one event, so behaviour is identical to the old
    single-event auto-detect. Opposite-field / co-sanctioned weeks (e.g. ISCO
    Championship + Genesis Scottish Open, both dated 2026-07-12) return every
    same-week event, so the caller can grade whichever one(s) actually have bets
    instead of whichever event DataGolf happens to list first.

    Same-week detection is date-based (from the event list), so it does NOT
    depend on every sibling's results being posted yet, and — critically — it
    does NOT read sim_inputs, which by grading time has already rolled forward
    to next week's tournament.

    Returns (multi, [(event_id, event_name, year), ...]) where `multi` is True
    on a multi-event week. Raises SystemExit(1) if no event has completed
    results yet (triggers the workflow's retry at the next scheduled run).
    """
    from grade_bets import fetch_event_list, fetch_historical_results
    from datetime import datetime

    print("\n  Detecting last completed event(s) from DataGolf...")
    events = fetch_event_list()
    if not events:
        print("  ERROR: Could not fetch event list from DataGolf.")
        sys.exit(1)

    year = datetime.now().year
    current = [e for e in events if e.get("calendar_year") == year]

    # Anchor = most recent current-year event with posted results. The event
    # list is ordered most-recent-first.
    anchor = None
    for event in current:
        results = fetch_historical_results(event.get("event_id"), year)
        if not results.empty and len(results) > 10:
            anchor = event
            break

    if anchor is None:
        print("  No completed event with results found. DataGolf may not have updated yet.")
        sys.exit(1)

    anchor_date = anchor.get("date")

    # Siblings = every current-year event sharing the anchor's date (incl.
    # anchor). Date-based, so a sibling whose results lag is still detected.
    siblings = [
        (e.get("event_id"), e.get("event_name"), year)
        for e in current
        if e.get("date") and e.get("date") == anchor_date
    ]

    multi = len(siblings) > 1
    if multi:
        names = ", ".join(f"{n} (ID {i})" for i, n, _ in siblings)
        print(f"  Multi-event week detected ({anchor_date}): {names}")
    else:
        i, n, _ = siblings[0]
        print(f"  Found: {n} (ID {i})")

    return multi, siblings


def _count_ungraded(spreadsheet, event_id):
    """Count total ungraded bets across all source tabs for this event."""
    from grade_bets import get_ungraded_bets
    import time

    # Include "Live" — exchange/live finish bets land there and must not be
    # skipped just because the standard tabs are already graded.
    tabs = ["Tournament Matchups", "Round Matchups", "Finish Positions", "Live"]
    total = 0

    for i, tab in enumerate(tabs):
        if i > 0:
            time.sleep(3)  # avoid Sheets rate limit
        df = get_ungraded_bets(spreadsheet, tab, event_id=event_id)
        n = len(df) if not df.empty else 0
        print(f"    {tab}: {n} ungraded")
        total += n

    return total


def _alert_orphan_ungraded(spreadsheet, week_event_ids):
    """Page on ungraded bets whose event_id matches NO event detected this week.

    A DP World bet (grading only queries tour=pga), a mis-stamped event_id, or a
    prior week left ungraded would otherwise print 'Already graded' forever and
    stay buried. Scans UNGRADED rows only (get_all_event_ids would drag in all
    history)."""
    import time
    from grade_bets import get_ungraded_bets
    week_ids = {str(e) for e in week_event_ids}
    orphans = {}
    for i, tab in enumerate(["Tournament Matchups", "Round Matchups",
                             "Finish Positions", "Live"]):
        if i > 0:
            time.sleep(3)
        try:
            df = get_ungraded_bets(spreadsheet, tab)   # all events
        except Exception as e:
            print(f"    orphan scan: {tab} failed ({e})")
            continue
        if df is None or df.empty or "event_id" not in df.columns:
            continue
        for eid, n in df["event_id"].astype(str).value_counts().items():
            if eid and eid not in week_ids and eid.lower() not in ("nan", "none", ""):
                orphans[eid] = orphans.get(eid, 0) + int(n)
    if orphans:
        msg = ("[monday_grading] ORPHAN ungraded bets, event(s) not detected this "
               "week (euro tour? mis-stamped id? prior week?): "
               + ", ".join(f"event {e}: {n} bets" for e, n in sorted(orphans.items())))
        print("  " + msg)
        try:
            from maker_alerts import send_telegram
            send_telegram(msg)
        except Exception as e:
            print(f"    orphan alert failed ({e})")
    return orphans


def _run_subprocess(cmd, label, ignore_failure=False):
    """Run a subprocess, printing output. Returns True on success."""
    print(f"\n  Running {label}...")
    print(f"  Command: {' '.join(cmd)}")
    print("  " + "-" * 50)

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        if ignore_failure:
            print(f"  [warn] {label} exited with code {result.returncode} (ignored)")
            return False
        else:
            print(f"  ERROR: {label} failed with exit code {result.returncode}")
            return False

    print(f"  {label} completed successfully.")
    return True


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Monday grading pipeline orchestrator")
    parser.add_argument("--dry-run", action="store_true", help="Preview checks without running grading")
    args = parser.parse_args()

    project_root = _setup_env()
    python = sys.executable

    print("\n" + "=" * 60)
    print("  MONDAY GRADING PIPELINE")
    print("=" * 60)

    if args.dry_run:
        print("  MODE: DRY RUN")

    # ------------------------------------------------------------------
    # Step 1: Detect completed event(s) for the week
    # ------------------------------------------------------------------
    multi, week_events = _detect_week()

    # ------------------------------------------------------------------
    # Step 2: Check which event(s) have ungraded bets (idempotent for retries)
    # ------------------------------------------------------------------
    print("\n  Connecting to Google Sheets...")
    from sheets_storage import get_spreadsheet
    spreadsheet = get_spreadsheet()

    if multi:
        print(f"\n  {len(week_events)} events finished this week — grading only the one(s) with bets.")

    to_grade = []  # [(event_id, event_name, ungraded_count)]
    for event_id, event_name, year in week_events:
        print(f"\n  Checking for ungraded bets — {event_name}...")
        n = _count_ungraded(spreadsheet, event_id)
        if n > 0:
            to_grade.append((event_id, event_name, n))

    # Ungraded bets for an event NOT in this week's (pga-detected) list would
    # otherwise be silently skipped forever: surface them loudly.
    _alert_orphan_ungraded(spreadsheet, [e for e, _, _ in week_events])

    if args.dry_run:
        if to_grade:
            for event_id, event_name, n in to_grade:
                print(
                    f"\n  [DRY RUN] Would grade {n} bets for "
                    f"{event_name} (ID {event_id})."
                )
        else:
            label = ", ".join(name for _, name, _ in week_events)
            print(f"\n  [DRY RUN] Already graded: {label}.")
        for event_id, event_name, year in week_events:
            slug = _diagnostic_tourney_slug(
                project_root, event_id, event_name
            )
            if slug:
                print(
                    f"  [DRY RUN] Would diagnose {event_name} "
                    f"(ID {event_id}, slug {slug})."
                )
        print("  [DRY RUN] No grading, diagnostics, or dashboard push executed.")
        print("=" * 60 + "\n")
        sys.exit(0)

    if not to_grade:
        label = ", ".join(name for _, name, _ in week_events)
        print(f"\n  Already graded: 0 ungraded bets for {label}.")
        print("  Skipping grade_bets.py — nothing to grade.")
    else:
        # ------------------------------------------------------------------
        # Step 3: Grade each event that has bets.
        #   Single-event weeks keep the original bare invocation (grade_bets
        #   auto-detects the same event). Multi-event weeks pass --event-id so
        #   we grade the event we bet on, not whichever DataGolf lists first.
        # ------------------------------------------------------------------
        for event_id, event_name, n in to_grade:
            print(f"\n  Found {n} ungraded bets for {event_name}.")
            cmd = [python, "grade_bets.py"]
            if multi:
                cmd += ["--event-id", str(event_id), "--event-name", str(event_name)]
            if not _run_subprocess(cmd, f"grade_bets.py ({event_name})"):
                print("  grade_bets.py reported failure.")
                sys.exit(1)

        # ------------------------------------------------------------------
        # Step 4: Verify grading succeeded
        # ------------------------------------------------------------------
        import time
        print("\n  Verifying grading results...")
        time.sleep(5)  # brief pause for Sheets propagation

        # Reconnect to get fresh data (cache may be stale)
        spreadsheet = get_spreadsheet()
        remaining = sum(_count_ungraded(spreadsheet, event_id)
                        for event_id, event_name, n in to_grade)

        if remaining > 0:
            print(f"\n  WARNING: {remaining} bets still ungraded after running grade_bets.py.")
            print("  This may indicate DataGolf data was incomplete.")
            sys.exit(1)

        print("\n  Verification passed: all bets graded.")

    # ------------------------------------------------------------------
    # Step 5: Run sg_diagnostic.py --no-email (always runs)
    # ------------------------------------------------------------------
    for event_id, event_name, year in week_events:
        slug = _diagnostic_tourney_slug(project_root, event_id, event_name)
        if not slug:
            continue
        _run_subprocess(
            [
                python,
                "sg_diagnostic.py",
                "--event-id", str(event_id),
                "--tourney", slug,
                "--year", str(year),
                "--no-email",
            ],
            f"sg_diagnostic.py ({event_name})",
            ignore_failure=True,
        )

    # ------------------------------------------------------------------
    # Step 6: Run push_dashboard_data.py (always runs)
    # ------------------------------------------------------------------
    _run_subprocess(
        [python, "push_dashboard_data.py"],
        "push_dashboard_data.py",
        ignore_failure=True,
    )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    done_events = to_grade if to_grade else week_events
    done_label = ", ".join(name for _, name, _ in done_events)
    print("\n" + "=" * 60)
    print(f"  GRADING PIPELINE COMPLETE — {done_label}")
    print("=" * 60 + "\n")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"\n  UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
