"""
monday_grading.py - Scheduled Monday Grading Pipeline

Orchestrates the post-tournament grading pipeline:
1. Detect last completed event from DataGolf
2. Check if grading already done (idempotent for retry)
3. Run grade_bets.py
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


def _get_event_and_results():
    """
    Detect last completed event and fetch results.

    Returns (event_id, event_name, year, results_df) or raises SystemExit.
    """
    from grade_bets import fetch_event_list, fetch_historical_results
    from datetime import datetime

    print("\n  Detecting last completed event from DataGolf...")
    events = fetch_event_list()
    if not events:
        print("  ERROR: Could not fetch event list from DataGolf.")
        sys.exit(1)

    current_year = datetime.now().year

    for event in events:
        if event.get("calendar_year") == current_year:
            event_id = event.get("event_id")
            event_name = event.get("event_name")

            results = fetch_historical_results(event_id, current_year)
            if not results.empty and len(results) > 10:
                print(f"  Found: {event_name} (ID: {event_id}, {len(results)} players)")
                return event_id, event_name, current_year, results

    print("  No completed event with results found. DataGolf may not have updated yet.")
    sys.exit(1)


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
    # Step 1: Detect completed event and verify results available
    # ------------------------------------------------------------------
    event_id, event_name, year, results_df = _get_event_and_results()

    # ------------------------------------------------------------------
    # Step 2: Check if already graded (idempotent for retry runs)
    # ------------------------------------------------------------------
    print("\n  Connecting to Google Sheets...")
    from sheets_storage import get_spreadsheet
    spreadsheet = get_spreadsheet()

    print("\n  Checking for ungraded bets...")
    ungraded_count = _count_ungraded(spreadsheet, event_id)

    if ungraded_count == 0:
        print(f"\n  Already graded: 0 ungraded bets for {event_name}.")
        print("  Skipping grade_bets.py — nothing to grade.")
    else:
        print(f"\n  Found {ungraded_count} ungraded bets for {event_name}.")

        if args.dry_run:
            print("\n  [DRY RUN] Would run: grade_bets.py, sg_diagnostic.py, push_dashboard_data.py")
            print("=" * 60 + "\n")
            sys.exit(0)

        # ------------------------------------------------------------------
        # Step 3: Run grade_bets.py
        # ------------------------------------------------------------------
        success = _run_subprocess(
            [python, "grade_bets.py"],
            "grade_bets.py",
        )

        if not success:
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
        remaining = _count_ungraded(spreadsheet, event_id)

        if remaining > 0:
            print(f"\n  WARNING: {remaining} bets still ungraded after running grade_bets.py.")
            print("  This may indicate DataGolf data was incomplete.")
            sys.exit(1)

        print(f"\n  Verification passed: all bets graded for {event_name}.")

    # ------------------------------------------------------------------
    # Step 5: Run sg_diagnostic.py --no-email (always runs)
    # ------------------------------------------------------------------
    _run_subprocess(
        [python, "sg_diagnostic.py", "--no-email"],
        "sg_diagnostic.py --no-email",
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
    print("\n" + "=" * 60)
    print(f"  GRADING PIPELINE COMPLETE — {event_name}")
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
