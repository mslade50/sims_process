"""
clear_tournament_bets.py — Wipe pre-tournament bets for a given event.

Clears Tournament Matchup + Finish Position rows from:
  1. Google Sheets (4 tabs)
  2. Local bet ledger (permanent_data/bet_ledger.parquet)

Usage:
  python clear_tournament_bets.py                  # Current tourney from sim_inputs
  python clear_tournament_bets.py --dry-run        # Preview what would be deleted
  python clear_tournament_bets.py --yes            # Skip confirmation prompt
  python clear_tournament_bets.py --event-name genesis --event-id 5
"""

import argparse
import os
import sys
import tempfile
from datetime import datetime

import pandas as pd

from sheets_storage import (
    get_spreadsheet,
    TAB_TOURNAMENT_MU,
    TAB_FINISH_POS,
    LEDGER_PATH,
)
from sim_inputs import tourney as default_tourney, event_ids as default_event_ids

# Bet types that belong to pre-tournament bets
PRE_TOURNEY_BET_TYPES = {"tournament_matchup", "finish_position"}


# ══════════════════════════════════════════════════════════════════════════════
# Google Sheets helpers
# ══════════════════════════════════════════════════════════════════════════════

def _find_matching_rows(records, event_name, year, bet_type_filter=False):
    """
    Return 1-indexed row numbers (skipping header) that match the event.

    Args:
        records:  list of lists from worksheet.get_all_values() (includes header row)
        event_name: event name to match (case-insensitive)
        year:     current year (int or str)
        bet_type_filter: if True, also require bet_type column to be in PRE_TOURNEY_BET_TYPES
    """
    if len(records) < 2:
        return []

    header = [h.strip().lower() for h in records[0]]
    try:
        name_idx = header.index("event_name")
    except ValueError:
        return []
    try:
        year_idx = header.index("year")
    except ValueError:
        return []

    bt_idx = None
    if bet_type_filter:
        try:
            bt_idx = header.index("bet_type")
        except ValueError:
            return []

    year_str = str(year)
    event_lower = event_name.lower().strip()
    matches = []

    for i, row in enumerate(records[1:], start=2):  # 1-indexed, row 1 = header
        if len(row) <= max(name_idx, year_idx):
            continue
        if row[name_idx].strip().lower() != event_lower:
            continue
        if row[year_idx].strip() != year_str:
            continue
        if bt_idx is not None:
            if len(row) <= bt_idx:
                continue
            if row[bt_idx].strip().lower() not in PRE_TOURNEY_BET_TYPES:
                continue
        matches.append(i)

    return matches


def _delete_rows_from_sheet(worksheet, row_numbers, dry_run=False):
    """
    Delete rows by 1-indexed row number, in reverse order to avoid index shift.
    Uses batch delete for contiguous ranges.
    """
    if not row_numbers:
        return 0

    if dry_run:
        return len(row_numbers)

    # Sort descending so deletions don't shift remaining indices
    sorted_rows = sorted(row_numbers, reverse=True)

    # Group into contiguous ranges for efficient batch deletion
    ranges = []
    start = end = sorted_rows[0]
    for r in sorted_rows[1:]:
        if r == start - 1:
            start = r
        else:
            ranges.append((start, end))
            start = end = r
    ranges.append((start, end))

    for rng_start, rng_end in ranges:
        worksheet.delete_rows(rng_start, rng_end)

    return len(row_numbers)


# ══════════════════════════════════════════════════════════════════════════════
# Main logic
# ══════════════════════════════════════════════════════════════════════════════

def clear_bets(event_name, event_id, dry_run=False, skip_confirm=False):
    """Clear pre-tournament bets from Sheets and local ledger."""
    year = datetime.now().year
    event_id_str = str(event_id)

    print(f"\nEvent:  {event_name}")
    print(f"ID:     {event_id_str}")
    print(f"Year:   {year}")
    print(f"Mode:   {'DRY RUN' if dry_run else 'LIVE'}")
    print()

    # ── Scan Google Sheets ──────────────────────────────────────────────────
    ss = get_spreadsheet()

    tabs_to_clear = [
        (TAB_TOURNAMENT_MU, False),
        (TAB_FINISH_POS, False),
    ]

    tab_matches = {}
    for tab_name, needs_bt_filter in tabs_to_clear:
        try:
            ws = ss.worksheet(tab_name)
            records = ws.get_all_values()
            rows = _find_matching_rows(records, event_name, year, bet_type_filter=needs_bt_filter)
            tab_matches[tab_name] = (ws, rows)
        except Exception as e:
            print(f"  Warning: could not read tab '{tab_name}': {e}")
            tab_matches[tab_name] = (None, [])

    # ── Scan local ledger ──────────────────────────────────────────────────
    ledger_count = 0
    if os.path.exists(LEDGER_PATH):
        try:
            ledger = pd.read_parquet(LEDGER_PATH)
            mask = (
                (ledger["event_id"].astype(str).str.strip() == event_id_str)
                & (ledger["bet_type"].str.strip().str.lower().isin(PRE_TOURNEY_BET_TYPES))
            )
            ledger_count = int(mask.sum())
        except Exception as e:
            print(f"  Warning: could not read ledger: {e}")
            ledger = None
            mask = None
    else:
        print("  Ledger file not found — skipping.")
        ledger = None
        mask = None

    # ── Summary ────────────────────────────────────────────────────────────
    print("Rows to delete:")
    total = 0
    for tab_name, (_, rows) in tab_matches.items():
        count = len(rows)
        total += count
        print(f"  {tab_name:30s}  {count}")
    print(f"  {'bet_ledger.parquet':30s}  {ledger_count}")
    total += ledger_count
    print(f"  {'TOTAL':30s}  {total}")
    print()

    if total == 0:
        print("Nothing to delete.")
        return

    # ── Confirmation ───────────────────────────────────────────────────────
    if not skip_confirm and not dry_run:
        answer = input("Proceed with deletion? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    # ── Delete from Sheets ─────────────────────────────────────────────────
    for tab_name, (ws, rows) in tab_matches.items():
        if not rows:
            continue
        deleted = _delete_rows_from_sheet(ws, rows, dry_run=dry_run)
        action = "Would delete" if dry_run else "Deleted"
        print(f"  {action} {deleted} rows from '{tab_name}'")

    # ── Delete from ledger ─────────────────────────────────────────────────
    if ledger is not None and mask is not None and ledger_count > 0:
        if dry_run:
            print(f"  Would delete {ledger_count} rows from bet_ledger.parquet")
        else:
            filtered = ledger[~mask]
            ledger_dir = os.path.dirname(LEDGER_PATH)
            fd, tmp_path = tempfile.mkstemp(suffix=".parquet", dir=ledger_dir)
            os.close(fd)
            try:
                filtered.to_parquet(tmp_path, index=False)
                os.replace(tmp_path, LEDGER_PATH)
                print(f"  Deleted {ledger_count} rows from bet_ledger.parquet "
                      f"({len(filtered)} remaining)")
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise

    print("\nDone." if not dry_run else "\nDry run complete — no changes made.")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Clear pre-tournament bets for an event from Sheets and local ledger."
    )
    parser.add_argument(
        "--event-name",
        default=default_tourney,
        help=f"Tournament name to match (default: '{default_tourney}' from sim_inputs)",
    )
    parser.add_argument(
        "--event-id",
        default=str(default_event_ids[0]),
        help=f"Event ID for ledger matching (default: {default_event_ids[0]} from sim_inputs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without making changes",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    args = parser.parse_args()

    clear_bets(
        event_name=args.event_name,
        event_id=args.event_id,
        dry_run=args.dry_run,
        skip_confirm=args.yes,
    )


if __name__ == "__main__":
    main()
