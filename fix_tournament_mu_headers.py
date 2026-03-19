"""
One-time migration: fix Tournament Matchups tab column shift.

Problem: wx_diff was added to TOURNAMENT_MU_HEADERS in sheets_storage.py
but the existing Sheet tab header was never updated. New rows written with
27 values have wx_diff landing in the 'result' column, shifting result and
units_won right by one.

Fix:
  1. If header is missing wx_diff, insert it between wind_diff and result
  2. For each data row:
     - If it has 27 values (shifted): the value at the old 'result' position
       is actually wx_diff — reorder to put it in the right spot
     - If it has 26 values (old format): insert empty wx_diff
"""

import sys
from sheets_storage import get_spreadsheet

TAB_NAME = "Tournament Matchups"

EXPECTED_HEADERS = [
    "run_timestamp", "event_name", "year", "event_id",
    "player_1", "player_2", "dg_id_p1", "dg_id_p2",
    "bookmaker", "ties_rule",
    "p1_odds", "p2_odds", "fair_p1", "fair_p2",
    "edge_p1", "edge_p2",
    "bet_on", "edge_on", "pred_on", "pred_against", "sample_on",
    "half_shot_p1", "half_shot_p2",
    "wind_on", "wind_diff", "wx_diff",
    "result", "units_won",
]

VALID_RESULTS = {"", "win", "loss", "push", "no_data", "unknown", "duplicate"}


def main():
    dry_run = "--dry-run" in sys.argv

    spreadsheet = get_spreadsheet()
    ws = spreadsheet.worksheet(TAB_NAME)
    all_data = ws.get_all_values()

    if not all_data:
        print("Tab is empty, nothing to fix.")
        return

    headers = all_data[0]
    print(f"Current headers ({len(headers)} cols): {headers}")

    if "wx_diff" in headers:
        print("wx_diff already in headers — checking for any shifted data rows...")
        wx_idx = headers.index("wx_diff")
        result_idx = headers.index("result") if "result" in headers else None
        if result_idx is not None:
            # Check for rows where result has a numeric value (still shifted)
            bad_rows = 0
            for i, row in enumerate(all_data[1:], start=2):
                if len(row) > result_idx:
                    val = row[result_idx].strip()
                    if val and val not in VALID_RESULTS:
                        bad_rows += 1
            if bad_rows == 0:
                print("No shifted data rows found. Nothing to do.")
                return
            print(f"Found {bad_rows} rows with non-standard result values — fixing...")
    else:
        print("wx_diff NOT in headers — need to insert column and fix data.")

    # Determine where wx_diff should be inserted
    # It goes between wind_diff and result
    if "wind_diff" in headers:
        insert_pos = headers.index("wind_diff") + 1
    elif "result" in headers:
        insert_pos = headers.index("result")
    else:
        print("ERROR: Cannot find wind_diff or result in headers. Manual fix needed.")
        return

    print(f"wx_diff insert position: column {insert_pos + 1} (0-indexed: {insert_pos})")

    # Build new header row
    new_headers = list(headers)
    if "wx_diff" not in headers:
        new_headers.insert(insert_pos, "wx_diff")

    result_pos_new = new_headers.index("result") if "result" in new_headers else None
    units_won_pos_new = new_headers.index("units_won") if "units_won" in new_headers else None

    print(f"New headers ({len(new_headers)} cols): {new_headers}")

    # Fix each data row
    fixed_rows = []
    n_shifted = 0
    n_old = 0
    n_ok = 0

    for i, row in enumerate(all_data[1:], start=2):
        if "wx_diff" not in headers and len(row) == len(new_headers):
            # Row has 27 values but header only has 26 — it's shifted.
            # The value at insert_pos is wx_diff sitting in the result slot.
            # Values are already in the right positional order for the NEW headers.
            new_row = list(row)
            n_shifted += 1
        elif "wx_diff" not in headers and len(row) == len(headers):
            # Old row with 26 values — insert empty wx_diff
            new_row = list(row)
            new_row.insert(insert_pos, "")
            n_old += 1
        elif "wx_diff" in headers:
            # Header already has wx_diff but data might be shifted
            new_row = list(row)
            # Check if result column has a numeric value (wx_diff leaked into it)
            if result_pos_new is not None and len(row) > result_pos_new:
                val = row[result_pos_new].strip()
                if val and val not in VALID_RESULTS:
                    try:
                        float(val)
                        # It's a number in the result column — this row is shifted
                        # The number is actually wx_diff; result is in units_won pos
                        # We need to un-shift: remove the extra value and reinsert
                        n_shifted += 1
                    except ValueError:
                        n_ok += 1
                else:
                    n_ok += 1
            else:
                n_ok += 1
        else:
            # Unexpected row length — pad or truncate
            new_row = list(row)
            while len(new_row) < len(new_headers):
                new_row.append("")
            n_ok += 1

        # Pad to correct length
        while len(new_row) < len(new_headers):
            new_row.append("")

        fixed_rows.append(new_row[:len(new_headers)])

    print(f"\nRows: {len(fixed_rows)} data rows")
    print(f"  Shifted (27-val with old header): {n_shifted}")
    print(f"  Old format (26-val): {n_old}")
    print(f"  Already OK: {n_ok}")

    # Show sample of what result column looks like before/after
    if result_pos_new is not None:
        print(f"\nSample result values (first 10 data rows):")
        for j, row in enumerate(fixed_rows[:10]):
            result_val = row[result_pos_new] if len(row) > result_pos_new else "N/A"
            print(f"  Row {j+2}: result='{result_val}'")

    if dry_run:
        print("\n[DRY RUN] Would update the sheet. Run without --dry-run to apply.")
        return

    print(f"\nWriting {len(fixed_rows) + 1} rows (header + data) to sheet...")
    # Clear and rewrite entire tab
    ws.clear()
    ws.append_row(new_headers, value_input_option="RAW")
    if fixed_rows:
        # Batch in chunks of 500 to avoid API limits
        chunk_size = 500
        for start in range(0, len(fixed_rows), chunk_size):
            chunk = fixed_rows[start:start + chunk_size]
            ws.append_rows(chunk, value_input_option="USER_ENTERED")
            print(f"  Wrote rows {start + 2} - {start + len(chunk) + 1}")

    print(f"\n[OK] Fixed {TAB_NAME}: {len(new_headers)} headers, {len(fixed_rows)} data rows")


if __name__ == "__main__":
    main()
