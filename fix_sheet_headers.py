"""Fix Sheet tab headers to match sheets_storage.py expected format.

Handles two cases:
  1. Old rows (written before type_on/wx_diff existed): mapped by old header names,
     new columns filled with empty strings.
  2. Shifted rows (written with new 29-col format into old 28-col header): the data
     is already in the expected column order, just needs the header to catch up.

Tournament Matchups: adds type_on, wx_diff; drops stale wx_edge header.
Finish Positions: adds type_on.
"""

import sys
from sheets_storage import get_spreadsheet, TOURNAMENT_MU_HEADERS, FINISH_POS_HEADERS

# Known archetype values that appear in type_on
ARCHETYPES = {
    "Ball Striker", "Long Accurate", "Long Wild", "Short Accurate",
    "Short Game Specialist", "Balanced", "Low Skill", "Elite Putter",
    "Stud", "Short Game Spe",  # truncated in sheet
}

TABS_TO_FIX = {
    "Tournament Matchups": TOURNAMENT_MU_HEADERS,
    "Finish Positions": FINISH_POS_HEADERS,
}


def _is_shifted_mu_row(row, current_headers):
    """Detect if a Tournament Matchups row was written with 29-col format into 28-col sheet.

    Key signal: the edge_on column (header index 17) contains an archetype string
    instead of a number — meaning type_on was inserted at position 17 and everything
    shifted right.
    """
    if "edge_on" not in current_headers:
        return False
    edge_idx = current_headers.index("edge_on")
    if edge_idx >= len(row):
        return False
    val = str(row[edge_idx]).strip()
    # Archetype string in what should be a numeric edge column
    if any(val.startswith(a) for a in ARCHETYPES):
        return True
    return False


def _is_shifted_fp_row(row, current_headers):
    """Detect if a Finish Positions row was written with 20-col format into 19-col sheet.

    Key signal: the type_on column doesn't exist in headers, and the value at the
    'result' position is an archetype string instead of win/loss/push.
    """
    if "type_on" in current_headers:
        return False  # header already has it
    if "result" not in current_headers:
        return False
    result_idx = current_headers.index("result")
    if result_idx >= len(row):
        return False
    val = str(row[result_idx]).strip()
    if any(val.startswith(a) for a in ARCHETYPES):
        return True
    return False


def fix_tab(spreadsheet, tab_name, expected_headers, dry_run=False):
    try:
        ws = spreadsheet.worksheet(tab_name)
    except Exception as e:
        print(f"  [{tab_name}] Tab not found: {e}")
        return

    all_data = ws.get_all_values()
    if not all_data:
        print(f"  [{tab_name}] Empty tab, nothing to fix.")
        return

    current_headers = all_data[0]

    print(f"  [{tab_name}] Current: {len(current_headers)} cols")
    print(f"  [{tab_name}] Expected: {len(expected_headers)} cols")
    print(f"  [{tab_name}] Missing: {set(expected_headers) - set(current_headers)}")
    print(f"  [{tab_name}] Extra: {set(current_headers) - set(expected_headers)}")

    # Build old-header column map (for non-shifted rows)
    old_col_map = {}
    for col in expected_headers:
        if col in current_headers:
            old_col_map[col] = current_headers.index(col)
        else:
            old_col_map[col] = None

    is_shifted_fn = _is_shifted_mu_row if tab_name == "Tournament Matchups" else _is_shifted_fp_row

    fixed_rows = []
    n_old = 0
    n_shifted = 0

    for row in all_data[1:]:
        if is_shifted_fn(row, current_headers):
            # Row was written with expected_headers column order — data is already correct
            # Just pad/truncate to expected length
            new_row = list(row[:len(expected_headers)])
            while len(new_row) < len(expected_headers):
                new_row.append("")
            n_shifted += 1
        else:
            # Old row — map by column name, fill missing with empty
            new_row = []
            for col in expected_headers:
                idx = old_col_map[col]
                if idx is not None and idx < len(row):
                    new_row.append(row[idx])
                else:
                    new_row.append("")
            n_old += 1

        fixed_rows.append(new_row)

    print(f"  [{tab_name}] {n_old} old-format rows, {n_shifted} shifted rows")

    # Validate: check result column has expected values
    result_idx = expected_headers.index("result") if "result" in expected_headers else None
    valid_results = {"", "win", "win_dh", "loss", "push", "no_data", "unknown", "duplicate", "0"}
    if result_idx is not None:
        bad_results = 0
        for i, row in enumerate(fixed_rows):
            val = row[result_idx].strip().lower()
            if val and val not in valid_results:
                bad_results += 1
                if bad_results <= 3:
                    print(f"    WARNING row {i+2}: result='{row[result_idx]}'")
        if bad_results > 0:
            print(f"    WARNING: {bad_results} rows have unexpected result values!")
        else:
            print(f"  [{tab_name}] Result column validation passed")

    # Show samples
    print(f"  [{tab_name}] Sample old row (first):")
    for row in all_data[1:]:
        if not is_shifted_fn(row, current_headers):
            for i, col in enumerate(expected_headers[:len(expected_headers)]):
                idx = old_col_map[col]
                val = row[idx] if idx is not None and idx < len(row) else ""
                if i >= 16:
                    print(f"    {col}: {val}")
            break

    print(f"  [{tab_name}] Sample shifted row (first):")
    for row in all_data[1:]:
        if is_shifted_fn(row, current_headers):
            for i, col in enumerate(expected_headers):
                val = row[i] if i < len(row) else ""
                if i >= 16:
                    print(f"    {col}: {val}")
            break

    if dry_run:
        print(f"  [{tab_name}] [DRY RUN] Would rewrite {len(fixed_rows) + 1} rows.")
        return

    # Clear and rewrite
    ws.clear()
    ws.append_row(expected_headers, value_input_option="RAW")
    if fixed_rows:
        chunk_size = 500
        for start in range(0, len(fixed_rows), chunk_size):
            chunk = fixed_rows[start:start + chunk_size]
            ws.append_rows(chunk, value_input_option="USER_ENTERED")
            print(f"    Wrote rows {start + 2} - {start + len(chunk) + 1}")

    print(f"  [{tab_name}] [OK] Fixed: {len(expected_headers)} headers, {len(fixed_rows)} data rows")


def main():
    dry_run = "--dry-run" in sys.argv
    spreadsheet = get_spreadsheet()

    for tab_name, expected_headers in TABS_TO_FIX.items():
        fix_tab(spreadsheet, tab_name, expected_headers, dry_run=dry_run)


if __name__ == "__main__":
    main()
