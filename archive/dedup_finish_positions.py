"""
One-time compaction of the 'Finish Positions' sheet tab.

The tab re-logs the same bet on every scraper/sim run, so a single
(event, player, market, odds, book) bet appears many times (varying
run_timestamp / kelly_stake / edge). This collapses each such group to a
single row to free up space.

Key       : (event_id, player_name, market_type, american_odds, sportsbook)
            all lowercased/stripped.
Keep-rule : keep the graded row (result in win/loss/win_dh/no_data/push) if
            present (verified <=1 per group), else keep the first row.

Safety:
  - Backs up the full current tab to archive/backups/ before writing.
  - Preserves exact cell strings (no pandas coercion) and column order.
  - Dry-run by default; pass --apply to rewrite the tab.
  - Re-reads and verifies graded-row counts are unchanged after the write.

Usage:
    python archive/dedup_finish_positions.py            # preview + backup
    python archive/dedup_finish_positions.py --apply    # rewrite the tab
"""
import argparse
import csv
import os
import sys
from datetime import datetime

sys.path.insert(0, __file__.rsplit("archive", 1)[0])
from sheets_storage import get_spreadsheet

TAB = "Finish Positions"
KEY_COLS = ["event_id", "player_name", "market_type", "american_odds", "sportsbook"]
GRADED = {"win", "loss", "win_dh", "no_data", "push"}
BACKUP_DIR = os.path.join(os.path.dirname(__file__), "backups")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="rewrite the tab (default: dry-run)")
    args = ap.parse_args()

    ss = get_spreadsheet()
    ws = ss.worksheet(TAB)
    data = ws.get_all_values()
    if len(data) < 2:
        print("Tab empty, nothing to do.")
        return

    header = data[0]
    rows = data[1:]
    col = {h: i for i, h in enumerate(header)}
    for c in KEY_COLS + ["result"]:
        if c not in col:
            print(f"FATAL: column '{c}' not found in tab header.")
            return

    def keyof(r):
        return "|".join(str(r[col[c]]).strip().lower() for c in KEY_COLS)

    def is_graded(r):
        return str(r[col["result"]]).strip().lower() in GRADED

    # Backup (always, even on dry-run, so it's available before --apply)
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"finish_positions_backup_{ts}.csv")
    with open(backup_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Backup written: {backup_path}  ({len(rows)} data rows)")

    # Group, pick keeper per group preserving original order of first appearance
    groups = {}        # key -> list of row indices (into `rows`)
    order = []         # keys in first-seen order
    for i, r in enumerate(rows):
        k = keyof(r)
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(i)

    keep_idx = []
    for k in order:
        idxs = groups[k]
        graded = [i for i in idxs if is_graded(rows[i])]
        keep_idx.append(graded[0] if graded else idxs[0])

    keep_idx_set = set(keep_idx)
    removed = len(rows) - len(keep_idx)

    # Report
    def result_counts(idxs):
        from collections import Counter
        return dict(Counter(str(rows[i][col["result"]]).strip().lower() for i in idxs))

    print(f"\n=== {TAB} dedup ===")
    print(f"  data rows now      : {len(rows)}")
    print(f"  unique key groups  : {len(order)}")
    print(f"  rows to remove     : {removed}")
    print(f"  rows remaining     : {len(keep_idx)}")
    print(f"  result counts before: {result_counts(range(len(rows)))}")
    print(f"  result counts kept  : {result_counts(keep_idx)}")

    # Verify no graded row is dropped
    graded_before = sum(1 for r in rows if is_graded(r))
    graded_kept = sum(1 for i in keep_idx if is_graded(rows[i]))
    print(f"  graded rows before/kept: {graded_before} / {graded_kept}", end="")
    if graded_before != graded_kept:
        print("  !!! MISMATCH — ABORT (would lose graded outcomes)")
        return
    print("  OK")

    if not args.apply:
        print("\n  DRY RUN — re-run with --apply to rewrite the tab.")
        return

    # Rewrite: clear values, write header + kept rows (preserve original order)
    new_data = [header] + [rows[i] for i in sorted(keep_idx_set)]
    ws.clear()
    ws.update(new_data, value_input_option="USER_ENTERED")
    print(f"\n  Wrote {len(new_data)-1} data rows to '{TAB}'.")

    # Verify read-back
    back = ws.get_all_values()
    bcol = {h: i for i, h in enumerate(back[0])}
    from collections import Counter
    bc = Counter(str(r[bcol["result"]]).strip().lower() for r in back[1:])
    print(f"  read-back rows     : {len(back)-1}")
    print(f"  read-back graded   : {sum(bc[g] for g in GRADED)} (expected {graded_kept})")
    print(f"  read-back result counts: {dict(bc)}")


if __name__ == "__main__":
    main()
