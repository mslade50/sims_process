"""
One-time retroactive fix for the dead-heat payout bug in grade_bets.py.

Bug (fixed at grade_bets.py:895): dead-heat wins were settled as
    units_won = units_wagered * (decimal_odds - 1) * factor
which treats the losing fraction (1 - factor) of the stake as PUSHED.
Standard dead-heat settlement loses that fraction:
    units_won = units_wagered * (decimal_odds * factor - 1)

Because the dead-heat factor was never persisted to the sheets, we recover it
from the stored (buggy) units_won and rewrite each cell using the exact
closed form (independent of factor):

    new = old * decimal_odds / (decimal_odds - 1) - units_wagered

which is algebraically identical to units_wagered * (decimal_odds*factor - 1).
The recovered factor is only used as a sanity guard (must be in (0, 1]).

Covers all three tabs that hold graded finish-position results:
Finish Positions, Live (round_sim live bets — not in grade_bets' regrade loop),
and Test Sim. Dry-run by default; pass --apply to write.

Usage:
    python archive/fix_dead_heat_payouts.py            # preview
    python archive/fix_dead_heat_payouts.py --apply    # write to sheets
"""
import argparse
import sys

import gspread

# import from repo root regardless of cwd
sys.path.insert(0, __file__.rsplit("archive", 1)[0])
from sheets_storage import get_spreadsheet
from grade_bets import read_sheet_as_df

UNIT_SIZE = 200.0
FLAT_BET_SIZE = 1.0
TABS = ["Finish Positions", "Live", "Test Sim"]


def to_float(x):
    try:
        s = str(x).strip()
        return float(s) if s != "" else None
    except (ValueError, TypeError):
        return None


def units_wagered_for(row):
    """Mirror grade_finish_position: kelly_stake/UNIT_SIZE, else FLAT_BET_SIZE."""
    ks = to_float(row.get("kelly_stake"))
    return ks / UNIT_SIZE if ks is not None else FLAT_BET_SIZE


def decimal_odds_for(row):
    """Mirror grade_finish_position decimal-odds resolution."""
    d = to_float(row.get("decimal_odds"))
    if d is not None:
        return d
    am = to_float(row.get("american_odds")) or to_float(row.get("odds"))
    if am is not None:
        return am / 100 + 1 if am > 0 else 100 / abs(am) + 1
    return 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write corrections (default: dry-run)")
    args = ap.parse_args()

    ss = get_spreadsheet()
    grand_old = grand_new = 0.0
    grand_rows = 0
    event_delta = {}  # event_name -> delta, for downstream summary repair

    for tab in TABS:
        df = read_sheet_as_df(ss, tab)
        if df.empty or "result" not in df.columns or "units_won" not in df.columns:
            print(f"\n=== {tab}: no gradeable rows, skipping ===")
            continue

        headers = list(df.columns)
        uw_col = headers.index("units_won") + 1
        dh = df[df["result"].astype(str).str.strip() == "win_dh"]
        if dh.empty:
            print(f"\n=== {tab}: 0 win_dh rows ===")
            continue

        print(f"\n=== {tab}: {len(dh)} win_dh rows ===")
        print(f"  {'row':>5} {'event':<12} {'player':<22} {'mkt':<7} "
              f"{'d':>6} {'uw':>7} {'f':>5} {'old':>8} {'new':>8} {'delta':>8}")

        cells = []
        flagged = []
        for idx, row in dh.iterrows():
            sheet_row = idx + 2
            uw = units_wagered_for(row)
            d = decimal_odds_for(row)
            old = to_float(row.get("units_won"))

            if old is None or d <= 1 or uw <= 0:
                flagged.append((sheet_row, "unparseable odds/units/wager"))
                continue

            f = old / (uw * (d - 1))
            if not (0 < f <= 1.05):
                flagged.append((sheet_row, f"recovered factor out of range: {f:.4f}"))
                continue

            new = round(old * d / (d - 1) - uw, 3)
            delta = new - old

            ev = str(row.get("event_name", "")).strip()
            event_delta[ev] = event_delta.get(ev, 0.0) + delta
            grand_old += old
            grand_new += new
            grand_rows += 1

            print(f"  {sheet_row:>5} {ev[:12]:<12} "
                  f"{str(row.get('player_name',''))[:22]:<22} "
                  f"{str(row.get('market_type',''))[:7]:<7} "
                  f"{d:>6.2f} {uw:>7.3f} {f:>5.3f} {old:>8.3f} {new:>8.3f} {delta:>+8.3f}")
            cells.append(gspread.Cell(sheet_row, uw_col, str(new)))

        for sr, why in flagged:
            print(f"  !! FLAGGED row {sr}: {why} (left unchanged)")

        if args.apply and cells:
            ss.worksheet(tab).update_cells(cells, value_input_option="USER_ENTERED")
            print(f"  -> wrote {len(cells)} cells to '{tab}'")
        elif cells:
            print(f"  [DRY RUN] would write {len(cells)} cells to '{tab}'")

    print("\n" + "=" * 60)
    print(f"  Rows corrected: {grand_rows}")
    print(f"  Sum units_won  old: {grand_old:+.3f}   new: {grand_new:+.3f}   "
          f"delta: {grand_new - grand_old:+.3f}")
    print("  Per-event delta (for Bet Results Summary repair):")
    for ev, dl in sorted(event_delta.items(), key=lambda kv: kv[1]):
        print(f"    {ev:<14} {dl:+.3f}")
    print("=" * 60)
    if not args.apply:
        print("  DRY RUN — re-run with --apply to write.")


if __name__ == "__main__":
    main()
