"""Apply the audit's book-of-record regrades to the Google Sheet tabs.

1. Finish Positions: the definitive 88-row exchange NO-side repair
   (inv/final_no_rows.csv, net +29.60u). Each row was graded with YES-side
   logic on a bet that was actually NO; fix market_type to the post-b878992
   side-suffix convention ({market}_no), result -> corr_result, units_won ->
   corr_units. FP-4814 (li haotong top_10, ambiguous +/-1.67u) is
   deliberately absent from the file and stays untouched.

2. Round/Tournament Matchups ties regrade: pushes stored under
   ties_rule == 'separate bet offered' (ties-LOSE books) regraded as losses
   (expected 11 RMU / 2 TMU, -13.84u total). Stake per the grading
   convention: plus-money risks 1u to win odds/100; minus-money risks
   |odds|/100 to win 1u.

Safety: every write is preceded by a row-identity check against the repair
file / expected query; mismatches are reported and skipped, never written.

Usage:  python apply_regrades.py [--live]     (default is dry-run)
"""
import argparse
import sys

import pandas as pd

REPO = r"C:\Users\McKinley Slade\dev\sims_process"
SC = r"C:\Users\McKinley Slade\AppData\Local\Temp\claude\C--Users-McKinley-Slade-dev-sims-process\44146807-e217-4d6a-8fb5-b76d150adb84\scratchpad"
sys.path.insert(0, REPO)

from sheets_storage import get_spreadsheet  # noqa: E402


def col_letter(idx0):
    letters = ""
    idx = idx0 + 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def norm(s):
    return str(s or "").strip().lower()


def stake_from_odds(odds_str):
    """grade_bets convention: plus-money risks 1u; minus-money risks |odds|/100."""
    try:
        odds = float(str(odds_str).replace("+", ""))
    except (ValueError, TypeError):
        return 1.0
    if odds >= 100:
        return 1.0
    if odds <= -100:
        return abs(odds) / 100.0
    return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="write to the Sheet (default dry-run)")
    args = ap.parse_args()
    dry = not args.live

    ss = get_spreadsheet()

    # ------------------------------------------------------------------
    # 1. Finish Positions 88-row NO-side repair
    # ------------------------------------------------------------------
    rep = pd.read_csv(f"{SC}\\inv\\final_no_rows.csv")
    assert len(rep) == 88 and (rep["final_side"] == "no").all()
    ws = ss.worksheet("Finish Positions")
    vals = ws.get_all_values()
    hdr = vals[0]
    ix = {h: i for i, h in enumerate(hdr)}
    for need in ("player_name", "market_type", "sportsbook", "result", "units_won"):
        assert need in ix, f"Finish Positions missing column {need}"

    updates, skipped = [], []
    total_delta = 0.0
    for _, r in rep.iterrows():
        srow = int(r["sheet_row"])            # 1-based sheet row
        row = vals[srow - 1]
        ok = (norm(row[ix["player_name"]]) == norm(r["player"])
              and norm(row[ix["market_type"]]) in (norm(r["market"]), norm(r["market"]) + "_no")
              and norm(row[ix["sportsbook"]]) == norm(r["book"])
              and norm(row[ix["event_id"]]) == norm(int(r["event_id"])))
        if not ok:
            skipped.append((srow, "identity mismatch",
                            row[ix["player_name"]], row[ix["market_type"]],
                            row[ix["sportsbook"]], row[ix["event_id"]]))
            continue
        cur_result = norm(row[ix["result"]])
        if cur_result != norm(r["cur_result"]):
            skipped.append((srow, f"result drift: tab={cur_result} expected={r['cur_result']}"))
            continue
        new_market = norm(r["market"]) + "_no"
        new_result = str(r["corr_result"])
        new_units = round(float(r["corr_units"]), 3)
        total_delta += float(r["delta"])
        updates.append((srow, ix["market_type"], new_market))
        updates.append((srow, ix["result"], new_result))
        updates.append((srow, ix["units_won"], new_units))
        if dry:
            print(f"  FP row {srow}: {r['player']} {r['market']}@{r['book']} "
                  f"{cur_result}/{row[ix['units_won']]} -> {new_market} "
                  f"{new_result}/{new_units} (delta {r['delta']:+.2f})")

    print(f"\n[FP] {len(updates)//3} of 88 rows verified; {len(skipped)} skipped; "
          f"applied delta {total_delta:+.2f}u (target +29.60)")
    for s in skipped:
        print("   SKIPPED:", s)

    if not dry and updates:
        data = [{"range": f"{col_letter(c)}{srow}", "values": [[v]]}
                for (srow, c, v) in updates]
        ws.batch_update(data, value_input_option="USER_ENTERED")
        print(f"[FP] wrote {len(data)} cells")

    # ------------------------------------------------------------------
    # 2. Ties-lose regrade (Round Matchups 11 rows, Tournament Matchups 2)
    # ------------------------------------------------------------------
    import time
    for tab, expected_n in (("Round Matchups", 11), ("Tournament Matchups", 2)):
        time.sleep(5)
        ws = ss.worksheet(tab)
        vals = ws.get_all_values()
        hdr = vals[0]
        ix = {h: i for i, h in enumerate(hdr)}
        for need in ("result", "units_won", "ties_rule", "bet_on",
                     "player_1", "player_2", "p1_odds", "p2_odds", "bookmaker"):
            assert need in ix, f"{tab} missing column {need}"
        hits = []
        for rnum, row in enumerate(vals[1:], start=2):
            if norm(row[ix["result"]]) == "push" and \
                    norm(row[ix["ties_rule"]]) == "separate bet offered":
                bet_on = norm(row[ix["bet_on"]])
                odds = (row[ix["p1_odds"]]
                        if bet_on == norm(row[ix["player_1"]])
                        else row[ix["p2_odds"]])
                stake = stake_from_odds(odds)
                hits.append((rnum, bet_on, row[ix["bookmaker"]], odds, stake))
        tot = sum(h[4] for h in hits)
        print(f"\n[{tab}] {len(hits)} ties-lose pushes (expected {expected_n}); "
              f"regrade delta {-tot:+.2f}u")
        for rnum, bet_on, book, odds, stake in hits:
            print(f"   row {rnum}: {bet_on} @ {book} odds={odds} -> loss, "
                  f"units_won {-stake:+.2f}")
        if len(hits) != expected_n:
            print(f"   COUNT MISMATCH vs audit ({expected_n}) — "
                  f"{'not writing this tab' if not dry else 'review before --live'}")
            continue
        if not dry:
            data = []
            for rnum, _, _, _, stake in hits:
                data.append({"range": f"{col_letter(ix['result'])}{rnum}",
                             "values": [["loss"]]})
                data.append({"range": f"{col_letter(ix['units_won'])}{rnum}",
                             "values": [[round(-stake, 3)]]})
            ws.batch_update(data, value_input_option="USER_ENTERED")
            print(f"[{tab}] wrote {len(data)} cells")

    print("\nDone." if not dry else "\nDRY RUN complete — rerun with --live to write.")


if __name__ == "__main__":
    main()
