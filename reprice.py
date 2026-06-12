"""reprice.py — cache-free round-matchup repricer.

Triggered by the golf_scraping sentinel (repository_dispatch) whenever fresh odds
land. Instead of restoring the 60 MB sim cache and re-running round_sim's full
pricing pipeline (Rust kernel, Excel export, predictions, weather — all of which
have failed in CI), this:

    1. Loads the committed round-H2H fair table  (round_h2h_r{N}.parquet, ~100 KB)
    2. Fetches fresh round-matchup odds          (odds_loader, GitHub -> DataGolf)
    3. Prices every offered matchup from the table  (reprice_core.price_from_h2h)
    4. Computes edges + the combined/sharp filter
    5. Dedups against the Round Matchups sheet, stores only NEW rows
    6. Telegram-alerts the new sharp-book edges (silent when there's nothing new)

The fair table is produced by the nightly sim (publish_sim_fairs.py
--round-h2h-only) and by live full round_sim runs, and is conclusive for H2H
pricing: (P(a<b), P(tie)) reconstructs round_sim.price_matchups EXACTLY.

Dependencies: pandas numpy requests gspread google-auth pyarrow python-dotenv.
NO Rust kernel, NO xlsxwriter, NO sim cache.

Usage:
    python reprice.py              # price + dedup + store + Telegram
    python reprice.py --dry-run    # price + print summary, no store/Telegram
"""

import argparse
import json
import os
import sys


def _setup_env():
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    return root


def main():
    ap = argparse.ArgumentParser(description="Cache-free round-matchup repricer")
    ap.add_argument("--dry-run", action="store_true",
                    help="Price + print summary; skip dedup/store/Telegram")
    args = ap.parse_args()

    root = _setup_env()
    import pandas as pd
    import reprice_core as rc
    from sheet_config import load_config

    try:
        from sim_inputs import name_replacements as NAME_REPL
    except Exception:
        NAME_REPL = {}

    cfg = load_config()
    tourney = cfg["tourney"]
    event_id = cfg["event_id"]
    round_num = cfg["round_num"]
    sim_round = round_num + 1 if round_num < 4 else 4

    print(f"\n{'='*60}")
    print(f"  REPRICE (cache-free) — {tourney} R{sim_round}")
    print(f"{'='*60}")

    # ── 1. Load the committed round-H2H fair table ───────────────────────
    pq = os.path.join(root, f"round_h2h_r{sim_round}.parquet")
    mj = os.path.join(root, f"round_h2h_r{sim_round}_meta.json")
    if not os.path.exists(pq) or not os.path.exists(mj):
        msg = (f"Reprice: no round-H2H fairs for R{sim_round} {tourney}. "
               f"Run the sim first.")
        print(f"  {msg}")
        if not args.dry_run:
            rc.send_telegram(msg)
        return 0

    with open(mj) as f:
        meta = json.load(f)
    if str(meta.get("round")) != str(sim_round) or meta.get("tourney") != tourney:
        msg = (f"Reprice: round-H2H artifact is R{meta.get('round')} "
               f"{meta.get('tourney')}, expected R{sim_round} {tourney}. "
               f"Run the sim first.")
        print(f"  {msg}")
        if not args.dry_run:
            rc.send_telegram(msg)
        return 0

    h2h_df = pd.read_parquet(pq)
    lookup, known = rc.build_h2h_lookup(h2h_df)
    pred_lookup = {k: float(v) for k, v in (meta.get("pred") or {}).items()}
    sample_lookup = {k: float(v) for k, v in (meta.get("sample") or {}).items()}
    wx_lookup = {k: float(v) for k, v in (meta.get("wx") or {}).items()}
    print(f"  Fairs: {len(h2h_df)} pairs, {len(known)} players "
          f"(sim @ {meta.get('generated_at', '?')})")

    # ── 2. Fetch fresh round-matchup odds ────────────────────────────────
    from odds_loader import load_matchup_odds
    matchup_df = load_matchup_odds("round_matchups", api_key=os.getenv("DATAGOLF_API_KEY"))
    if matchup_df is None or matchup_df.empty:
        print("  No round-matchup odds available — nothing to price.")
        return 0
    print(f"  Loaded {len(matchup_df)} matchup lines")

    # ── 3-4. Price from the table + edges + combined/sharp filter ─────────
    matchup_df = rc.price_from_h2h(matchup_df, lookup, known, repl=NAME_REPL)
    matchup_df = rc.calculate_edges(matchup_df)
    combined, sharp = rc.build_matchup_outputs(
        matchup_df, sim_round, pred_lookup, sample_lookup, wx_lookup=wx_lookup
    )

    if args.dry_run:
        print(f"\n  [dry-run] {len(combined)} combined / {len(sharp)} sharp edges")
        if not sharp.empty:
            show = [c for c in ["bet_on", "Player 1", "Player 2", "Bookmaker",
                                "edge_on", "Fair_p1", "Fair_p2", "P1 Odds", "P2 Odds"]
                    if c in sharp.columns]
            print(sharp[show].head(25).to_string(index=False))
        return 0

    # ── 5. Dedup against Sheets + store only new rows ─────────────────────
    from sheets_storage import get_spreadsheet, store_round_matchups, load_dg_id_lookup
    spreadsheet = get_spreadsheet()
    new_mu = rc.dedup_round_matchups(combined, spreadsheet, event_id, sim_round)

    try:
        dg_id_lookup = load_dg_id_lookup(tourney, NAME_REPL)
    except Exception:
        dg_id_lookup = {}

    if new_mu is not None and not new_mu.empty:
        store_round_matchups(
            new_mu, sim_round, tourney, event_id,
            dg_id_lookup=dg_id_lookup, spreadsheet=spreadsheet,
        )
    else:
        print("  [reprice] No new matchup rows to store.")

    # ── 6. Telegram-alert new sharp edges (silent when nothing new) ───────
    rc.send_matchup_alert(new_mu, sim_round, tourney)

    n_new = len(new_mu) if new_mu is not None and not new_mu.empty else 0
    print(f"\n  [reprice] Done. Stored {n_new} new matchups. "
          f"{'Telegram sent.' if n_new else 'No alert (nothing new).'}")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
