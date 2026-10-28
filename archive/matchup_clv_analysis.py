"""
matchup_clv_analysis.py — Adverse-selection / CLV backtest for tournament matchups.

Joins our stored tournament-matchup bets (Google Sheet "Tournament Matchups",
carries our entry price + fair as of bet time) to DataGolf historical matchup
odds (opening + closing sharp lines + settled outcomes), then measures:

  CLV                = de-vigged sharp CLOSE vs our entry price   (did we beat close?)
  edge_vs_sharp_open = our fair vs de-vigged sharp OPEN           (lever-2 signal, known at bet time)

Usage:
    python archive/matchup_clv_analysis.py --recent 4     # pull N most-recent events
    python archive/matchup_clv_analysis.py --events 34 26 # specific event_ids
Pulls are cached to permanent_data/hist_matchup_odds.parquet (append/merge).
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api_utils import fetch_historical_matchup_odds

CACHE = "permanent_data/hist_matchup_odds.parquet"
BOOKS = ["pinnacle", "betcris"]


def impl(american):
    """American odds -> raw implied probability."""
    a = pd.to_numeric(american, errors="coerce")
    return np.where(a < 0, -a / (100 - a), 100 / (100 + a))


def load_bets():
    """Our graded tournament-matchup bets from the cached Sheet dump."""
    b = pd.read_parquet("/tmp/tm_sheet.parquet")
    b = b[b["result"].isin(["win", "loss", "push"])].copy()
    b["event_id"] = pd.to_numeric(b["event_id"], errors="coerce").astype("Int64")
    for c in ["dg_id_p1", "dg_id_p2", "p1_odds", "p2_odds",
              "fair_p1", "fair_p2", "edge_on", "units_won"]:
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b["ts"] = pd.to_datetime(b["run_timestamp"], errors="coerce")
    b["dow"] = b["ts"].dt.day_name()
    # our side: is bet_on player_1?
    on1 = (b["bet_on"].astype(str).str.lower().str.strip()
           == b["player_1"].astype(str).str.lower().str.strip())
    b["bet_dg"] = pd.Series(np.where(on1, b["dg_id_p1"], b["dg_id_p2"]), index=b.index).astype("Int64")
    b["opp_dg"] = pd.Series(np.where(on1, b["dg_id_p2"], b["dg_id_p1"]), index=b.index).astype("Int64")
    b["entry_odds"] = np.where(on1, b["p1_odds"], b["p2_odds"])
    b["our_fair"] = np.where(on1, b["fair_p1"], b["fair_p2"])
    b["our_fair"] = np.where(b["our_fair"] > 1.5, b["our_fair"] / 100, b["our_fair"])
    b["entry_impl"] = impl(b["entry_odds"])
    b["stake"] = np.where(b["entry_odds"] < 0, np.abs(b["entry_odds"]) / 100, 1.0)
    b["won"] = (b["result"] == "win").astype(int)
    return b


def pull_odds(event_ids):
    frames = []
    for eid in event_ids:
        for bk in BOOKS:
            d = fetch_historical_matchup_odds(2026, bk, event_id=int(eid))
            if isinstance(d, str) and d == "RATE_LIMITED":
                print("[abort] hit rate-limit suspension — stopping to let it clear. "
                      f"Pulled {len(frames)} book-frames so far.")
                eid = None
                break
            if isinstance(d, pd.DataFrame) and not d.empty:
                frames.append(d)
            time.sleep(3)
        if eid is None:
            break
    if not frames:
        return pd.DataFrame()
    hist = pd.concat(frames, ignore_index=True)
    hist["event_id"] = pd.to_numeric(hist["event_id"], errors="coerce").astype("Int64")
    for c in ["p1_dg_id", "p2_dg_id"]:
        hist[c] = pd.to_numeric(hist[c], errors="coerce").astype("Int64")
    # merge into cache (dedup on event/book/pair)
    if os.path.exists(CACHE):
        old = pd.read_parquet(CACHE)
        hist = pd.concat([old, hist], ignore_index=True).drop_duplicates(
            subset=["event_id", "book", "p1_dg_id", "p2_dg_id"], keep="last")
    hist.to_parquet(CACHE)
    return hist


def dg_long(hist):
    """One row per (matchup, side, book) with own+opp open/close for de-vigging."""
    rows = []
    for _, r in hist.iterrows():
        for own, opp in (("p1", "p2"), ("p2", "p1")):
            rows.append({
                "event_id": r["event_id"], "book": r["book"],
                "dg": r[f"{own}_dg_id"], "opp_dg": r[f"{opp}_dg_id"],
                "own_open": r[f"{own}_open"], "own_close": r[f"{own}_close"],
                "opp_open": r[f"{opp}_open"], "opp_close": r[f"{opp}_close"],
            })
    d = pd.DataFrame(rows)
    for c in ["event_id", "dg", "opp_dg"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype("Int64")
    io, ic = impl(d["own_open"]), impl(d["own_close"])
    oo, oc = impl(d["opp_open"]), impl(d["opp_close"])
    d["open_devig"] = io / (io + oo)
    d["close_devig"] = ic / (ic + oc)
    return d


def roi(f):
    return pd.Series({"n": len(f), "win%": round(100 * f["won"].mean(), 1),
                      "ROI%": round(100 * f["units_won"].sum() / f["stake"].sum(), 1)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recent", type=int, default=4)
    ap.add_argument("--events", type=int, nargs="*")
    args = ap.parse_args()

    bets = load_bets()
    if args.events:
        eids = args.events
    else:
        recent = (bets.groupby("event_id")["ts"].max()
                  .sort_values(ascending=False).head(args.recent).index.tolist())
        eids = [int(e) for e in recent]
    print(f"[pull] events: {eids}")
    hist = pull_odds(eids)
    print(f"[pull] cached {len(hist)} matchup-book rows\n")
    if hist.empty or "event_id" not in hist.columns:
        print("[abort] no odds pulled (likely rate-limited). Wait and re-run; "
              "the cache merges so completed events are kept.")
        return

    bets = bets[bets["event_id"].isin(eids)].copy()
    dl = dg_long(hist)
    # join our bet -> sharp line (prefer pinnacle, else betcris)
    dl["bk_rank"] = dl["book"].map({"pinnacle": 0, "betcris": 1}).fillna(9)
    dl = dl.sort_values("bk_rank")
    key = ["event_id", "dg", "opp_dg"]
    dl1 = dl.drop_duplicates(subset=key, keep="first")
    m = bets.merge(dl1, left_on=["event_id", "bet_dg", "opp_dg"],
                   right_on=key, how="left")
    matched = m["close_devig"].notna()
    print(f"[join] {matched.sum()}/{len(m)} bets matched to sharp odds "
          f"({100*matched.mean():.0f}%)\n")
    m = m[matched].copy()

    m["clv"] = (m["close_devig"] - m["entry_impl"]) * 100          # + = beat close
    m["edge_open"] = (m["our_fair"] - m["open_devig"]) * 100       # our edge vs sharp OPEN

    print("OVERALL:", roi(m).to_dict())
    print(f"mean CLV: {m['clv'].mean():+.2f}pp   ({(m['clv']>0).mean()*100:.0f}% positive)\n")

    print("[T2] ROI by CLV bucket (vs sharp close):")
    m["clvb"] = pd.cut(m["clv"], [-99, -2, 0, 2, 99],
                       labels=["<-2", "-2..0", "0..2", ">2"])
    print(m.groupby("clvb", observed=True).apply(roi, include_groups=False).to_string())

    print("\n[T3] ROI by edge-vs-sharp-OPEN (lever-2 signal, known at bet time):")
    m["eob"] = pd.cut(m["edge_open"], [-99, 0, 3, 99],
                      labels=["<0 (sharp disagrees)", "0-3", "3+"])
    print(m.groupby("eob", observed=True).apply(roi, include_groups=False).to_string())
    print("\n  filter: keep only edge_vs_sharp_open >= 0 :", roi(m[m["edge_open"] >= 0]).to_dict())
    print("  (dropped)                                 :", roi(m[m["edge_open"] < 0]).to_dict())


if __name__ == "__main__":
    main()
