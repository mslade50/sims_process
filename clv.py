"""
clv.py — Closing Line Value (CLV) annotation for graded bets.

Pulls archived opening/closing odds from DataGolf's historical-odds feeds
(available shortly after an event settles) and annotates each bet:

    open_odds   — bet-side OPENING line (American) at clv_book
    close_odds  — bet-side CLOSING line (American) at clv_book
    tot_clv     — implied(close) - implied(open), in percentage points.
                  Positive = line moved toward our side from open to close.
    clv         — implied(close) - implied(our bet line), in percentage points.
                  Positive = we beat the close. (Same convention as clv_alert.py)
    clv_book    — book the archived open/close came from. Same book as the bet
                  when DataGolf archives it; otherwise first available fallback
                  (sharp books first).

Scope: round matchups, tournament matchups, and PRE-TOURNAMENT finish
position bets at DataGolf-covered sportsbooks. Exchange bets (kalshi/novig)
and live in-play bets get no CLV — in-play prices can't be compared to the
pre-tournament close, and DG doesn't archive exchange prices.

Usage:
    python clv.py --event-id 541 --year 2026              # preview CSV only
    python clv.py --event-id 541 --year 2026 --write      # write sheet + ledger
    python clv.py --backfill --write                      # all events in sheet

grade_bets.py calls annotate_event_clv() automatically after grading.

Notes / caveats:
- Implied probabilities are computed from raw prices (vig NOT removed). For
  same-book comparisons the vig largely cancels; for fallback-book matches
  small cross-book vig differences leak into clv.
- Not every book archives every market (pinnacle/betcris archive win only
  for outrights); fallback covers the rest.
"""

import os
import argparse
import time
from datetime import datetime

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from api_utils import fetch_historical_matchup_odds, fetch_historical_outright_odds
from sim_inputs import name_replacements

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# DataGolf book slugs, in fallback priority order (sharp first). The bet's own
# book is always tried first; these are tried when it isn't archived.
MATCHUP_BOOK_PRIORITY = [
    "betcris", "pinnacle", "betonline", "bovada", "bet365",
    "unibet", "draftkings", "fanduel", "caesars", "betmgm", "pointsbet",
]
# pinnacle/betcris archive outrights for "win" only; the matcher skips books
# with no row for the bet's market, so top_5/10/20 fall through to betonline+.
OUTRIGHT_BOOK_PRIORITY = [
    "pinnacle", "betcris", "betonline", "bovada", "bet365",
    "unibet", "draftkings", "fanduel", "caesars", "betmgm", "pointsbet",
]

# Books DataGolf archives — bets at any other book (kalshi, novig, ...) are
# skipped entirely per design: no CLV on exchange bets.
DG_BOOKS = set(MATCHUP_BOOK_PRIORITY)

OUTRIGHT_MARKETS = ["win", "top_5", "top_10", "top_20"]

# sheet/ledger book name -> DataGolf book slug (identity for most)
BOOK_NAME_MAP = {
    "bet online": "betonline",
    "bookmaker": "betcris",
}

# DataGolf bet_type -> our round key ("tournament" or round number as str)
DG_BET_TYPE_MAP = {
    "72-hole Match": "tournament",
    "R1 Match-Up": "1",
    "R2 Match-Up": "2",
    "R3 Match-Up": "3",
    "R4 Match-Up": "4",
}

# Seconds between DataGolf requests (limit is 45/min across all endpoints)
REQUEST_PACING = 1.5

CLV_SHEET_COLS = ["open_odds", "close_odds", "tot_clv", "clv", "clv_book"]


# ---------------------------------------------------------------------------
# Odds math (same conventions as clv_alert.py)
# ---------------------------------------------------------------------------

def american_to_implied(odds) -> float | None:
    """American odds -> implied probability (0-1). Raw price, vig included."""
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None
    if odds == 0 or np.isnan(odds):
        return None
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def normalize_book(book) -> str:
    b = str(book).lower().strip()
    return BOOK_NAME_MAP.get(b, b)


MARKET_ALIASES = {"winner": "win", "top5": "top_5", "top10": "top_10", "top20": "top_20"}


def normalize_market(market) -> tuple[str, bool]:
    """'Top 20' -> ('top_20', False); 'winner_no' -> ('win', True)."""
    m = str(market).lower().strip().replace(" ", "_")
    is_no = m.endswith("_no")
    if is_no:
        m = m[:-3]
    return MARKET_ALIASES.get(m, m), is_no


def normalize_dg_name(name: str) -> str:
    """DataGolf 'Last, First' -> lowercase 'first last' run through name_replacements."""
    s = str(name).strip()
    if "," in s:
        last, first = s.split(",", 1)
        s = f"{first.strip()} {last.strip()}"
    s = s.lower().strip()
    return name_replacements.get(s, s)


# ---------------------------------------------------------------------------
# Odds tape builders (one DataGolf call per book / book+market)
# ---------------------------------------------------------------------------

def build_matchup_tape(event_id, year, books) -> pd.DataFrame:
    """
    Fetch archived matchup odds for each book. A single matchups call returns
    ALL bet types (72-hole + R1-R4 match-ups + 3-balls), so it's one request
    per book. Returns long df:
        book, round_key, p1_dg_id, p2_dg_id, p1_name, p2_name,
        p1_open, p1_close, p2_open, p2_close
    """
    frames = []
    for book in books:
        df = fetch_historical_matchup_odds(year, book, event_id=event_id)
        time.sleep(REQUEST_PACING)
        if isinstance(df, str):  # "RATE_LIMITED"
            print(f"  [clv] Rate-limited fetching matchups for {book} — stopping fetches")
            break
        if df is None or df.empty:
            continue
        df = df[df["bet_type"].isin(DG_BET_TYPE_MAP)].copy()
        if df.empty:
            continue
        df["round_key"] = df["bet_type"].map(DG_BET_TYPE_MAP)
        df["book"] = book
        df = df.rename(columns={
            "p1_player_name": "p1_name", "p2_player_name": "p2_name",
        })
        frames.append(df[[
            "book", "round_key", "p1_dg_id", "p2_dg_id", "p1_name", "p2_name",
            "p1_open", "p1_close", "p2_open", "p2_close",
        ]])

    if not frames:
        return pd.DataFrame()
    tape = pd.concat(frames, ignore_index=True)
    tape["p1_norm_name"] = tape["p1_name"].apply(normalize_dg_name)
    tape["p2_norm_name"] = tape["p2_name"].apply(normalize_dg_name)
    print(f"  [clv] Matchup tape: {len(tape)} rows across {tape['book'].nunique()} books")
    return tape


def build_outright_tape(event_id, year, books, markets=None) -> pd.DataFrame:
    """
    Fetch archived outright (finish position) odds, one request per
    book x market. Returns long df:
        book, market, dg_id, player_name, norm_name, open_odds, close_odds
    """
    markets = markets or OUTRIGHT_MARKETS
    frames = []
    rate_limited = False
    for book in books:
        for market in markets:
            df = fetch_historical_outright_odds(year, book, event_id=event_id, market=market)
            time.sleep(REQUEST_PACING)
            if isinstance(df, str):  # "RATE_LIMITED"
                print(f"  [clv] Rate-limited fetching outrights ({book}/{market}) — stopping fetches")
                rate_limited = True
                break
            if df is None or df.empty:
                continue
            frames.append(df)
        if rate_limited:
            break

    if not frames:
        return pd.DataFrame()
    tape = pd.concat(frames, ignore_index=True)
    tape["norm_name"] = tape["player_name"].apply(normalize_dg_name)
    print(f"  [clv] Outright tape: {len(tape)} rows across {tape['book'].nunique()} books")
    return tape


# ---------------------------------------------------------------------------
# Bet matching
# ---------------------------------------------------------------------------

def _clv_fields(bet_implied, open_am, close_am, book):
    """Compute the CLV output fields from matched open/close American odds."""
    open_imp = american_to_implied(open_am)
    close_imp = american_to_implied(close_am)
    out = {
        "open_odds": open_am if open_am is not None and not pd.isna(open_am) else None,
        "close_odds": close_am if close_am is not None and not pd.isna(close_am) else None,
        "clv_book": book,
        "tot_clv": None,
        "clv": None,
    }
    if open_imp is not None and close_imp is not None:
        out["tot_clv"] = round((close_imp - open_imp) * 100, 2)
    if bet_implied is not None and close_imp is not None:
        out["clv"] = round((close_imp - bet_implied) * 100, 2)
    return out


def _empty_clv(note=""):
    return {"open_odds": None, "close_odds": None, "clv_book": None,
            "tot_clv": None, "clv": None, "clv_note": note}


def match_matchup_bet(bet, tape) -> dict:
    """
    Match one matchup bet to the archived tape and compute CLV fields.

    bet: dict-like with round_key ("tournament"/"1".."4"), bookmaker,
         dg_id_bet_on, dg_id_opponent, bet_on_name, opponent_name, book_odds.
    """
    if tape.empty:
        return _empty_clv("no tape")

    cands = tape[tape["round_key"] == bet["round_key"]]
    if cands.empty:
        return _empty_clv("round not archived")

    dg_on, dg_opp = bet.get("dg_id_bet_on"), bet.get("dg_id_opponent")
    have_ids = pd.notna(dg_on) and pd.notna(dg_opp) and float(dg_on) > 0 and float(dg_opp) > 0

    if have_ids:
        dg_on, dg_opp = float(dg_on), float(dg_opp)
        fwd = (cands["p1_dg_id"] == dg_on) & (cands["p2_dg_id"] == dg_opp)
        rev = (cands["p1_dg_id"] == dg_opp) & (cands["p2_dg_id"] == dg_on)
    else:
        n_on, n_opp = bet.get("bet_on_name", ""), bet.get("opponent_name", "")
        fwd = (cands["p1_norm_name"] == n_on) & (cands["p2_norm_name"] == n_opp)
        rev = (cands["p1_norm_name"] == n_opp) & (cands["p2_norm_name"] == n_on)

    hits = cands[fwd | rev]
    if hits.empty:
        return _empty_clv("matchup not archived")

    bet_book = normalize_book(bet.get("bookmaker", ""))
    book_order = [bet_book] + [b for b in MATCHUP_BOOK_PRIORITY if b != bet_book]
    bet_implied = american_to_implied(bet.get("book_odds"))

    for book in book_order:
        row = hits[hits["book"] == book]
        if row.empty:
            continue
        r = row.iloc[0]
        if have_ids:
            on_is_p1 = r["p1_dg_id"] == dg_on
        else:
            on_is_p1 = r["p1_norm_name"] == bet.get("bet_on_name", "")
        side = "p1" if on_is_p1 else "p2"
        fields = _clv_fields(bet_implied, r[f"{side}_open"], r[f"{side}_close"], book)
        fields["clv_note"] = "" if book == bet_book else "fallback book"
        return fields

    return _empty_clv("no book match")


def match_finish_bet(bet, tape) -> dict:
    """
    Match one pre-tournament finish position bet to the archived outright tape.

    bet: dict-like with market_type, bookmaker, dg_id, player_name
         (normalized), book_odds (American).
    """
    if tape.empty:
        return _empty_clv("no tape")

    base_market, is_no = normalize_market(bet.get("market_type", ""))
    if is_no:
        # NO-side markets only exist on exchanges, which are excluded upstream
        return _empty_clv("exchange NO market — skipped")

    cands = tape[tape["market"] == base_market]
    if cands.empty:
        return _empty_clv("market not archived")

    dg_id = bet.get("dg_id")
    have_id = pd.notna(dg_id) and float(dg_id) > 0
    if have_id:
        hits = cands[cands["dg_id"] == float(dg_id)]
    else:
        hits = cands[cands["norm_name"] == bet.get("player_name", "")]
    if hits.empty:
        return _empty_clv("player not archived")

    bet_book = normalize_book(bet.get("bookmaker", ""))
    book_order = [bet_book] + [b for b in OUTRIGHT_BOOK_PRIORITY if b != bet_book]
    bet_implied = american_to_implied(bet.get("book_odds"))

    for book in book_order:
        row = hits[hits["book"] == book]
        if row.empty:
            continue
        r = row.iloc[0]
        fields = _clv_fields(bet_implied, r["open_odds"], r["close_odds"], book)
        fields["clv_note"] = "" if book == bet_book else "fallback book"
        return fields

    return _empty_clv("no book match")


# ---------------------------------------------------------------------------
# Bet loaders — normalize to one frame per bet family
# ---------------------------------------------------------------------------
# Normalized matchup bet columns:
#   event_id, year, round_key, bookmaker, book_odds, dg_id_bet_on,
#   dg_id_opponent, bet_on_name, opponent_name, source_tab, sheet_row
# Normalized finish bet columns:
#   event_id, year, market_type, bookmaker, book_odds, dg_id, player_name,
#   source_tab, sheet_row


def _drop_uncovered_books(df, label):
    """Drop bets at books DataGolf doesn't archive (exchanges etc.)."""
    if df is None or df.empty:
        return df
    covered = df["bookmaker"].apply(normalize_book).isin(DG_BOOKS)
    dropped = int((~covered).sum())
    if dropped:
        skipped_books = sorted(df.loc[~covered, "bookmaker"].astype(str).str.lower().unique())
        print(f"  [clv] Skipping {dropped} {label} bets at uncovered books: {skipped_books}")
    return df[covered].copy()


def load_bets_from_ledger(event_id) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load matchup + pre-tournament finish bets for an event from the ledger."""
    path = os.path.join("permanent_data", "bet_ledger.parquet")
    ledger = pd.read_parquet(path)
    ev = ledger[ledger["event_id"].astype(str) == str(event_id)].copy()

    mu = ev[ev["bet_type"].isin(["round_matchup", "tournament_matchup"])].copy()
    if not mu.empty:
        mu["round_key"] = np.where(
            mu["bet_type"] == "tournament_matchup", "tournament",
            mu["round"].astype(str).str.strip(),
        )
        mu["bet_on_name"] = mu["bet_on"].astype(str).str.lower().str.strip()
        mu["opponent_name"] = mu["opponent"].astype(str).str.lower().str.strip()
        mu["dg_id_bet_on"] = pd.to_numeric(mu["dg_id_bet_on"], errors="coerce")
        mu["dg_id_opponent"] = pd.to_numeric(mu["dg_id_opponent"], errors="coerce")
        mu["source_tab"] = "ledger"
        mu["sheet_row"] = None
        mu = mu[["event_id", "year", "round_key", "bookmaker", "book_odds",
                 "dg_id_bet_on", "dg_id_opponent", "bet_on_name",
                 "opponent_name", "source_tab", "sheet_row"]]

    # finish_position_live is excluded: in-play prices can't be compared to
    # the pre-tournament close DataGolf archives.
    fin = ev[ev["bet_type"] == "finish_position"].copy()
    if not fin.empty:
        # Ledger finish rows store the market (win/top_5/...) in `opponent`
        fin["market_type"] = fin["opponent"].astype(str)
        fin["player_name"] = fin["bet_on"].astype(str).str.lower().str.strip()
        fin["dg_id"] = pd.to_numeric(fin["dg_id_bet_on"], errors="coerce")
        fin["source_tab"] = "ledger"
        fin["sheet_row"] = None
        fin = fin[["event_id", "year", "market_type", "bookmaker", "book_odds",
                   "dg_id", "player_name", "source_tab", "sheet_row"]]

    return _drop_uncovered_books(mu, "matchup"), _drop_uncovered_books(fin, "finish")


def load_bets_from_sheet(event_id=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load bets from the Google Sheet tabs (read-only) and normalize.
    event_id=None loads ALL events (for backfill).
    """
    from sheets_storage import get_spreadsheet

    spreadsheet = get_spreadsheet()
    mu_frames, fin_frames = [], []

    def read_tab(tab):
        import gspread
        try:
            ws = spreadsheet.worksheet(tab)
            data = ws.get_all_values()
        except gspread.exceptions.WorksheetNotFound:
            return pd.DataFrame()
        if len(data) < 2:
            return pd.DataFrame()
        df = pd.DataFrame(data[1:], columns=data[0])
        df["sheet_row"] = df.index + 2
        if event_id is not None and "event_id" in df.columns:
            df = df[df["event_id"].astype(str).str.strip() == str(event_id)]
        return df

    # Matchup tabs
    for tab, is_round in [("Tournament Matchups", False), ("Round Matchups", True)]:
        df = read_tab(tab)
        time.sleep(2)
        if df.empty:
            continue
        out = pd.DataFrame(index=df.index)
        p1 = df["player_1"].astype(str).str.lower().str.strip()
        p2 = df["player_2"].astype(str).str.lower().str.strip()
        bet_on = df["bet_on"].astype(str).str.lower().str.strip()
        on_is_p1 = bet_on == p1
        out["event_id"] = df["event_id"].astype(str).str.strip()
        out["year"] = pd.to_numeric(df.get("year", ""), errors="coerce")
        out["round_key"] = df["round"].astype(str).str.strip() if is_round else "tournament"
        out["bookmaker"] = df["bookmaker"]
        out["book_odds"] = np.where(on_is_p1, df["p1_odds"], df["p2_odds"])
        out["dg_id_bet_on"] = pd.to_numeric(
            np.where(on_is_p1, df.get("dg_id_p1", ""), df.get("dg_id_p2", "")), errors="coerce")
        out["dg_id_opponent"] = pd.to_numeric(
            np.where(on_is_p1, df.get("dg_id_p2", ""), df.get("dg_id_p1", "")), errors="coerce")
        out["bet_on_name"] = bet_on
        out["opponent_name"] = np.where(on_is_p1, p2, p1)
        out["source_tab"] = tab
        out["sheet_row"] = df["sheet_row"].values
        mu_frames.append(out)

    # Finish position tab. "Live" (in-play finish bets) is deliberately NOT
    # read: in-play prices can't be compared to the pre-tournament close.
    for tab in ["Finish Positions"]:
        df = read_tab(tab)
        time.sleep(2)
        if df.empty:
            continue
        out = pd.DataFrame(index=df.index)
        out["event_id"] = df["event_id"].astype(str).str.strip()
        out["year"] = pd.to_numeric(df.get("year", ""), errors="coerce")
        out["market_type"] = df["market_type"]
        out["bookmaker"] = df["sportsbook"]
        out["book_odds"] = df["american_odds"]
        out["dg_id"] = pd.to_numeric(df.get("dg_id", ""), errors="coerce")
        out["player_name"] = (
            df["player_name"].astype(str).str.lower().str.strip().replace(name_replacements)
        )
        out["source_tab"] = tab
        out["sheet_row"] = df["sheet_row"].values
        fin_frames.append(out)

    mu = pd.concat(mu_frames, ignore_index=True) if mu_frames else pd.DataFrame()
    fin = pd.concat(fin_frames, ignore_index=True) if fin_frames else pd.DataFrame()
    return _drop_uncovered_books(mu, "matchup"), _drop_uncovered_books(fin, "finish")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def compute_event_clv(event_id, year, mu_bets, fin_bets) -> pd.DataFrame:
    """
    Compute CLV for all bets of an event. Returns one combined DataFrame with
    the normalized bet columns + CLV columns. Purely local — no writes.
    """
    results = []

    if mu_bets is not None and not mu_bets.empty:
        books = sorted({normalize_book(b) for b in mu_bets["bookmaker"].dropna()}
                       & set(MATCHUP_BOOK_PRIORITY))
        # Always include the top sharp books so fallback matching has a tape
        for b in MATCHUP_BOOK_PRIORITY[:3]:
            if b not in books:
                books.append(b)
        print(f"  [clv] Fetching matchup tape for books: {books}")
        tape = build_matchup_tape(event_id, year, books)
        for _, bet in mu_bets.iterrows():
            row = bet.to_dict()
            row.update(match_matchup_bet(row, tape))
            row["bet_family"] = "matchup"
            results.append(row)

    if fin_bets is not None and not fin_bets.empty:
        books = sorted({normalize_book(b) for b in fin_bets["bookmaker"].dropna()}
                       & set(OUTRIGHT_BOOK_PRIORITY))
        for b in OUTRIGHT_BOOK_PRIORITY[:3]:
            if b not in books:
                books.append(b)
        markets = sorted(
            {normalize_market(m)[0] for m in fin_bets["market_type"].dropna()}
            & set(OUTRIGHT_MARKETS)
        )
        print(f"  [clv] Fetching outright tape for books: {books}, markets: {markets}")
        tape = build_outright_tape(event_id, year, books, markets)
        for _, bet in fin_bets.iterrows():
            row = bet.to_dict()
            row.update(match_finish_bet(row, tape))
            row["bet_family"] = "finish"
            results.append(row)

    return pd.DataFrame(results)


def print_summary(df: pd.DataFrame):
    if df.empty:
        print("  No CLV-eligible bets found for event.")
        return
    matched = df[df["clv"].notna()]
    print(f"\n  CLV SUMMARY")
    print(f"  {'-'*44}")
    print(f"  Bets:            {len(df)}")
    print(f"  Matched to tape: {len(matched)} ({len(matched)/len(df)*100:.0f}%)")
    if not matched.empty:
        print(f"  Avg CLV:         {matched['clv'].mean():+.2f}pp")
        print(f"  Positive CLV:    {(matched['clv'] > 0).sum()}/{len(matched)}")
        print(f"  Avg tot_clv:     {matched['tot_clv'].mean():+.2f}pp")
        by_fam = matched.groupby("bet_family")["clv"].agg(["count", "mean"])
        for fam, r in by_fam.iterrows():
            print(f"    {fam:<10} n={int(r['count']):<4} avg clv {r['mean']:+.2f}pp")
    unmatched = df[df["clv"].isna()]
    if not unmatched.empty:
        print(f"\n  Unmatched reasons:")
        for note, n in unmatched["clv_note"].value_counts().items():
            print(f"    {note or '(none)'}: {n}")


SHARP_CLV_BOOKS = {"pinnacle", "betcris", "betonline"}


def clv_summary_stats(df) -> dict | None:
    """
    Compact CLV stats for the Monday results email. `clv_book` tells which
    book's close each bet was measured against; "vs sharp close" covers rows
    matched to a sharp book's archived line. Returns None if nothing matched.
    """
    if df is None or df.empty or "clv" not in df.columns:
        return None
    m = df[df["clv"].notna()]
    if m.empty:
        return None
    sharp = m[m["clv_book"].isin(SHARP_CLV_BOOKS)]
    stats = {
        "n": len(m),
        "avg_clv": float(m["clv"].mean()),
        "pos_pct": float((m["clv"] > 0).mean() * 100),
        "sharp_n": len(sharp),
        "sharp_avg_clv": float(sharp["clv"].mean()) if not sharp.empty else None,
    }
    for fam, sub in m.groupby("bet_family"):
        stats[f"{fam}_n"] = len(sub)
        stats[f"{fam}_avg_clv"] = float(sub["clv"].mean())
    return stats


# ---------------------------------------------------------------------------
# Writers — Google Sheet + Parquet ledger
# ---------------------------------------------------------------------------

def write_clv_to_sheet(spreadsheet, df):
    """
    Write CLV columns back to the source sheet tabs. Adds the CLV header
    columns to a tab (at the end of row 1) if they don't exist yet.
    Only rows that matched (clv_book set) are written.
    """
    import gspread

    if df.empty:
        return

    for tab in [t for t in df["source_tab"].unique() if t != "ledger"]:
        rows = df[(df["source_tab"] == tab) & df["clv_book"].notna()]
        if rows.empty:
            continue

        ws = spreadsheet.worksheet(tab)
        headers = ws.row_values(1)

        missing = [c for c in CLV_SHEET_COLS if c not in headers]
        if missing:
            if ws.col_count < len(headers) + len(missing):
                ws.add_cols(len(headers) + len(missing) - ws.col_count)
            header_cells = [
                gspread.Cell(1, len(headers) + i + 1, name)
                for i, name in enumerate(missing)
            ]
            ws.update_cells(header_cells, value_input_option="RAW")
            headers = headers + missing
            print(f"  [clv] Added columns {missing} to '{tab}'")

        col_idx = {c: headers.index(c) + 1 for c in CLV_SHEET_COLS}

        cells = []
        for _, r in rows.iterrows():
            sheet_row = int(r["sheet_row"])
            for c in CLV_SHEET_COLS:
                val = r[c]
                if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
                    continue
                if isinstance(val, (np.floating, np.integer, float, int)):
                    val = float(val)
                    if val == int(val):
                        val = int(val)
                cells.append(gspread.Cell(sheet_row, col_idx[c], val))

        if cells:
            ws.update_cells(cells, value_input_option="USER_ENTERED")
            print(f"  [clv] Wrote CLV for {len(rows)} rows to '{tab}'")
        time.sleep(2)


def build_ledger_records(df) -> list[dict]:
    """Convert matched CLV rows to ledger-update records keyed by LEDGER_DEDUP_COLS."""
    records = []
    for _, r in df[df["clv_book"].notna()].iterrows():
        if r["bet_family"] == "matchup":
            is_tourney = r["round_key"] == "tournament"
            rec = {
                "event_id": r["event_id"],
                "bet_type": "tournament_matchup" if is_tourney else "round_matchup",
                "round": "0" if is_tourney else str(r["round_key"]),
                "bet_on": r["bet_on_name"],
                "opponent": r["opponent_name"],
                "bookmaker": r["bookmaker"],
            }
        else:
            rec = {
                "event_id": r["event_id"],
                "bet_type": "finish_position",
                "round": "0",
                "bet_on": r["player_name"],
                "opponent": r["market_type"],
                "bookmaker": r["bookmaker"],
            }
        for c in CLV_SHEET_COLS:
            rec[c] = r[c]
        records.append(rec)
    return records


def annotate_event_clv(spreadsheet, event_id, year):
    """
    Full CLV pass for one event: load bets from the sheet, match against
    DataGolf's archive, write CLV columns to the sheet tabs and the ledger.
    Called by grade_bets.py after grading; safe to re-run (overwrites in place).
    """
    from sheets_storage import update_ledger_clv

    mu, fin = load_bets_from_sheet(event_id)
    n_mu = 0 if mu is None or mu.empty else len(mu)
    n_fin = 0 if fin is None or fin.empty else len(fin)
    print(f"  [clv] Loaded {n_mu} matchup bets, {n_fin} finish bets for event {event_id}")
    if n_mu + n_fin == 0:
        return pd.DataFrame()

    df = compute_event_clv(event_id, year, mu, fin)
    print_summary(df)

    write_clv_to_sheet(spreadsheet, df)
    update_ledger_clv(build_ledger_records(df))
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute CLV for bets from DataGolf's odds archive")
    parser.add_argument("--event-id", type=str, help="Single event to process")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--source", choices=["ledger", "sheet"], default="sheet",
                        help="Where to read bets from (preview mode only)")
    parser.add_argument("--backfill", action="store_true",
                        help="Process every event found in the sheet tabs")
    parser.add_argument("--events", type=str, default=None,
                        help="Comma-separated event_id filter for --backfill (resume/chunk)")
    parser.add_argument("--write", action="store_true",
                        help="Write results to the Google Sheet + ledger (default: preview CSV only)")
    parser.add_argument("--out", type=str, default=None, help="Preview CSV path")
    args = parser.parse_args()

    if not args.backfill and not args.event_id:
        parser.error("Provide --event-id or --backfill")

    print("\n" + "=" * 60)
    print(f"  CLV ANNOTATOR ({'WRITE' if args.write else 'preview only'})")
    print("=" * 60)

    if args.backfill:
        mu_all, fin_all = load_bets_from_sheet(None)
        pairs = pd.concat([
            f[["event_id", "year"]] for f in (mu_all, fin_all)
            if f is not None and not f.empty
        ], ignore_index=True)
        events = (pairs.groupby("event_id")["year"]
                  .agg(lambda s: int(s.mode().iloc[0]) if s.notna().any() else datetime.now().year)
                  .to_dict())
        if args.events:
            wanted = {e.strip() for e in args.events.split(",")}
            events = {k: v for k, v in events.items() if k in wanted}
        print(f"  Backfilling {len(events)} events: {events}")

        if args.write:
            from sheets_storage import get_spreadsheet, update_ledger_clv
            spreadsheet = get_spreadsheet()

        all_dfs = []
        for i, (eid, yr) in enumerate(sorted(events.items())):
            print(f"\n  --- Event {eid} ({yr}) [{i+1}/{len(events)}] ---", flush=True)
            mu = mu_all[mu_all["event_id"] == eid] if not mu_all.empty else mu_all
            fin = fin_all[fin_all["event_id"] == eid] if not fin_all.empty else fin_all
            df = compute_event_clv(eid, yr, mu, fin)
            print_summary(df)
            # Write per event so an interrupted backfill keeps its progress
            # (safe to re-run — overwrites the same cells/ledger fields).
            if args.write and not df.empty:
                write_clv_to_sheet(spreadsheet, df)
                update_ledger_clv(build_ledger_records(df))
            all_dfs.append(df)
            time.sleep(5)  # breathing room between events (DG rate limit)

        combined = pd.concat([d for d in all_dfs if not d.empty], ignore_index=True) \
            if any(not d.empty for d in all_dfs) else pd.DataFrame()
        if args.write:
            print(f"\n  Backfill complete: {len(events)} events processed.")
            return
    else:
        year = args.year or datetime.now().year
        if args.source == "ledger" and not args.write:
            mu, fin = load_bets_from_ledger(args.event_id)
        else:
            mu, fin = load_bets_from_sheet(args.event_id)
        n_mu = 0 if mu is None or mu.empty else len(mu)
        n_fin = 0 if fin is None or fin.empty else len(fin)
        print(f"  Loaded {n_mu} matchup bets, {n_fin} finish bets")
        combined = compute_event_clv(args.event_id, year, mu, fin)
        print_summary(combined)

    if combined.empty:
        print("\n  Nothing to write.")
        return

    if args.write:
        from sheets_storage import get_spreadsheet, update_ledger_clv
        spreadsheet = get_spreadsheet()
        write_clv_to_sheet(spreadsheet, combined)
        update_ledger_clv(build_ledger_records(combined))
    else:
        out_path = args.out or f"clv_preview_{args.event_id or 'backfill'}.csv"
        combined.to_csv(out_path, index=False)
        print(f"\n  Preview written to {out_path} (no sheet/ledger writes)")


if __name__ == "__main__":
    main()
