"""
clv_alert.py — Morning closing-line-value check for round matchup bets.

Reads stored round matchup bets from Google Sheets, fetches fresh
(closing) odds for those same matchups, computes CLV, and sends a
Telegram summary.

Usage:
    python clv_alert.py              # Auto-detect round from sheet config
    python clv_alert.py --round 2    # Force specific round

Scheduled via .github/workflows/clv-alert.yml at 6:30 AM EST on
Fri/Sat/Sun (morning of R1-R3 bets closing before the next round).
"""

import os
import sys
import json
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def american_to_implied(odds):
    """American odds -> implied probability (0-1)."""
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None
    if odds == 0:
        return None
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def implied_to_american(prob):
    """Implied probability (0-1) -> American odds (int)."""
    if prob is None or prob <= 0 or prob >= 1:
        return None
    if prob >= 0.5:
        return int(round(-100 * prob / (1 - prob)))
    return int(round(100 * (1 - prob) / prob))


def send_telegram(text):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("  Warning: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        print(f"  Warning: Telegram alert failed: {e}")


# ---------------------------------------------------------------------------
# Load bets from Sheets
# ---------------------------------------------------------------------------

def _drop_excluded_bets(df):
    """Remove audit-preserved invalid bets while supporting legacy sheets."""
    if df is None or df.empty or "result" not in df.columns:
        return df
    result = df["result"].fillna("").astype(str).str.lower().str.strip()
    return df[~result.str.startswith("excluded_")].copy()


def load_round_bets(sim_round, event_id=None):
    """Read stored round matchup bets for a specific round from Google Sheets."""
    from sheets_storage import get_spreadsheet

    spreadsheet = get_spreadsheet()
    ws = spreadsheet.worksheet("Round Matchups")
    rows = ws.get_all_records()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Invalidated rows remain in Sheets for audit history, but are not bets and
    # must be removed before the report's keep-first dedup can select one.
    df = _drop_excluded_bets(df)

    # Filter to this round
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    df = df[df["round"] == sim_round].copy()

    # Filter to this event if specified
    if event_id is not None:
        df = df[df["event_id"].astype(str) == str(event_id)]

    if df.empty:
        return df

    # Normalize names — same lower/strip + name_replacements as the closing
    # feed, so a mapped name (e.g. an accent variant) still pair-matches.
    from sim_inputs import name_replacements
    df["player_1"] = df["player_1"].str.lower().str.strip().replace(name_replacements)
    df["player_2"] = df["player_2"].str.lower().str.strip().replace(name_replacements)
    df["bet_on"] = df["bet_on"].str.lower().str.strip().replace(name_replacements)
    df["bookmaker"] = df["bookmaker"].str.lower().str.strip()

    # Only include sharp books for CLV tracking
    sharp_books = {"betonline", "betcris", "pinnacle"}
    df = df[df["bookmaker"].isin(sharp_books)]

    # Dedup: same matchup + book → keep first (earliest bet placed)
    df = df.drop_duplicates(subset=["player_1", "player_2", "bet_on", "bookmaker"], keep="first")

    return df


# ---------------------------------------------------------------------------
# Fetch closing odds
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BOARD_CLOSING_PATH = os.path.join(
    PROJECT_ROOT, "permanent_data", "board_closing.json"
)


class FrozenClosingUnavailable(RuntimeError):
    """Raised when an automated CLV run cannot use its frozen close."""


def _load_local_board_closing(path=None):
    """Load a Wrangler-downloaded closing snapshot, if one is available."""
    configured_path = path or os.getenv("CLV_BOARD_CLOSING_PATH")
    snapshot_path = configured_path or DEFAULT_BOARD_CLOSING_PATH
    if not os.path.isfile(snapshot_path):
        if configured_path:
            print(f"  board closing snapshot not found: {snapshot_path}")
        return None
    try:
        with open(snapshot_path, "r", encoding="utf-8") as handle:
            closing = json.load(handle)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  board closing snapshot is unreadable ({e})")
        return None
    if not isinstance(closing, dict):
        print("  board closing snapshot is not a JSON object")
        return None
    return closing


def fetch_board_closing(sim_round, event_id=None, snapshot_path=None):
    """The board's FROZEN closing snapshot for this round, from R2 closing.json.

    This is the actual line set at the freeze cutoff — strictly better than
    'whatever is live at 11:30 UTC' (on UK weeks the live feed is 7-12h stale by
    then and gets freshness-rejected, losing the round's CLV entirely). Returns
    a DataFrame in the load_matchup_odds schema, or empty on any validation or
    read failure. CI downloads this file with Wrangler before Python starts;
    interactive runs may still fall back to the live feed when it is absent.
    """
    closing = _load_local_board_closing(snapshot_path)
    if closing is None:
        return pd.DataFrame()
    if event_id is not None and str(closing.get("event_id")) != str(event_id):
        print(f"  board closing is for event {closing.get('event_id')}, "
              f"not {event_id} — ignoring")
        return pd.DataFrame()
    rounds = closing.get("rounds")
    if not isinstance(rounds, dict) or str(sim_round) not in rounds:
        print(f"  board closing has no frozen R{sim_round} snapshot — ignoring")
        return pd.DataFrame()
    snap = rounds[str(sim_round)]
    if not isinstance(snap, dict):
        print(f"  board closing R{sim_round} snapshot is malformed — ignoring")
        return pd.DataFrame()
    rows = []
    for mt, rws in ((snap.get("rmatch") or {}).get("markets") or {}).items():
        if mt != "round_matchup":
            continue
        for r in rws:
            row_round = r.get("round")
            if row_round is not None and str(row_round) != str(sim_round):
                print(f"  board closing contains an R{row_round} row in the "
                      f"R{sim_round} snapshot — ignoring")
                return pd.DataFrame()
            for bk, entry in (r.get("books") or {}).items():
                if not (entry.get("a") and entry.get("b")):
                    continue
                rows.append({"Player 1": str(r.get("player_a", "")).lower().strip(),
                             "Player 2": str(r.get("player_b", "")).lower().strip(),
                             "Bookmaker": str(bk).lower(),
                             "P1 Odds": entry.get("a"), "P2 Odds": entry.get("b")})
    df = pd.DataFrame(rows)
    if not df.empty:
        from sim_inputs import name_replacements
        df["Player 1"] = df["Player 1"].replace(name_replacements)
        df["Player 2"] = df["Player 2"].replace(name_replacements)
        print(f"  Using board FROZEN closing for R{sim_round}: {len(df)} lines")
    return df


def fetch_closing_odds(sim_round):
    """Fetch current round matchup odds from odds_loader, scoped to the round
    being graded so a prior/next round's lines can never stand in as closes."""
    from odds_loader import load_matchup_odds
    from sim_inputs import name_replacements

    df = load_matchup_odds("round_matchups", round=sim_round)
    if df.empty:
        return df

    df["Player 1"] = df["Player 1"].str.lower().str.strip().replace(name_replacements)
    df["Player 2"] = df["Player 2"].str.lower().str.strip().replace(name_replacements)
    df["Bookmaker"] = df["Bookmaker"].str.lower().str.strip()

    return df


def resolve_closing_odds(sim_round, event_id=None, require_frozen=False):
    """Use the frozen board close, optionally refusing any live fallback."""
    closing_odds = fetch_board_closing(sim_round, event_id)
    if not closing_odds.empty:
        return closing_odds
    if require_frozen:
        raise FrozenClosingUnavailable(
            f"authoritative frozen closing is unavailable for event "
            f"{event_id}, R{sim_round}"
        )
    return fetch_closing_odds(sim_round)


# ---------------------------------------------------------------------------
# CLV calculation
# ---------------------------------------------------------------------------

def compute_clv(bets, closing_odds):
    """Match stored bets to closing odds at the SAME book and compute CLV.

    CLV = implied_prob(closing_line) - implied_prob(opening/bet_line)
    Positive CLV means we got a better price than the closing line.
    Bets whose book no longer shows the line are reported as unmatched
    rather than graded against another book's price.
    """
    results = []

    for _, bet in bets.iterrows():
        p1 = bet["player_1"]
        p2 = bet["player_2"]
        bet_on = bet["bet_on"]
        book = bet["bookmaker"]

        # What odds did we bet at?
        if bet_on == p1:
            bet_odds = bet.get("p1_odds")
        else:
            bet_odds = bet.get("p2_odds")

        bet_implied = american_to_implied(bet_odds)
        if bet_implied is None:
            continue

        # Find closing line at the SAME book the bet was placed at (either
        # player order). No cross-book fallback: if the bet book isn't showing
        # the line anymore, the honest answer is "no close", not another
        # book's (or a DataGolf aggregate's) price.
        match = closing_odds[
            (closing_odds["Bookmaker"] == book) & (
                ((closing_odds["Player 1"] == p1) & (closing_odds["Player 2"] == p2)) |
                ((closing_odds["Player 1"] == p2) & (closing_odds["Player 2"] == p1))
            )
        ]

        if match.empty:
            results.append({
                "bet_on": bet_on,
                "opponent": p2 if bet_on == p1 else p1,
                "book": book,
                "bet_odds": int(float(bet_odds)),
                "bet_implied": round(bet_implied * 100, 1),
                "close_odds": None,
                "close_implied": None,
                "close_book": None,
                "clv": None,
                "edge_on": bet.get("edge_on", ""),
            })
            continue

        close_row = match.iloc[0]

        # Get closing odds for the player we bet on
        if bet_on == close_row["Player 1"].lower().strip():
            close_odds_val = close_row["P1 Odds"]
        elif bet_on == close_row["Player 2"].lower().strip():
            close_odds_val = close_row["P2 Odds"]
        else:
            # Name mismatch after normalization — try partial
            close_odds_val = None

        close_implied = american_to_implied(close_odds_val)

        # CLV = closing_implied - bet_implied (positive = we got better price)
        clv = None
        if close_implied is not None:
            clv = round((close_implied - bet_implied) * 100, 1)

        results.append({
            "bet_on": bet_on,
            "opponent": p2 if bet_on == p1 else p1,
            "book": book,
            "bet_odds": int(float(bet_odds)),
            "bet_implied": round(bet_implied * 100, 1),
            "close_odds": int(float(close_odds_val)) if close_odds_val is not None else None,
            "close_implied": round(close_implied * 100, 1) if close_implied is not None else None,
            "close_book": close_row["Bookmaker"] if close_row is not None else None,
            "clv": clv,
            "edge_on": bet.get("edge_on", ""),
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Format + send
# ---------------------------------------------------------------------------

def format_alert(clv_df, sim_round, tourney_name):
    """Build Telegram message from CLV results."""
    lines = []
    title = f"<b>R{sim_round} CLV Report -- {tourney_name.replace('_', ' ').title()}</b>"
    lines.append(title)
    lines.append("")

    if clv_df.empty:
        lines.append("No bets found for this round.")
        return "\n".join(lines)

    matched = clv_df[clv_df["clv"].notna()].copy()
    unmatched = clv_df[clv_df["clv"].isna()]

    if not matched.empty:
        # One row per (matchup, book) — same-book CLV means each book's bet on
        # the same pairing is its own data point, so don't collapse across books.
        best = matched.sort_values("clv", ascending=False).drop_duplicates(
            subset=["bet_on", "opponent", "book"], keep="first"
        )
        avg_clv = matched["clv"].mean()
        positive = (matched["clv"] > 0).sum()
        total = len(matched)

        lines.append(f"<b>Avg CLV: {avg_clv:+.1f}pp ({positive}/{total} positive)</b>")
        lines.append("")

        display_limit = 12
        for _, r in best.head(display_limit).iterrows():
            clv_val = r["clv"]
            marker = "+" if clv_val > 0 else ""
            lines.append(
                f"  {r['bet_on']} vs {r['opponent']} [{r['book']}]"
                f"  {int(r['bet_odds'])} -> {int(r['close_odds'])}"
                f"  <b>{marker}{clv_val:.1f}pp</b>"
            )
        if len(best) > display_limit:
            lines.append(f"  ... +{len(best) - display_limit} more")

    if not unmatched.empty:
        lines.append("")
        lines.append(f"<i>{len(unmatched)} bet(s) -- no closing line at bet book</i>")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Morning CLV alert for round matchup bets")
    parser.add_argument("--round", type=int, help="Force specific round (default: auto-detect)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Detect round + tourney from sheet config
    from sheet_config import load_config
    cfg = load_config()
    tourney = cfg["tourney"]
    event_id = cfg.get("event_id")

    if args.round:
        sim_round = args.round
    else:
        # Sheet round_num = last completed round
        # Bets placed last night are for round_num + 1 (today's round)
        # Closing odds in the feed are also for today's round
        round_num = cfg["round_num"]
        if round_num < 0:
            print("  Pre-event — no round bets to check CLV on.")
            sys.exit(0)
        if round_num >= 4:
            print("  Tournament complete — no more rounds to check CLV on.")
            sys.exit(0)
        sim_round = round_num + 1

    print(f"\n  CLV Alert: R{sim_round} — {tourney}")
    print(f"  Event ID: {event_id}")

    # Load stored bets
    bets = load_round_bets(sim_round, event_id)
    if bets.empty:
        print(f"  No R{sim_round} bets found in Sheets.")
        send_telegram(f"R{sim_round} CLV — {tourney}: no bets found.")
        sys.exit(0)

    print(f"  Found {len(bets)} stored R{sim_round} matchup bets")

    # Automated runs must grade against the authoritative frozen snapshot.
    # Interactive runs retain the historical live-feed fallback for diagnosis.
    require_frozen = os.getenv("CLV_REQUIRE_FROZEN_CLOSING", "").lower() in {
        "1", "true", "yes", "on"
    }
    try:
        closing_odds = resolve_closing_odds(
            sim_round, event_id, require_frozen=require_frozen
        )
    except FrozenClosingUnavailable as e:
        print(f"  ERROR: {e}")
        sys.exit(1)
    if closing_odds.empty:
        print("  No closing odds available.")
        send_telegram(f"R{sim_round} CLV — {tourney}: no closing odds available.")
        sys.exit(0)

    print(f"  Fetched {len(closing_odds)} closing odds lines")

    # Compute CLV
    clv_df = compute_clv(bets, closing_odds)
    print(f"  CLV computed for {len(clv_df)} bets")

    matched = clv_df[clv_df["clv"].notna()]
    if not matched.empty:
        avg = matched["clv"].mean()
        print(f"  Average CLV: {avg:+.1f}pp")

    # Format and send
    msg = format_alert(clv_df, sim_round, tourney)
    print(f"\n{msg}\n")
    send_telegram(msg)
    print("  Done.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"\n  UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        # A crashed CLV run must not be silent — the report simply not arriving
        # reads as "no bets this round", which is exactly wrong.
        send_telegram(f"❌ CLV alert CRASHED before reporting: {type(e).__name__}: {e}")
        sys.exit(1)
