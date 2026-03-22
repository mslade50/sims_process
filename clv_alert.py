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

def load_round_bets(sim_round, event_id=None):
    """Read stored round matchup bets for a specific round from Google Sheets."""
    from sheets_storage import get_spreadsheet

    spreadsheet = get_spreadsheet()
    ws = spreadsheet.worksheet("Round Matchups")
    rows = ws.get_all_records()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Filter to this round
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    df = df[df["round"] == sim_round].copy()

    # Filter to this event if specified
    if event_id is not None:
        df = df[df["event_id"].astype(str) == str(event_id)]

    if df.empty:
        return df

    # Normalize names
    df["player_1"] = df["player_1"].str.lower().str.strip()
    df["player_2"] = df["player_2"].str.lower().str.strip()
    df["bet_on"] = df["bet_on"].str.lower().str.strip()
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

def fetch_closing_odds():
    """Fetch current round matchup odds from odds_loader."""
    from odds_loader import load_matchup_odds
    from sim_inputs import name_replacements

    df = load_matchup_odds("round_matchups")
    if df.empty:
        return df

    df["Player 1"] = df["Player 1"].str.lower().str.strip().replace(name_replacements)
    df["Player 2"] = df["Player 2"].str.lower().str.strip().replace(name_replacements)
    df["Bookmaker"] = df["Bookmaker"].str.lower().str.strip()

    return df


# ---------------------------------------------------------------------------
# CLV calculation
# ---------------------------------------------------------------------------

def compute_clv(bets, closing_odds):
    """Match stored bets to closing odds and compute CLV.

    CLV = implied_prob(closing_line) - implied_prob(opening/bet_line)
    Positive CLV means we got a better price than the closing line.
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

        # Find closing line — match on player pair (any book order)
        match = closing_odds[
            ((closing_odds["Player 1"] == p1) & (closing_odds["Player 2"] == p2)) |
            ((closing_odds["Player 1"] == p2) & (closing_odds["Player 2"] == p1))
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
                "clv": None,
                "edge_on": bet.get("edge_on", ""),
            })
            continue

        # Use the sharpest available book for closing line, preferring pinnacle
        sharp_order = ["pinnacle", "betcris", "betonline", "draftkings", "fanduel"]
        close_row = None
        for sharp in sharp_order:
            sharp_match = match[match["Bookmaker"] == sharp]
            if not sharp_match.empty:
                close_row = sharp_match.iloc[0]
                break
        if close_row is None:
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
        # Deduplicate display: best CLV per unique matchup (bet_on + opponent)
        best = matched.sort_values("clv", ascending=False).drop_duplicates(
            subset=["bet_on", "opponent"], keep="first"
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
                f"  {r['bet_on']} vs {r['opponent']}"
                f"  {int(r['bet_odds'])} -> {int(r['close_odds'])}"
                f"  <b>{marker}{clv_val:.1f}pp</b>"
            )
        if len(best) > display_limit:
            lines.append(f"  ... +{len(best) - display_limit} more")

    if not unmatched.empty:
        lines.append("")
        lines.append(f"<i>{len(unmatched)} bet(s) -- no closing line found</i>")

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

    # Fetch closing odds
    closing_odds = fetch_closing_odds()
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
        sys.exit(1)
