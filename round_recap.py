"""round_recap.py - Telegram recap of the prior round's matchup results.

Two groups for one completed round of the current event:
  (a) recommended: Round Matchups rows the bot alerted (sharp books, one per
      pairing+side), and
  (b) killed: edges > 5% that the reprice gates filtered out
      (``Killed Round Matchups`` tab, written by reprice.py), with kill reason.

Both are graded from DataGolf live-tournament-stats round scores through
grade_bets.grade_round_matchup (spread, ties rule, units), so settlement logic
is not duplicated.  Units are grade_bets' flat convention (risk 1u at plus
money, win 1u at minus money); killed-edge units are hypothetical.

Sent at most once per (event_id, round): the marker lives in the
``Round Recap Log`` tab.  If any involved player is still on course (thru < 18)
the send is skipped without a marker and the next run retries; WD/DQ and
players with no score are graded void ("no score").  Rounds are matchup-only (no 3-balls).

Usage:
    python round_recap.py --event-id 123 --round 2 --dry-run   # print only
    python round_recap.py --event-id 123 --round 2             # send + mark
    python round_recap.py --event-id 123 --round 2 --force     # resend
"""

import argparse
import html
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd

MIN_SCORED_PLAYERS = 10
DATAGOLF_TIMEOUT = 20
MESSAGE_MAX_CHARS = 3500
RESULT_LETTER = {"win": "W", "loss": "L", "push": "P"}


# ── records ──────────────────────────────────────────────────────────────────
def load_recommended(event_id, rnd, spreadsheet):
    """Alerted bets for event+round: stored sharp-book rows, first per pairing+side."""
    import reprice_core as rc
    from sheets_storage import TAB_ROUND_MU, is_excluded_or_invalid_result

    rows = spreadsheet.worksheet(TAB_ROUND_MU).get_all_records()
    def unverified_betcris(r):
        # Same skip as dedup_round_matchups: legacy rows cannot identify the wager.
        book = str(r.get("bookmaker", "")).lower().strip()
        return ("betcris" in book or "bookmaker" in book) and not rc._truthy_value(
            r.get("line_verified"))

    rows = sorted(
        (r for r in rows
         if str(r.get("event_id", "")).strip() == str(event_id).strip()
         and str(r.get("round", "")).strip() == str(rnd).strip()
         and not is_excluded_or_invalid_result(r.get("result", ""))
         and not unverified_betcris(r)
         and str(r.get("bookmaker", "")).lower().strip() in rc.TELEGRAM_BOOKS),
        key=lambda r: str(r.get("run_timestamp", "")),
    )
    seen, out = set(), []
    for r in rows:
        key = rc.alerted_key(r.get("player_1", ""), r.get("player_2", ""),
                             r.get("bet_on", ""), rc.selected_spread_line(r))
        if key not in seen:
            seen.add(key)
            out.append(dict(r))
    return out


# ── scores + grading ─────────────────────────────────────────────────────────
def fetch_round_scores(rnd, api_key, expected_event_name=None):
    """Official round results as a grade_bets results frame, or None if unusable.

    Uses the same fields as official_round_history.completed_strokes: ``round``
    (the round score to par, not the tournament total) and ``thru``.  Columns:
    player_name, round_{rnd} (NaN unless the player finished 18), and status:
    final (thru 18), terminal (WD/DQ/CUT before 18 holes) or live.
    """
    from api_utils import fetch_live_stats

    df = fetch_live_stats(rnd, api_key, timeout=DATAGOLF_TIMEOUT)
    if df is None or not {"player_name", "position", "thru", "round"}.issubset(df.columns):
        return None
    if expected_event_name and "event_name" in df.columns:
        names = df["event_name"].dropna().astype(str).str.lower().str.strip()
        if not names.empty and names.iloc[0] != str(expected_event_name).lower().strip():
            print(f"  [recap] live stats are for '{names.iloc[0]}', not "
                  f"'{expected_event_name}'; skipping")
            return None
    thru = pd.to_numeric(df["thru"], errors="coerce")
    score = pd.to_numeric(df["round"], errors="coerce")
    final = thru.eq(18) & score.notna()
    terminal = (~final & ~thru.eq(18)
                & df["position"].astype(str).str.contains("WD|DQ|CUT|W/D", case=False))
    if int(final.sum()) < MIN_SCORED_PLAYERS:
        return None
    return pd.DataFrame({
        "player_name": df["player_name"].astype(str).str.lower().str.strip(),
        f"round_{rnd}": score.where(final),
        "status": np.where(final, "final", np.where(terminal, "terminal", "live")),
    })


def grade_records(records, results_df, rnd):
    """Grade stored/killed records; returns a list of display dicts.

    ``result`` is win/loss/push, or "void" when a score is unavailable
    (terminal WD/DQ or missing player); the caller has already confirmed no
    involved player is still playing.
    """
    from grade_bets import grade_round_matchup

    graded = []
    for rec in records:
        row = {**rec, "round": rnd}
        g = grade_round_matchup(row, results_df)
        if g.get("skip_write"):
            continue  # unknowable contract (e.g. legacy unverified BetCRIS)
        p1 = str(rec.get("player_1", "")).lower().strip()
        bet_on = str(rec.get("bet_on", "")).lower().strip()
        is_p1 = bet_on == p1
        opp = rec.get("player_2", "") if is_p1 else rec.get("player_1", "")
        scores = (g.get("p1_score", ""), g.get("p2_score", ""))
        graded.append({
            "bet_on": bet_on,
            "opponent": str(opp).lower().strip(),
            "book": str(rec.get("bookmaker", "")),
            "odds": rec.get("p1_odds") if is_p1 else rec.get("p2_odds"),
            "edge": rec.get("edge_on", ""),
            "bet_score": scores[0] if is_p1 else scores[1],
            "opp_score": scores[1] if is_p1 else scores[0],
            "result": "void" if g["result"] == "no_data" else g["result"],
            "units": float(g.get("units_won") or 0.0),
            "reason": str(rec.get("kill_reason", "")),
        })
    return graded


def summarize(bets):
    """(wins, losses, pushes, units) over graded bets, ignoring no_data."""
    counts = {"win": 0, "loss": 0, "push": 0}
    units = 0.0
    for b in bets:
        if b["result"] in counts:
            counts[b["result"]] += 1
            units += b["units"]
    return counts["win"], counts["loss"], counts["push"], round(units, 2)


# ── formatting ───────────────────────────────────────────────────────────────
def _odds_text(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"+{int(number)}" if number > 0 else str(int(number))


def _score_text(value):
    """Round scores arrive to-par (small) or absolute; show to-par with a sign."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "?"
    if abs(number) <= 20:
        return "E" if number == 0 else f"{int(number):+d}"
    return str(int(number))


def _record_text(bets):
    w, l, p, u = summarize(bets)
    return f"{w}-{l}-{p}  {u:+.2f}u"


def _bet_line(b):
    letter = RESULT_LETTER.get(b["result"], "?")
    edge = b["edge"]
    try:
        edge_text = f"{float(edge):.1f}%"
    except (TypeError, ValueError):
        edge_text = "?%"
    if b["result"] in RESULT_LETTER:
        tail = (f"{_score_text(b['bet_score'])} v {_score_text(b['opp_score'])}"
                f"  {letter} {b['units']:+.2f}u")
    else:
        tail = "no score"
    return html.escape(
        f"{b['bet_on']} v {b['opponent']} | {b['book']} {_odds_text(b['odds'])} "
        f"| edge {edge_text} | {tail}"
    )


def format_recap_messages(tourney, rnd, recommended, killed, max_chars=MESSAGE_MAX_CHARS):
    """Telegram-sized HTML messages for a graded recap."""
    title = html.escape(str(tourney).replace("_", " ").title())
    lines = [f"<b>R{rnd} Recap - {title}</b>",
             "Scores are bet side v opponent; units use grade_bets flat convention.", ""]

    lines.append(f"<b>Recommended (alerted): {_record_text(recommended)}</b>")
    if recommended:
        lines += [_bet_line(b) for b in sorted(
            recommended, key=lambda b: -(_num(b["edge"])))]
    else:
        lines.append("none")
    lines.append("")

    lines.append(f"<b>Killed edges >5% (hypothetical): {_record_text(killed)}</b>")
    if killed:
        by_reason = OrderedDict()
        for b in sorted(killed, key=lambda b: (b["reason"], -_num(b["edge"]))):
            by_reason.setdefault(b["reason"] or "unknown", []).append(b)
        for reason, bets in by_reason.items():
            lines.append("")
            lines.append(f"<b>{html.escape(reason)}: {_record_text(bets)}</b>")
            lines += [_bet_line(b) for b in bets]
    else:
        lines.append("none")

    messages, chunk = [], []
    for line in lines:
        if chunk and len("\n".join(chunk + [line])) > max_chars:
            messages.append("\n".join(chunk))
            chunk = []
        chunk.append(line)
    if chunk:
        messages.append("\n".join(chunk))
    return messages


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# ── orchestration ────────────────────────────────────────────────────────────
def run_recap(event_id, tourney, rnd, *, spreadsheet=None, api_key=None,
              dry_run=False, force=False,
              expected_event_name=None, fetch_scores=None, sender=None):
    """Grade and send the recap for one completed round.

    Returns a status string: sent, dry_run, already_sent, nothing_to_recap,
    incomplete, or no_round.  Raises on Telegram/Sheets errors; callers in the
    pricing path swallow them.
    """
    import sheets_storage as ss

    if int(rnd) < 1:
        return "no_round"
    spreadsheet = spreadsheet or ss.get_spreadsheet()
    if not force and not dry_run and ss.round_recap_sent(event_id, rnd, spreadsheet):
        return "already_sent"

    import reprice_core as rc

    recommended = load_recommended(event_id, rnd, spreadsheet)
    alerted = {rc.alerted_key(r.get("player_1", ""), r.get("player_2", ""),
                              r.get("bet_on", ""), rc.selected_spread_line(r))
               for r in recommended}
    # An edge killed in one run may be alerted in a later run: count it once.
    killed = [k for k in ss.load_killed_round_matchups(event_id, rnd, spreadsheet)
              if rc.alerted_key(k.get("player_1", ""), k.get("player_2", ""),
                                k.get("bet_on", ""), rc.selected_spread_line(k))
              not in alerted]
    if not recommended and not killed:
        if not dry_run:
            ss.mark_round_recap_sent(event_id, rnd, 0, 0, spreadsheet)
        return "nothing_to_recap"

    fetch = fetch_scores or fetch_round_scores
    results_df = fetch(rnd, api_key or os.getenv("DATAGOLF_API_KEY"),
                       expected_event_name=expected_event_name)
    if results_df is None:
        return "incomplete"

    # Final only when no involved player is still playing; terminal WD/DQ and
    # players without a score are graded void ("no score").
    involved = {str(r.get(k, "")).lower().strip()
                for r in recommended + killed for k in ("player_1", "player_2")}
    live = results_df["player_name"].isin(involved) & results_df["status"].eq("live")
    if live.any():
        print(f"  [recap] R{rnd}: {int(live.sum())} involved player(s) not finished; will retry")
        return "incomplete"

    rec_bets = grade_records(recommended, results_df, rnd)
    kill_bets = grade_records(killed, results_df, rnd)
    messages = format_recap_messages(tourney, rnd, rec_bets, kill_bets)
    if dry_run:
        for message in messages:
            print(message + "\n")
        return "dry_run"

    send = sender or rc.send_telegram
    chat_id = os.getenv("TELEGRAM_ROUND_BETS_CHAT_ID", "") or None
    # The first chunk is required; once it is accepted the recap counts as sent
    # (marker written) so a later chunk failure cannot cause a full resend.
    send(messages[0], chat_id=chat_id, required=True)
    ss.mark_round_recap_sent(event_id, rnd, len(rec_bets), len(kill_bets), spreadsheet)
    for message in messages[1:]:
        try:
            send(message, chat_id=chat_id, required=True)
        except Exception as exc:
            print(f"  [recap] R{rnd}: follow-up chunk not delivered ({type(exc).__name__})")
    return "sent"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Prior-round matchup recap")
    ap.add_argument("--event-id", required=True)
    ap.add_argument("--round", type=int, required=True, dest="rnd")
    ap.add_argument("--tourney", default="", help="Display name (default: event id)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the message; no Telegram, no marker")
    ap.add_argument("--force", action="store_true", help="Ignore the send-once marker")
    args = ap.parse_args(argv)

    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    status = run_recap(args.event_id, args.tourney or args.event_id, args.rnd,
                       dry_run=args.dry_run, force=args.force)
    print(f"  [recap] R{args.rnd} event {args.event_id}: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
