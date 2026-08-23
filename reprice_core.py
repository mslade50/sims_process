"""reprice_core.py — pure matchup pricing / edge / dedup / alert helpers for the
cache-free repricer (reprice.py).

These functions MIRROR round_sim.py's matchup pipeline exactly so the lightweight
repricer prices identically to the full sim. The one new piece is price_from_h2h:
instead of computing P(a beats b) from per-sim draws (round_sim.price_matchups),
it looks the probabilities up in the committed round-H2H fair table — which was
built from those same joint draws, so it reconstructs the four my_odds_* columns
EXACTLY (P(a<b) and P(tie) are sufficient; everything else is algebra).

This module deliberately has NO heavy / module-load side effects (no Google Sheet
read, no Rust kernel, no Excel) so importing it is cheap and can't fail the way
importing round_sim can.

KEEP IN SYNC with round_sim.py:
  - SHARP_BOOKS / HALF_SHOT_ADJ            (round_sim.py:71-72)
  - american_to_implied / implied_to_american (round_sim.py:1910-1925)
  - calculate_edges                        (round_sim.py:2339)
  - build_matchup_outputs                  (round_sim.py:2416)
  - _dedup_round_matchups                  (round_sim.py:3707)
  - Telegram message formatting            (round_sim.py:1928)

The cache-free reprice transport is intentionally stricter than round_sim's
best-effort sender: a bet row cannot be stored until Telegram accepts its alert.
"""

import html
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

# ── constants (mirror round_sim.py:71-72) ──────────────────────────────────────
SHARP_BOOKS = ["pinnacle", "betonline", "betcris"]
HALF_SHOT_ADJ = {"betonline": 25, "betcris": 30}
# Only sharp books generate Telegram matchup alerts (mirror round_sim:3942).
TELEGRAM_BOOKS = {"betonline", "pinnacle", "betcris"}
TELEGRAM_MESSAGE_MAX_CHARS = 3500


class TelegramDeliveryError(RuntimeError):
    """A required Telegram message was not accepted by the Telegram API."""


# ── odds helpers (mirror round_sim.py:1910-1925) ───────────────────────────────
def american_to_implied(odds):
    """American odds → implied probability (0–1)."""
    if pd.isna(odds) or odds == 0:
        return None
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def implied_to_american(prob):
    """Implied probability (0–1) → American odds (int)."""
    if prob is None or pd.isna(prob) or prob <= 0 or prob >= 1:
        return None
    if prob >= 0.5:
        return int(round(-100 * prob / (1 - prob)))
    return int(round(100 * (1 - prob) / prob))


def _norm_name(name, repl):
    """lower + strip + name_replacements — both sides of the join must use this."""
    n = str(name).strip().lower()
    return repl.get(n, n)


# ── round-H2H table → matchup probabilities ────────────────────────────────────
def build_h2h_lookup(h2h_df):
    """{(player_a, player_b): (P(a<b), P(tie))} keyed in the table's canonical
    (lexicographic) player order. Also returns the set of known players."""
    lookup = {}
    known = set()
    for a, b, p_lt, p_tie in zip(
        h2h_df["player_a"], h2h_df["player_b"], h2h_df["p_a_lt_b"], h2h_df["p_tie"]
    ):
        a, b = str(a), str(b)
        lookup[(a, b)] = (float(p_lt), float(p_tie))
        known.add(a)
        known.add(b)
    return lookup, known


def price_from_h2h(matchup_df, lookup, known, repl=None):
    """Attach fair win probabilities to each matchup row from the H2H table.

    Drop-in replacement for round_sim.price_matchups(matchup_df, sim_dict): produces
    the identical my_odds_p1 / my_odds_p2 (ties push) and my_odds_p1_tl /
    my_odds_p2_tl (ties loss) columns, and tracks name mismatches the same way.
    """
    repl = repl or {}
    cols = {"fair_p1": [], "fair_p2": [], "tl_p1": [], "tl_p2": []}
    name_mismatches = defaultdict(set)

    for _, row in matchup_df.iterrows():
        p1 = _norm_name(row["Player 1"], repl)
        p2 = _norm_name(row["Player 2"], repl)
        book = row.get("Bookmaker", "unknown")

        if p1 not in known or p2 not in known:
            if p1 not in known:
                name_mismatches[p1].add(book)
            if p2 not in known:
                name_mismatches[p2].add(book)
            for k in cols:
                cols[k].append(None)
            continue

        # Look up in canonical (sorted) orientation, then orient to p1.
        a, b = (p1, p2) if p1 <= p2 else (p2, p1)
        hit = lookup.get((a, b))
        if hit is None:
            for k in cols:
                cols[k].append(None)
            continue
        p_a_lt_b, p_tie = hit
        p1_lt = p_a_lt_b if p1 <= p2 else (1.0 - p_a_lt_b - p_tie)   # P(p1 < p2)
        p2_lt = 1.0 - p1_lt - p_tie                                  # P(p2 < p1)
        non_tie = p1_lt + p2_lt

        cols["fair_p1"].append(p1_lt / non_tie if non_tie else 0.5)
        cols["fair_p2"].append(p2_lt / non_tie if non_tie else 0.5)
        cols["tl_p1"].append(p1_lt)
        cols["tl_p2"].append(p2_lt)

    matchup_df["my_odds_p1"] = cols["fair_p1"]
    matchup_df["my_odds_p2"] = cols["fair_p2"]
    matchup_df["my_odds_p1_tl"] = cols["tl_p1"]
    matchup_df["my_odds_p2_tl"] = cols["tl_p2"]

    if name_mismatches:
        print(f"  Warning: {len(name_mismatches)} scraped players not found in fairs")
        matchup_df.attrs["name_mismatches"] = dict(name_mismatches)

    return matchup_df


# ── edges (verbatim mirror of round_sim.calculate_edges) ───────────────────────
def calculate_edges(df):
    """Calculate edges, fair odds, half-shot spreads for all matchup rows."""
    df = df.dropna(subset=["my_odds_p1", "my_odds_p2"]).copy()

    df["p1_dec"] = np.where(
        df["P1 Odds"] > 0,
        df["P1 Odds"] / 100 + 1,
        100 / df["P1 Odds"].abs() + 1,
    )
    df["p2_dec"] = np.where(
        df["P2 Odds"] > 0,
        df["P2 Odds"] / 100 + 1,
        100 / df["P2 Odds"].abs() + 1,
    )

    use_tl = df["Ties"] == "separate bet offered"
    prob_p1 = np.where(use_tl, df["my_odds_p1_tl"], df["my_odds_p1"])
    prob_p2 = np.where(use_tl, df["my_odds_p2_tl"], df["my_odds_p2"])

    df["edge_p1"] = (prob_p1 * (df["p1_dec"] - 1) - (1 - prob_p1)) * 100
    df["edge_p2"] = (prob_p2 * (df["p2_dec"] - 1) - (1 - prob_p2)) * 100

    df["Fair_p1"] = df["my_odds_p1"].apply(
        lambda p: implied_to_american(p) if pd.notna(p) else None
    )
    df["Fair_p2"] = df["my_odds_p2"].apply(
        lambda p: implied_to_american(p) if pd.notna(p) else None
    )

    df["p1_implied"] = df["P1 Odds"].apply(
        lambda o: round(american_to_implied(o) * 100, 1) if pd.notna(o) else None
    )
    df["p2_implied"] = df["P2 Odds"].apply(
        lambda o: round(american_to_implied(o) * 100, 1) if pd.notna(o) else None
    )

    df["half_shot_p1"] = (df["my_odds_p1"] - df["my_odds_p1_tl"]) * 400
    df["half_shot_p2"] = (df["my_odds_p2"] - df["my_odds_p2_tl"]) * 400

    df["p1_pushwins"] = (1 - df["my_odds_p2_tl"]) * 100
    df["p2_pushwins"] = (1 - df["my_odds_p1_tl"]) * 100
    df["p1_nopush"] = df["my_odds_p1_tl"] * 100
    df["p2_nopush"] = df["my_odds_p2_tl"] * 100

    for book, adj in HALF_SHOT_ADJ.items():
        mask = df["Bookmaker"].str.lower() == book
        if not mask.any():
            continue
        for side, odds_col in [("p1", "P1 Odds"), ("p2", "P2 Odds")]:
            pw_imp = (df.loc[mask, odds_col] - adj).apply(
                lambda o: round(american_to_implied(o) * 100, 1) if pd.notna(o) else None
            )
            np_imp = (df.loc[mask, odds_col] + adj).apply(
                lambda o: round(american_to_implied(o) * 100, 1) if pd.notna(o) else None
            )
            df.loc[mask, f"{side}_pushwins_imp"] = pw_imp
            df.loc[mask, f"{side}_nopush_imp"] = np_imp
            df.loc[mask, f"{side}_+0.5"] = df.loc[mask, f"{side}_pushwins"] - pw_imp
            df.loc[mask, f"{side}_-0.5"] = df.loc[mask, f"{side}_nopush"] - np_imp

    return df


# ── combined/sharp split (verbatim mirror of round_sim.build_matchup_outputs) ───
def build_matchup_outputs(df, sim_round, pred_lookup, sample_lookup, wx_lookup=None):
    """Filter, annotate, and split matchup DataFrame into combined + sharp."""
    df["p1_pred"] = df["Player 1"].map(pred_lookup)
    df["p2_pred"] = df["Player 2"].map(pred_lookup)
    df["Sample_P1"] = df["Player 1"].map(sample_lookup)
    df["Sample_P2"] = df["Player 2"].map(sample_lookup)
    df["Round"] = f"r{sim_round}"

    df["edge_on"] = df[["edge_p1", "edge_p2"]].max(axis=1).round(1)
    df["bet_on"] = df.apply(
        lambda r: r["Player 1"] if r["edge_p1"] > r["edge_p2"] else r["Player 2"],
        axis=1,
    )
    df["pred_on"] = df.apply(
        lambda r: r["p1_pred"] if r["edge_p1"] > r["edge_p2"] else r["p2_pred"],
        axis=1,
    )
    df["pred_against"] = df.apply(
        lambda r: r["p2_pred"] if r["edge_p1"] > r["edge_p2"] else r["p1_pred"],
        axis=1,
    )
    df["sample_on"] = df.apply(
        lambda r: r["Sample_P1"] if r["edge_p1"] > r["edge_p2"] else r["Sample_P2"],
        axis=1,
    )

    if wx_lookup:
        df["wx_on"] = df["bet_on"].map(wx_lookup).fillna(0)
        df["wx_against"] = df.apply(
            lambda r: wx_lookup.get(
                r["Player 2"] if r["bet_on"] == r["Player 1"] else r["Player 1"], 0
            ),
            axis=1,
        )
        df["wx_diff"] = df["wx_on"] - df["wx_against"]

    combined = df[df["edge_on"] > 3].copy()
    combined = combined[combined["sample_on"].fillna(0) >= 20]
    combined = combined[
        ((combined["pred_on"] > 0) & (combined["edge_on"] > 7))
        | (combined["pred_on"] > 1)
    ]
    combined = combined[
        ~((combined["edge_on"] < 5) & (combined["pred_on"] < 1))
    ]

    sharp = combined[combined["Bookmaker"].str.lower().isin(SHARP_BOOKS)].copy()
    # Every sportsbook quote is a separately actionable bet.  The historical
    # pair-only dedup silently dropped (for example) a qualifying BetCris quote
    # whenever the same golfers also had a slightly larger BetOnline edge.  Only
    # collapse literal feed duplicates, including reversed player ordering.
    sharp = retain_unique_actionable_quotes(sharp, player_count=2)

    for out in [combined, sharp]:
        out["p1_pred"] = out["p1_pred"].round(2)
        out["p2_pred"] = out["p2_pred"].round(2)
        out["edge_p1"] = out["edge_p1"].round(1)
        out["edge_p2"] = out["edge_p2"].round(1)

    display_cols = [
        "Player 1", "Player 2", "Round", "Bookmaker", "Ties",
        "P1 Odds", "P2 Odds", "Fair_p1", "Fair_p2",
        "edge_p1", "edge_p2", "edge_on", "bet_on",
        "p1_pred", "p2_pred", "pred_on", "pred_against",
        "Sample_P1", "Sample_P2", "sample_on",
        "half_shot_p1", "half_shot_p2",
    ]
    for col in ["wx_diff"]:
        if col in combined.columns:
            display_cols.append(col)
    for col in ["p1_+0.5", "p2_+0.5", "p1_-0.5", "p2_-0.5"]:
        if col in combined.columns:
            display_cols.append(col)

    combined = combined[[c for c in display_cols if c in combined.columns]]
    sharp = sharp[[c for c in display_cols if c in sharp.columns]]

    print(f"  Combined matchups: {len(combined)} rows")
    print(f"  Sharp filtered:    {len(sharp)} rows")
    return combined, sharp


# ── dedup vs Sheets (mirror of round_sim._dedup_round_matchups) ─────────────────
def alerted_key(p1, p2, bet_on):
    """Alert-level identity of a matchup edge: order-insensitive pairing + which
    player the bet is on. A price/edge-size change on the same bet maps to the
    same key; the edge flipping to the other player maps to a new one."""
    a, b = sorted([str(p1).lower().strip(), str(p2).lower().strip()])
    return (a, b, str(bet_on).lower().strip())


def _canonical_mu_key(p1, p2, book, o1, o2, bet_on):
    """Order-insensitive bet identity with each price attached to its player.

    The selected side is deliberately part of the storage identity.  If the
    fair crosses at unchanged market odds, betting the other player is a new bet
    and must survive both Sheet dedup and Telegram alert dedup.
    """
    a = (str(p1).lower().strip(), _canonical_quote_price(o1))
    b = (str(p2).lower().strip(), _canonical_quote_price(o2))
    lo, hi = (a, b) if a[0] <= b[0] else (b, a)
    return (
        lo[0], hi[0], str(book).lower().strip(), lo[1], hi[1],
        str(bet_on).lower().strip(),
    )


def _canonical_quote_price(value):
    """Stable text identity for an American-odds value."""
    try:
        number = float(value)
        if not np.isfinite(number):
            return ""
        if number == int(number):
            return str(int(number))
        return format(number, ".6f").rstrip("0").rstrip(".")
    except (TypeError, ValueError, OverflowError):
        return str(value).strip().lower()


def _canonical_actionable_quote_key(row, player_count):
    """Canonical identity of one actionable 2-ball or 3-ball quote.

    Player order from an upstream feed is irrelevant, but each price remains
    attached to its player.  Book, settlement rule, and selected side are part
    of the identity because any of them makes the row a distinct wager.
    """
    quoted_players = tuple(sorted(
        (
            str(row.get(f"Player {idx}", "")).lower().strip(),
            _canonical_quote_price(row.get(f"P{idx} Odds", "")),
        )
        for idx in range(1, player_count + 1)
    ))
    return (
        str(row.get("Bookmaker", "")).lower().strip(),
        str(row.get("Ties", "")).lower().strip(),
        str(row.get("bet_on", "")).lower().strip(),
        quoted_players,
    )


def retain_unique_actionable_quotes(frame, *, player_count):
    """Keep every distinct book/side/price while collapsing true feed repeats.

    When duplicate representations disagree slightly on calculated edge, the
    highest-edge copy wins deterministically.  Stable sorting preserves source
    order for exact ties.
    """
    if frame is None or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    if player_count not in (2, 3):
        raise ValueError("player_count must be 2 or 3")

    unique = frame.copy()
    unique["_actionable_quote_key"] = unique.apply(
        lambda row: _canonical_actionable_quote_key(row, player_count), axis=1
    )
    unique = (
        unique.sort_values("edge_on", ascending=False, kind="mergesort")
        .drop_duplicates("_actionable_quote_key", keep="first")
        .drop(columns="_actionable_quote_key")
    )
    return unique


def dedup_round_matchups(combined, spreadsheet, event_id, sim_round):
    """Split `combined` against the Round Matchups sheet for this event+round.

    Returns (new_rows, seen_alert_keys):
      new_rows        — rows not already stored. The store key includes the odds
                        and selected side (players, bookmaker, prices, bet_on), so
                        either a price move or an edge flip still stores.
      seen_alert_keys — alerted_key() of every already-stored row, so the Telegram
                        layer can suppress edges previously surfaced to the user
                        and ping only pairings (or flipped sides) that are new.
    """
    if combined is None or combined.empty:
        return combined, set()

    from sheets_storage import (
        TAB_ROUND_MU,
        ROUND_MU_HEADERS,
        _get_or_create_tab,
        is_excluded_or_invalid_result,
    )
    ws = _get_or_create_tab(spreadsheet, TAB_ROUND_MU, ROUND_MU_HEADERS)
    existing = ws.get_all_records()

    existing_keys = set()
    seen_alert_keys = set()
    for row in existing:
        # Invalidated bad-run rows remain visible for audit, but they are not a
        # historical bet/alert and therefore cannot suppress a corrected retry.
        if is_excluded_or_invalid_result(row.get("result", "")):
            continue
        if str(row.get("event_id", "")) == str(event_id) and str(row.get("round", "")) == str(sim_round):
            existing_keys.add(_canonical_mu_key(
                row.get("player_1", ""), row.get("player_2", ""),
                row.get("bookmaker", ""), row.get("p1_odds", ""),
                row.get("p2_odds", ""), row.get("bet_on", ""),
            ))
            # Only a row from a Telegram-eligible book can prove this edge was
            # previously surfaced.  A soft-book row may be stored independently
            # while a sharp-book alert is failing; counting it here would suppress
            # the sharp alert on the workflow retry.
            if str(row.get("bookmaker", "")).lower().strip() in TELEGRAM_BOOKS:
                seen_alert_keys.add(alerted_key(
                    row.get("player_1", ""), row.get("player_2", ""), row.get("bet_on", "")))

    if not existing_keys:
        return combined, seen_alert_keys

    mask = []
    for _, r in combined.iterrows():
        key = _canonical_mu_key(
            r.get("Player 1", ""), r.get("Player 2", ""),
            r.get("Bookmaker", ""), r.get("P1 Odds", ""), r.get("P2 Odds", ""),
            r.get("bet_on", ""),
        )
        mask.append(key not in existing_keys)

    new_rows = combined[mask].copy()
    print(f"  [reprice] Matchups: {len(combined)} total, {len(new_rows)} new "
          f"(deduped {len(combined) - len(new_rows)})")
    return new_rows, seen_alert_keys


# ── Telegram (strict for bet delivery; best-effort for diagnostics) ───────────
def _telegram_failure(reason, required):
    message = f"Telegram delivery failed: {reason}"
    if required:
        raise TelegramDeliveryError(message)
    print(f"  Warning: {message}")
    return False


def send_telegram(text, chat_id=None, *, required=False):
    """Send one Telegram message and return whether Telegram accepted it.

    A successful HTTP request is not sufficient: Telegram can return HTTP 200
    with ``{"ok": false}``.  Required bet alerts therefore validate credentials,
    the HTTP status, JSON response, and API ``ok`` flag, then raise on any failure.
    Diagnostic notices retain their historical best-effort behavior by leaving
    ``required=False``.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return _telegram_failure("missing bot token or chat ID", required)
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        # Do not include the requests exception in logs: its URL contains the bot
        # token, which must never be exposed in a workflow traceback.
        return _telegram_failure("request error", required)

    status = getattr(response, "status_code", None)
    if not isinstance(status, int) or not 200 <= status < 300:
        return _telegram_failure(f"HTTP {status if status is not None else 'unknown'}", required)

    try:
        payload = response.json()
    except Exception:
        return _telegram_failure("invalid JSON response", required)
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        description = payload.get("description") if isinstance(payload, dict) else None
        reason = "Telegram API returned ok=false"
        if description:
            reason += f" ({description})"
        return _telegram_failure(reason, required)
    return True


def _fmt_odds(v):
    if isinstance(v, (int, float)) and not pd.isna(v):
        return f"+{int(v)}" if v > 0 else str(int(v))
    return str(v)


def partition_matchup_alert_rows(new_mu, seen_alert_keys=None):
    """Split new rows into (must_alert_before_store, may_store_without_alert).

    Sharp-book rows need a Telegram delivery only when their pairing+side has not
    already been alerted.  Soft books and sharp price moves for an already-alerted
    side are safe to store without another notification.
    """
    if new_mu is None or new_mu.empty:
        return new_mu, new_mu

    seen = seen_alert_keys or set()
    needs_alert = []
    for _, row in new_mu.iterrows():
        book = str(row.get("Bookmaker", "")).lower().strip()
        key = alerted_key(
            row.get("Player 1", ""), row.get("Player 2", ""), row.get("bet_on", "")
        )
        needs_alert.append(book in TELEGRAM_BOOKS and key not in seen)

    mask = np.asarray(needs_alert, dtype=bool)
    return new_mu.iloc[mask].copy(), new_mu.iloc[~mask].copy()


def _matchup_alert_messages(tg, sim_round, tourney_name, max_chars=TELEGRAM_MESSAGE_MAX_CHARS):
    """Build conservative Telegram-sized HTML messages for matchup rows."""
    title = html.escape(str(tourney_name).replace("_", " ").title())
    header = f"<b>R{sim_round} Reprice — {title}</b>\n\n"
    entries = []
    for _, r in tg.iterrows():
        bet = html.escape(str(r.get("bet_on", "?")))
        edge = html.escape(str(r.get("edge_on", "?")))
        book = html.escape(str(r.get("Bookmaker", "?")))
        is_p1 = str(r.get("bet_on", "")).lower() == str(r.get("Player 1", "")).lower()
        opp = html.escape(str(r.get("Player 2", "") if is_p1 else r.get("Player 1", "")))
        mkt_odds = r.get("P1 Odds", "?") if is_p1 else r.get("P2 Odds", "?")
        fair_odds = r.get("Fair_p1", "?") if is_p1 else r.get("Fair_p2", "?")
        entries.append(
            f"  {bet} vs {opp}\n"
            f"    {book} {html.escape(_fmt_odds(mkt_odds))} "
            f"(fair {html.escape(_fmt_odds(fair_odds))}) edge={edge}%"
        )

    messages = []
    chunk = []
    for entry in entries:
        label = f"<b>New Matchups ({len(tg)}):</b>\n"
        candidate = header + label + "\n".join(chunk + [entry])
        if chunk and len(candidate) > max_chars:
            messages.append(header + label + "\n".join(chunk))
            chunk = [entry]
        else:
            chunk.append(entry)
    if chunk:
        messages.append(header + f"<b>New Matchups ({len(tg)}):</b>\n" + "\n".join(chunk))
    return messages


def send_matchup_alert(new_mu, sim_round, tourney_name, seen_alert_keys=None):
    """Telegram alert for newly-priced round matchups (sharp books only).

    Sends only when there is something new (the cache-free repricer fires on every
    scrape, so a 'nothing new' ping every time would be noise). `seen_alert_keys`
    (from dedup_round_matchups) suppresses edges the user has already been shown:
    a re-store caused by a price/edge-size move on a previously-seen bet doesn't
    re-alert — only a new pairing, or the edge flipping to the other player, does.
    Returns the number of rows delivered and raises ``TelegramDeliveryError`` if
    any required message is not accepted.
    """
    tg, _ = partition_matchup_alert_rows(new_mu, seen_alert_keys=seen_alert_keys)
    if new_mu is None or new_mu.empty:
        return 0
    n_dropped = len(new_mu[new_mu["Bookmaker"].str.lower().isin(TELEGRAM_BOOKS)]) - len(tg)
    if n_dropped:
        print(f"  [reprice] Alert: suppressed {n_dropped} previously-seen edge(s) "
              f"(price moved, same bet)")
    if tg.empty:
        return 0

    round_bets_chat_id = os.getenv("TELEGRAM_ROUND_BETS_CHAT_ID", "")
    messages = _matchup_alert_messages(tg, sim_round, tourney_name)
    for message in messages:
        send_telegram(
            message,
            chat_id=round_bets_chat_id or None,
            required=True,
        )
    print(f"  [reprice] Telegram delivered {len(tg)} new sharp edge(s) "
          f"in {len(messages)} message(s).")
    return len(tg)
