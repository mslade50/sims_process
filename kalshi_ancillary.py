"""kalshi_ancillary.py — price ANCILLARY Kalshi golf markets from round_sim outputs.

The outright pipeline (round_sim.price_kalshi_outrights) already covers winner +
top 5/10/20. This module covers everything else Kalshi offers for a single PGA
tournament that our strokes-gained sim can price:

  LIVE round markets (need end-of-round standings from the cascade):
    KXPGAR2LEAD / KXPGAR3LEAD     — leader at the end of round N (co-leaders = Yes)
    KXPGAR2TOP10 / KXPGAR3TOP10   — top 10 (incl. ties) after round N
    KXPGAR3TOP5                   — top 5 (incl. ties) after round 3
  Event markets (from final_scores):
    KXPGAPLAYOFF                  — will there be a playoff (>=2 tied at the low)
    KXPGAWINNINGSCORE             — winning 72-hole total threshold
    KXPGASTROKEMARGIN             — victory margin threshold
  Group / pairwise markets (from the single-round score sim, when offered):
    KXPGAH2H                      — round head-to-head
    KXPGA3BALL / KXPGA5BALL       — lowest in a 3-/5-ball group
    KXPGAGOLFERSCORE              — round strokes over/under

NOT priceable here (need a hole-by-hole model we don't have): KXPGAEAGLE,
KXPGAROUNDBIRDIES, KXPGAHOLEINONE, KXPGAHOLE.

Inclusion rules (a market only makes the table if at least one passes):
  TAKER : we can fill TAKER_TARGET (300) contracts walking from the best ask,
          and our fair STILL beats that VWAP (incl. Kalshi taker fee) by >= edge.
  MAKER : the market is tight (yes spread < MAKER_MAX_SPREAD, 4c) AND our fair
          beats (ask - 1c) with no fee by >= edge (i.e. we'd profit posting there).

Everything else is too thin to matter and is dropped.
"""
import re
import time
from collections import Counter, defaultdict

import httpx
import numpy as np
import pandas as pd

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"

TAKER_TARGET = 300         # contracts we want filled as a taker
MAKER_MAX_SPREAD = 0.04    # only consider posting if the yes spread is < 4c
MAKER_IMPROVE = 0.01       # post at ask - 1c
DEFAULT_EDGE = 2.0         # min edge in percentage points (matches outright threshold)
PRICE_EPS = 1e-4

# Kalshi position sizing: 0.4 fractional Kelly on a $50k bankroll, 1 unit = $500.
KALSHI_BANKROLL = 50_000.0
KALSHI_KELLY = 0.4
KALSHI_UNIT = 500.0
KALSHI_MAX_RISK_FRAC = 0.05   # cap any single bet's risk at 5% of bankroll ($2,500 = 5u)


def kelly_contracts(fair_prob, cost):
    """0.4-Kelly stake on a $50k bankroll, expressed as Kalshi contracts.

    fair_prob = our probability the position resolves in our favor.
    cost       = price paid per contract ($0-1) — also the per-contract risk.
    Returns (contracts:int, dollars:float, units:float) where units = dollars/$500.
    """
    try:
        c = float(cost)
        p = float(fair_prob)
    except (TypeError, ValueError):
        return 0, 0.0, 0.0
    if not (0.0 < c < 1.0) or not (0.0 < p <= 1.0):
        return 0, 0.0, 0.0
    b = (1.0 - c) / c                       # net decimal odds (payout/stake)
    f = (b * p - (1.0 - p)) / b if b > 0 else 0.0
    f = max(f, 0.0)
    dollars = KALSHI_BANKROLL * KALSHI_KELLY * f
    dollars = min(dollars, KALSHI_MAX_RISK_FRAC * KALSHI_BANKROLL)  # 5% risk cap
    contracts = int(round(dollars / c))
    return contracts, dollars, dollars / KALSHI_UNIT


def _taker_fee(price):
    """Kalshi taker fee: 7% of price*(1-price). Mirrors price_kalshi_outrights."""
    if price <= 0 or price >= 1:
        return 0.0
    return 0.07 * price * (1 - price)


# ──────────────────────────────────────────────────────────────────────────────
# Fair-probability builders (pure functions of the sim arrays)
# ──────────────────────────────────────────────────────────────────────────────
def leader_topn_fairs(standings, player_names, made_cut=None):
    """Leader (co-leader) + top 5/10/20 probabilities for an (n, sims) standings
    matrix, via the integer-exact Rust aggregator. `made_cut` (n, sims bool) marks
    missed-cut sims as ineligible (pinned above any real score) for R3 standings.

    Returns DataFrame[player_name, leader, top5, top10, top20].

    Leader uses the DEAD-HEAT-split rank-1 prob (`prob_u`): Kalshi pays a tied YES
    leader $1/N (rules_secondary), so the fair YES = E[payout] = sum_N P(N-way tie
    for the lead)*(1/N). This sums to exactly 1.0 across the field. Top-N uses the
    NO-dead-heat count ("including ties" -> a tied top-N finish pays full $1).
    """
    import sims_kernel as sk
    s = np.ascontiguousarray(np.asarray(standings, dtype=np.int64))
    if made_cut is not None:
        s = s.copy()
        s[~np.asarray(made_cut, dtype=bool)] = 10_000  # excluded from the ranking
    prob_u, _prob_ndh, _top_dh = sk.aggregate(s)        # prob_u[:,0] = dead-heat leader
    _praw, _tdh, top_nodh = sk.aggregate_round(s)        # top_nodh = top-N incl. ties
    prob_u = np.ascontiguousarray(prob_u)
    top_nodh = np.ascontiguousarray(top_nodh)
    return pd.DataFrame({
        "player_name": list(player_names),
        "leader": prob_u[:, 0],            # dead-heat $1/N expected payout
        "top5": top_nodh[:, 0],
        "top10": top_nodh[:, 1],
        "top20": top_nodh[:, 2],
    })


def playoff_prob(final_scores):
    """P(>=2 players tied at the lowest 72-hole total) = P(a playoff)."""
    fs = np.asarray(final_scores)
    mins = fs.min(axis=0)
    n_at_min = (fs == mins[None, :]).sum(axis=0)
    return float((n_at_min >= 2).mean())


def winning_score_cdf(final_scores):
    """Distribution of the winning (minimum) 72-hole total.
    Returns (values_sorted, P(win_score <= v)) plus the raw winners array."""
    fs = np.asarray(final_scores)
    winners = fs.min(axis=0)
    vals, counts = np.unique(winners, return_counts=True)
    cdf = np.cumsum(counts) / winners.size
    return vals, cdf, winners


def stroke_margin_pmf(final_scores):
    """PMF of the victory margin (2nd-best minus best) per sim."""
    fs = np.asarray(final_scores)
    part = np.partition(fs, 1, axis=0)
    margin = part[1, :] - part[0, :]
    vals, counts = np.unique(margin, return_counts=True)
    return {int(v): c / margin.size for v, c in zip(vals, counts)}


def h2h_prob(sim_dict, p1, p2):
    """P(p1 beats p2) excluding ties (push), from single-round score arrays."""
    s1, s2 = sim_dict[p1], sim_dict[p2]
    w1 = int((s1 < s2).sum())
    w2 = int((s1 > s2).sum())
    non_tie = w1 + w2
    return (w1 / non_tie) if non_tie else 0.5


def nball_probs(sim_dict, players):
    """P(each player has the lowest score in the group), ties split evenly."""
    arrs = [np.asarray(sim_dict[p]) for p in players]
    M = np.vstack(arrs)
    mins = M.min(axis=0)
    is_min = (M == mins[None, :])
    tie_ct = is_min.sum(axis=0)
    probs = (is_min / tie_ct).mean(axis=1)
    return dict(zip(players, probs))


# ──────────────────────────────────────────────────────────────────────────────
# Kalshi API helpers
# ──────────────────────────────────────────────────────────────────────────────
def _make_client():
    return httpx.Client(timeout=15.0, headers={"Accept": "application/json"})


def _get_markets(client, series_ticker):
    out, cursor = [], None
    while True:
        params = {"limit": 200, "status": "open", "series_ticker": series_ticker}
        if cursor:
            params["cursor"] = cursor
        resp = client.get(f"{KALSHI_API}/markets", params=params)
        resp.raise_for_status()
        data = resp.json()
        mkts = data.get("markets", [])
        out.extend(mkts)
        cursor = data.get("cursor")
        if not cursor or len(mkts) < 200:
            break
    return out


def _get_orderbook(client, ticker):
    resp = client.get(f"{KALSHI_API}/markets/{ticker}/orderbook")
    resp.raise_for_status()
    data = resp.json()
    ob = data.get("orderbook", data.get("orderbook_fp", data))

    def parse(raw):
        out = []
        for item in (raw or []):
            if isinstance(item, list) and len(item) == 2:
                out.append((float(item[0]), int(float(item[1]))))
        return out

    return {
        "yes": parse(ob.get("yes_dollars") or ob.get("yes", [])),
        "no": parse(ob.get("no_dollars") or ob.get("no", [])),
    }


def _market_quote(market):
    """Best yes bid/ask (dollars) from the market summary fields only — no network.
    Used as a cheap liquidity pre-filter so we skip the orderbook fetch on the many
    quote-less (thin) markets."""
    bid = float(market.get("yes_bid_dollars") or 0) or (float(market.get("yes_bid", 0) or 0) / 100.0)
    ask = float(market.get("yes_ask_dollars") or 0) or (float(market.get("yes_ask", 0) or 0) / 100.0)
    return bid, ask


def _best_bid_ask(market, ob):
    """Best yes bid/ask in dollars. Prefer the market quote; fall back to the book.
    yes bid = highest resting yes order; yes ask = 1 - highest resting no order."""
    bid, ask = _market_quote(market)
    if bid <= 0 and ob["yes"]:
        bid = max(p for p, _ in ob["yes"])
    if ask <= 0 and ob["no"]:
        ask = 1.0 - max(p for p, _ in ob["no"])
    return bid, ask


def _walk_target(levels, target):
    """Walk same-direction yes-equivalent levels [(yes_price, qty)] to fill `target`.
    Returns (vwap_incl_fee, filled) or None if nothing fillable."""
    fill = sorted(levels)
    if not fill:
        return None
    filled, cost, fee_sum = 0, 0.0, 0.0
    for price, qty in fill:
        take = min(qty, target - filled)
        if take <= 0:
            break
        filled += take
        cost += take * price
        fee_sum += take * _taker_fee(price)
        if filled >= target:
            break
    if filled == 0:
        return None
    return (cost + fee_sum) / filled, filled


def _match_tournament(markets, tourney):
    """Keep only markets for the configured tourney, so simultaneously-open events
    don't bleed in. Kalshi tags golf inconsistently (winner/leader use RBBCAN26 but
    3-balls use RBCO26 and their TITLE omits the tournament name), so match against
    title + rules_primary + event_ticker, not the title alone."""
    GENERIC = {"the", "a", "an", "of", "at", "in", "tournament", "championship",
               "open", "classic", "invitational", "cup", "pga", "tour", "round",
               "end", "will", "lead", "finish", "top", "playoff", "ball", "matchup"}
    words = {w for w in re.split(r"[\s_]+", tourney.lower()) if w}
    distinct = words - GENERIC

    def ok(m):
        blob = " ".join([m.get("title", ""), m.get("rules_primary", "") or "",
                         m.get("event_ticker", "")]).lower()
        blob_ns = re.sub(r"\s+", "", blob)
        if distinct:
            return any(w in blob or w in blob_ns for w in distinct)
        return all(w in blob or w in blob_ns for w in words)

    return [m for m in markets if ok(m)]


def _norm_name(s, name_replacements):
    """'Ben Kohles' -> 'kohles, ben' (sim player_name convention), then remap."""
    x = (s or "").strip().lower()
    if "," not in x:
        parts = x.rsplit(" ", 1)
        if len(parts) == 2:
            x = f"{parts[1]}, {parts[0]}"
    return name_replacements.get(x, x)


# ──────────────────────────────────────────────────────────────────────────────
# Edge engine
# ──────────────────────────────────────────────────────────────────────────────
def _edge_rows(base, fair_yes, bid, ask, ob, edge_threshold):
    """Apply the taker (fill 300 @ ask+fee) and maker (post ask-1c, spread<4c)
    rules to one market with a known fair YES probability. Returns 0..N rows."""
    rows = []
    fair_no = 1.0 - fair_yes
    spread = ask - bid if (bid > 0 and ask > 0) else None

    # ---- TAKER: fill 300 contracts and still beat our fair ----
    # Buy YES by walking the NO book (each no level -> yes price 1-p).
    yes_fill = _walk_target([(1.0 - p, q) for p, q in ob["no"]], TAKER_TARGET)
    if yes_fill:
        eff, filled = yes_fill
        if filled >= TAKER_TARGET and 0 < eff < 1:
            edge = (fair_yes - eff) * 100
            if edge >= edge_threshold:
                rows.append({**base, "side": "yes", "pricing": "taker",
                             "fair_prob": fair_yes, "cost": eff,
                             "edge_pp": round(edge, 2), "fill": filled,
                             "yes_bid": bid, "yes_ask": ask})
    # Buy NO by walking the YES book.
    no_fill = _walk_target([(1.0 - p, q) for p, q in ob["yes"]], TAKER_TARGET)
    if no_fill:
        eff, filled = no_fill
        if filled >= TAKER_TARGET and 0 < eff < 1:
            edge = (fair_no - eff) * 100
            if edge >= edge_threshold:
                rows.append({**base, "side": "no", "pricing": "taker",
                             "fair_prob": fair_no, "cost": eff,
                             "edge_pp": round(edge, 2), "fill": filled,
                             "yes_bid": bid, "yes_ask": ask})

    # ---- MAKER: tight market, post at ask-1c (no fee) ----
    if spread is not None and 0 < spread < MAKER_MAX_SPREAD:
        yes_cost = ask - MAKER_IMPROVE
        if 0 < yes_cost < 1:
            edge = (fair_yes - yes_cost) * 100
            if edge >= edge_threshold:
                rows.append({**base, "side": "yes", "pricing": "maker",
                             "fair_prob": fair_yes, "cost": yes_cost,
                             "edge_pp": round(edge, 2), "fill": np.nan,
                             "yes_bid": bid, "yes_ask": ask})
        no_cost = (1.0 - bid) - MAKER_IMPROVE   # post NO at its ask - 1c
        if 0 < no_cost < 1:
            edge = (fair_no - no_cost) * 100
            if edge >= edge_threshold:
                rows.append({**base, "side": "no", "pricing": "maker",
                             "fair_prob": fair_no, "cost": no_cost,
                             "edge_pp": round(edge, 2), "fill": np.nan,
                             "yes_bid": bid, "yes_ask": ask})
    return rows


# series_ticker -> (market_type label, fair source key, fair column)
PLAYER_SERIES = {
    "KXPGAR2LEAD":  ("r2_leader", "standings_r2", "leader"),
    "KXPGAR3LEAD":  ("r3_leader", "standings_r3", "leader"),
    "KXPGAR2TOP10": ("r2_top10",  "standings_r2", "top10"),
    "KXPGAR3TOP10": ("r3_top10",  "standings_r3", "top10"),
    "KXPGAR3TOP5":  ("r3_top5",   "standings_r3", "top5"),
    "KXPGAR2TOP5":  ("r2_top5",   "standings_r2", "top5"),
}
EVENT_SERIES = {"KXPGAPLAYOFF": "playoff"}


def price_ancillary_markets(sim_data, tourney, name_replacements=None,
                            edge_threshold=DEFAULT_EDGE, verbose=True):
    """Pull every priceable ancillary Kalshi series for `tourney`, compute our fair
    from `sim_data`, and emit only rows that clear the taker/maker fill rules.

    sim_data keys (all optional except player_names):
      player_names, standings_r2, standings_r3, made_cut, final_scores,
      pred_lookup, sim_dict (single-round scores for H2H / n-ball, when offered)

    Returns a DataFrame (possibly empty) of fillable ancillary edges.
    """
    name_replacements = name_replacements or {}
    players = list(sim_data["player_names"])
    pred_lookup = sim_data.get("pred_lookup", {})
    client = _make_client()

    # Build per-player fair tables once (leader/top-N per round) ----------------
    fair_tables = {}
    if sim_data.get("standings_r2") is not None:
        fair_tables["standings_r2"] = leader_topn_fairs(
            sim_data["standings_r2"], players).set_index("player_name")
    if sim_data.get("standings_r3") is not None:
        fair_tables["standings_r3"] = leader_topn_fairs(
            sim_data["standings_r3"], players, made_cut=sim_data.get("made_cut")
        ).set_index("player_name")

    rows = []
    mismatches = set()

    # ---- Player-per-market series (leader, top-N) ----------------------------
    for series, (mtype, src, col) in PLAYER_SERIES.items():
        table = fair_tables.get(src)
        if table is None:
            continue
        try:
            markets = _match_tournament(_get_markets(client, series), tourney)
        except Exception as e:
            if verbose:
                print(f"  [ancillary] fetch {series} failed: {e}")
            continue
        if not markets:
            continue
        if verbose:
            print(f"  [ancillary] {series} ({mtype}): {len(markets)} markets")
        for m in markets:
            player = _norm_name(m.get("yes_sub_title") or "", name_replacements)
            if player not in table.index:
                if player:
                    mismatches.add(player)
                continue
            fair_yes = float(table.loc[player, col])
            if fair_yes <= 0:
                continue
            q_bid, q_ask = _market_quote(m)
            if q_bid <= 0 and q_ask <= 0:
                continue  # no quote -> too thin to fill; skip the orderbook fetch
            try:
                ob = _get_orderbook(client, m["ticker"])
                time.sleep(0.03)
            except Exception:
                continue
            bid, ask = _best_bid_ask(m, ob)
            if ask <= 0:
                continue
            base = {"market_type": mtype, "series": series, "ticker": m["ticker"],
                    "player_name": player, "my_pred": pred_lookup.get(player)}
            rows.extend(_edge_rows(base, fair_yes, bid, ask, ob, edge_threshold))

    # ---- Event markets (playoff) ---------------------------------------------
    if "playoff" in EVENT_SERIES.values() and sim_data.get("final_scores") is not None:
        fair_yes = playoff_prob(sim_data["final_scores"])
        try:
            markets = _match_tournament(_get_markets(client, "KXPGAPLAYOFF"), tourney)
        except Exception:
            markets = []
        for m in markets:
            q_bid, q_ask = _market_quote(m)
            if q_bid <= 0 and q_ask <= 0:
                continue
            try:
                ob = _get_orderbook(client, m["ticker"])
                time.sleep(0.03)
            except Exception:
                continue
            bid, ask = _best_bid_ask(m, ob)
            if ask <= 0:
                continue
            base = {"market_type": "playoff", "series": "KXPGAPLAYOFF",
                    "ticker": m["ticker"], "player_name": "(field)", "my_pred": None}
            rows.extend(_edge_rows(base, fair_yes, bid, ask, ob, edge_threshold))
        if verbose:
            print(f"  [ancillary] KXPGAPLAYOFF (playoff): fair P(playoff)={fair_yes:.3f}, "
                  f"{len(markets)} market(s)")

    # ---- 3-ball groups (lowest score in the round; dead-heat $1/N like leader) ----
    sim_dict = sim_data.get("sim_dict")
    if sim_dict is not None:
        try:
            markets = _match_tournament(_get_markets(client, "KXPGA3BALL"), tourney)
        except Exception:
            markets = []
        groups = defaultdict(list)
        for m in markets:
            groups[m.get("event_ticker", "")].append(m)
        priceable = 0
        for ev, gms in groups.items():
            # Each market's yes_sub_title is "<Player> beats <X> and <Y>".
            members = [_norm_name((m.get("yes_sub_title") or "").split(" beats ")[0],
                                  name_replacements) for m in gms]
            missing = [p for p in members if p not in sim_dict]
            if len(members) < 2 or missing:
                mismatches.update(p for p in missing if p)
                continue
            probs = nball_probs(sim_dict, members)
            priceable += 1
            for m in gms:
                player = _norm_name((m.get("yes_sub_title") or "").split(" beats ")[0],
                                    name_replacements)
                fair_yes = float(probs.get(player, 0.0))
                if fair_yes <= 0:
                    continue
                q_bid, q_ask = _market_quote(m)
                if q_bid <= 0 and q_ask <= 0:
                    continue
                try:
                    ob = _get_orderbook(client, m["ticker"])
                    time.sleep(0.03)
                except Exception:
                    continue
                bid, ask = _best_bid_ask(m, ob)
                if ask <= 0:
                    continue
                base = {"market_type": "3ball", "series": "KXPGA3BALL",
                        "ticker": m["ticker"], "player_name": player,
                        "my_pred": pred_lookup.get(player)}
                rows.extend(_edge_rows(base, fair_yes, bid, ask, ob, edge_threshold))
        if verbose:
            print(f"  [ancillary] KXPGA3BALL (3ball): {len(groups)} groups, "
                  f"{priceable} priceable from sim")

    client.close()

    cols = ["market_type", "series", "player_name", "side", "pricing", "fair_prob",
            "cost", "edge_pp", "stake", "units", "fill", "yes_bid", "yes_ask",
            "my_pred", "ticker"]
    for r in rows:
        contracts, _d, units = kelly_contracts(r["fair_prob"], r["cost"])
        r["stake"] = contracts
        r["units"] = round(units, 2)
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df = df.sort_values(["pricing", "edge_pp"], ascending=[True, False]).reset_index(drop=True)
    df.attrs["name_mismatches"] = mismatches
    return df
