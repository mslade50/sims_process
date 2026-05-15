"""Portfolio-Kelly solver for outright maker sizing.

Wired into kalshi_maker.py --preview. Wrapped in a try/except inside the
caller so a solver failure falls back to flat-33 sizing rather than killing
the preview pipeline.

Design contract (from 2026-05-13 discussion):
    1. Bankroll       = Kalshi's `balance_dollars` (cash actually available,
                         already net of cash reserved for resting orders).
    2. Kelly fraction = 0.5  (half-Kelly).
    3. Correlation    = handled per-player via discrete-outcome Kelly
                         over 5 mutually-exclusive finish-position buckets.
    4. Existing       = filled positions + ALL resting golf orders
       exposure         (script-placed AND manual hand-clicks) count
                         toward our target stake. New orders only top up
                         the delta — never cancel-and-replace.
    5. Bankroll cap   = if sum(new_stakes) > balance, scale proportionally.
    6. Matchups       = OUT OF SCOPE; their cash is subtracted from the
                         pre-Kelly budget so the optimizer doesn't try to
                         deploy it.
    7. Sizes are       = solver can only ADD contracts. If target shrinks,
       monotone-up       existing exposure stays.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


# Mutually-exclusive finish buckets per player.
BUCKET_LABELS = ("winner", "rank2-5", "rank6-10", "rank11-20", "rank21+")
N_BUCKETS = len(BUCKET_LABELS)

# For each (market, side) bet, the set of bucket indices in which the bet
# wins ($1 per contract).
_WIN_SET = {
    ("winner", "yes"): {0},
    ("winner", "no"):  {1, 2, 3, 4},
    ("top_5",  "yes"): {0, 1},
    ("top_5",  "no"):  {2, 3, 4},
    ("top_10", "yes"): {0, 1, 2},
    ("top_10", "no"):  {3, 4},
    ("top_20", "yes"): {0, 1, 2, 3},
    ("top_20", "no"):  {4},
}


def player_buckets_from_top_probs(top_probs):
    """Convert {top_5, top_10, top_20, winner} into a length-5 bucket array.

    top_probs: dict-like with keys 'winner', 'top_5', 'top_10', 'top_20'.
               These are non-mutually-exclusive (nested) probabilities.
    Returns: np.array of [p_winner, p_2-5, p_6-10, p_11-20, p_21+] summing to ~1.
    """
    p_winner = float(top_probs.get("winner", 0) or 0)
    p_top5 = float(top_probs.get("top_5", 0) or 0)
    p_top10 = float(top_probs.get("top_10", 0) or 0)
    p_top20 = float(top_probs.get("top_20", 0) or 0)
    return np.array([
        p_winner,
        max(p_top5 - p_winner, 0.0),
        max(p_top10 - p_top5, 0.0),
        max(p_top20 - p_top10, 0.0),
        max(1.0 - p_top20, 0.0),
    ])


def bet_return_vector(market, side, post_price):
    """5-vector of multiplicative returns per dollar staked on bet b in each bucket."""
    win_set = _WIN_SET.get((market, side))
    if win_set is None:
        return None
    win_return = (1.0 - post_price) / post_price
    return np.array([
        win_return if i in win_set else -1.0
        for i in range(N_BUCKETS)
    ])


def optimize_player_kelly(candidates, bucket_probs, kelly_fraction=0.5,
                          max_total_fraction=0.25):
    """Solve discrete-outcome Kelly for one player's candidate bets.

    Args:
        candidates: list of dicts with market, side, post_price.
        bucket_probs: length-5 array.
        kelly_fraction: post-solve haircut (0.5 = half-Kelly).
        max_total_fraction: hard cap on Σ f_i per player (25% of bankroll).

    Returns:
        list of bankroll-fractions, same order as candidates, post-haircut.
        Empty list if optimization fails or no usable bets.
    """
    from scipy.optimize import minimize

    n = len(candidates)
    if n == 0:
        return []

    rows = []
    valid_idx = []
    for i, c in enumerate(candidates):
        v = bet_return_vector(c.get("market", ""), c.get("side", ""),
                              float(c.get("post_price") or 0))
        if v is None or float(c.get("post_price", 0)) <= 0:
            continue
        rows.append(v)
        valid_idx.append(i)
    if not rows:
        return [0.0] * n

    R = np.stack(rows)
    p = np.asarray(bucket_probs, dtype=float)

    def neg_log_growth(f):
        growth = 1.0 + R.T @ f
        if np.any(growth <= 1e-9):
            return 1e6
        return -float(np.sum(p * np.log(growth)))

    bounds = [(0.0, max_total_fraction) for _ in range(len(rows))]
    cons = ({"type": "ineq", "fun": lambda f: max_total_fraction - np.sum(f)},)
    f0 = np.full(len(rows), max_total_fraction / (2 * len(rows)))

    try:
        res = minimize(neg_log_growth, f0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-8})
        f_full = np.clip(res.x, 0.0, None) * kelly_fraction
    except Exception:
        return [0.0] * n

    out = [0.0] * n
    for j, orig_i in enumerate(valid_idx):
        out[orig_i] = float(f_full[j])
    return out


# ── Authenticated data fetches ────────────────────────────────────────
def fetch_balance():
    """GET /portfolio/balance → balance_dollars (free cash, Kalshi already
    nets out cash reserved for resting orders)."""
    from kalshi_maker import _authed_request
    r = _authed_request("GET", "/trade-api/v2/portfolio/balance")
    r.raise_for_status()
    body = r.json()
    # Kalshi exposes either `balance` (cents) or `balance_dollars` (dollars).
    if "balance_dollars" in body:
        return float(body["balance_dollars"])
    return float(body.get("balance", 0)) / 100.0


def fetch_positions():
    """GET /portfolio/positions → {ticker: {yes_contracts, no_contracts}}.
    Uses signed `position_fp`: positive = long YES, negative = long NO."""
    from kalshi_maker import _authed_request
    out = {}
    cursor = None
    while True:
        path = "/trade-api/v2/portfolio/positions?limit=200"
        if cursor:
            path += f"&cursor={cursor}"
        r = _authed_request("GET", path)
        r.raise_for_status()
        data = r.json()
        for p in data.get("market_positions", []):
            t = p.get("ticker", "")
            try:
                pos_fp = float(p.get("position_fp", 0) or 0)
            except (TypeError, ValueError):
                pos_fp = 0.0
            yes = int(round(pos_fp)) if pos_fp > 0 else 0
            no = int(round(-pos_fp)) if pos_fp < 0 else 0
            out[t] = {"yes_contracts": yes, "no_contracts": no}
        cursor = data.get("cursor")
        if not cursor:
            break
    return out


def existing_contracts_per_bet(positions, resting_orders):
    """Combine filled positions + resting orders into a per-(ticker, side)
    contract count. Manual hand-clicks AND script orders both count."""
    out = defaultdict(int)
    for ticker, pos in positions.items():
        out[(ticker, "yes")] += int(pos.get("yes_contracts", 0))
        out[(ticker, "no")]  += int(pos.get("no_contracts", 0))
    for o in resting_orders:
        side = o.get("side", "")
        ticker = o.get("ticker", "")
        try:
            qty = int(round(float(o.get("remaining_count_fp") or 0)))
        except (TypeError, ValueError):
            qty = 0
        out[(ticker, side)] += qty
    return dict(out)


# ── Master orchestration ──────────────────────────────────────────────
def compute_optimal_outright_sizing(
    outright_candidates,           # list of candidate dicts from kalshi_maker.scan
    matchup_candidates,            # list — used only to net out matchup cash
    prob_lookup,                   # {player_lower: {winner, top_5, top_10, top_20}}
    kelly_fraction=0.5,
    max_total_fraction_per_player=0.25,
    verbose=False,
):
    """Returns dict {(ticker, side, milli_dollars_price): int_new_contracts}.

    Drops any entry where new_contracts < 1.
    """
    from kalshi_maker import list_resting_orders

    balance = fetch_balance()
    positions = fetch_positions()
    resting = list_resting_orders(golf_only=True)

    # Subtract matchup cash from balance — matchups are sized flat and
    # consume cash outside the Kelly optimization.
    matchup_cash = sum(
        float(c.get("post_price", 0)) * int(c.get("contracts", 0))
        for c in matchup_candidates
    )
    available_cash = max(balance - matchup_cash, 0.0)

    existing = existing_contracts_per_bet(positions, resting)

    by_player = defaultdict(list)
    for c in outright_candidates:
        by_player[c.get("player", "")].append(c)

    target_contracts = {}
    players_solved = 0
    players_skipped = 0
    for player, cands in by_player.items():
        rec = prob_lookup.get(player)
        if rec is None:
            players_skipped += 1
            continue
        buckets = player_buckets_from_top_probs(rec)
        if not (0.99 <= buckets.sum() <= 1.01):
            players_skipped += 1
            continue

        # Group the player's candidates by (ticker, side). The 1-3 entries
        # in each group are different price levels (near_ask / mid / bid) of
        # the SAME underlying bet — they have perfectly correlated payoffs.
        # Joint Kelly treats them as independent and naturally collapses all
        # sizing onto the cheapest level (same win-bucket, lower cost). So
        # we feed the optimizer one representative per group, then split the
        # resulting contract count evenly across the levels that passed the
        # edge gate. The representative uses the AVERAGE price across the
        # group's levels — that way, dividing the Kelly target contracts by
        # n_levels and laddering deploys exactly the Kelly-target $ amount
        # (cheapest-rep would systematically over-deploy when fills land at
        # higher levels, average is unbiased in expectation).
        groups = defaultdict(list)
        for c in cands:
            groups[(c["ticker"], c["side"])].append(c)
        reps = []
        group_keys = []
        for gk, glist in groups.items():
            prices = [float(x.get("post_price", 0)) for x in glist]
            avg_price = sum(prices) / len(prices) if prices else 0.0
            rep = dict(glist[0])
            rep["post_price"] = avg_price
            reps.append(rep)
            group_keys.append(gk)

        fractions = optimize_player_kelly(
            reps, buckets,
            kelly_fraction=kelly_fraction,
            max_total_fraction=max_total_fraction_per_player,
        )

        for gk, rep, f in zip(group_keys, reps, fractions):
            if f <= 0:
                continue
            stake_dollars = balance * f
            avg_price = float(rep.get("post_price") or 0)
            if avg_price <= 0:
                continue
            total_contracts = stake_dollars / avg_price
            glist = groups[gk]
            n_levels = len(glist)
            per_level = total_contracts / n_levels
            for c in glist:
                target_contracts[_cand_key(c)] = per_level
        players_solved += 1

    # Subtract existing contracts; only post the delta.
    new_contracts = {}
    cand_by_key = {_cand_key(c): c for c in outright_candidates}
    for key, target in target_contracts.items():
        c = cand_by_key[key]
        already = existing.get((c["ticker"], c["side"]), 0)
        delta = max(target - already, 0.0)
        if delta >= 1.0:
            new_contracts[key] = delta

    # Bankroll-cap scale-down.
    total_new_dollars = sum(
        new_contracts[k] * float(cand_by_key[k]["post_price"])
        for k in new_contracts
    )
    scaled = False
    if total_new_dollars > available_cash and available_cash > 0:
        scale = available_cash / total_new_dollars
        new_contracts = {k: v * scale for k, v in new_contracts.items()}
        scaled = True

    if verbose:
        print(f"  [kelly] balance=${balance:,.2f}  matchup_reserved=${matchup_cash:,.2f}"
              f"  available=${available_cash:,.2f}")
        print(f"  [kelly] players solved={players_solved} skipped={players_skipped}"
              f"  candidates_with_target={len(target_contracts)}"
              f"  after_existing_subtraction={len(new_contracts)}"
              f"  scaled_down={scaled}")
        if scaled:
            print(f"  [kelly] proposed total=${total_new_dollars:,.2f} > cash, "
                  f"scaling all stakes by {scale:.4f}")

    return {k: int(round(v)) for k, v in new_contracts.items() if v >= 1}


def _cand_key(c):
    """(ticker, side, milli-dollars) — matches kalshi_maker._resting_key."""
    return (c["ticker"], c["side"], int(round(c["post_price"] * 1000)))
