"""Build passive working-quote candidates from model fairs + live books, using
quote_engine.plan_quote — the new outright quoting brain (replaces the static
edge-rung ladder). One working order per (market, side) the model likes, pegged
at/near the touch and capped at fair minus the side's minimum edge. Pure +
unit-testable; the maker's scan() feeds it per-market snapshots and the existing
reconcile_and_post() posts/maintains the results (a 'replace' at a new price
naturally consolidates old rungs into a single working order).

Pricing floor is Kelly-based per market type: the quote ceiling is the highest
price whose fill still keeps a Kelly fraction >= the market type's minimum
(KELLY_MIN_BY_TYPE) — so the required cents-edge scales with price level
(tight near 0/100, wide mid-book) instead of a flat cents gate. The old flat
side-aware margins (YES 0.5c, NO 0.3c) survive as absolute floors so extreme
prices never get looser than the audited behavior. H2H matchups are NOT
handled here — scan_matchups applies the same h2h Kelly floor itself.
"""
from __future__ import annotations

import os

import quote_engine as qe

EDGE_YES = 0.005   # absolute min edge to bid YES (floor under the Kelly margin)
EDGE_NO = 0.003    # absolute min edge to bid NO  (floor under the Kelly margin)


def _envf(name: str, default: float) -> float:
    try:
        v = os.getenv(name)
        return float(v) if v not in (None, "") else default
    except (TypeError, ValueError):
        return default


# Minimum Kelly fraction a fill must keep, by market type. A quote improves
# toward the touch only while the Kelly edge at that price stays >= this floor.
KELLY_MIN_BY_TYPE = {
    "winner": _envf("MAKER_KELLY_MIN_WINNER", 0.25),
    "top_5": _envf("MAKER_KELLY_MIN_TOP5", 0.25),
    "top_10": _envf("MAKER_KELLY_MIN_TOP10", 0.25),
    "top_20": _envf("MAKER_KELLY_MIN_TOP20", 0.15),
    "h2h": _envf("MAKER_KELLY_MIN_H2H", 0.05),
}


def kelly_edge_margin(fair: float, kelly_min: float) -> float:
    """Edge (price units) below `fair` needed so a fill at fair - margin keeps a
    Kelly fraction >= kelly_min. Kelly f = (fair - P)/(1 - P) >= k solves to
    P <= (fair - k)/(1 - k). Returns `fair` (no postable price) when no price
    can achieve the floor (fair <= kelly_min)."""
    if kelly_min <= 0:
        return 0.0
    if kelly_min >= 1 or fair <= kelly_min:
        return fair
    return fair - (fair - kelly_min) / (1.0 - kelly_min)

DEFAULT_ENGINE_CFG = {
    "mode": "improve",          # step ahead of the best bid by a tick
    "display_size": 40,         # iceberg slice shown at once
    "reprice_ticks": 1,         # hysteresis: don't move the order for < 1 tick
    "quote_below_touch": True,  # if best bid is above our edge ceiling, sit at the ceiling
    "min_price": 0.01,
}


def _no_book(yes_bid, yes_ask):
    """NO-side book from the YES book: a NO bid at P == a YES offer at 1-P."""
    no_bid = (1.0 - yes_ask) if yes_ask else None
    no_ask = (1.0 - yes_bid) if yes_bid else None
    return no_bid, no_ask


def plan_market(*, ticker, market_type, player, title, yes_bid, yes_ask, sim_yes,
                tick, held_yes=0, held_no=0, resting_yes=None, resting_no=None,
                target_usd=50.0, engine_cfg=None, edge_yes=EDGE_YES, edge_no=EDGE_NO,
                kelly_min=None):
    """Plan the working quote(s) for one outright market. Returns a list of 0-2
    candidate dicts (one per side that clears its edge), shaped for
    reconcile_and_post. 'keep'/'replace' actions become candidates (keyed by
    price so reconcile keeps or re-posts correctly); 'cancel'/'none' yield no
    candidate (so reconcile pulls any stale order on that side).

    `kelly_min` defaults to KELLY_MIN_BY_TYPE[market_type]; pass 0.0 to fall
    back to the flat edge_yes/edge_no margins only."""
    cfg = engine_cfg or DEFAULT_ENGINE_CFG
    k_min = KELLY_MIN_BY_TYPE.get(market_type, 0.0) if kelly_min is None else float(kelly_min)
    no_bid, no_ask = _no_book(yes_bid, yes_ask)
    # Single-sided accumulation toward the model's view: quote YES when the model
    # is at/above the market mid (YES looks underpriced), else quote NO. We never
    # rest both sides of the same market.
    mid = ((yes_bid or 0) + (yes_ask or 0)) / 2.0 if (yes_bid or yes_ask) else float(sim_yes)
    if float(sim_yes) >= mid:
        chosen = [("yes", float(sim_yes), edge_yes, yes_bid, yes_ask, held_yes, resting_yes)]
    else:
        chosen = [("no", 1.0 - float(sim_yes), edge_no, no_bid, no_ask, held_no, resting_no)]
    out = []
    for side, fair, flat_edge, bid, ask, held, resting in chosen:
        edge = max(flat_edge, kelly_edge_margin(fair, k_min))
        ceiling = fair - edge
        if ceiling <= cfg.get("min_price", 0.01):
            continue  # no edge room on this side
        target = max(1, int(target_usd / max(ceiling, tick)))
        dec = qe.plan_quote(
            side=side, fair=fair, target=target, held=int(held or 0),
            resting=resting, book={"bid": bid, "ask": ask, "tick": tick},
            cfg={**cfg, "edge_margin": edge})
        if dec["action"] not in ("replace", "keep"):
            continue
        price, size = dec["price"], dec["size"]
        if not price or not size or size < 1:
            continue
        out.append({
            "ticker": ticker, "title": title, "player": player, "market": market_type,
            "side": side, "post_price": float(price), "contracts": int(size),
            "post_type": "engine",
            "best_bid": float(bid or 0.0), "best_ask": float(ask or 1.0),
            "mid": float(((bid or 0) + (ask or 0)) / 2) if (bid or ask) else float(fair),
            "tick": float(tick), "sim_prob": float(fair),
            "edge_pp": (float(fair) - float(price)) * 100.0,
            "kelly_f": max(0.0, (float(fair) - float(price)) / (1.0 - float(price))) if price < 1 else 0.0,
            "kelly_min": k_min,
            "engine_action": dec["action"], "engine_reason": dec["reason"],
            "target": target, "held": int(held or 0),
        })
    return out
