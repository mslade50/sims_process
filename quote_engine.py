"""Passive accumulation quoting engine for the golf maker.

GOAL: accumulate a TARGET position by providing liquidity near the top of book —
never taking it. The engine pegs a resting bid to the touch, moves with the book,
never bids above the model fair (so every fill keeps edge), never crosses the
opposite offer (pure maker), and works the order in slices (iceberg) so it
accumulates gradually instead of flashing one big, easily-faded order.

`plan_quote()` is PURE: it takes a market snapshot and returns a decision
(keep / replace / cancel / none). An order-management loop applies the decision;
the engine itself does no I/O, so it is unit-testable and can be dry-run against a
simulated book with zero live orders. See test_quote_engine.py and quote_sim.py.

All prices are dollars in [0, 1]. "side" is the side we are ACCUMULATING (we post
a resting bid for that side); `book.bid`/`book.ask` are that side's best bid and
best offer (ask = the no-cross ceiling).
"""
from __future__ import annotations

import math


def round_tick(p, tick):
    if not tick or tick <= 0:
        return p
    return round(round(p / tick) * tick, 10)


def floor_tick(p, tick):
    if not tick or tick <= 0:
        return p
    return round(math.floor(p / tick + 1e-9) * tick, 10)


def _decision(action, reason, price=None, size=None):
    return {"action": action, "price": price, "size": size, "reason": reason}


def plan_quote(*, side, fair, target, held, resting, book, cfg):
    """Decide the resting quote for one (market, side).

    side     'yes' | 'no'   — the side we are accumulating (informational; the
             caller passes the matching `fair` and `book` for that side).
    fair     our fair PRICE for that side (dollars, 0..1).
    target   target contracts to HOLD on this side.
    held     contracts already held on this side.
    resting  {'price': float, 'size': int} | None — our current resting order.
    book     {'bid': float|None, 'ask': float|None, 'tick': float} for THIS side.
    cfg      {
       'edge_margin':   float  required edge vs fair (price units); ceiling = fair - this
       'mode':          'improve' (best_bid + 1 tick) | 'join' (sit on best_bid)
       'display_size':  int     iceberg slice shown at once (replenished as it fills)
       'reprice_ticks': float   hysteresis band — don't move the order for < this many ticks
       'quote_below_touch': bool if the best bid is above our edge ceiling, sit at the
                                 ceiling (True) or pull and wait (False)
       'min_price':     float   lowest postable price (default = one tick)
    }

    Returns {'action': 'keep'|'replace'|'cancel'|'none', 'price', 'size', 'reason'}.
    """
    tick = float(book.get("tick") or 0.01)
    min_price = float(cfg.get("min_price", tick))

    def stop(reason):
        return _decision("cancel", reason) if resting else _decision("none", reason)

    remaining = int(target) - int(held)
    if remaining <= 0:
        return stop("target reached")

    # Edge ceiling: the highest price that still leaves >= edge_margin of edge.
    # FLOOR (not round) to the tick so we never bid above fair - edge_margin.
    ceiling = floor_tick(float(fair) - float(cfg.get("edge_margin", 0.0)), tick)
    if ceiling < min_price:
        return stop("no edge (ceiling below min price)")

    bid = book.get("bid")
    ask = book.get("ask")
    # Never cross: stay strictly below the opposite offer.
    no_cross = (round_tick(ask, tick) - tick) if ask is not None else ceiling
    cap = min(ceiling, no_cross)
    if cap < min_price:
        return stop("cannot post below ask while keeping edge")

    mode = cfg.get("mode", "improve")
    if mode == "improve":
        want = (bid + tick) if bid is not None else cap
    else:  # join — sit on the best bid
        want = bid if bid is not None else cap
    desired = max(floor_tick(min(want, cap), tick), min_price)

    behind_touch = bid is not None and desired < bid - 1e-9
    if behind_touch and not cfg.get("quote_below_touch", True):
        return stop("best bid above edge ceiling — not chasing")

    size = min(int(cfg.get("display_size", remaining) or remaining), remaining)
    if size < 1:
        return stop("no size")

    # Hysteresis: keep an existing order that's close enough and the right size,
    # as long as it's still valid (not crossing, still has edge). Avoids churn.
    if resting:
        rp = float(resting.get("price", 0))
        rs = int(resting.get("size", 0))
        band = float(cfg.get("reprice_ticks", 1)) * tick + 1e-9
        if abs(rp - desired) <= band and rs == size and rp <= no_cross + 1e-9 and rp <= ceiling + 1e-9:
            return _decision("keep", "within reprice band", rp, rs)

    return _decision("replace", f"{mode} at touch", desired, size)
