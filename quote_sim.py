"""Dry-run simulation of quote_engine against a moving order book. NO live orders,
no network - purely illustrative so you can watch the passive-accumulation
behavior. Run: python quote_sim.py
"""
import random

import quote_engine as qe

random.seed(7)

TICK = 0.01
FAIR = 0.30                       # our model fair for YES
TARGET = 200                      # contracts we want to accumulate
CFG = {"edge_margin": 0.02,       # ceiling = 0.28 - never bid above this
       "mode": "improve",         # step ahead of the best bid by a tick
       "display_size": 40,        # iceberg slice
       "reprice_ticks": 1,        # hysteresis: don't move for < 1 tick
       "quote_below_touch": True,
       "min_price": 0.01}

# A synthetic YES book that drifts UP over time (the player's price firming),
# so we can watch the bot follow the bid up and then refuse to pay above fair-edge.
bid, ask = 0.23, 0.28
held = 0
resting = None          # {'price','size'}
cost = 0.0              # $ spent (for avg price)

print(f"target {TARGET} @ fair {FAIR:.2f}  ceiling(fair-edge) {FAIR-CFG['edge_margin']:.2f}  "
      f"display {CFG['display_size']}  mode {CFG['mode']}")
print(f"{'t':>2} {'book bid/ask':>13} {'our quote':>11} {'act':>7} {'fill':>5} "
      f"{'held/target':>12} {'avg':>6}")

for t in range(1, 26):
    # ── book drifts up with noise ──
    drift = 0.01 if t % 3 == 0 else 0.0
    bid = round(min(0.40, max(0.05, bid + drift + random.choice([-0.01, 0, 0, 0.01]))), 2)
    ask = round(max(bid + 0.01, bid + random.choice([0.01, 0.02, 0.02])), 2)

    book = {"bid": bid, "ask": ask, "tick": TICK}
    d = qe.plan_quote(side="yes", fair=FAIR, target=TARGET, held=held,
                      resting=resting, book=book, cfg=CFG)

    # ── apply decision ──
    if d["action"] == "replace":
        resting = {"price": d["price"], "size": d["size"]}
    elif d["action"] == "cancel" or d["action"] == "none":
        resting = None
    # 'keep' leaves resting unchanged

    # ── invariant: we are a MAKER - never at/above the offer ──
    if resting:
        assert resting["price"] < ask + 1e-9, f"CROSS! {resting['price']} >= ask {ask}"

    # ── stochastic passive fill: a seller hits our bid when we're at/ahead of
    # the touch and the spread is tight. We fill up to our shown size. ──
    fill = 0
    if resting and resting["price"] >= bid - 1e-9 and random.random() < 0.55:
        fill = min(resting["size"], random.choice([10, 20, 30, 40]), TARGET - held)
        if fill > 0:
            held += fill
            cost += fill * resting["price"]
            resting["size"] -= fill
            if resting["size"] <= 0:
                resting = None   # slice exhausted; engine reposts a fresh slice next tick

    q = f"{resting['price']*100:.0f}c x{resting['size']}" if resting else "-"
    avg = (cost / held) if held else 0.0
    print(f"{t:>2} {f'{bid*100:.0f}/{ask*100:.0f}c':>13} {q:>11} {d['action']:>7} "
          f"{fill if fill else '':>5} {f'{held}/{TARGET}':>12} {avg*100:>5.1f}c")
    if held >= TARGET:
        print(f"\n  TARGET REACHED at t={t}: {held} contracts, avg {avg*100:.1f}c "
              f"(fair {FAIR*100:.0f}c, ceiling {(FAIR-CFG['edge_margin'])*100:.0f}c) - "
              f"every fill kept edge, nothing crossed.")
        break
