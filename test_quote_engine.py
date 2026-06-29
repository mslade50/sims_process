"""Unit tests for quote_engine.plan_quote. No I/O, no orders.
Run: python test_quote_engine.py"""
import quote_engine as qe

_p = _f = 0


def check(name, cond):
    global _p, _f
    if cond:
        _p += 1
    else:
        _f += 1
        print(f"FAIL {name}")


def eq(name, got, want):
    check(f"{name} (got {got!r} want {want!r})", got == want)


CFG = {"edge_margin": 0.02, "mode": "improve", "display_size": 40,
       "reprice_ticks": 1, "quote_below_touch": True, "min_price": 0.01}
BOOK = {"bid": 0.25, "ask": 0.30, "tick": 0.01}  # YES side: bid 25c, ask 30c


def plan(**kw):
    base = dict(side="yes", fair=0.30, target=200, held=0, resting=None, book=BOOK, cfg=CFG)
    base.update(kw)
    return qe.plan_quote(**base)


# improve: posts best_bid + 1 tick = 26c (below ceiling 28c, below ask 30c)
d = plan()
eq("improve action", d["action"], "replace")
eq("improve price = bid+tick", round(d["price"], 2), 0.26)
eq("improve size = display", d["size"], 40)
check("never crosses ask", d["price"] < BOOK["ask"])
check("never above ceiling (fair-edge=0.28)", d["price"] <= 0.28 + 1e-9)

# join: sits on the best bid = 25c
d = plan(cfg={**CFG, "mode": "join"})
eq("join price = bid", round(d["price"], 2), 0.25)

# ceiling caps it: fair 0.27, edge 0.02 -> ceiling 0.25; improve wants 0.26 -> capped to 0.25
d = plan(fair=0.27)
eq("capped at ceiling", round(d["price"], 2), 0.25)

# best bid above ceiling: bid 0.29, ceiling 0.28 -> sit at ceiling (quote_below_touch True)
d = plan(book={"bid": 0.29, "ask": 0.34, "tick": 0.01})
eq("sit at ceiling below touch", round(d["price"], 2), 0.28)
# ... or pull when quote_below_touch False
d = plan(book={"bid": 0.29, "ask": 0.34, "tick": 0.01}, cfg={**CFG, "quote_below_touch": False})
eq("pull when not chasing", d["action"], "none")

# no edge: fair below min postable -> stop
d = plan(fair=0.02)
check("no-edge stops", d["action"] in ("none", "cancel"))

# target reached -> cancel any resting, else none
d = plan(held=200, resting={"price": 0.26, "size": 40})
eq("target reached cancels resting", d["action"], "cancel")
d = plan(held=200, resting=None)
eq("target reached none", d["action"], "none")

# iceberg: remaining < display -> size = remaining
d = plan(held=180, target=200)  # remaining 20 < display 40
eq("iceberg trims to remaining", d["size"], 20)

# no-cross in a 1-tick spread: bid 0.29 ask 0.30 -> can't improve into 0.30, joins 0.29
d = plan(fair=0.40, book={"bid": 0.29, "ask": 0.30, "tick": 0.01})
check("1-tick spread never crosses", d["price"] < 0.30)
eq("1-tick spread joins bid", round(d["price"], 2), 0.29)

# hysteresis: existing order 1 tick away, same size -> keep (no churn)
d = plan(resting={"price": 0.26, "size": 40})
eq("keep within band", d["action"], "keep")
# moved 3 ticks away -> replace
d = plan(resting={"price": 0.23, "size": 40})
eq("replace when book moved", d["action"], "replace")
# right price but wrong size -> replace (e.g. need to top up the slice)
d = plan(resting={"price": 0.26, "size": 10})
eq("replace on size change", d["action"], "replace")

# empty bid side: still posts at the cap (fair-edge), below ask
d = plan(book={"bid": None, "ask": 0.30, "tick": 0.01})
eq("empty bid posts at cap", round(d["price"], 2), 0.28)

print(f"\n{_p} passed, {_f} failed")
raise SystemExit(1 if _f else 0)
