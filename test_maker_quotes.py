"""Unit tests for maker_quotes.plan_market (the outright quoting brain). No I/O.
Run: python test_maker_quotes.py"""
import maker_quotes as mq

_p = _f = 0


def eq(name, got, want):
    global _p, _f
    if got == want:
        _p += 1
    else:
        _f += 1
        print(f"FAIL {name}: got {got!r} want {want!r}")


def truthy(name, got):
    global _p, _f
    if got:
        _p += 1
    else:
        _f += 1
        print(f"FAIL {name}: falsy {got!r}")


def base(**kw):
    d = dict(ticker="KXPGATOP10-TRAV26-SCHE", market_type="top_10", player="scottie scheffler",
             title="t", yes_bid=0.25, yes_ask=0.30, sim_yes=0.35, tick=0.01, target_usd=50.0)
    d.update(kw)
    return mq.plan_market(**d)


# model loves YES (fair 0.35): bid YES at touch+tick=0.26, below ceiling 0.345 and ask 0.30
cs = base()
yes = [c for c in cs if c["side"] == "yes"]
truthy("yes side quoted", yes)
eq("yes price improves touch", round(yes[0]["post_price"], 2), 0.26)
truthy("yes price below ask", yes[0]["post_price"] < 0.30)
truthy("yes keeps edge (price < fair-0.005)", yes[0]["post_price"] <= 0.35 - 0.005 + 1e-9)
eq("post_type engine", yes[0]["post_type"], "engine")

# single-sided: model is above mid (0.35 > 0.275) -> YES only, no NO on same market
eq("single-sided (prefer yes)", len(cs), 1)
truthy("no NO when model prefers yes", not [c for c in cs if c["side"] == "no"])

# model BELOW mid -> accumulate NO. sim_yes 0.20 < mid 0.275; fair_no 0.80.
csno = base(sim_yes=0.20)
eq("no-pref single-sided", len(csno), 1)
eq("no side chosen", csno[0]["side"], "no")
truthy("no price <= fair_no - 0.003", csno[0]["post_price"] <= 0.80 - 0.003 + 1e-9)
truthy("no price below no_ask 0.75", csno[0]["post_price"] < 0.75)

# never crosses: every candidate strictly below its side's ask
for c in cs + csno:
    truthy(f"{c['side']} no-cross", c["post_price"] < c["best_ask"])

# thin YES edge (model just above mid): still posts at/below ceiling, never above fair-0.005
cs2 = base(sim_yes=0.27, yes_bid=0.25, yes_ask=0.27)
yes2 = [c for c in cs2 if c["side"] == "yes"]
truthy("thin yes edge keeps edge", yes2 and yes2[0]["post_price"] <= 0.27 - 0.005 + 1e-9)

# target reached on YES (held >= target) -> no YES candidate (engine returns cancel/none)
cs3 = base(held_yes=100000)
truthy("held past target drops yes", not any(c["side"] == "yes" for c in cs3))

# hysteresis: existing yes order 1 tick from desired, right size -> 'keep' candidate present
cs4 = base(resting_yes={"price": 0.26, "size": 40})
yk = [c for c in cs4 if c["side"] == "yes"]
truthy("keep candidate emitted", yk and yk[0]["engine_action"] == "keep")

# target scales with target_usd / price (bigger budget -> bigger target)
cbig = base(target_usd=200.0)
csm = base(target_usd=20.0)
truthy("target scales with budget", [c for c in cbig if c['side']=='yes'][0]["target"]
       > [c for c in csm if c['side']=='yes'][0]["target"])

print(f"\n{_p} passed, {_f} failed")
raise SystemExit(1 if _f else 0)
