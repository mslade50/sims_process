"""Unit tests for maker_guard (kill switch + exposure governor). No network, no
orders. Run: python test_maker_guard.py"""
import os

import maker_guard as mg

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
        print(f"FAIL {name}: expected truthy, got {got!r}")


# ── kill switch ────────────────────────────────────────────────────────────────
os.environ.pop("MAKER_KILL", None)
eq("no kill by default", mg.env_or_file_kill(), None)
os.environ["MAKER_KILL"] = "1"
truthy("env kill fires", mg.env_or_file_kill())
os.environ["MAKER_KILL"] = "no"
eq("env kill falsy ignored", mg.env_or_file_kill(), None)
os.environ.pop("MAKER_KILL", None)

# sheet toggle (inject get_param)
eq("sheet enabled", mg.sheet_disabled(lambda n, default=None: "yes"), None)
truthy("sheet off halts", mg.sheet_disabled(lambda n, default=None: "no"))
eq("sheet absent -> no block", mg.sheet_disabled(lambda n, default=None: None), None)
eq("sheet blank -> no block", mg.sheet_disabled(lambda n, default=None: ""), None)


def _raises(n, default=None):
    raise RuntimeError("sheet down")


eq("sheet error -> no block", mg.sheet_disabled(_raises), None)

ok, why = mg.should_trade(lambda n, default=None: "yes")
truthy("should_trade enabled", ok)
ok, why = mg.should_trade(lambda n, default=None: "off")
eq("should_trade disabled", ok, False)

# pull_script_quotes
resting = [
    {"order_id": "a", "ticker": "KXPGATOUR-X-A", "_script": True},
    {"order_id": "b", "ticker": "KXPGATOUR-X-B", "_script": False},  # manual
    {"order_id": "c", "ticker": "KXPGATOUR-X-C", "_script": True},
]
cancelled = []
n = mg.pull_script_quotes(
    lambda: resting,
    lambda o: o["_script"],
    lambda oid, tk, reason="": (cancelled.append(oid) or True),
)
eq("pull cancels only script", n, 2)
eq("pull left manual alone", "b" in cancelled, False)

# ── build_exposure ─────────────────────────────────────────────────────────────
positions = [
    {"ticker": "KXPGATOUR-TRAV26-SCHE", "position_fp": "500", "market_exposure_dollars": "90"},
    {"ticker": "KXPGATOP5-TRAV26-MCIL", "position_fp": "-300", "market_exposure_dollars": "120"},
    {"ticker": "KXPGATOUR-TRAV26-ZERO", "position_fp": "0", "market_exposure_dollars": "0"},
]
resting2 = [
    {"ticker": "KXPGATOUR-TRAV26-SCHE", "side": "yes", "remaining_count_fp": "100", "yes_price_dollars": "0.20"},
    {"ticker": "KXPGATOP10-PGC26-RAHM", "side": "no", "remaining_count_fp": "50", "no_price_dollars": "0.30"},
]
exp = mg.build_exposure(positions, resting2)
eq("exp SCHE yes = held 90 + resting 20", round(exp["per_key"][("KXPGATOUR-TRAV26-SCHE", "yes")], 2), 110.0)
eq("exp MCIL no held", exp["per_key"][("KXPGATOP5-TRAV26-MCIL", "no")], 120.0)
eq("exp event TRAV26", round(exp["per_event"]["TRAV26"], 2), 230.0)  # 90+20+120
eq("exp event PGC26", exp["per_event"]["PGC26"], 15.0)  # 50*0.30
eq("exp total", round(exp["total"], 2), 245.0)
truthy("zero position skipped", ("KXPGATOUR-TRAV26-ZERO", "yes") not in exp["per_key"])

# ── apply_exposure_caps ────────────────────────────────────────────────────────
CAPS = {"per_market_usd": 50.0, "per_event_usd": 400.0, "total_usd": 1000.0,
        "max_new_usd_run": 300.0, "max_orders_run": 40}

# per-market cap trims: SCHE yes already at $110 held+resting; cap $50 -> 0 room -> drop
cands = [{"ticker": "KXPGATOUR-TRAV26-SCHE", "side": "yes", "post_price": 0.20, "contracts": 100, "kelly_f": 0.9}]
kept, rep = mg.apply_exposure_caps(cands, exp, CAPS)
eq("over-cap market dropped", len(kept), 0)
eq("drop reason cap", rep["dropped_detail"][0]["why"], "cap reached")

# fresh market: cap $50 at 10c -> max 500 contracts; want 800 -> trimmed to 500
cands = [{"ticker": "KXPGATOP20-TRAV26-NEW", "side": "yes", "post_price": 0.10, "contracts": 800, "kelly_f": 0.5}]
kept, rep = mg.apply_exposure_caps(cands, {"per_key": {}, "per_event": {}, "total": 0}, CAPS)
eq("trimmed to market cap", kept[0]["contracts"], 500)
eq("trim recorded", rep["trimmed"], 1)

# total cap: many fresh markets, total cap $1000 at $0.50 each market capped $50 -> 100 each... use total to bind
caps2 = dict(CAPS, total_usd=120.0, per_market_usd=1000.0, per_event_usd=1000.0)
many = [{"ticker": f"KXPGATOP20-E-{i}", "side": "yes", "post_price": 0.50, "contracts": 200, "kelly_f": 1.0 - i * 0.01} for i in range(10)]
kept, rep = mg.apply_exposure_caps(many, {"per_key": {}, "per_event": {}, "total": 0}, caps2)
truthy("total cap binds new_usd<=120", rep["new_usd"] <= 120.0 + 1e-9)

# max orders/run
caps3 = dict(CAPS, max_orders_run=3, per_market_usd=1000.0)
many2 = [{"ticker": f"KXPGATOP20-E-{i}", "side": "yes", "post_price": 0.10, "contracts": 5, "kelly_f": 1.0 - i * 0.01} for i in range(8)]
kept, rep = mg.apply_exposure_caps(many2, {"per_key": {}, "per_event": {}, "total": 0}, caps3)
eq("max orders/run honored", rep["kept"], 3)

# ranking: highest kelly gets budget first when total binds
caps4 = dict(CAPS, total_usd=10.0, per_market_usd=1000.0, max_new_usd_run=10.0)
two = [
    {"ticker": "KXPGATOP20-E-LO", "side": "yes", "post_price": 0.10, "contracts": 100, "kelly_f": 0.1},
    {"ticker": "KXPGATOP20-E-HI", "side": "yes", "post_price": 0.10, "contracts": 100, "kelly_f": 0.9},
]
kept, rep = mg.apply_exposure_caps(two, {"per_key": {}, "per_event": {}, "total": 0}, caps4)
eq("highest kelly funded first", kept[0]["ticker"], "KXPGATOP20-E-HI")

print(f"\n{_p} passed, {_f} failed")
raise SystemExit(1 if _f else 0)
