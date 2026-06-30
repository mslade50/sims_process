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

# ── Guard #1: fairs freshness ──────────────────────────────────────────────────
NOW = 1_000_000.0
eq("no fair file -> halt", mg.check_fairs_fresh(None, None, NOW)[0], False)
eq("fresh fairs ok", mg.check_fairs_fresh("f.parquet", NOW - 3600, NOW, max_age_hours=48)[0], True)
eq("stale fairs halt", mg.check_fairs_fresh("f.parquet", NOW - 60 * 3600, NOW, max_age_hours=48)[0], False)

# ── Guard #2: datagolf_live (asymmetric) ───────────────────────────────────────
eq("dg fresh -> live", mg.datagolf_live(NOW - 5 * 60, NOW), True)
eq("dg ambiguous -> unknown", mg.datagolf_live(NOW - 60 * 60, NOW), None)  # 60m: >30 <120
eq("dg confidently stale -> not live", mg.datagolf_live(NOW - 180 * 60, NOW), False)
eq("dg none -> unknown", mg.datagolf_live(None, NOW), None)
eq("dg future (skew) -> live", mg.datagolf_live(NOW + 600, NOW), True)

# ── Guard #2: schedule_live ────────────────────────────────────────────────────
ft, lt = NOW + 3600, NOW + 4 * 3600   # first tee +1h, last tee +4h
eq("before first tee -> not live", mg.schedule_live(ft, lt, NOW, round_hours=6), False)
eq("during play -> live", mg.schedule_live(ft, lt, ft + 1800, round_hours=6), True)
eq("after last tee + round_hours -> not live", mg.schedule_live(ft, lt, lt + 7 * 3600, round_hours=6), False)
eq("tee times unknown -> None", mg.schedule_live(None, None, NOW), None)

# ── Guard #2: resolve_live ─────────────────────────────────────────────────────
eq("agree not-live -> trade", mg.resolve_live(False, False)[0], False)
eq("schedule live, dg not-live -> dg wins (resume)", mg.resolve_live(True, False)[0], False)
eq("schedule clear, dg live -> dg wins (halt)", mg.resolve_live(False, True)[0], True)
eq("dg unavailable -> schedule", mg.resolve_live(True, None)[0], True)
eq("both blind -> fail-closed live", mg.resolve_live(None, None)[0], True)
truthy("override is logged", "override" in mg.resolve_live(False, True)[1].lower())

# ── Pre-Wednesday rule ─────────────────────────────────────────────────────────
import datetime as _dt
def D(weekday, hour):  # a datetime with a given weekday (0=Mon) and hour
    base = _dt.datetime(2026, 6, 29)  # Monday
    return (base + _dt.timedelta(days=weekday)).replace(hour=hour)

eq("Mon -> before cutoff", mg.before_wed_cutoff(now=D(0, 10)), True)
eq("Tue -> before cutoff", mg.before_wed_cutoff(now=D(1, 23)), True)
eq("Wed 3pm -> before cutoff", mg.before_wed_cutoff(now=D(2, 15)), True)
eq("Wed 5pm -> after cutoff", mg.before_wed_cutoff(now=D(2, 17)), False)
eq("Thu -> after cutoff", mg.before_wed_cutoff(now=D(3, 9)), False)
eq("Sun -> after cutoff", mg.before_wed_cutoff(now=D(6, 12)), False)

# block_yes_outright: only YES outrights, only before cutoff, only when enabled
eq("block yes winner pre-wed", mg.block_yes_outright("yes", "winner", now=D(1, 10)), True)
eq("block yes top_10 pre-wed", mg.block_yes_outright("yes", "top_10", now=D(0, 9)), True)
eq("NO never blocked", mg.block_yes_outright("no", "winner", now=D(1, 10)), False)
eq("h2h not an outright", mg.block_yes_outright("yes", "h2h", now=D(1, 10)), False)
eq("yes after cutoff allowed", mg.block_yes_outright("yes", "winner", now=D(3, 10)), False)
eq("rule disabled -> allowed", mg.block_yes_outright("yes", "winner", now=D(1, 10), rule_enabled=False), False)

# ── Published sim_fairs.json freshness ─────────────────────────────────────────
_gen = "2026-06-30 02:05:04 UTC"
_ts = mg.parse_fairs_ts(_gen)
eq("parse generated_at", _ts is not None, True)
eq("fresh fairs ok", mg.check_fairs_fresh_payload({"tourney": "deere", "generated_at": _gen}, _ts + 3600, tourney="deere")[0], True)
eq("stale fairs blocked", mg.check_fairs_fresh_payload({"tourney": "deere", "generated_at": _gen}, _ts + 100 * 3600, tourney="deere")[0], False)
eq("wrong tourney blocked", mg.check_fairs_fresh_payload({"tourney": "us_open", "generated_at": _gen}, _ts + 3600, tourney="deere")[0], False)
eq("no payload blocked", mg.check_fairs_fresh_payload(None, _ts, tourney="deere")[0], False)
eq("missing generated_at -> present ok", mg.check_fairs_fresh_payload({"tourney": "deere"}, _ts, tourney="deere")[0], True)

print(f"\n{_p} passed, {_f} failed")
raise SystemExit(1 if _f else 0)
