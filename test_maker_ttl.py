"""Unit tests for the maker's rolling-TTL dead-man's switch in
reconcile_and_post. Fully offline: list_resting_orders / cancel_order /
post_limit are monkeypatched to record calls — NOTHING touches Kalshi.
Run: python test_maker_ttl.py"""
import os
import time

# Deterministic TTL window before importing (env is read at import time).
os.environ["MAKER_QUOTE_TTL_SEC"] = "600"
os.environ["MAKER_QUOTE_REFRESH_SEC"] = "300"

import kalshi_maker as km

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


TICK = "KXPGATOUR-26XYZ-ABC"
NOW = int(time.time())

# Schedule-aware expiry: pin the round + tee times so reconcile is offline.
# Next tee 8h out, buffer 900s -> every post expires at exactly TEE - 900.
TEE = NOW + 8 * 3600
km._current_play_round = lambda: 1
km._teetimes_memo[1] = [TEE]


def _resting(price, cid, order_id, exp=None):
    """Build a fake YES resting order at `price` dollars. cid='' => manual."""
    o = {
        "ticker": TICK,
        "side": "yes",
        "yes_price_dollars": f"{price:.4f}",
        "order_id": order_id,
        "client_order_id": cid,
    }
    if exp is not None:
        o["expiration_ts"] = exp
    return o


def _cand(price):
    return {"ticker": TICK, "side": "yes", "post_price": price, "contracts": 33}


# ── Scenario: one reconcile pass over a mixed book ──────────────────────────────
# 0.30 new (no resting)         -> post, deterministic id
# 0.40 script, healthy TTL      -> keep (no cancel, no post)
# 0.50 script, expiring soon    -> re-arm (cancel + fresh-id post)
# 0.60 script, legacy no-TTL    -> re-arm (cancel + fresh-id post)
# 0.70 manual hand-click        -> skip (never cancel, never stack)
# 0.80 script, NOT a candidate  -> stale cancel (existing behavior), no post
resting = [
    _resting(0.40, "smk_c2", "oid_c2", exp=NOW + 36000),  # healthy
    _resting(0.50, "smk_c3", "oid_c3", exp=NOW + 60),      # expiring
    _resting(0.60, "smk_c4", "oid_c4"),                    # no expiration_ts
    _resting(0.70, "", "oid_manual", exp=NOW + 36000),     # manual
    _resting(0.80, "smk_s6", "oid_s6", exp=NOW + 36000),   # stale (not a candidate)
]
candidates = [_cand(0.30), _cand(0.40), _cand(0.50), _cand(0.60), _cand(0.70)]

posts, cancels = [], []
km.list_resting_orders = lambda golf_only=True: list(resting)
km.cancel_order = lambda order_id, ticker, reason="": (cancels.append(order_id) or True)
km.post_limit = (
    lambda ticker, side, price_dollars, count, expiration_ts=None, client_order_id=None:
    (posts.append({"price": round(price_dollars, 4), "exp": expiration_ts,
                   "cid": client_order_id}) or (True, "fake_oid"))
)

km.reconcile_and_post(candidates)

posted_prices = sorted(p["price"] for p in posts)
by_price = {p["price"]: p for p in posts}

# Which prices were posted: new(0.30) + two re-arms(0.50, 0.60). NOT kept(0.40) or manual(0.70).
eq("posted exactly the new + re-armed prices", posted_prices, [0.30, 0.50, 0.60])
eq("healthy quote not re-posted", 0.40 in by_price, False)
eq("manual price not stacked", 0.70 in by_price, False)

# Cancels: stale 0.80 (step 1) + the two re-arms. Manual + healthy untouched.
eq("cancelled stale + both re-arms", sorted(cancels), ["oid_c3", "oid_c4", "oid_s6"])
eq("manual never cancelled", "oid_manual" in cancels, False)
eq("healthy never cancelled", "oid_c2" in cancels, False)

# Every post carries the schedule-pinned native expiration: next tee - buffer.
for pr, p in by_price.items():
    truthy(f"expiration set on {pr}", p["exp"] is not None)
    eq(f"expiration pinned to tee-buffer on {pr}", p["exp"], TEE - 900)

# New post keeps the deterministic id (None override); re-arms get a fresh,
# still-script-prefixed id so the re-post is well-defined after the cancel.
eq("new post uses deterministic id", by_price[0.30]["cid"], None)
for pr in (0.50, 0.60):
    cid = by_price[pr]["cid"]
    truthy(f"re-arm {pr} has an explicit id", cid is not None)
    truthy(f"re-arm {pr} id is script-recognizable",
           str(cid or "").startswith(km.SCRIPT_ORDER_PREFIX))

# ── Scenario 2: expiry pinned at an imminent tee — expiring quote is KEPT, not
# re-armed (re-arming can't extend it; it lapses on schedule before golf). ──────
km._teetimes_memo[1] = [NOW + 300]          # tee 5 min out (inside the 900s buffer)
resting2 = [_resting(0.45, "smk_p1", "oid_p1", exp=NOW + 250)]  # needs refresh
posts2, cancels2 = [], []
km.list_resting_orders = lambda golf_only=True: list(resting2)
km.cancel_order = lambda order_id, ticker, reason="": (cancels2.append(order_id) or True)
km.post_limit = (
    lambda ticker, side, price_dollars, count, expiration_ts=None, client_order_id=None:
    (posts2.append(round(price_dollars, 4)) or (True, "fake_oid"))
)
r2 = km.reconcile_and_post([_cand(0.45)])
eq("pinned quote not cancelled", cancels2, [])
eq("pinned quote not re-posted", posts2, [])
eq("pinned quote counted as kept", r2["kept"], 1)
km._teetimes_memo[1] = [TEE]  # restore

# ── Pure-helper checks: maker_guard.quote_expiry ────────────────────────────────
KW = dict(ttl_max=43200, ttl_fallback=3600, tee_buffer=900, ttl_short=600)
gq = km.maker_guard.quote_expiry
eq("expiry: schedule blind -> fallback", gq(NOW, [], **KW), NOW + 3600)
eq("expiry: quiet window -> tee - buffer", gq(NOW, [TEE], **KW), TEE - 900)
eq("expiry: far tee capped at ttl_max", gq(NOW, [NOW + 86400], **KW), NOW + 43200)
eq("expiry: inside buffer -> short leash, hard-stop at tee",
   gq(NOW, [NOW + 300], **KW), NOW + 300)
eq("expiry: inside buffer, tee past short -> short leash",
   gq(NOW, [NOW + 850], **KW), NOW + 600)
eq("expiry: all tees past -> short leash", gq(NOW, [NOW - 7200], **KW), NOW + 600)
eq("expiry: picks next future tee", gq(NOW, [NOW - 7200, TEE], **KW), TEE - 900)

# ── Pure-helper checks ──────────────────────────────────────────────────────────
eq("no-expiration => needs refresh", km._needs_refresh({}, NOW), True)
eq("expiring-soon => needs refresh", km._needs_refresh({"expiration_ts": NOW + 100}, NOW), True)
eq("healthy => no refresh", km._needs_refresh({"expiration_ts": NOW + 4000}, NOW), False)
eq("expiration parse", km._order_expiration_ts({"expiration_ts": "123"}), 123)
eq("expiration absent", km._order_expiration_ts({}), None)
eq("expiration garbage", km._order_expiration_ts({"expiration_ts": "x"}), None)
# Re-arm id stays inside Kalshi's 32-char client_order_id budget.
eq("re-arm id length ok",
   len(km._rearm_client_order_id(TICK, "yes", 0.5, NOW)) <= 32, True)

print(f"\n{_p} passed, {_f} failed")
raise SystemExit(1 if _f else 0)
