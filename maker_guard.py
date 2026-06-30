"""Safety layer for the automated maker — kill switch + exposure governor.

Both are checked BEFORE any order is posted; neither sends orders. The decision
logic and the cap math are pure (no I/O) so they're unit-tested in
test_maker_guard.py.

KILL SWITCH — any one of these halts `--live` posting:
  * env  MAKER_KILL          truthy  (CI / hard override; always readable)
  * file permanent_data/MAKER_HALT   (local panic button; always readable)
  * Google Sheet round_config `maker_enabled` explicitly falsy (phone toggle).
    Best-effort: an ABSENT row or an unreadable sheet does NOT halt — so a sheet
    outage or an un-migrated sheet never trips it, and the env/file layers stay
    the reliable kill. Set `maker_enabled` to no/0/false to stop from your phone.
A halt also PULLS the bot's own resting (script-placed) quotes to remove live
risk; manual hand-clicks are never touched (that's still `--cancel-all`).

EXPOSURE GOVERNOR — trims/drops candidates before posting so that
held + resting + new exposure stays within (env-overridable) caps:
  * MAKER_CAP_MARKET_USD       per (ticker, side)   default $50
  * MAKER_CAP_EVENT_USD        per event_code       default $400
  * MAKER_CAP_TOTAL_USD        all golf             default $1000
  * MAKER_MAX_NEW_USD_PER_RUN  new $ this run       default $300
  * MAKER_MAX_ORDERS_PER_RUN   new orders this run  default 40
Because the per-(ticker, side) cap counts CURRENT held + resting exposure, it
doubles as inventory control: once a side is at its cap, the governor won't add
to it. Highest-Kelly candidates get the budget first.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

HALT_FILE = Path("permanent_data") / "MAKER_HALT"
_TRUTHY = {"1", "true", "yes", "on", "y", "t"}
_FALSY = {"0", "false", "no", "off", "n", "f"}


# ── config ────────────────────────────────────────────────────────────────────
def _envf(name, default):
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def caps_from_env():
    """Current cap config (env-overridable). Conservative defaults."""
    return {
        "per_market_usd": _envf("MAKER_CAP_MARKET_USD", 50.0),
        "per_event_usd": _envf("MAKER_CAP_EVENT_USD", 400.0),
        "total_usd": _envf("MAKER_CAP_TOTAL_USD", 1000.0),
        "max_new_usd_run": _envf("MAKER_MAX_NEW_USD_PER_RUN", 300.0),
        "max_orders_run": int(_envf("MAKER_MAX_ORDERS_PER_RUN", 40)),
    }


# ── kill switch ────────────────────────────────────────────────────────────────
def env_or_file_kill():
    """Hard kill from always-readable sources. Returns a reason str or None."""
    if (os.getenv("MAKER_KILL") or "").strip().lower() in _TRUTHY:
        return "MAKER_KILL env set"
    if HALT_FILE.exists():
        return f"halt file present: {HALT_FILE}"
    return None


def sheet_disabled(get_param=None):
    """Best-effort phone toggle. Returns a reason str if the sheet EXPLICITLY
    says maker_enabled is off; None if enabled, absent, blank, or unreadable."""
    if get_param is None:
        try:
            from sheet_config import get_param as _gp
            get_param = _gp
        except Exception:
            return None
    try:
        raw = get_param("maker_enabled", default=None)
    except Exception as e:  # sheet/network failure — don't block on it
        print(f"[guard] sheet 'maker_enabled' unreadable ({e}); relying on env/file kill")
        return None
    if raw is None or str(raw).strip() == "":
        return None  # row absent/blank -> not configured, don't block
    if str(raw).strip().lower() in _FALSY:
        return "sheet maker_enabled is OFF"
    return None


def should_trade(get_param=None):
    """KILL-SWITCH-ONLY check. (ok, reason). The full precondition set (kill +
    fairs-fresh + not-live) is assembled by the maker via check_preconditions(),
    which calls this plus the pure guards below with live I/O."""
    r = env_or_file_kill()
    if r:
        return (False, r)
    r = sheet_disabled(get_param)
    if r:
        return (False, r)
    return (True, "enabled")


# ── Guard #1: sim-fairs freshness (fail-closed) ────────────────────────────────
def check_fairs_fresh(path, mtime, now_ts, max_age_hours=None):
    """Pure. (ok, reason). Refuse to quote unless the sim's fair file exists and
    is recent. The file name is tourney-specific (rank_probs_*_{tourney}), so a
    present-and-fresh file is implicitly for the current event."""
    if not path or mtime is None:
        return (False, "no sim fair file found — run the sim")
    max_age = _envf("MAKER_FAIRS_MAX_AGE_HRS", 48.0) if max_age_hours is None else max_age_hours
    age_h = max(0.0, (now_ts - mtime) / 3600.0)
    if age_h > max_age:
        return (False, f"sim fairs stale: {age_h:.1f}h old > {max_age:.0f}h cap — re-run the sim")
    return (True, f"fairs {age_h:.1f}h old")


def active_fair_file(tourney):
    """I/O. (path, mtime) of the freshest tourney fair file, else (None, None).
    Matches load_sim_probs(): live (post-round) preferred, then pre-event."""
    for name in (f"rank_probs_live_{tourney}.parquet", f"rank_probs_updated_{tourney}.parquet"):
        if os.path.exists(name):
            return (name, os.path.getmtime(name))
    return (None, None)


# ── Guard #2: live-round detection (schedule default + DataGolf override) ───────
def datagolf_live(last_updated_ts, now_ts):
    """Pure. live / not-live / unknown from the DataGolf live feed's last_updated.
      True  : feed fresh (<= MAKER_LIVE_FRESH_MIN) -> a round is actively scoring.
      False : feed CONFIDENTLY stale (>= MAKER_LIVE_STALE_MIN) -> not live.
      None  : in-between / unparseable -> unknown (caller falls back to schedule).
    Asymmetric on purpose: quick to call 'live' (halt), cautious to call 'not
    live' (resume), so a brief feed hiccup never resumes us mid-round."""
    if last_updated_ts is None:
        return None
    age_min = (now_ts - last_updated_ts) / 60.0
    if age_min < 0:
        return True  # clock/timezone skew -> assume live (safe)
    if age_min <= _envf("MAKER_LIVE_FRESH_MIN", 30.0):
        return True
    if age_min >= _envf("MAKER_LIVE_STALE_MIN", 120.0):
        return False
    return None


def schedule_live(first_tee_ts, last_tee_ts, now_ts, round_hours=None):
    """Pure. Is the current round live per the tee-time schedule? Live window =
    [first tee, last tee + round_hours]. None if tee times are unknown."""
    if first_tee_ts is None or last_tee_ts is None:
        return None
    rh = _envf("MAKER_ROUND_HOURS", 6.0) if round_hours is None else round_hours
    return (now_ts >= first_tee_ts) and (now_ts <= last_tee_ts + rh * 3600.0)


# ── Pre-Wednesday rule: no YES outrights before field lock (withdrawal risk) ───
OUTRIGHT_MARKETS = {"winner", "top_5", "top_10", "top_20"}


def before_wed_cutoff(now=None, cutoff_hour=16):
    """True if it's before THIS week's Wednesday at cutoff_hour ET. A player can
    withdraw before the field locks, so a YES outright bought pre-Wednesday carries
    WD risk; NO doesn't (a WD only helps NO). `now` may be passed for tests."""
    import datetime as _dt
    if now is None:
        try:
            from zoneinfo import ZoneInfo
            now = _dt.datetime.now(ZoneInfo("America/New_York"))
        except Exception:
            now = _dt.datetime.now()  # fall back to local time if no tz database
    days_to_wed = 2 - now.weekday()  # Mon=0 .. Sun=6, Wed=2
    wed = (now + _dt.timedelta(days=days_to_wed)).replace(
        hour=int(cutoff_hour), minute=0, second=0, microsecond=0)
    return now < wed


def block_yes_outright(side, market_type, now=None, rule_enabled=True, cutoff_hour=16):
    """Should this candidate be skipped by the pre-Wednesday rule? True only for a
    YES OUTRIGHT quote, while the rule is enabled and it's before Wed cutoff."""
    if not rule_enabled:
        return False
    if side != "yes" or market_type not in OUTRIGHT_MARKETS:
        return False
    return before_wed_cutoff(now=now, cutoff_hour=cutoff_hour)


def resolve_live(sched, dg):
    """Pure. (is_live, reason). DataGolf wins on a confident signal; the schedule
    is the default/fallback; if both are blind, fail closed (assume live)."""
    if dg is not None:
        if sched is not None and bool(dg) != bool(sched):
            return (bool(dg), f"DataGolf override: schedule={sched} datagolf={dg} -> trust DataGolf (fix the Sheet)")
        return (bool(dg), f"datagolf live={dg}")
    if sched is not None:
        return (bool(sched), f"schedule live={sched} (datagolf unavailable)")
    return (True, "live status unknown (schedule + datagolf both blind) -> fail-closed: assume live")


def pull_script_quotes(list_resting, is_script, cancel_order):
    """On halt, cancel the bot's own resting (script) quotes; leave manual
    orders alone. Injected callables keep this testable. Returns count cancelled."""
    n = 0
    for o in list_resting():
        if not is_script(o):
            continue
        try:
            if cancel_order(o.get("order_id"), o.get("ticker", ""), reason="HALT"):
                n += 1
        except Exception as e:
            print(f"[guard] cancel failed for {o.get('order_id')}: {e}")
    return n


# ── exposure governor ──────────────────────────────────────────────────────────
def event_of(ticker):
    parts = str(ticker).split("-", 2)
    return parts[1] if len(parts) >= 2 else ""


def _price_to_dollars(raw):
    try:
        px = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return px / 100.0 if px > 1 else px


def build_exposure(positions, resting):
    """Committed $ by (ticker, side), by event_code, and total — from live
    positions (held) + resting orders (working). Pure given the raw API lists.

    positions: Kalshi /portfolio/positions `market_positions` dicts.
    resting:   Kalshi resting order dicts.
    """
    per_key, per_event, total = {}, {}, 0.0

    def add(ticker, side, usd):
        nonlocal total
        per_key[(ticker, side)] = per_key.get((ticker, side), 0.0) + usd
        ev = event_of(ticker)
        per_event[ev] = per_event.get(ev, 0.0) + usd
        total += usd

    for p in positions or []:
        tk = p.get("ticker", "")
        try:
            pf = float(p.get("position_fp") or p.get("position") or 0)
        except (TypeError, ValueError):
            pf = 0.0
        if abs(pf) < 1e-9:
            continue
        try:
            exp = float(p.get("market_exposure_dollars") or p.get("market_exposure") or 0)
        except (TypeError, ValueError):
            exp = 0.0
        add(tk, "yes" if pf > 0 else "no", exp)

    for o in resting or []:
        tk = o.get("ticker", "")
        side = o.get("side", "")
        try:
            rem = float(o.get("remaining_count_fp") or o.get("remaining_count") or 0)
        except (TypeError, ValueError):
            rem = 0.0
        px = _price_to_dollars(o.get(f"{side}_price_dollars") or o.get(f"{side}_price") or 0)
        add(tk, side, rem * px)

    return {"per_key": per_key, "per_event": per_event, "total": total}


def _brief(c):
    return {"ticker": c.get("ticker"), "side": c.get("side"),
            "post_price": c.get("post_price"), "contracts": c.get("contracts")}


def apply_exposure_caps(candidates, exposure, caps=None):
    """Trim/drop candidates so held + resting + new exposure stays within caps.
    Highest-Kelly candidates get the budget first. Pure.

    Returns (kept, report). Each kept candidate is a shallow copy whose
    'contracts' may be reduced. report carries counts + dropped/trimmed detail.
    """
    if caps is None:
        caps = caps_from_env()
    per_key = dict(exposure.get("per_key", {}))
    per_event = dict(exposure.get("per_event", {}))
    total = float(exposure.get("total", 0.0))
    new_usd, n_orders = 0.0, 0

    ranked = sorted(candidates, key=lambda c: c.get("kelly_f", 0) or 0, reverse=True)
    kept, dropped, trimmed = [], [], []

    for c in ranked:
        tk, side = c["ticker"], c["side"]
        px = float(c["post_price"])
        want = int(c.get("contracts", 0) or 0)
        if want < 1 or px <= 0:
            dropped.append({**_brief(c), "why": "zero qty/price"})
            continue
        if n_orders >= caps["max_orders_run"]:
            dropped.append({**_brief(c), "why": "max orders/run"})
            continue
        key, ev = (tk, side), event_of(tk)
        room = min(
            caps["per_market_usd"] - per_key.get(key, 0.0),
            caps["per_event_usd"] - per_event.get(ev, 0.0),
            caps["total_usd"] - total,
            caps["max_new_usd_run"] - new_usd,
        )
        if room <= 0:
            dropped.append({**_brief(c), "why": "cap reached"})
            continue
        allowed = min(want, int(math.floor(room / px)))
        if allowed < 1:
            dropped.append({**_brief(c), "why": "no room for 1 contract"})
            continue
        spend = allowed * px
        per_key[key] = per_key.get(key, 0.0) + spend
        per_event[ev] = per_event.get(ev, 0.0) + spend
        total += spend
        new_usd += spend
        n_orders += 1
        cc = dict(c)
        cc["contracts"] = allowed
        kept.append(cc)
        if allowed < want:
            trimmed.append({**_brief(cc), "from": want, "to": allowed})

    report = {
        "kept": len(kept), "dropped": len(dropped), "trimmed": len(trimmed),
        "new_usd": round(new_usd, 2), "orders": n_orders,
        "dropped_detail": dropped, "trimmed_detail": trimmed, "caps": caps,
    }
    return kept, report
