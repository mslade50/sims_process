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


def parse_fairs_ts(s):
    """Parse a sim_fairs timestamp ('YYYY-MM-DD HH:MM:SS UTC' or ISO-8601, assumed
    UTC when naive) to epoch, or None."""
    import datetime as _dt
    if not s:
        return None
    raw = str(s).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S UTC", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = _dt.datetime.strptime(raw, fmt)
            return dt.replace(tzinfo=_dt.timezone.utc).timestamp()
        except ValueError:
            continue
    try:
        dt = _dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def check_fairs_fresh_payload(payload, now_ts, tourney=None, max_age_hours=None,
                              current_round=None):
    """Pure. (ok, reason) for a PUBLISHED sim_fairs.json payload. FAIL-CLOSED on
    every hole: the payload must exist, be for the current tourney, carry a
    parseable `sim_run_at` (stamped by publish_sim_fairs from the sim's source-file
    mtimes — `generated_at` is publish wall-clock, which any machine re-stamps
    'fresh', so it is NOT trusted), be recent, and — once a round has completed —
    be built from the round currently in play (pre-cut fairs must never price
    post-cut markets).

    `current_round` is the round in play / next to play (1..4), or None to skip
    the round gate. Intra-event (current_round >= 2) the freshness cap tightens to
    MAKER_FAIRS_MAX_AGE_LIVE_HRS (default 8h): fairs must postdate the previous
    round, not just the week."""
    if not payload:
        return (False, "no sim_fairs.json (GitHub or local) — publish the sim")
    pj = str(payload.get("tourney") or "").strip().lower()
    if tourney and pj and pj != str(tourney).strip().lower():
        return (False, f"sim_fairs is for '{pj}', not current '{tourney}' — publish the sim")
    ts = parse_fairs_ts(payload.get("sim_run_at"))
    if ts is None:
        return (False, "sim_fairs has no parseable sim_run_at (generated_at is publish "
                       "wall-clock, not sim time — untrusted) — re-publish the sim")
    intra_event = current_round is not None and int(current_round) >= 2
    if intra_event:
        prnd = payload.get("round")
        if prnd is None or int(prnd) != int(current_round):
            return (False, f"sim_fairs round={prnd} but round {int(current_round)} is in "
                           f"play — re-run round_sim + publish")
    if max_age_hours is None:
        max_age = (_envf("MAKER_FAIRS_MAX_AGE_LIVE_HRS", 8.0) if intra_event
                   else _envf("MAKER_FAIRS_MAX_AGE_HRS", 48.0))
    else:
        max_age = max_age_hours
    age_h = max(0.0, (now_ts - ts) / 3600.0)
    if age_h > max_age:
        return (False, f"sim fairs stale: sim ran {age_h:.1f}h ago > {max_age:.0f}h cap — "
                       f"re-run + publish the sim")
    return (True, f"fairs {age_h:.1f}h old (sim_run_at)")


# ── Guard #3: realized-loss circuit breaker + balance-vs-caps sanity ───────────
def _money(rec, key):
    """Dollars from a Kalshi record: prefer the `{key}_dollars` decimal-string
    field, fall back to the integer-cents `{key}` field."""
    v = rec.get(f"{key}_dollars")
    if v not in (None, ""):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(rec.get(key) or 0) / 100.0
    except (TypeError, ValueError):
        return 0.0


def _et_date(ts):
    """ET calendar date of an epoch ts (trading-day boundary)."""
    import datetime as _dt
    try:
        from zoneinfo import ZoneInfo
        return _dt.datetime.fromtimestamp(ts, ZoneInfo("America/New_York")).date()
    except Exception:
        return _dt.datetime.fromtimestamp(ts, _dt.timezone(_dt.timedelta(hours=-4))).date()


def realized_pnl_today(settlements, now_ts):
    """Pure. Realized $ PnL of settlements dated today (ET). Each Kalshi
    settlement record's PnL = revenue - (yes_total_cost + no_total_cost)."""
    import datetime as _dt
    today = _et_date(now_ts)
    pnl = 0.0
    for s in settlements or []:
        raw = s.get("settled_time") or s.get("settled_ts")
        if isinstance(raw, (int, float)):
            ts = float(raw)
        else:
            try:
                ts = _dt.datetime.fromisoformat(
                    str(raw).replace("Z", "+00:00")).timestamp()
            except (TypeError, ValueError):
                continue  # undated settlement — can't attribute to a day
        if _et_date(ts) != today:
            continue
        pnl += _money(s, "revenue") - _money(s, "yes_total_cost") - _money(s, "no_total_cost")
    return pnl


def check_daily_loss(settlements, now_ts, max_loss_usd=None):
    """Pure. (ok, reason). Halt for the rest of the ET day once realized losses
    breach the cap. Without this, settled losses free up open-exposure cap room
    and the maker can re-lose the same dollars daily forever."""
    max_loss = _envf("MAKER_MAX_DAILY_LOSS_USD", 200.0) if max_loss_usd is None else max_loss_usd
    if max_loss <= 0:
        return (True, "daily-loss breaker disabled (MAKER_MAX_DAILY_LOSS_USD<=0)")
    pnl = realized_pnl_today(settlements, now_ts)
    if pnl <= -max_loss:
        return (False, f"daily realized PnL ${pnl:+.0f} breaches -${max_loss:.0f} cap — "
                       f"halted for the day (resets at ET midnight)")
    return (True, f"daily realized PnL ${pnl:+.0f} (cap -${max_loss:.0f})")


def check_balance_caps(balance_usd, exposure_total_usd, caps=None):
    """Pure. (ok, reason). Startup sanity: the caps must be fundable. If available
    cash can't cover what the caps still allow this run, the caps are fiction —
    lower them to bankroll (or set MAKER_ALLOW_CAPS_OVER_BALANCE=1 consciously)."""
    if caps is None:
        caps = caps_from_env()
    if balance_usd is None:
        return (False, "Kalshi balance unreadable — cannot verify caps vs bankroll")
    if (os.getenv("MAKER_ALLOW_CAPS_OVER_BALANCE") or "").strip().lower() in _TRUTHY:
        return (True, f"balance ${balance_usd:.0f} (caps-over-balance override on)")
    room = max(0.0, caps["total_usd"] - float(exposure_total_usd or 0.0))
    need = min(room, caps["max_new_usd_run"])
    if balance_usd + 1e-9 < need:
        return (False, f"balance ${balance_usd:.0f} < ${need:.0f} the caps allow this run "
                       f"(total room ${room:.0f}, per-run ${caps['max_new_usd_run']:.0f}) — "
                       f"lower MAKER_CAP_* to bankroll or set MAKER_ALLOW_CAPS_OVER_BALANCE=1")
    return (True, f"balance ${balance_usd:.0f} covers ${need:.0f} allowable this run")


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


def quote_expiry(now_ts, tee_times, *, ttl_max, ttl_fallback, tee_buffer, ttl_short):
    """Pure. Native expiration_ts for a quote posted now, pinned to the schedule.

    Quotes only rest while golf is NOT being played, so let an untouched quote
    keep its queue priority for the whole quiet window: expire it just before
    the next round's first tee (minus tee_buffer), capped at ttl_max. The native
    expiry doubles as the dead-man's switch — pinned to tee-off, a dead box's
    quotes cannot survive into live play.

    - tee times unknown (not posted yet / fetch failed) -> now + ttl_fallback
      (schedule-blind: keep the leash short enough to bound overnight news risk)
    - all tee times past (round underway or sheet round stale) -> now + ttl_short
    - inside the pre-tee buffer -> now + ttl_short, hard-stopped at tee-off
    """
    if not tee_times:
        return int(now_ts + ttl_fallback)
    nxt = min((t for t in tee_times if t > now_ts), default=None)
    if nxt is None:
        return int(now_ts + ttl_short)
    exp = min(now_ts + ttl_max, nxt - tee_buffer)
    if exp <= now_ts:  # already inside the pre-tee buffer
        exp = min(now_ts + ttl_short, nxt)
    return int(exp)


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


# Mutually-exclusive outright markets: Kalshi nets collateral across players in
# the same event+market because at most N can "win" (winner=1, top_5=5, top_10=10,
# top_20=20). A basket of NO positions there can lose on AT MOST N of them, so its
# worst-case loss is the sum of the N largest — not the gross sum. We charge that
# worst-case (an upper bound) to the per-event / total caps so winner-NO baskets
# (n=1) count as ~their single largest position, while top-N still scales with N.
MUTEX_PREFIX = {"KXPGATOUR": "winner", "KXPGATOP5": "top_5",
                "KXPGATOP10": "top_10", "KXPGATOP20": "top_20"}
MUTEX_NWIN = {"winner": 1, "top_5": 5, "top_10": 10, "top_20": 20}


def market_type_of(ticker):
    """Outright market_type from a Kalshi ticker prefix (KXPGATOUR-... -> winner), else None."""
    return MUTEX_PREFIX.get(str(ticker).split("-", 1)[0])


def _topn_sum(notionals, n):
    """Worst-case loss of a mutually-exclusive NO basket = the n largest notionals
    (at most n can lose). n=None -> not mutually exclusive, sum all."""
    if n is None:
        return float(sum(notionals))
    return float(sum(sorted(notionals, reverse=True)[:max(1, int(n))]))


def _netted_event(flat_ev, groups_ev):
    """Event $: flat (non-mutex) exposure + worst-case loss of each mutex NO group."""
    total = float(flat_ev)
    for mt, by_ticker in groups_ev.items():
        total += _topn_sum(list(by_ticker.values()), MUTEX_NWIN.get(mt))
    return total


def _per_event_total(flat_event, groups):
    events = set(flat_event) | {ev for (ev, _mt) in groups}
    per_event = {}
    for ev in events:
        groups_ev = {mt: by for (e, mt), by in groups.items() if e == ev}
        per_event[ev] = _netted_event(flat_event.get(ev, 0.0), groups_ev)
    return per_event, float(sum(per_event.values()))


def build_exposure(positions, resting):
    """Committed $ by (ticker, side), netted by event, and netted total — from live
    positions (held, using Kalshi's reported per-market exposure) + resting orders.
    Mutually-exclusive NO outright baskets are netted per (event, market_type) so a
    winner-NO basket counts as ~its largest single position, not the gross sum.
    Pure given the raw API lists.

    positions: Kalshi /portfolio/positions `market_positions` dicts.
    resting:   Kalshi resting order dicts.
    """
    per_key = {}
    flat_event = {}   # event -> $ for everything except mutex-NO (yes, matchups, …)
    groups = {}       # (event, market_type) -> {ticker: notional} for NO outrights

    def add(ticker, side, usd):
        per_key[(ticker, side)] = per_key.get((ticker, side), 0.0) + usd
        ev = event_of(ticker)
        mt = market_type_of(ticker)
        if side == "no" and mt in MUTEX_NWIN:
            groups.setdefault((ev, mt), {})
            groups[(ev, mt)][ticker] = groups[(ev, mt)].get(ticker, 0.0) + usd
        else:
            flat_event[ev] = flat_event.get(ev, 0.0) + usd

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

    per_event, total = _per_event_total(flat_event, groups)
    return {"per_key": per_key, "per_event": per_event, "total": total,
            "_flat_event": flat_event, "_groups": groups}


def _brief(c):
    return {"ticker": c.get("ticker"), "side": c.get("side"),
            "post_price": c.get("post_price"), "contracts": c.get("contracts")}


def apply_exposure_caps(candidates, exposure, caps=None, own_resting=None):
    """Trim/drop candidates so held + resting + new exposure stays within caps.
    Highest-Kelly candidates get the budget first. Pure.

    `own_resting` maps (ticker, side, milli_dollars) -> resting $ for the BOT'S OWN
    script quotes. A candidate at the same exact price REPLACES (or keeps) that
    resting quote in reconcile rather than stacking on it, so its resting $ is
    netted out before the cap math. Without this the steady-state quote is charged
    twice (once in `exposure`, again as candidate spend), which halves cap room and
    — worse — makes the governor drop the re-quote, reconcile cancel the resting
    order as stale, and the next run re-post it: a cancel/repost oscillator that
    burns queue priority.

    Returns (kept, report). Each kept candidate is a shallow copy whose
    'contracts' may be reduced. report carries counts + dropped/trimmed detail.
    """
    if caps is None:
        caps = caps_from_env()
    per_key = dict(exposure.get("per_key", {}))
    flat_event = dict(exposure.get("_flat_event") or {})
    groups = {k: dict(v) for k, v in (exposure.get("_groups") or {}).items()}
    own = dict(own_resting or {})
    new_usd, n_orders = 0.0, 0

    def netted_event(ev):
        groups_ev = {mt: by for (e, mt), by in groups.items() if e == ev}
        return _netted_event(flat_event.get(ev, 0.0), groups_ev)

    def netted_total():
        evs = set(flat_event) | {e for (e, _mt) in groups}
        return float(sum(netted_event(e) for e in evs))

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
        mt = market_type_of(tk)
        is_mutex_no = (side == "no" and mt in MUTEX_NWIN)
        # Net out our own resting quote at this exact price — the candidate
        # replaces it (reconcile keeps/re-arms it), it doesn't stack on it.
        # `before` is captured FIRST so new_usd charges only dollars beyond what
        # this quote already had committed. pop() so the credit applies once.
        before = netted_total()
        offset = own.pop((tk, side, int(round(px * 1000))), 0.0)
        if offset > 0.0:
            per_key[key] = max(0.0, per_key.get(key, 0.0) - offset)
            if is_mutex_no:
                by = groups.get((ev, mt))
                if by and tk in by:
                    by[tk] = max(0.0, by[tk] - offset)
            else:
                flat_event[ev] = max(0.0, flat_event.get(ev, 0.0) - offset)
        # per-market cap is naive (per player); per-event/total use NETTED exposure
        # (mutually-exclusive NO baskets count as worst-case loss, not gross sum).
        room = min(
            caps["per_market_usd"] - per_key.get(key, 0.0),
            caps["per_event_usd"] - netted_event(ev),
            caps["total_usd"] - netted_total(),
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
        if is_mutex_no:
            groups.setdefault((ev, mt), {})
            groups[(ev, mt)][tk] = groups[(ev, mt)].get(tk, 0.0) + spend
        else:
            flat_event[ev] = flat_event.get(ev, 0.0) + spend
        # Charge the run/total budget the NETTED marginal (≈0 for a mutex-NO once
        # its basket is established, since only the largest can lose).
        new_usd += max(0.0, netted_total() - before)
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
