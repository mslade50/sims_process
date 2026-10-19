"""kalshi_preflight.py — pre-week Kalshi pickup sanity check.

Run BEFORE the week (e.g. Wednesday, after init_weekly.py / cat_dists_player.py)
so a renamed sponsor, a major's differently-tagged series, or slug/name drift is
caught while it's still fixable — not discovered as a silent empty table on Sunday.

It does three things, all against the LIVE Kalshi API:
  1. RESOLVE — confirm sim_inputs.tourney maps to exactly ONE Kalshi event by
     counting how many of OUR field appear per event (sim-player overlap, via
     kalshi_match). Flags 0-overlap (wrong slug) and low overlap (name drift).
  2. COVERAGE — for the resolved event, check each per-tournament golf series and
     report which have open markets we DON'T price (the missing-edge signal).
  3. DISCOVERY — list every open KXPGA*/golf series Kalshi offers and flag any not
     in our handled set (catches brand-new / renamed series, incl. major-specific).

Alerts (loud print + Telegram when TELEGRAM_* env set) fire on a stale
sim_inputs (configured event already settled, or not this week's DataGolf
event), a real resolution failure (winner markets we cannot read), low overlap,
or unhandled series with open markets for our event. A field whose event is
simply not listed yet is a warning and a green run only after sim_inputs is
confirmed current: Kalshi opens the weekly book after DataGolf has a field.

    python kalshi_preflight.py [tourney]
"""
import json
import os
import re
import sys
import time

import httpx
from dotenv import load_dotenv

import kalshi_match as km

load_dotenv()
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"

# Series the pricers handle today (ancillary + outrights + maker H2H).
# R1LEAD/R1TOP10 have no automated pricer (no standings_r1 fair source) but are
# priced manually later in the week — counted as handled so they don't fail the
# build daily Mon-Thu while their markets sit open.
HANDLED = {
    "KXPGATOUR", "KXPGAWIN", "KXPGATOP5", "KXPGATOP10", "KXPGATOP20",
    "KXPGAR1LEAD", "KXPGAR2LEAD", "KXPGAR3LEAD",
    "KXPGAR1TOP10", "KXPGAR2TOP10", "KXPGAR3TOP10", "KXPGAR3TOP5", "KXPGAR2TOP5",
    "KXPGAPLAYOFF", "KXPGA3BALL", "KXPGAH2H",
}
# Intentionally NOT priceable — need hole-by-hole modeling our SG sim doesn't have.
# Listed so they don't cry wolf in the unhandled-series alert every week.
SKIP = {
    "KXPGAHOLEINONE", "KXPGAEAGLE", "KXPGAROUNDBIRDIES", "KXPGAHOLE", "KXPGAHOLESCORE",
    "KXPGABIRDIES", "KXPGABOGEYFREE",
}
# Per-tournament series worth checking coverage on (one event at a time).
CANDIDATE = [
    "KXPGATOUR", "KXPGATOP5", "KXPGATOP10", "KXPGATOP20",
    "KXPGAR1LEAD", "KXPGAR2LEAD", "KXPGAR3LEAD",
    "KXPGAR1TOP10", "KXPGAR2TOP10", "KXPGAR3TOP10", "KXPGAR2TOP5", "KXPGAR3TOP5",
    "KXPGAPLAYOFF", "KXPGA3BALL", "KXPGA5BALL", "KXPGAH2H",
    "KXPGAGOLFERSCORE", "KXPGAWINNINGSCORE", "KXPGASTROKEMARGIN",
    "KXPGAROUNDBIRDIES", "KXPGAEAGLE", "KXPGAHOLEINONE",
]
# Per-group series (3-ball etc.) have a distinct event code per group, so they're
# matched by member overlap, not a single event_code.
GROUP_SERIES = {"KXPGA3BALL", "KXPGA5BALL"}
# Title carries the player (vs yes_sub_title) for these outright/winner series.
TITLE_SERIES = {"KXPGATOUR", "KXPGAWIN", "KXPGATOP5", "KXPGATOP10", "KXPGATOP20"}
# A weekly winner book covers essentially the whole field (Baycurrent 72/72,
# Bank of Utah 117/120). Masters futures cleared the old absolute floor of 10
# (20 of a 72-name file on Oct 13-16) and then overlapped the next live field
# by 0 (Oct 17-18). Half the field separates those from a real listing.
FIELD_COVERAGE_MIN = 0.5
_RESOLUTION_FAILED = (
    "RESOLUTION FAILED: no Kalshi event overlaps our field — check "
    "sim_inputs.tourney / name_replacements."
)


def _client():
    return httpx.Client(timeout=25.0, headers={"Accept": "application/json"})


def _get_open(client, series):
    out, cur = [], None
    while True:
        p = {"limit": 200, "status": "open", "series_ticker": series}
        if cur:
            p["cursor"] = cur
        r = client.get(f"{KALSHI_API}/markets", params=p)
        if r.status_code == 429:
            time.sleep(0.6)
            continue
        if r.status_code != 200:
            return out
        d = r.json()
        out += d.get("markets", [])
        cur = d.get("cursor")
        if not cur or len(d.get("markets", [])) < 200:
            break
        time.sleep(0.05)
    return out


def discover_golf_series(client):
    """All golf-ish series Kalshi lists -> {ticker: title}."""
    found = {}
    for cat in ("Sports", "Golf"):
        try:
            r = client.get(f"{KALSHI_API}/series", params={"category": cat})
            if r.status_code != 200:
                continue
            for s in (r.json().get("series") or []):
                t = s.get("ticker", "")
                if re.search(r"PGA|GOLF|LIV|LPGA", t, re.I):
                    found[t] = s.get("title", "")
        except Exception:
            pass
    return found


def _event_id_token(value):
    """Comparable event id across DataGolf's int and string JSON."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError, OverflowError):
        pass
    return text


def _empty_field(source, detail):
    return {"names": set(), "source": source, "event_id": None,
            "event_name": None, "detail": detail}


def load_field(tourney):
    """Player names plus which event they came from.

    With DATAGOLF_API_KEY set, this is DataGolf `/field-updates`: the current
    event, not the sim_inputs tourney. `tourney` is only the local-file fallback
    (`player_names_live_{tourney}.json`, then the pre-course / model CSVs) when
    that live field is missing. Oct 17-18 used the live field (120 players, tee
    times not posted). Oct 13-16 had no live field and fell back to the
    configured tourney's local file (72 names).
    """
    import pandas as pd
    api_key = os.getenv("DATAGOLF_API_KEY")
    if api_key:
        try:
            from api_utils import fetch_field_updates
            fdf = fetch_field_updates(api_key)
            if fdf is not None and "player_name" in getattr(fdf, "columns", []) and len(fdf):
                names = set(fdf["player_name"].dropna().astype(str).str.lower().str.strip())
                if names:
                    attrs = getattr(fdf, "attrs", {}) or {}
                    return {
                        "names": names,
                        "source": "datagolf_live",
                        "event_id": attrs.get("event_id"),
                        "event_name": attrs.get("event_name"),
                        "detail": "DataGolf field-updates, the current event",
                    }
        except Exception as e:
            print(f"  DataGolf field fetch failed ({e!r}); falling back to local files")
    p = f"player_names_live_{tourney}.json"
    if os.path.exists(p):
        return {"names": set(json.load(open(p))), "source": "local",
                "event_id": None, "event_name": None, "detail": p}
    for path in (f"pre_course_fit_{tourney}.csv", "model_predictions_r2.csv",
                 "model_predictions_r1.csv"):
        if os.path.exists(path):
            df = pd.read_csv(path)
            col = "player_name" if "player_name" in df.columns else (
                "my_name" if "my_name" in df.columns else None)
            if col:
                names = set(df[col].dropna().astype(str).str.lower().str.strip())
                return {"names": names, "source": "local", "event_id": None,
                        "event_name": None, "detail": path}
    return _empty_field("none", "no live field and no local file")


def format_field_source(info, tourney):
    """One log line. Says whether the names are the live week or the configured file."""
    info = info or _empty_field("none", "")
    n = len(info.get("names") or ())
    if info.get("source") == "datagolf_live":
        return (f"field source: DataGolf live current event "
                f"event_id={info.get('event_id')} name={info.get('event_name')!r} "
                f"({n} players). sim_inputs.tourney={tourney!r} is not the field "
                f"that was loaded.")
    if info.get("source") == "local":
        return (f"field source: local file {info.get('detail')} for configured "
                f"tourney {tourney!r} ({n} players). DataGolf's current-week field "
                f"was not available.")
    return f"field source: none ({n} players)."


def _telegram(text):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        return
    try:
        httpx.post(f"https://api.telegram.org/bot{token}/sendMessage",
                   json={"chat_id": chat, "text": text, "parse_mode": "HTML"}, timeout=10)
    except Exception:
        print("  Warning: Telegram failed")


def _strict_preflight():
    """KALSHI_PREFLIGHT_STRICT=1 keeps the old loud failure for a missing listing."""
    return (os.getenv("KALSHI_PREFLIGHT_STRICT") or "").strip().lower() in ("1", "true", "yes")


def _id_set(values):
    out = set()
    for value in values or []:
        if isinstance(value, dict):
            value = value.get("event_id")
        token = _event_id_token(value)
        if token is not None:
            out.add(token)
    return out


def schedule_config_status(configured_event_ids, week_events):
    """'unknown', 'off_week', 'mismatch', or 'current'.

    Unknown when the schedule lookup failed. Off week when DataGolf has no PGA
    event starting this Monday-Sunday. Mismatch when this week has an event and
    sim_inputs.event_ids does not contain it.
    """
    if week_events is None:
        return "unknown"
    if not week_events:
        return "off_week"
    live = _id_set(week_events)
    configured = _id_set(configured_event_ids)
    if live and configured and not (live & configured):
        return "mismatch"
    if live and configured and (live & configured):
        return "current"
    return "unknown"


def stale_config_reason(configured_event_ids, week_events):
    """SC-07 schedule relationship, or None when it is not a stale-config case.

    Kept for the existing guard tests. A mismatch here is a loud STALE CONFIG
    failure in classify_preflight, not a skip. KALSHI_PREFLIGHT_STRICT=1 makes
    this helper return None; it does not silence that failure.
    """
    if _strict_preflight():
        return None
    status = schedule_config_status(configured_event_ids, week_events)
    if status == "off_week":
        return "no PGA event this week (off week)"
    if status == "mismatch":
        live = sorted(_id_set(week_events))
        configured = sorted(_id_set(configured_event_ids))
        return (f"sim_inputs.event_ids {configured} is not this week's PGA event "
                f"{live} (config not rolled forward)")
    return None


def _tourney_in_text(tourney, text):
    slug = re.sub(r"[^a-z0-9]+", " ", (tourney or "").lower()).strip()
    if len(slug) < 3:
        return False
    hay = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    return f" {slug} " in f" {hay} "


def settled_tourney_reason(tourney, winner_events):
    """STALE CONFIG detail when every Kalshi book matching tourney has settled.

    winner_events items need title/sub_title/event_ticker and book='open' or
    'settled'. An open match means the configured event is still listed.
    None when the slug matches nothing or a status is unknown.
    """
    if not tourney or winner_events is None:
        return None
    matched = []
    for event in winner_events:
        blob = " ".join(str(event.get(key) or "") for key in ("title", "sub_title", "event_ticker"))
        if _tourney_in_text(tourney, blob):
            matched.append(event)
    if not matched:
        return None
    books = {str(event.get("book") or "").lower() for event in matched}
    if "open" in books or not books <= {"settled"}:
        return None
    tickers = sorted({str(event.get("event_ticker") or "") for event in matched if event.get("event_ticker")})
    shown = ", ".join(tickers) or "winner book"
    return (f"sim_inputs.tourney={tourney!r} is the settled Kalshi event {shown}. "
            "Roll sim_inputs forward; this is not a missing listing.")


def stale_config_issue(configured_event_ids, week_events, field_event_id=None,
                       tourney=None, winner_events=None):
    """Loud STALE CONFIG text, or None when config is not shown to be stale.

    Fires when the live field's event id, this week's DataGolf schedule, or a
    settled Kalshi winner book disagrees with sim_inputs. An off week alone
    does not. Strict mode does not suppress this.
    """
    bits = []
    configured = _id_set(configured_event_ids)
    field_token = _event_id_token(field_event_id)
    if field_token and configured and field_token not in configured:
        bits.append(
            f"DataGolf's live field is event {field_token}, sim_inputs.event_ids "
            f"is {sorted(configured)}")
    if schedule_config_status(configured_event_ids, week_events) == "mismatch":
        live = sorted(_id_set(week_events))
        bits.append(
            f"sim_inputs.event_ids {sorted(configured)} is not this week's PGA "
            f"event {live}")
    settled = settled_tourney_reason(tourney, winner_events)
    if settled:
        bits.append(settled)
    if not bits:
        return None
    return "STALE CONFIG: " + "; ".join(bits) + "."


def not_listed_is_benign(configured_event_ids, week_events, field_event_id=None,
                         tourney=None, winner_events=None):
    """True only when sim_inputs is confirmed current, or it is an off week and
    the configured tourney is not a settled Kalshi book."""
    if stale_config_issue(configured_event_ids, week_events, field_event_id,
                          tourney, winner_events):
        return False
    configured = _id_set(configured_event_ids)
    field_token = _event_id_token(field_event_id)
    if field_token and configured and field_token in configured:
        return True
    status = schedule_config_status(configured_event_ids, week_events)
    if status == "current":
        return True
    if status == "off_week" and winner_events is not None:
        return settled_tourney_reason(tourney, winner_events) is None
    return False


def assess_winner_resolution(markets, field, name_replacements=None, min_overlap=10):
    """Classify open winner markets against our field.

    status
      resolved    — one event covers at least half the field (and min_overlap)
      low_overlap — a few of our players match; the existing name-drift alert
      not_listed  — markets are absent, or they name players but do not cover
                    this field (another event, usually futures)
      failed      — markets are present and most of them do not name a player
                    (title/format or lookup break; Oct 1 2026 was this shape)

    Read-only: this only inspects market dicts already fetched. It does not
    place, amend, or cancel orders.
    """
    markets = list(markets or [])
    field = set(field or [])
    raw = len(markets)
    named = sum(1 for m in markets if km.player_from_market(m, kind="title"))
    _, rep = km.match_markets(
        markets, field, name_replacements, kind="title", min_overlap=min_overlap)
    leader = int(rep.get("leader") or 0)
    target = rep.get("target")
    field_n = len(field)
    coverage = (leader / field_n) if field_n else 0.0
    parse_rate = (named / raw) if raw else 1.0
    base = {
        "target": target, "leader": leader, "counts": rep.get("counts") or {},
        "raw": raw, "named": named, "coverage": coverage, "parse_rate": parse_rate,
        "reason": rep.get("reason") or "",
    }
    if field_n and leader >= min_overlap and coverage >= FIELD_COVERAGE_MIN:
        return dict(base, status="resolved", message="")
    if target and 0 < leader < min_overlap:
        return dict(base, status="low_overlap", message=(
            f"LOW OVERLAP: best event {target} has only {leader} of our "
            f"players — likely name drift; check name_replacements."
        ))
    if raw == 0 or parse_rate >= 0.5 or leader >= min_overlap:
        if raw == 0:
            message = (
                f"No open winner markets. This {field_n}-player field's Kalshi "
                "event is not listed yet."
            )
        elif leader:
            message = (
                f"best open event {target} overlaps {leader} of {field_n} field "
                f"players ({coverage:.0%}), under half the field. {named} of {raw} "
                "open winner markets name a player, so the lookup can read Kalshi, "
                "but this field's event is not listed yet."
            )
        else:
            message = (
                f"{named} of {raw} open winner markets name a player and none are "
                f"in the {field_n}-player field. This field's Kalshi event is not "
                "listed yet."
            )
        return dict(base, status="not_listed", message=message)
    return dict(base, status="failed", message=(
        _RESOLUTION_FAILED +
        f" Only {named} of {raw} open winner markets named a player, so this is "
        "a lookup or title-format break, not a missing listing."
    ))


def classify_preflight(assessment, *, configured_event_ids, week_events,
                      field_event_id=None, tourney=None, winner_events=None,
                      strict=False):
    """Issues that fail the run, and the not-listed warning if the run stays green.

    A stale sim_inputs wins over 'not listed yet'. The warning is only for a
    confirmed-current config whose Kalshi book is not up yet.
    """
    stale = stale_config_issue(
        configured_event_ids, week_events, field_event_id, tourney, winner_events)
    if stale:
        return {"issues": [stale], "warning": None}
    if not assessment:
        return {"issues": [], "warning": None}
    if assessment.get("status") == "not_listed":
        if not_listed_is_benign(configured_event_ids, week_events, field_event_id,
                                tourney, winner_events):
            return {"issues": [], "warning": assessment.get("message")}
        return {"issues": [_RESOLUTION_FAILED + " sim_inputs was not confirmed as "
                           "this week's event, so this is not treated as not listed yet."],
                "warning": None}
    issue = resolution_failure_issue(assessment, strict=strict)
    return {"issues": [issue] if issue else [], "warning": None}


def resolution_failure_issue(assessment, *, strict=False):
    """Issue string that fails the run, or None when the run stays green.

    'not listed yet' is a warning, not a failure, unless strict mode is on.
    Unreadable markets and low overlap still fail.
    """
    status = assessment.get("status")
    if status == "resolved":
        return None
    if status == "not_listed" and not strict:
        return None
    if status == "not_listed":
        return _RESOLUTION_FAILED + " " + assessment.get("message", "")
    if status == "low_overlap":
        return assessment.get("message") or "LOW OVERLAP"
    return assessment.get("message") or _RESOLUTION_FAILED


def _list_winner_events(client):
    """Read-only KXPGATOUR event list. None if the request fails."""
    found, seen, cur = [], set(), None
    try:
        for _ in range(5):
            params = {"limit": 200, "series_ticker": "KXPGATOUR"}
            if cur:
                params["cursor"] = cur
            response = client.get(f"{KALSHI_API}/events", params=params)
            if response.status_code == 429:
                time.sleep(0.6)
                continue
            if response.status_code != 200:
                return None if not found else found
            payload = response.json()
            for event in payload.get("events") or []:
                ticker = event.get("event_ticker") or ""
                if ticker and ticker not in seen:
                    seen.add(ticker)
                    found.append(event)
            cur = payload.get("cursor")
            if not cur:
                break
    except Exception as exc:
        print(f"  Kalshi event list failed ({type(exc).__name__})")
        return None if not found else found
    return found


def _event_book(client, event_ticker):
    """'open' if any winner market is open, else 'settled' if one has settled, else 'unknown'."""
    for status, book in (("open", "open"), ("settled", "settled")):
        try:
            response = client.get(f"{KALSHI_API}/markets", params={
                "limit": 1, "event_ticker": event_ticker, "status": status})
            if response.status_code != 200:
                continue
            if response.json().get("markets"):
                return book
        except Exception:
            return "unknown"
    return "unknown"


def _tourney_winner_books(client, tourney):
    """Kalshi winner events whose title matches tourney, each tagged open or settled.

    None when the event list could not be read. Read-only: events and markets GETs.
    """
    events = _list_winner_events(client)
    if events is None:
        return None
    matched = []
    for event in events:
        blob = f"{event.get('title') or ''} {event.get('sub_title') or ''}"
        if not _tourney_in_text(tourney, blob):
            continue
        ticker = event.get("event_ticker") or ""
        matched.append({
            "event_ticker": ticker,
            "title": event.get("title") or "",
            "sub_title": event.get("sub_title") or "",
            "book": _event_book(client, ticker) if ticker else "unknown",
        })
    return matched


def _week_events():
    key = os.getenv("DATAGOLF_API_KEY")
    if not key:
        return None
    try:
        from api_utils import fetch_pga_events_this_week
        return fetch_pga_events_this_week(key)
    except Exception as exc:
        print(f"  PGA schedule lookup failed ({type(exc).__name__}); not skipping")
        return None


def main():
    try:
        from sim_inputs import tourney as _slug, name_replacements
    except Exception:
        _slug, name_replacements = "", {}
    try:
        from sim_inputs import event_ids as _cfg_event_ids
    except Exception:
        _cfg_event_ids = []
    tourney = sys.argv[1] if len(sys.argv) > 1 else _slug
    client = _client()
    field_info = load_field(tourney)
    field = field_info["names"]
    print(f"=== Kalshi preflight: tourney='{tourney}', field={len(field)} players ===")
    print("  " + format_field_source(field_info, tourney))
    issues = []
    if not field:
        issues.append("No field loaded (no player_names_live / pre_course_fit / model_predictions).")
        print("  " + issues[-1])

    # 1) RESOLVE via the winner series (open well before the event).
    #    Stale sim_inputs is decided first: the live field is DataGolf's current
    #    event, so a forgotten baycurrent config can otherwise look like "not listed"
    #    or even resolve to next week's book.
    resolved = None
    not_listed = None
    week_events = _week_events()
    winner_events = _tourney_winner_books(client, tourney)
    assessment = None
    if field:
        winner = _get_open(client, "KXPGATOUR") or _get_open(client, "KXPGAWIN")
        assessment = assess_winner_resolution(
            winner, field, name_replacements, min_overlap=10)
        resolved = assessment["target"] if assessment["status"] == "resolved" else None
        print(f"\nRESOLUTION: event={resolved}  overlap={assessment['leader']}  "
              f"(raw={assessment['raw']}, named={assessment['named']})")
        print(f"  overlap by event: {assessment['counts']}")
    verdict = classify_preflight(
        assessment, configured_event_ids=_cfg_event_ids, week_events=week_events,
        field_event_id=field_info.get("event_id"), tourney=tourney,
        winner_events=winner_events, strict=_strict_preflight())
    issues.extend(verdict["issues"])
    if verdict["warning"]:
        not_listed = verdict["warning"]
        print(f"::warning title=Kalshi not listed yet::{not_listed}")
        print(f"  NOT LISTED YET: {not_listed}")
    elif any(item.startswith("STALE CONFIG:") for item in verdict["issues"]):
        resolved = None
        print("::error title=Kalshi stale config::" + verdict["issues"][0])

    # 2) COVERAGE for the resolved event + 3) DISCOVERY of unhandled series.
    print("\nCOVERAGE (open markets for our event):")
    for series in CANDIDATE:
        mk = _get_open(client, series)
        if not mk:
            continue
        if series in GROUP_SERIES:
            n = sum(1 for m in mk
                    if km.norm_name(km.player_from_market(m, "subtitle"), name_replacements, field) in field)
        elif resolved:
            n = sum(1 for m in mk if km.ticker_event_code(m.get("ticker", "")) == resolved)
        else:
            n = 0
        if n == 0:
            continue
        if series in HANDLED:
            print(f"  [OK ] {series:18s} {n} open markets for our event")
        elif series in SKIP:
            print(f"  [skip] {series:18s} {n} open (hole-level — not priceable)")
        else:
            print(f"  [UNHANDLED] {series:18s} {n} open markets for our event")
            issues.append(f"UNHANDLED series with {n} open markets for our event: {series}")

    discovered = discover_golf_series(client)
    new_series = sorted(t for t in discovered if t not in HANDLED and t not in CANDIDATE
                        and t not in SKIP and len(t) > 6 and t.startswith("KXPGA"))
    if new_series:
        print(f"\nDISCOVERY: {len(new_series)} golf series not in handled/candidate lists "
              f"(review if any are per-tournament): {', '.join(new_series[:20])}")

    # ---- Alerts ----
    print("\n" + "=" * 60)
    if issues:
        print(f"  {len(issues)} ISSUE(S):")
        for i in issues:
            print("   - " + i)
        lines = [f"<b>Kalshi Preflight — {tourney}</b>", ""] + [f"• {i}" for i in issues]
        _telegram("\n".join(lines))
        print("  (Telegram alert sent if TELEGRAM_* configured)")
        return 1
    if not_listed:
        print("  NO FAILURE — winner event is not listed yet. Run stays green; "
              "a real lookup or title-format break still fails.")
        return 0
    print(f"  ALL CLEAR — resolved to {resolved}, all open per-event series handled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
