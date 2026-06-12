"""kalshi_preflight.py — pre-week Kalshi pickup sanity check.

Run BEFORE the week (e.g. Wednesday, after init_weekly.py / cat_dists_player.py)
so a renamed sponsor, a major's differently-tagged series, or slug/name drift is
caught while it's still fixable — not discovered as a silent empty table on Sunday.

It does three things, all against the LIVE Kalshi API:
  1. RESOLVE — confirm sim_inputs.tourney maps to exactly ONE Kalshi event by
     counting how many of OUR field appear per event (sim-player overlap, via
     kalshi_match). Flags 0-overlap (wrong slug / name drift) and ambiguous ties.
  2. COVERAGE — for the resolved event, check each per-tournament golf series and
     report which have open markets we DON'T price (the missing-edge signal).
  3. DISCOVERY — list every open KXPGA*/golf series Kalshi offers and flag any not
     in our handled set (catches brand-new / renamed series, incl. major-specific).

Alerts (loud print + Telegram when TELEGRAM_* env set) fire on resolution failure,
low/ambiguous overlap, or unhandled series with open markets for our event.

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
HANDLED = {
    "KXPGATOUR", "KXPGAWIN", "KXPGATOP5", "KXPGATOP10", "KXPGATOP20",
    "KXPGAR2LEAD", "KXPGAR3LEAD", "KXPGAR2TOP10", "KXPGAR3TOP10", "KXPGAR3TOP5", "KXPGAR2TOP5",
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


def load_field(tourney):
    """Field player_names (sim 'last, first' lowercase). Prefer the LIVE DataGolf
    field — authoritative for the CURRENT event, so a CI run never resolves a stale
    week — then fall back to committed local files for offline/dev use."""
    import pandas as pd
    api_key = os.getenv("DATAGOLF_API_KEY")
    if api_key:
        try:
            from api_utils import fetch_field_updates
            fdf = fetch_field_updates(api_key)
            if fdf is not None and "player_name" in getattr(fdf, "columns", []) and len(fdf):
                names = set(fdf["player_name"].dropna().astype(str).str.lower().str.strip())
                if names:
                    print(f"  field source: DataGolf live ({len(names)} players)")
                    return names
        except Exception as e:
            print(f"  DataGolf field fetch failed ({e!r}); falling back to local files")
    p = f"player_names_live_{tourney}.json"
    if os.path.exists(p):
        return set(json.load(open(p)))
    for path in (f"pre_course_fit_{tourney}.csv", "model_predictions_r2.csv",
                 "model_predictions_r1.csv"):
        if os.path.exists(path):
            df = pd.read_csv(path)
            col = "player_name" if "player_name" in df.columns else (
                "my_name" if "my_name" in df.columns else None)
            if col:
                return set(df[col].dropna().astype(str).str.lower().str.strip())
    return set()


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


def main():
    try:
        from sim_inputs import tourney as _slug, name_replacements
    except Exception:
        _slug, name_replacements = "", {}
    tourney = sys.argv[1] if len(sys.argv) > 1 else _slug
    client = _client()
    field = load_field(tourney)
    print(f"=== Kalshi preflight: tourney='{tourney}', field={len(field)} players ===")
    issues = []
    if not field:
        issues.append("No field loaded (no player_names_live / pre_course_fit / model_predictions).")
        print("  " + issues[-1])

    # 1) RESOLVE via the winner series (open well before the event).
    resolved = None
    if field:
        winner = _get_open(client, "KXPGATOUR") or _get_open(client, "KXPGAWIN")
        _, rep = km.match_markets(winner, field, name_replacements, kind="title", min_overlap=10)
        resolved = rep["target"]
        print(f"\nRESOLUTION: event={resolved}  overlap={rep['leader']}  (raw={rep['raw']})")
        print(f"  overlap by event: {rep['counts']}")
        if resolved is None:
            issues.append("RESOLUTION FAILED: no Kalshi event overlaps our field — check "
                          "sim_inputs.tourney / name_replacements.")
        elif rep["leader"] < 10:
            issues.append(f"LOW OVERLAP: best event {resolved} has only {rep['leader']} of our "
                          f"players — likely name drift; check name_replacements.")
        elif rep["ambiguous"]:
            issues.append(f"AMBIGUOUS: {resolved} vs a near-tie event ({rep['counts']}).")

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
    print(f"  ALL CLEAR — resolved to {resolved}, all open per-event series handled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
