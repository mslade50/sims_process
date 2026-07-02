"""Kalshi maker — phase 2.

Reads the most recent sim's persisted probability outputs, re-fetches Kalshi
live, and either dry-runs (default) or posts resting limit orders.

Inclusion gates (per side, per ladder rung). Raw-edge floor is tiered by
post price — at extreme favorite prices the structural EV/$ ceiling is
so low that even 1pp is meaningful, while at long odds we tolerate
smaller raw edges because the leverage on a fill is large. EV-per-dollar
gate kicks in below 0.60 to filter rungs whose raw edge looks acceptable
but whose return per dollar staked is thin.
    post_price > 0.98:   raw_edge >= 1.0 pp
    0.45 - 0.98:         raw_edge >= 1.5 pp
    0.30 - 0.45:         raw_edge >= 1.0 pp
    0.15 - 0.30:         raw_edge >= 0.5 pp
    post_price < 0.15:   raw_edge >= 0.15 pp
    post_price < 0.60 (in addition to above): kelly_ev_per_$ >= 5%

Fill-probability filter: rungs are also cut if estimated fill probability
falls below MIN_FILL_PCT (20%). Estimate drops 5pp per 1c the post is
below the ask, +/-5pp by market lifetime volume tier (<1k / 1k-10k / >10k).
Surviving rungs absorb the dropped budget via kelly_solver's even split.

Outright quoting (default) = the passive-accumulation engine (quote_engine.py via
maker_quotes.plan_market): one working order per market on the side the model
favors, pegged near the touch, capped at fair minus the side's min edge (YES 0.5c
/ NO 0.3c), never crossing, worked in iceberg slices. Legacy static edge rungs are
available via --rungs. H2H matchups still use scan_matchups (5pp gate).

Modes:
    (default)        scan + print intents only
    --live           reconcile + auto-cancel stale + POST new intents
    --rungs          use the legacy edge-rung ladder for outrights (not the engine)
    --cancel-all     cancel every resting order owned by our key, then exit
    --cancel-script  cancel only the bot's script-placed orders (client_order_id
                     prefix); manual hand-clicked orders are left resting

Automation safety layer (maker_guard.py) — applies to --live AND the dry-run:
    KILL SWITCH (any halts --live; a halt also pulls the bot's own quotes):
        env  MAKER_KILL=1               hard override (CI / GitHub Actions var)
        file permanent_data/MAKER_HALT  local panic button
        sheet round_config 'maker_enabled' = no/0/false  (phone toggle; an
            absent row or unreadable sheet does NOT halt — env/file are the
            reliable kill)
    EXPOSURE GOVERNOR (trims/drops candidates; counts held + resting so the
    per-(ticker,side) cap also bounds inventory; env-overridable):
        MAKER_CAP_MARKET_USD=50  MAKER_CAP_EVENT_USD=400  MAKER_CAP_TOTAL_USD=1000
        MAKER_MAX_NEW_USD_PER_RUN=300  MAKER_MAX_ORDERS_PER_RUN=40

Safety invariants (enforced in code, not just convention):
    1. post_price is clamped to be STRICTLY below the opposite best ask, in
       1¢ ticks (no maker → taker math bug).
    2. Before each POST, the orderbook is re-fetched and the assert is run
       AGAIN against the live ask, so a tick between scan and post can't
       turn an order into a cross.
    3. client_order_id is a deterministic hash of (ticker, side, price), so
       Kalshi rejects duplicates if our reconciliation misses one.
"""
from __future__ import annotations

import argparse
import datetime as _dtmod
import hashlib
import json as _json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dotenv import load_dotenv

import httpx
import numpy as np
import pandas as pd

from sim_inputs import tourney, name_replacements
import maker_guard

load_dotenv()

# ── Gates ──────────────────────────────────────────────────────────────
# Tiered raw-edge floor by post price. See _raw_edge_floor() below for the
# table. Extreme favorites have a tiny EV/$ ceiling so we accept 1pp; mid-range
# bets need 1.5pp; long-odd rungs tolerate smaller raw edges because the
# leverage on a fill is large.
HIGH_PRICE_THRESHOLD   = 0.98   # post > this  -> 1.0pp raw
TIER_DEFAULT_LOW       = 0.45   # 0.45 - 0.98  -> 1.5pp raw
TIER_LOW_PRICE         = 0.30   # 0.30 - 0.45  -> 1.0pp raw
TIER_LONGSHOT          = 0.15   # 0.15 - 0.30  -> 0.5pp raw
                                # post < 0.15  -> 0.15pp raw
# EV gate kicks in below this price (in addition to whatever raw floor applies).
LOW_PRICE_THRESHOLD    = 0.60
MIN_KELLY_EV_LOW_PRICE = 0.05   # 5% EV/$ required when post_price < 0.60


def _raw_edge_floor(post_price):
    """Return the minimum raw edge (in pp) required for this post price."""
    if post_price > HIGH_PRICE_THRESHOLD:
        return 1.0
    if post_price >= TIER_DEFAULT_LOW:
        return 1.5
    if post_price >= TIER_LOW_PRICE:
        return 1.0
    if post_price >= TIER_LONGSHOT:
        return 0.5
    return 0.15


# ── Fill-probability filter ────────────────────────────────────────────
# Posting a tight maker bid 16c below the ask captures big edge per fill
# but rarely fills. Estimate fill probability based on distance from ask
# and market lifetime volume; cut rungs below MIN_FILL_PCT. The surviving
# rungs absorb the cut budget naturally because kelly_solver splits each
# bet's $ target evenly across (ticker, side) entries — fewer rungs ->
# bigger contracts per surviving rung.
MIN_FILL_PCT = 20.0


def _estimate_fill_pct(post_price, best_ask, volume):
    """Heuristic fill probability (0-100). Drops 5pp per 1c the post sits
    below the ask, then +/-5pp based on lifetime volume tier."""
    distance_cents = (best_ask - post_price) * 100.0
    drop = distance_cents * 5.0  # 5pp per 1c below ask
    base = 100.0
    if volume < 1000:
        base -= 5.0
    elif volume > 10000:
        base += 5.0
    return max(0.0, min(100.0, base - drop))

# Per-level allocation (3 levels per market+side: near_ask, mid, bid).
# Outrights stay flat-33 until the portfolio-Kelly solver (kelly_solver.py)
# is wired in. Matchups stay flat-500 regardless — they're out of scope
# for the Kelly solver per design discussion.
LEVEL_CONTRACTS = 33               # outright: top_5/top_10/top_20/winner
MATCHUP_LEVEL_CONTRACTS = 500      # H2H matchups
MATCHUP_MIN_EDGE_PP = 5.0          # matchup-specific raw-edge gate (no Kelly gate)
MATCHUP_SERIES = "KXPGAH2H"

# Pre-tournament fallback for matchup final_scores. Off by default — mid-event
# we want to refuse stale pre-tournament probs rather than silently mispricing.
# Set True via --allow-pre-matchup when running the maker before R1 begins.
_allow_pre_matchup_fallback = False

# Tick handling: Kalshi markets carry a `price_ranges` field with per-price
# tick sizes (e.g. winner markets use 0.1¢ ticks above 90¢ and below 10¢, but
# 1¢ ticks in the middle). All tick math is per-market — never a global
# default — so we don't lose precision near the bounds.
DEFAULT_TICK = 0.01        # fallback only if a market doesn't report ranges


def _env_int(name, default):
    """Parse an int env override; fall back to default on unset/garbage."""
    try:
        v = os.getenv(name)
        return int(v) if v not in (None, "") else default
    except (TypeError, ValueError):
        return default


# ── Rolling quote TTL (dead-man's switch) ──────────────────────────────
# Every live maker quote is posted with a native Kalshi expiration_ts of
# now + QUOTE_TTL_SEC, and re-armed each reconcile cycle once its remaining
# life drops below QUOTE_REFRESH_SEC. If this process dies (crash, dropped
# network, VPS reboot, kill -9 mid-deploy), every resting quote self-cancels
# within QUOTE_TTL_SEC — with NO dependence on the reconcile loop, the cron
# Worker, or anything else staying up. Manual hand-clicks never get a TTL here.
#
# INVARIANT: run the maker MORE OFTEN than QUOTE_REFRESH_SEC, or quotes will
# lapse between normal cycles. A lapse is fail-safe (no quote is safe) but it
# defeats the maker. With the defaults below, schedule the live loop to run at
# least every ~4 min; on a crash, all exposure is pulled within 10 min.
QUOTE_TTL_SEC     = _env_int("MAKER_QUOTE_TTL_SEC", 600)      # native expiry: 10 min
QUOTE_REFRESH_SEC = _env_int("MAKER_QUOTE_REFRESH_SEC", 300)  # re-arm when < 5 min left

# ── Kalshi public API (no auth) ────────────────────────────────────────
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
OUTRIGHT_SERIES = {
    "KXPGATOP5": "top_5",
    "KXPGATOP10": "top_10",
    "KXPGATOP20": "top_20",
    "KXPGATOUR": "winner",
}

# ── Golf scope guard ──────────────────────────────────────────────────
# Hard invariant: this script must NEVER touch a Kalshi order outside golf.
# Every code path that lists, cancels, or posts orders runs each ticker
# through _is_golf_ticker() and skips non-matches. Update this prefix tuple
# if Kalshi ever adds a new PGA series ticker (e.g. KXPGAMU for matchups).
GOLF_TICKER_PREFIXES = ("KXPGA",)


def _is_golf_ticker(ticker):
    return isinstance(ticker, str) and ticker.startswith(GOLF_TICKER_PREFIXES)

_client = httpx.Client(timeout=15.0, headers={"Accept": "application/json"})

# Throttle between authenticated calls (kalshi allows ~10/sec sustained)
_AUTH_SLEEP = 0.12


def _authed_request(method, path, json_body=None):
    """Sign + execute an authenticated request. Path includes query string."""
    from kalshi_auth import sign_headers
    headers = dict(_client.headers)
    headers.update(sign_headers(method, path))
    if json_body is not None:
        headers["Content-Type"] = "application/json"
    url = f"https://api.elections.kalshi.com{path}"
    r = _client.request(method, url, headers=headers, json=json_body)
    time.sleep(_AUTH_SLEEP)
    return r


def _get_markets(series_ticker):
    out, cursor = [], None
    while True:
        params = {"limit": 200, "status": "open", "series_ticker": series_ticker}
        if cursor:
            params["cursor"] = cursor
        r = _client.get(f"{KALSHI_API}/markets", params=params)
        r.raise_for_status()
        data = r.json()
        mkts = data.get("markets", [])
        out.extend(mkts)
        cursor = data.get("cursor")
        if not cursor or len(mkts) < 200:
            break
    return out


# ── Tournament filter ─────────────────────────────────────────────────
# Title-based slug matching was replaced by sim-player-overlap detection
# (_detect_target_event_code below). See _apply_event_filter for the
# replacement logic.


def _extract_player(title):
    m = re.match(r".*:\s*Will (.+?) (?:finish|make|miss|lead|win)", title)
    if m:
        return m.group(1).strip()
    m = re.match(r"Will (.+?) win the ", title)
    if m:
        return m.group(1).strip()
    return ""


def _extract_matchup(title):
    """Parse 'Will <player> beat <opponent> in the <event>?' titles.
    Returns (player_raw, opponent_raw) or ("", "")."""
    m = re.match(r"Will (.+?) beat (.+?) in the", title)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", ""


def _norm(s):
    x = s.strip().lower()
    if "," not in x:
        parts = x.rsplit(" ", 1)
        if len(parts) == 2:
            x = f"{parts[1]}, {parts[0]}"
    return name_replacements.get(x, x)


def _ticker_event_code(ticker):
    """Pull the event code segment out of a Kalshi golf ticker.
    e.g. 'KXPGATOUR-THCCBN26-WKIM' -> 'THCCBN26'.
    Returns '' if the shape doesn't match."""
    if not isinstance(ticker, str):
        return ""
    parts = ticker.split("-", 2)
    return parts[1] if len(parts) >= 2 else ""


def _detect_target_event_code(markets, sim_player_set):
    """Auto-detect which Kalshi event code we're targeting by counting how
    many sim-known players appear in each event's markets.

    Replaces title-based slug matching, which is fragile when the Kalshi
    title doesn't contain the slug as a single word (e.g. slug 'cjcup' vs
    title 'THE CJ CUP Byron Nelson'). Since the maker only knows what to do
    with markets whose players are in our sim, the event with the highest
    sim-player overlap is by definition the one we should be posting on.

    Handles both outright titles ("Will X win the EVENT?") and matchup titles
    ("Will X beat Y in the EVENT?"). Returns (target_event_code, counts_dict)
    or (None, {}) if no event has any matching players.
    """
    counts = defaultdict(int)
    for m in markets:
        event_code = _ticker_event_code(m.get("ticker", ""))
        if not event_code:
            continue
        title = m.get("title", "")
        # Outright title path.
        player_raw = _extract_player(title)
        if player_raw:
            if _norm(player_raw) in sim_player_set:
                counts[event_code] += 1
            continue
        # Matchup title path.
        p1_raw, p2_raw = _extract_matchup(title)
        if p1_raw and p2_raw:
            if _norm(p1_raw) in sim_player_set or _norm(p2_raw) in sim_player_set:
                counts[event_code] += 1
    if not counts:
        return None, {}
    target = max(counts, key=counts.get)
    return target, dict(counts)


# Fail-closed thresholds for event auto-detection (env-overridable). Quoting the
# wrong event (e.g. Sunday night with unrotated sim_inputs: last week's markets
# gone, next week's open, and last week's sim players overlapping next week's
# field) is strictly worse than quoting nothing.
def _env_float(name, default):
    try:
        v = os.getenv(name)
        return float(v) if v not in (None, "") else default
    except (TypeError, ValueError):
        return default


EVENT_MIN_MATCHES = _env_int("MAKER_EVENT_MIN_MATCHES", 5)
EVENT_MIN_MATCH_FRAC = _env_float("MAKER_EVENT_MIN_MATCH_FRAC", 0.70)
EVENT_AMBIGUITY_RATIO = _env_float("MAKER_EVENT_AMBIGUITY_RATIO", 0.80)


def _apply_event_filter(markets, sim_player_set, label):
    """Detect the target Kalshi event from sim-player overlap and filter markets
    to that event. Used by both scan() (outrights) and scan_matchups().

    FAIL-CLOSED: returns [] (quote nothing; --live then runs a cancel-only
    reconcile) instead of guessing when
      * no event has any sim-player overlap,
      * the best event has too few matches (< EVENT_MIN_MATCHES),
      * the best event's markets match the sim too weakly (< EVENT_MIN_MATCH_FRAC
        of its markets — the true event's field ≈ the sim's field, so a weak
        fraction means the sim is for a DIFFERENT event), or
      * a runner-up event is close enough to be ambiguous (>= EVENT_AMBIGUITY_RATIO
        of the leader's matches).
    """
    target, counts = _detect_target_event_code(markets, sim_player_set)
    if target is None:
        print(f"  [{label}] FAIL-CLOSED: no sim-player overlap on any event "
              f"({len(markets)} markets) — quoting nothing. Rotate sim_inputs?")
        return []
    totals = Counter(_ticker_event_code(m.get("ticker", "")) for m in markets)
    leader = counts[target]
    frac = leader / max(1, totals.get(target, 0))
    runners = sorted(
        ((ec, c) for ec, c in counts.items() if ec != target),
        key=lambda x: -x[1],
    )
    if leader < EVENT_MIN_MATCHES:
        print(f"  [{label}] FAIL-CLOSED: best event {target} has only {leader} "
              f"sim-player match(es) (< {EVENT_MIN_MATCHES}) — quoting nothing")
        return []
    if frac < EVENT_MIN_MATCH_FRAC:
        print(f"  [{label}] FAIL-CLOSED: best event {target} matches only "
              f"{leader}/{totals.get(target, 0)} of its markets ({frac:.0%} < "
              f"{EVENT_MIN_MATCH_FRAC:.0%}) — the sim is likely for a different "
              f"event (unrotated sim_inputs?). Quoting nothing")
        return []
    if runners and runners[0][1] >= EVENT_AMBIGUITY_RATIO * leader:
        print(f"  [{label}] FAIL-CLOSED: ambiguous event detection — "
              f"{target}={leader} vs {runners[0][0]}={runners[0][1]} matches. "
              f"Quoting nothing")
        return []
    filtered = [m for m in markets if _ticker_event_code(m.get("ticker", "")) == target]
    msg = (f"  [{label}] target event={target} ({leader}/{totals.get(target, 0)} "
           f"sim-player matches, {frac:.0%}) — filtered {len(markets)} -> "
           f"{len(filtered)} markets")
    print(msg)
    if len(counts) > 1:
        other = ", ".join(f"{ec}={c}" for ec, c in runners[:4])
        print(f"  [{label}] other event codes seen: {other}")
    return filtered


# ── Sim probability load ───────────────────────────────────────────────
_SIM_FAIRS_CACHE = None
_SIM_FAIRS_META = {}


def _fetch_sim_fairs_github():
    repo = os.getenv("SIMS_PROCESS_REPO", "mslade50/sims_process")
    branch = os.getenv("SIMS_PROCESS_BRANCH", "main")
    token = os.getenv("SIMS_PROCESS_PAT") or os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    try:
        import requests
        headers = {"Accept": "application/vnd.github.raw+json", "User-Agent": _BROWSER_UA}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        url = f"https://api.github.com/repos/{repo}/contents/sim_fairs.json?ref={branch}"
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [probs] GitHub sim_fairs fetch failed ({e})")
        return None


def _read_sim_fairs_local():
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_fairs.json")
    try:
        with open(local, encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return None


def _fetch_sim_fairs(force=False):
    """Published sim fairs (sim_fairs.json). Reads BOTH GitHub (origin) and the local
    working-tree copy and uses whichever has the newer generated_at — so a failed
    publish-push (fresh local, stale origin) can't strand the maker on stale fairs,
    and a machine without the repo still gets origin. Cached per run.

    The sim publishes sim_fairs.json (publish_sim_fairs.py commits + pushes it); the
    maker just reads it, so NO local new_sim run is needed. Set SIMS_PROCESS_PAT (or
    GH_TOKEN) to read the repo from a machine without it cloned."""
    global _SIM_FAIRS_CACHE, _SIM_FAIRS_META
    if _SIM_FAIRS_CACHE is not None and not force:
        return _SIM_FAIRS_CACHE
    options = [(p, src) for p, src in ((_fetch_sim_fairs_github(), "github"),
                                       (_read_sim_fairs_local(), "local")) if p]
    if not options:
        print("  [probs] sim_fairs.json unavailable (GitHub + local both failed)")
        payload, source = None, None
    else:
        # Prefer-fresher ranks on sim_run_at (when the SIM ran — publish_sim_fairs
        # stamps it from source-file mtimes). generated_at is publish wall-clock and
        # only breaks ties among legacy payloads; a payload with neither ranks last.
        def _fairs_rank(p):
            return (maker_guard.parse_fairs_ts(p.get("sim_run_at"))
                    or maker_guard.parse_fairs_ts(p.get("generated_at"))
                    or 0.0)
        payload, source = max(options, key=lambda ps: _fairs_rank(ps[0]))
        print(f"  [probs] sim_fairs.json via {source} (freshest of "
              f"{[s for _, s in options]}): tourney={payload.get('tourney')}, "
              f"sim_run_at={payload.get('sim_run_at')}, "
              f"generated_at={payload.get('generated_at')}")
    _SIM_FAIRS_CACHE = payload
    _SIM_FAIRS_META = {"source": source, "generated_at": (payload or {}).get("generated_at"),
                       "sim_run_at": (payload or {}).get("sim_run_at"),
                       "round": (payload or {}).get("round"),
                       "tourney": (payload or {}).get("tourney")}
    return payload


def _prob_lookup_from_fairs(payload):
    """Transpose a sim_fairs payload into {player: {market_type: yes_prob}}. Uses
    outrights_nodh for top-N (Kalshi settles a clean binary, no dead-heat) and
    outrights for winner. Player keys are already canonical ('last, first' lower)."""
    outr = payload.get("outrights") or {}
    nodh = payload.get("outrights_nodh") or {}
    sources = (
        ("winner", outr.get("winner")),
        ("top_5", nodh.get("top_5") or outr.get("top_5")),
        ("top_10", nodh.get("top_10") or outr.get("top_10")),
        ("top_20", nodh.get("top_20") or outr.get("top_20")),
    )
    lookup = {}
    for market, src in sources:
        if not isinstance(src, dict):
            continue
        for player, prob in src.items():
            try:
                lookup.setdefault(player, {})[market] = float(prob)
            except (TypeError, ValueError):
                continue
    return lookup


def load_sim_probs():
    """{player: {market_type: yes_prob}} from the PUBLISHED sim fairs (sim_fairs.json
    — GitHub repo then local). Kalshi no-dead-heat top-N (outrights_nodh) + dead-heat
    winner. No local sim run needed: whoever runs the sim publishes the fairs; the
    maker reads them. Raises FileNotFoundError if absent / wrong event."""
    payload = _fetch_sim_fairs()
    if payload is None:
        raise FileNotFoundError("no sim_fairs.json (GitHub or local) — run + publish the sim")
    pj = str(payload.get("tourney") or "").strip().lower()
    if pj and tourney and pj != str(tourney).strip().lower():
        raise FileNotFoundError(
            f"sim_fairs.json is for '{pj}', not current '{tourney}' — publish the sim for {tourney}")
    lookup = _prob_lookup_from_fairs(payload)
    print(f"  [probs] {len(lookup)} players from sim_fairs.json "
          f"(tourney={pj or '?'}, generated_at={payload.get('generated_at')})")
    return lookup


# Live matchup tape published by publish_sim_fairs as a `sim-data` release asset —
# lets a machine that never ran round_sim (the VPS) price H2H matchups. Cached on
# disk keyed by the asset's updated_at so the 2-min cadence costs one cheap
# release-metadata GET, not a tens-of-MB download.
_MATCHUP_TAPE_ASSET = "matchup_scores_live.parquet"
_MATCHUP_TAPE_CACHE = os.path.join("permanent_data", "matchup_tape_cache.parquet")
_MATCHUP_TAPE_CACHE_META = os.path.join("permanent_data", "matchup_tape_cache_meta.json")


def _load_matchup_tape_file(path, want_tourney):
    """Parse a matchup-tape parquet -> (scores (players, draws), names, meta dict).
    Fail-closed on tourney mismatch or malformed shape: wrong-event or wrong-name
    H2H prices are worse than skipping matchups. Returns (None, None, meta)."""
    import pyarrow.parquet as pq
    tbl = pq.read_table(path)
    md = {k.decode(): v.decode() for k, v in (tbl.schema.metadata or {}).items()
          if isinstance(k, bytes) and isinstance(v, bytes)}
    tape_tourney = (md.get("tourney") or "").strip().lower()
    if tape_tourney != str(want_tourney or "").strip().lower():
        print(f"    [matchups] release tape is for '{tape_tourney}', not "
              f"'{want_tourney}' — skipping matchups")
        return None, None, md
    df = tbl.to_pandas()
    names = [str(i) for i in df.index]
    scores = df.to_numpy()
    if scores.ndim != 2 or scores.shape[0] != len(names) or not names:
        print(f"    [matchups] release tape malformed ({scores.shape} vs "
              f"{len(names)} names) — skipping matchups")
        return None, None, md
    return scores, names, md


def _fetch_matchup_tape_release():
    """Fetch (with on-disk cache) the published live matchup tape. Returns
    (final_scores, player_names) or (None, None). Fail-closed when the tape is
    unstamped or materially older than the sim fairs the maker already gated on
    (they're published by the same run — a big gap means a stale asset)."""
    repo = os.getenv("SIMS_PROCESS_REPO", "mslade50/sims_process")
    token = os.getenv("SIMS_PROCESS_PAT") or os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    try:
        import tempfile as _tempfile
        import requests
        H = {"Accept": "application/vnd.github+json", "User-Agent": _BROWSER_UA}
        if token:
            H["Authorization"] = f"Bearer {token}"
        r = requests.get(f"https://api.github.com/repos/{repo}/releases/tags/sim-data",
                         headers=H, timeout=15)
        r.raise_for_status()
        asset = next((a for a in r.json().get("assets", [])
                      if a.get("name") == _MATCHUP_TAPE_ASSET), None)
        if asset is None:
            print(f"    [matchups] no {_MATCHUP_TAPE_ASSET} on the sim-data release "
                  f"— skipping matchups")
            return None, None
        cache_key = f"{asset.get('id')}:{asset.get('updated_at')}"
        try:
            with open(_MATCHUP_TAPE_CACHE_META, encoding="utf-8") as f:
                cached_key = _json.load(f).get("key")
        except Exception:
            cached_key = None
        if cache_key != cached_key or not os.path.exists(_MATCHUP_TAPE_CACHE):
            dr = requests.get(asset["url"],
                              headers={**H, "Accept": "application/octet-stream"},
                              timeout=300)
            dr.raise_for_status()
            os.makedirs(os.path.dirname(_MATCHUP_TAPE_CACHE), exist_ok=True)
            fd, tmp = _tempfile.mkstemp(dir=os.path.dirname(_MATCHUP_TAPE_CACHE))
            with os.fdopen(fd, "wb") as f:
                f.write(dr.content)
            os.replace(tmp, _MATCHUP_TAPE_CACHE)
            with open(_MATCHUP_TAPE_CACHE_META, "w", encoding="utf-8") as f:
                _json.dump({"key": cache_key}, f)
            print(f"    [matchups] downloaded matchup tape "
                  f"({len(dr.content) // 1_000_000}MB, updated {asset.get('updated_at')})")
        scores, names, md = _load_matchup_tape_file(_MATCHUP_TAPE_CACHE, tourney)
        if scores is None:
            return None, None
        tape_ts = maker_guard.parse_fairs_ts(md.get("sim_run_at"))
        fairs_ts = maker_guard.parse_fairs_ts(_SIM_FAIRS_META.get("sim_run_at"))
        if tape_ts is None or (fairs_ts and tape_ts < fairs_ts - 12 * 3600):
            print(f"    [matchups] release tape stale/unstamped "
                  f"(sim_run_at={md.get('sim_run_at')!r} vs fairs "
                  f"{_SIM_FAIRS_META.get('sim_run_at')!r}) — skipping matchups")
            return None, None
        print(f"    [matchups] using RELEASE matchup tape: {scores.shape[0]} players x "
              f"{scores.shape[1]} draws (sim_run_at={md.get('sim_run_at')})")
        return scores, names
    except Exception as e:
        print(f"    [matchups] release tape fetch failed ({e}) — skipping matchups")
        return None, None


def load_matchup_sim_data(allow_pre_fallback=False):
    """Load final_scores + player_names so we can compute P(p1 beats p2) on
    demand for any H2H matchup.

    Prefers the live post-round final_scores written by round_sim.py
    (`final_scores_live_{tourney}.npy`), then the tape the sim machine published
    to the sim-data release (VPS path), then — only with allow_pre_fallback —
    the pre-tournament `{tourney}/final_scores.npy`. Falling back to
    pre-tournament scores mid-event would produce stale prices, which is worse
    than no proposal, so that path stays opt-in.
    """
    import json as _json
    live_fs = f"final_scores_live_{tourney}.npy"
    live_pn = f"player_names_live_{tourney}.json"
    pre_fs = os.path.join(".", tourney, "final_scores.npy")
    pn_path = os.path.join(".", tourney, "player_names.json")

    if os.path.exists(live_fs):
        # The live npy's rows are in round_sim's live field order. Pairing it
        # with new_sim's alphabetical player_names.json silently computes
        # P(wrong player beats wrong player) — only the sidecar written by the
        # same round_sim run is a valid label set.
        if not os.path.exists(live_pn):
            print(f"    [matchups] {live_fs} present but {live_pn} missing — "
                  f"refusing to pair live scores with pre-tournament name order; "
                  f"skipping matchups (rerun round_sim to regenerate both)")
            return None, None
        final_scores = np.load(live_fs)
        with open(live_pn) as f:
            player_names = _json.load(f)
        if len(player_names) != final_scores.shape[0]:
            print(f"    [matchups] live name/score mismatch "
                  f"({len(player_names)} names vs {final_scores.shape[0]} rows) — "
                  f"skipping matchups")
            return None, None
        print(f"    [matchups] using live final_scores: {live_fs} "
              f"({final_scores.shape[0]} players)")
        return final_scores, player_names

    # No local live sim on this machine — try the published release tape (VPS path)
    final_scores, player_names = _fetch_matchup_tape_release()
    if final_scores is not None:
        return final_scores, player_names

    if allow_pre_fallback and os.path.exists(pre_fs) and os.path.exists(pn_path):
        print(f"    [matchups] using PRE-TOURNAMENT final_scores: {pre_fs} "
              f"(allow_pre_fallback=True)")
        final_scores = np.load(pre_fs)
        with open(pn_path) as f:
            player_names = _json.load(f)
        return final_scores, player_names

    return None, None


def matchup_sim_prob(p1, p2, final_scores, name_to_idx):
    """P(p1 beats p2) from final_scores, ties pushed (matches new_sim.py)."""
    i1 = name_to_idx.get(p1)
    i2 = name_to_idx.get(p2)
    if i1 is None or i2 is None:
        return None
    s1 = final_scores[i1]
    s2 = final_scores[i2]
    wins = int(np.sum(s1 < s2))   # lower score wins in golf
    losses = int(np.sum(s1 > s2))
    denom = wins + losses
    return wins / denom if denom > 0 else None


# ── Post-price + Kelly math ────────────────────────────────────────────
def _parse_price_ranges(market):
    """Return a list of (start, end, step) floats, sorted by start. Falls
    back to a single 0¢→100¢ range with DEFAULT_TICK if the market lacks
    price_ranges (shouldn't happen on golf markets, but defensive)."""
    raw = market.get("price_ranges") or []
    out = []
    for r in raw:
        try:
            out.append((float(r["start"]), float(r["end"]), float(r["step"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not out:
        return [(0.0, 1.0, DEFAULT_TICK)]
    return sorted(out, key=lambda x: x[0])


def _tick_at(price_ranges, price):
    """Tick size at this price level, using the market's price_ranges."""
    for start, end, step in price_ranges:
        # Use a tiny tolerance on the upper bound — ranges are usually
        # exclusive on `end`, so 0.0999 falls in [0,0.1) and 0.1 in [0.1,0.9)
        if start <= price < end + 1e-9:
            return step
    return price_ranges[-1][2] if price_ranges else DEFAULT_TICK


def _floor_to_tick(price, tick):
    """Floor a price down to the nearest valid tick. The +1e-9 absorbs
    binary-float drift so 0.99 doesn't accidentally floor to 0.98 when
    tick=0.01."""
    return math.floor(price / tick + 1e-9) * tick


def _round_to_tick(price, tick):
    """Round to nearest valid tick (used for canonicalizing prices in
    client_order_id and the resting-order dedup key)."""
    return round(round(price / tick) * tick, 6)


def _maker_intent(yes_bid, yes_ask, sim_yes, price_ranges):
    """For each of YES and NO sides, generate one intent per valid tick
    price between best_bid (inclusive) and best_ask − 1 tick (inclusive).

    Each rung is an independent intent — the edge / EV gates filter rungs
    individually, and kelly_solver groups by (ticker, side) and splits its
    Kelly-target contracts evenly across whichever rungs survive the gates.

    Wide markets produce many rungs (one per tick across the spread). Tight
    markets (ask = bid + 1 tick) produce a single rung. The market's tick
    size can change by price region (e.g. 0.1¢ near 0/100 and 1¢ in the
    middle for winner markets), so we recompute the step at each rung.

    Posts are always strictly below the opposite ask, so no cross risk.

    Labels: rungs are tagged sequentially as L01, L02, ... (lowest price
    first). The label is display-only — no downstream logic depends on it.
    """
    yes_mid = (yes_bid + yes_ask) / 2.0
    no_bid = 1.0 - yes_ask
    no_ask = 1.0 - yes_bid
    no_mid = (no_bid + no_ask) / 2.0

    out = []
    for side, p_mid, p_ask, p_bid, sim_p in [
        ("yes", yes_mid, yes_ask, yes_bid, sim_yes),
        ("no", no_mid, no_ask, no_bid, 1.0 - sim_yes),
    ]:
        # Walk from the bid upward, stepping by the local tick. Stop one
        # tick below the ask so we can never cross.
        post = round(p_bid, 6)
        rung_idx = 0
        seen = set()
        # Safety counter — _tick_at should never return <= 0, but cap to
        # avoid an infinite loop if a malformed price_ranges row ever sneaks in.
        max_iters = 10_000
        for _ in range(max_iters):
            post_r = round(post, 6)
            if post_r >= round(p_ask, 6) - 1e-9:
                break
            if post_r > 0 and post_r not in seen:
                seen.add(post_r)
                edge_pp = (sim_p - post_r) * 100
                kelly_f = ((sim_p - post_r) / (1.0 - post_r)
                           if post_r < 1.0 else 0.0)
                rung_idx += 1
                out.append({
                    "side": side,
                    "post_type": f"L{rung_idx:02d}",
                    "best_bid": p_bid,
                    "best_ask": p_ask,
                    "mid": p_mid,
                    "post_price": post_r,
                    "tick": _tick_at(price_ranges, post_r),
                    "sim_prob": sim_p,
                    "edge_pp": edge_pp,
                    "kelly_f": kelly_f,
                    # `contracts` is intentionally not set here — caller (scan
                    # for outrights, scan_matchups for matchups) assigns size
                    # based on its own sizing rule (Kelly vs flat).
                })
            tick = _tick_at(price_ranges, post_r)
            if tick <= 0:
                break
            post = round(post_r + tick, 6)
    return out


# ── Authenticated trading helpers ──────────────────────────────────────
# All client_order_ids placed by this script start with this marker. Used
# to distinguish our orders from manual hand-clicks (which have an empty
# client_order_id) — manual orders are NEVER auto-cancelled.
SCRIPT_ORDER_PREFIX = "smk_"


def _client_order_id(ticker, side, post_price_dollars):
    """Deterministic per-(ticker,side,price), prefixed so we can recognize
    our own orders when reconciling. Kalshi rejects duplicate
    client_order_ids, so this gives us idempotency for free even if the
    reconciliation step misses an in-flight order.

    Uses milli-dollar precision (e.g. 0.995 → 995) so two different
    sub-cent posts (99.5¢ vs 99.6¢) don't hash to the same id.
    """
    milli = int(round(post_price_dollars * 1000))
    raw = f"sims_maker|{ticker}|{side}|{milli}"
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    # 4-char prefix + 28-char hex = 32 chars total (within Kalshi's limit)
    return SCRIPT_ORDER_PREFIX + h[:28]


def _rearm_client_order_id(ticker, side, post_price_dollars, now_ts):
    """A script-prefixed but time-UNIQUE id for a TTL re-arm re-post. The normal
    id is deterministic per (ticker,side,price) for idempotency; on a re-arm we
    cancel the twin and re-post at the same price, so a fresh id makes the re-post
    well-defined regardless of whether Kalshi frees the cancelled client_order_id.
    Still starts with SCRIPT_ORDER_PREFIX so _is_script_order recognizes it."""
    milli = int(round(post_price_dollars * 1000))
    raw = f"sims_maker|{ticker}|{side}|{milli}|{int(now_ts)}"
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return SCRIPT_ORDER_PREFIX + h[:28]


def _is_script_order(o):
    """True if this resting order was placed by this script."""
    return str(o.get("client_order_id", "")).startswith(SCRIPT_ORDER_PREFIX)


def list_resting_orders(golf_only=True):
    """Resting orders for our key. Paginates.

    By default returns ONLY golf-series orders (KXPGA-prefixed tickers).
    Pass golf_only=False for diagnostic dumps that include other markets.
    """
    out, cursor = [], None
    while True:
        path = "/trade-api/v2/portfolio/orders?status=resting&limit=200"
        if cursor:
            path += f"&cursor={cursor}"
        r = _authed_request("GET", path)
        r.raise_for_status()
        data = r.json()
        out.extend(data.get("orders", []))
        cursor = data.get("cursor")
        if not cursor:
            break
    if golf_only:
        out = [o for o in out if _is_golf_ticker(o.get("ticker", ""))]
    return out


def list_positions():
    """All open portfolio positions (paginated). Raw Kalshi market_positions
    dicts — used by the exposure governor to count held risk."""
    out, cursor = [], None
    while True:
        path = "/trade-api/v2/portfolio/positions?limit=200"
        if cursor:
            path += f"&cursor={cursor}"
        r = _authed_request("GET", path)
        r.raise_for_status()
        data = r.json()
        out.extend(data.get("market_positions", []))
        cursor = data.get("cursor")
        if not cursor:
            break
    return out


def list_settlements(max_pages=3):
    """Recent settlements (newest first, paginated). A few pages is plenty for the
    daily realized-loss window."""
    out, cursor = [], None
    for _ in range(max_pages):
        path = "/trade-api/v2/portfolio/settlements?limit=200"
        if cursor:
            path += f"&cursor={cursor}"
        r = _authed_request("GET", path)
        r.raise_for_status()
        data = r.json()
        out.extend(data.get("settlements", []))
        cursor = data.get("cursor")
        if not cursor:
            break
    return out


def get_balance_usd():
    """Available cash balance in dollars (Kalshi reports integer cents)."""
    r = _authed_request("GET", "/trade-api/v2/portfolio/balance")
    r.raise_for_status()
    data = r.json()
    raw = data.get("balance_dollars")
    if raw not in (None, ""):
        return float(raw)
    return float(data.get("balance") or 0) / 100.0


def list_fills(max_pages=2):
    """Recent fills (newest first, paginated). Used by maker_alerts to notify on
    new inventory."""
    out, cursor = [], None
    for _ in range(max_pages):
        path = "/trade-api/v2/portfolio/fills?limit=200"
        if cursor:
            path += f"&cursor={cursor}"
        r = _authed_request("GET", path)
        r.raise_for_status()
        data = r.json()
        out.extend(data.get("fills", []))
        cursor = data.get("cursor")
        if not cursor:
            break
    return out


def _resting_key(o):
    """(ticker, side, milli_dollars) — the dedup grain for reconciliation.

    Kalshi returns prices as decimal-dollar strings ("0.6700" = 67¢) under
    `yes_price_dollars` / `no_price_dollars`. We store as integer
    milli-dollars (0.995 → 995) so sub-cent levels stay distinct.
    """
    side = o.get("side", "")
    if side == "yes":
        raw = o.get("yes_price_dollars") or o.get("yes_price") or 0
    else:
        raw = o.get("no_price_dollars") or o.get("no_price") or 0
    try:
        milli = int(round(float(raw) * 1000))
    except (TypeError, ValueError):
        milli = 0
    return (o.get("ticker", ""), side, milli)


def _resting_count(o):
    """Remaining contracts on a resting order."""
    raw = o.get("remaining_count_fp") or o.get("remaining_count") or 0
    try:
        return int(round(float(raw)))
    except (TypeError, ValueError):
        return 0


def _order_expiration_ts(o):
    """Native Kalshi expiration for a resting order as a Unix-seconds int, or
    None when the order carries no expiration (legacy unlimited-GTC) or the value
    is unreadable. Kalshi returns `expiration_ts` as integer Unix seconds."""
    raw = o.get("expiration_ts")
    if raw in (None, "", 0):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _needs_refresh(o, now_ts):
    """True if a resting SCRIPT quote must be re-armed: it has no native TTL
    (legacy unlimited GTC) or its expiration is within QUOTE_REFRESH_SEC of now.
    Re-arming renews the rolling dead-man's switch so live quotes never lapse
    while the maker runs, yet die on their own within QUOTE_TTL_SEC if it stops."""
    exp = _order_expiration_ts(o)
    if exp is None:
        return True
    return (exp - now_ts) <= QUOTE_REFRESH_SEC


def _shadow_mode():
    """MAKER_SHADOW truthy => HARD no-orders: post_limit and cancel_order become
    no-ops at the API boundary, so even an accidental --live can't touch the book.
    The shadow runner sets it; it's belt-and-suspenders to plain dry-run."""
    return (os.getenv("MAKER_SHADOW") or "").strip().lower() in ("1", "true", "yes", "on")


def cancel_order(order_id, ticker, reason=""):
    """Cancel a single order. ENFORCES golf-scope at the API boundary —
    raises if the ticker isn't a golf series ticker. Every caller passes
    the order's ticker so this assertion can't be skipped by accident.
    """
    if not _is_golf_ticker(ticker):
        raise RuntimeError(
            f"REFUSING to cancel non-golf order: ticker={ticker!r} id={order_id}"
        )
    if _shadow_mode():
        print(f"  [shadow] WOULD cancel {order_id} {ticker} — NOT sent (MAKER_SHADOW)")
        return False
    path = f"/trade-api/v2/portfolio/orders/{order_id}"
    r = _authed_request("DELETE", path)
    ok = 200 <= r.status_code < 300
    tag = f" ({reason})" if reason else ""
    print(f"  [cancel] {order_id} {ticker} status={r.status_code}{tag}")
    if not ok:
        print(f"  [cancel] body: {r.text[:200]}")
    return ok


def post_limit(ticker, side, price_dollars, count, expiration_ts=None, client_order_id=None):
    """POST a single resting limit order. Re-fetches the orderbook FIRST and
    re-asserts no-cross against the live ask. Returns (ok, order_id_or_err).

    `price_dollars` is the full-precision post price (e.g. 0.995 for 99.5¢
    on a deci-cent market). The exact field name Kalshi expects for sub-cent
    prices is verified before phase-2 goes live — see SUB_CENT_POST_NOTE.

    `expiration_ts` is an optional Unix timestamp in seconds. When provided,
    Kalshi auto-cancels any unfilled remainder at that wall-clock time. Per
    Kalshi docs, the field must be paired with time_in_force=good_till_canceled
    (omitting expiration_ts under GTC = true unlimited GTC, the prior default).

    `client_order_id` overrides the default deterministic per-(ticker,side,price)
    id. The reconcile loop passes a fresh one only on a TTL re-arm (cancel +
    re-post at the same price); every other post keeps the deterministic id so
    Kalshi's duplicate rejection still guards against a missed reconciliation.
    """
    if not _is_golf_ticker(ticker):
        print(f"  [skip] {ticker} non-golf ticker — refusing")
        return (False, f"REFUSING to post non-golf ticker: {ticker!r}")
    if _shadow_mode():
        print(f"  [shadow] WOULD post {ticker} {side.upper()} @{price_dollars*100:.1f}c "
              f"x {int(count)} — NOT sent (MAKER_SHADOW)")
        return (False, "shadow mode — no order placed")
    # Live re-validate — orderbook may have moved since scan
    try:
        ob_path = f"/trade-api/v2/markets/{ticker}/orderbook"
        r_ob = _client.get(f"https://api.elections.kalshi.com{ob_path}")
        r_ob.raise_for_status()
        ob_resp = r_ob.json()
        # Kalshi wraps levels under `orderbook_fp` (floating-point variant)
        # in the v2 API; fall back to legacy `orderbook` or flat shapes.
        ob = ob_resp.get("orderbook_fp") or ob_resp.get("orderbook") or ob_resp
        yes_levels = ob.get("yes_dollars") or ob.get("yes", [])
        no_levels = ob.get("no_dollars") or ob.get("no", [])
        def _best(levels):
            return max((float(p) for p, _ in levels), default=None) if levels else None
        best_yes_bid = _best(yes_levels)
        best_no_bid = _best(no_levels)
        if best_yes_bid is not None and best_yes_bid > 1:
            best_yes_bid /= 100.0
        if best_no_bid is not None and best_no_bid > 1:
            best_no_bid /= 100.0
        if side == "yes":
            live_ask = (1.0 - best_no_bid) if best_no_bid is not None else None
        else:
            live_ask = (1.0 - best_yes_bid) if best_yes_bid is not None else None
        if live_ask is None:
            print(f"  [skip] {ticker} {side.upper()} — could not derive live ask. "
                  f"orderbook keys={list(ob.keys())} "
                  f"yes_levels={yes_levels[:2]} no_levels={no_levels[:2]}")
            return (False, "no orderbook")
        if price_dollars >= live_ask - 1e-9:
            print(f"  [skip] {ticker} {side.upper()} @{price_dollars*100:.1f}c "
                  f"would cross live_ask={live_ask*100:.1f}c — book moved since scan")
            return (False, f"would cross: post={price_dollars:.4f} ask={live_ask:.4f}")
    except Exception as e:
        print(f"  [skip] {ticker} prevalidate exception: {e}")
        return (False, f"prevalidate failed: {e}")

    # Always use the `_dollars` decimal-string field. Kalshi accepts it for
    # both whole-cent and deci-cent markets (the integer `yes_price` /
    # `no_price` fields only support whole cents 1-99 and would be rejected
    # on sub-cent markets). Per Kalshi spec, provide exactly one of the
    # four price fields per order — never mix.
    # Source: https://docs.kalshi.com/api-reference/orders/create-order
    body = {
        "ticker": ticker,
        "client_order_id": client_order_id or _client_order_id(ticker, side, price_dollars),
        "type": "limit",
        "action": "buy",
        "side": side,
        "count": int(count),
        f"{side}_price_dollars": f"{price_dollars:.4f}",
    }
    if expiration_ts is not None:
        body["time_in_force"] = "good_till_canceled"
        body["expiration_ts"] = int(expiration_ts)

    r = _authed_request("POST", "/trade-api/v2/portfolio/orders", json_body=body)
    if 200 <= r.status_code < 300:
        oid = r.json().get("order", {}).get("order_id", "?")
        exp_tag = f"  expires={int(expiration_ts)}" if expiration_ts is not None else ""
        print(f"  [post] {ticker} {side.upper()} @{price_dollars*100:.1f}c x {count} "
              f"-> order_id={oid}  live_ask={live_ask*100:.1f}c{exp_tag}")
        return (True, oid)
    print(f"  [post-fail] {ticker} {side.upper()} @{price_dollars*100:.1f}c "
          f"status={r.status_code} body={r.text[:200]}")
    return (False, r.text)


def reconcile_and_post(candidates):
    """1. List resting golf orders, split into script-placed vs manual.
    2. Auto-cancel SCRIPT orders whose (ticker, side, price) is NOT in the
       fresh candidate set. Manual orders are NEVER auto-cancelled.
    3. Skip candidates already resting at the exact same price (whether
       from this script or a manual hand-click — avoids double-posting on
       the same price level).
    4. Re-validate + POST the rest.
    """
    print("\n[live] reconciling against resting orders…")
    resting = list_resting_orders(golf_only=True)
    script_orders = [o for o in resting if _is_script_order(o)]
    manual_orders = [o for o in resting if not _is_script_order(o)]
    print(f"  Found {len(resting)} golf resting order(s): "
          f"{len(script_orders)} script-placed, {len(manual_orders)} manual")

    # Candidate key uses milli-dollar precision to match _resting_key, so
    # sub-cent levels stay distinct.
    candidate_keys = {(c["ticker"], c["side"], int(round(c["post_price"] * 1000)))
                      for c in candidates}

    # ── 1. Auto-cancel stale SCRIPT orders (manual orders are immune) ───
    stale = [o for o in script_orders if _resting_key(o) not in candidate_keys]
    if stale:
        print(f"  Cancelling {len(stale)} stale script order(s)…")
        for o in stale:
            cancel_order(o.get("order_id"), o.get("ticker", ""),
                         reason=f"{o.get('ticker')} {o.get('side')} "
                                f"@{_resting_key(o)[2]}c")
    else:
        print("  No stale script orders to cancel.")
    if manual_orders:
        print(f"  (Manual orders left alone: "
              f"{', '.join(o.get('ticker','?') for o in manual_orders[:5])}"
              f"{'…' if len(manual_orders) > 5 else ''})")

    # ── 2. Bucket candidates that match a resting order. Manual hand-clicks
    # are immune (never touched, never stacked). A matching SCRIPT quote is
    # KEPT when its rolling TTL is still healthy (preserves queue priority), or
    # RE-ARMED (cancel + re-post with a fresh native expiration) when its expiry
    # is within QUOTE_REFRESH_SEC — so the dead-man's switch never lapses while
    # we're alive. ───────────────────────────────────────────────────────────
    now_ts = int(time.time())
    expiration_ts = now_ts + QUOTE_TTL_SEC

    still_resting = [o for o in resting if o not in stale]
    script_by_key, manual_keys = {}, set()
    for o in still_resting:
        if _is_script_order(o):
            script_by_key.setdefault(_resting_key(o), o)
        else:
            manual_keys.add(_resting_key(o))

    to_post, to_rearm, kept = [], [], 0
    for c in candidates:
        key = (c["ticker"], c["side"], int(round(c["post_price"] * 1000)))
        if key in manual_keys:
            continue  # sitting on your hand-click — leave it, never stack on top
        ro = script_by_key.get(key)
        if ro is None:
            to_post.append(c)
        elif _needs_refresh(ro, now_ts):
            to_rearm.append((c, ro))
        else:
            kept += 1
    if kept:
        print(f"  Keeping {kept} healthy script quote(s) (> {QUOTE_REFRESH_SEC}s TTL left)")

    # ── 3. Re-arm aging script quotes: cancel first, then re-post below. Only
    # re-post the ones we actually cancelled, so a cancel failure can never
    # leave a stacked duplicate. A skipped re-arm still self-expires on its
    # native TTL — fail-safe. ────────────────────────────────────────────────
    rearm_post = []
    if to_rearm:
        print(f"  Re-arming {len(to_rearm)} script quote(s) nearing expiry…")
        for c, ro in to_rearm:
            if cancel_order(ro.get("order_id"), ro.get("ticker", ""),
                            reason=f"TTL refresh {ro.get('ticker')} {ro.get('side')}"):
                rearm_post.append(c)

    # ── 4. POST fresh quotes (deterministic id) + re-armed quotes (fresh id),
    # each carrying the rolling native expiration. Re-validation happens inside
    # post_limit. ────────────────────────────────────────────────────────────
    print(f"  Posting {len(to_post) + len(rearm_post)} order(s) "
          f"({len(to_post)} new, {len(rearm_post)} re-armed; TTL {QUOTE_TTL_SEC}s)…")
    n_ok = n_fail = 0
    for c in to_post:
        ok, _ = post_limit(c["ticker"], c["side"], c["post_price"], c["contracts"],
                           expiration_ts=expiration_ts)
        n_ok += int(ok)
        n_fail += int(not ok)
    for c in rearm_post:
        ok, _ = post_limit(c["ticker"], c["side"], c["post_price"], c["contracts"],
                           expiration_ts=expiration_ts,
                           client_order_id=_rearm_client_order_id(
                               c["ticker"], c["side"], c["post_price"], now_ts))
        n_ok += int(ok)
        n_fail += int(not ok)
    print(f"\n[live] posted={n_ok}  failed={n_fail}  re-armed={len(rearm_post)}  "
          f"kept={kept}  cancelled_stale={len(stale)}")
    return {"posted": n_ok, "failed": n_fail, "rearmed": len(rearm_post),
            "kept": kept, "cancelled_stale": len(stale)}


def cancel_all():
    """--cancel-all entry point: PANIC BUTTON — cancel every golf order,
    including manual hand-clicks. Non-golf orders never touched.
    """
    all_orders = list_resting_orders(golf_only=False)
    golf = [o for o in all_orders if _is_golf_ticker(o.get("ticker", ""))]
    non_golf = len(all_orders) - len(golf)
    script_count = sum(1 for o in golf if _is_script_order(o))
    manual_count = len(golf) - script_count
    print(f"[cancel-all] PANIC: cancelling all {len(golf)} golf order(s) "
          f"({script_count} script, {manual_count} manual). "
          f"{non_golf} non-golf order(s) untouched.")
    for o in golf:
        cancel_order(o.get("order_id"), o.get("ticker", ""),
                     reason=f"{o.get('ticker')} {o.get('side')} "
                            f"@{_resting_key(o)[2]}c")


def cancel_script():
    """--cancel-script entry point: cancel only SCRIPT-placed golf orders
    (identified by the client_order_id prefix). Manual hand-clicked orders are
    left resting — the softer sibling of --cancel-all for pulling the bot
    without disturbing positions worked by hand.
    """
    golf = list_resting_orders(golf_only=True)
    manual = sum(1 for o in golf if not _is_script_order(o))
    print(f"[cancel-script] pulling script quotes: {len(golf) - manual} script, "
          f"{manual} manual golf order(s) left untouched.")
    n = maker_guard.pull_script_quotes(lambda: golf, _is_script_order, cancel_order)
    print(f"[cancel-script] cancelled {n} script order(s).")


def list_orders_smoke():
    """--list-orders entry point: pure GET, no DELETE. Auth smoke test."""
    print("[list-orders] GET /portfolio/orders?status=resting (no side effects)")
    all_orders = list_resting_orders(golf_only=False)
    golf = [o for o in all_orders if _is_golf_ticker(o.get("ticker", ""))]
    other = [o for o in all_orders if not _is_golf_ticker(o.get("ticker", ""))]
    script_golf = [o for o in golf if _is_script_order(o)]
    manual_golf = [o for o in golf if not _is_script_order(o)]
    print(f"  Total resting orders on account: {len(all_orders)}")
    print(f"    golf in scope:    {len(golf)} "
          f"({len(script_golf)} script-placed, {len(manual_golf)} manual)")
    print(f"    non-golf (immune): {len(other)}")
    print(f"  Under --live, auto-cancel would consider only script-placed orders.")
    if all_orders:
        print("\n  Sample (up to 10):")
        for o in all_orders[:10]:
            if _is_golf_ticker(o.get("ticker", "")):
                scope = "scr" if _is_script_order(o) else "man"
            else:
                scope = "OTH"
            _, side, milli = _resting_key(o)
            count = _resting_count(o)
            print(f"    [{scope:>3}] {o.get('ticker', '?'):<32} "
                  f"{side:<3} @{milli/10:.1f}c × {count}")


# ── H2H matchup scan ──────────────────────────────────────────────────
def scan_matchups():
    """Scan KXPGAH2H matchup markets for maker candidates.

    Gates differ from outright scan:
      - raw edge >= MATCHUP_MIN_EDGE_PP (default 5pp), no Kelly gate
      - level size = MATCHUP_LEVEL_CONTRACTS (default 500) per intent

    Uses the same 3-level intent structure (near_ask/mid/bid) and the
    same no-cross safety as outrights.
    """
    final_scores, player_names = load_matchup_sim_data(
        allow_pre_fallback=_allow_pre_matchup_fallback
    )
    if final_scores is None:
        live_fs = f"final_scores_live_{tourney}.npy"
        print(f"    [matchups] no live final_scores at {live_fs} — skipping. "
              f"round_sim.py must persist final_scores_live_{{tourney}}.npy "
              f"to enable mid-event matchup pricing. "
              f"(Use --allow-pre-matchup pre-tournament only.)")
        return []
    name_to_idx = {p: i for i, p in enumerate(player_names)}

    try:
        all_mkts = _get_markets(MATCHUP_SERIES)
    except Exception as e:
        print(f"    [matchups] fetch failed: {e}")
        return []
    print(f"    Fetched {len(all_mkts)} H2H markets")

    all_mkts = _apply_event_filter(all_mkts, set(name_to_idx.keys()), label="matchups")

    candidates = []
    skipped_name = 0
    for m in all_mkts:
        ticker = m.get("ticker", "")
        title = m.get("title", "")
        bid = float(m.get("yes_bid_dollars") or 0)
        ask = float(m.get("yes_ask_dollars") or 0)
        if bid == 0 and ask == 0:
            bid = float(m.get("yes_bid", 0) or 0) / 100.0
            ask = float(m.get("yes_ask", 0) or 0) / 100.0
        if bid <= 0 or ask <= 0:
            continue

        player_raw, opp_raw = _extract_matchup(title)
        if not player_raw or not opp_raw:
            continue
        player = _norm(player_raw)
        opp = _norm(opp_raw)
        sim_yes = matchup_sim_prob(player, opp, final_scores, name_to_idx)
        if sim_yes is None or sim_yes <= 0:
            skipped_name += 1
            continue

        price_ranges = _parse_price_ranges(m)
        for intent in _maker_intent(bid, ask, sim_yes, price_ranges):
            if intent["edge_pp"] < MATCHUP_MIN_EDGE_PP:
                continue
            assert intent["post_price"] < intent["best_ask"], (
                f"SAFETY: post {intent['post_price']} >= ask {intent['best_ask']}"
            )
            # Matchups: flat sizing (user spec — leave flat-staked for now).
            intent["contracts"] = MATCHUP_LEVEL_CONTRACTS
            # Compact "player v opponent" using last names only; full player
            # name is preserved in the underlying ticker for reconciliation.
            display = f"{player.split(',')[0]} v {opp.split(',')[0]}"
            candidates.append({
                "ticker": ticker,
                "title": title,
                "player": display,
                "market": "h2h",
                **intent,
            })

    if skipped_name:
        print(f"    [matchups] {skipped_name} sides skipped — player name "
              f"not in sim's player_names")
    return candidates


# ── Main scan ──────────────────────────────────────────────────────────
def _engine_outright_candidates(all_mkts, prob_lookup):
    """Build passive working-quote candidates via maker_quotes.plan_market — the
    new quoting brain (replaces the static edge-rung ladder). One working order
    per (market, side) the model favors, pegged near the touch and capped at fair
    minus the side's min edge. Fetches current positions + our resting script
    orders so the engine knows held inventory (for hysteresis, target, and to stop
    accumulating once full)."""
    import maker_quotes
    target_usd = maker_guard.caps_from_env()["per_market_usd"]
    # Pre-Wednesday rule (toggle from the dashboard; fail-safe ON): skip YES
    # outright quotes before this week's Wed 4pm ET (withdrawal risk).
    cfg = _fetch_maker_config()
    rule_on = bool(cfg.get("block_yes_outrights_pre_wed", True))
    blocked_yes = 0

    held = {}
    try:
        for p in list_positions():
            pf = float(p.get("position_fp") or p.get("position") or 0)
            if abs(pf) < 1e-9:
                continue
            k = (p.get("ticker", ""), "yes" if pf > 0 else "no")
            held[k] = held.get(k, 0) + abs(int(round(pf)))
    except Exception as e:
        print(f"  [engine] positions unavailable ({e}) — assuming flat")
    resting_q = {}
    try:
        for o in list_resting_orders(golf_only=True):
            if not _is_script_order(o):
                continue
            k = (o.get("ticker", ""), o.get("side", ""))
            _, _, milli = _resting_key(o)
            entry = {"price": milli / 1000.0, "size": _resting_count(o)}
            if k not in resting_q or entry["size"] > resting_q[k]["size"]:
                resting_q[k] = entry
    except Exception as e:
        print(f"  [engine] resting orders unavailable ({e})")

    out = []
    for m in all_mkts:
        ticker = m.get("ticker", "")
        title = m.get("title", "")
        mtype = m.get("_market_type", "")
        bid = float(m.get("yes_bid_dollars") or 0)
        ask = float(m.get("yes_ask_dollars") or 0)
        if bid == 0 and ask == 0:
            bid = float(m.get("yes_bid", 0) or 0) / 100.0
            ask = float(m.get("yes_ask", 0) or 0) / 100.0
        if bid <= 0 or ask <= 0:
            continue
        player_raw = _extract_player(title)
        if not player_raw:
            continue
        player = _norm(player_raw)
        rec = prob_lookup.get(player)
        if rec is None or mtype not in rec:
            continue
        sim_yes = float(rec[mtype])
        if sim_yes <= 0:
            continue
        volume = float(m.get("volume_fp", m.get("volume", 0)) or 0)
        # Same whole-market illiquid skip as the rung path.
        if (1.0 - bid) >= 0.95 and (ask - bid) > 0.03 and volume < 10000:
            continue
        tick = _tick_at(_parse_price_ranges(m), bid) or 0.01
        for c in maker_quotes.plan_market(
                ticker=ticker, market_type=mtype, player=player, title=title,
                yes_bid=bid, yes_ask=ask, sim_yes=sim_yes, tick=tick,
                held_yes=held.get((ticker, "yes"), 0), held_no=held.get((ticker, "no"), 0),
                resting_yes=resting_q.get((ticker, "yes")), resting_no=resting_q.get((ticker, "no")),
                target_usd=target_usd):
            if maker_guard.block_yes_outright(c["side"], c["market"], rule_enabled=rule_on):
                blocked_yes += 1
                continue
            c["volume"] = volume
            assert c["post_price"] < c["best_ask"], (
                f"SAFETY cross {c['post_price']} >= ask {c['best_ask']}")
            out.append(c)
    globals()["_rule_status"] = {
        "block_yes_pre_wed": {
            "enabled": rule_on,
            "active_now": bool(maker_guard.before_wed_cutoff()),
            "blocked": blocked_yes,
        }
    }
    if blocked_yes:
        print(f"  [rule] pre-Wed: blocked {blocked_yes} YES outright quote(s) "
              f"(no YES outrights before Wed 4pm ET)")
    print(f"  [engine] {len(out)} working quote(s) from {len(all_mkts)} markets "
          f"(edge YES>={maker_quotes.EDGE_YES*100:.1f}c / NO>={maker_quotes.EDGE_NO*100:.1f}c, "
          f"target ${target_usd:.0f}/market)")
    return out


def scan():
    print(f"[maker] tourney={tourney}  "
          f"raw_edge: 1.0pp(>{HIGH_PRICE_THRESHOLD}) / 1.5pp({TIER_DEFAULT_LOW}-{HIGH_PRICE_THRESHOLD}) "
          f"/ 1.0pp({TIER_LOW_PRICE}-{TIER_DEFAULT_LOW}) "
          f"/ 0.5pp({TIER_LONGSHOT}-{TIER_LOW_PRICE}) / 0.15pp(<{TIER_LONGSHOT})  "
          f"ev/$>={MIN_KELLY_EV_LOW_PRICE*100:.0f}%(price<{LOW_PRICE_THRESHOLD})  "
          f"fill_pct>={MIN_FILL_PCT:.0f}%  "
          f"per-tick laddering")
    prob_lookup = load_sim_probs()
    print(f"  Loaded sim probs for {len(prob_lookup)} players from sim_fairs.json")

    all_mkts = []
    for series, mtype in OUTRIGHT_SERIES.items():
        try:
            mkts = _get_markets(series)
            for m in mkts:
                m["_market_type"] = mtype
            all_mkts.extend(mkts)
        except Exception as e:
            print(f"  [warn] fetch {series} failed: {e}")
    print(f"  Fetched {len(all_mkts)} Kalshi markets")

    all_mkts = _apply_event_filter(all_mkts, set(prob_lookup.keys()), label="outrights")

    candidates = []
    # Default: the passive-accumulation engine (maker_quotes.plan_market). The
    # legacy edge-rung ladder is available via --rungs (sets _use_engine False),
    # in which case the loop below runs over all markets; otherwise it's skipped.
    use_engine = globals().get("_use_engine", True)
    if use_engine:
        candidates.extend(_engine_outright_candidates(all_mkts, prob_lookup))
    rung_markets = [] if use_engine else all_mkts
    for m in rung_markets:
        ticker = m.get("ticker", "")
        title = m.get("title", "")
        mtype = m.get("_market_type", "")
        bid = float(m.get("yes_bid_dollars") or 0)
        ask = float(m.get("yes_ask_dollars") or 0)
        if bid == 0 and ask == 0:
            bid = float(m.get("yes_bid", 0) or 0) / 100.0
            ask = float(m.get("yes_ask", 0) or 0) / 100.0
        if bid <= 0 or ask <= 0:
            continue
        player_raw = _extract_player(title)
        if not player_raw:
            continue
        player = _norm(player_raw)
        rec = prob_lookup.get(player)
        if rec is None or mtype not in rec:
            continue
        sim_yes = float(rec[mtype])
        if sim_yes <= 0:
            continue
        price_ranges = _parse_price_ranges(m)
        # Kalshi exposes lifetime volume as `volume_fp` (floating-point);
        # falls back to `volume` (centable int) if absent.
        volume = float(m.get("volume_fp", m.get("volume", 0)) or 0)

        # Whole-market skip: a 95c+ NO ask (i.e. yes_bid <= 5c) on a wide
        # (>3c) spread with low volume is too illiquid to ladder
        # meaningfully on either side. These are deep-longshot markets
        # where the YES side sits near the floor and the NO side has
        # little fill room above 95c. Drop the entire market.
        if (1.0 - bid) >= 0.95 and (ask - bid) > 0.03 and volume < 10000:
            continue

        for intent in _maker_intent(bid, ask, sim_yes, price_ranges):
            post_price = intent["post_price"]
            # Tiered raw-edge floor by post price.
            if intent["edge_pp"] < _raw_edge_floor(post_price):
                continue
            # Additional Kelly-EV floor for low-price (long-odds) bets only.
            if post_price < LOW_PRICE_THRESHOLD:
                ev_per_dollar = (intent["sim_prob"] - post_price) / post_price
                if ev_per_dollar < MIN_KELLY_EV_LOW_PRICE:
                    continue
            # Fill-probability filter: cut rungs we'd never realistically
            # fill at, then let kelly_solver redistribute the budget across
            # surviving rungs (it splits per-bet target evenly per level).
            fill_pct = _estimate_fill_pct(post_price, intent["best_ask"], volume)
            if fill_pct < MIN_FILL_PCT:
                continue
            # Universal sub-3c cull on non-tight markets. 1-2c YES bids
            # on anything wider than a 2c spread rarely fill — they sit
            # at the back of the deep queue and the rest of our ladder
            # captures any walking flow first.
            if (post_price < 0.03
                    and (intent["best_ask"] - intent["best_bid"]) > 0.02):
                continue
            # Wide-spread tiny-price cull for top_N markets. Deep YES bids
            # on wide top_5/10/20 markets get crowded out and rarely fill
            # meaningfully even when fill_pct passes. The volume condition
            # spares well-traded markets where flow eventually walks down
            # the queue.
            if (mtype in ("top_5", "top_10", "top_20")
                    and post_price < 0.05
                    and (intent["best_ask"] - intent["best_bid"]) > 0.05
                    and volume < 10000):
                continue
            intent["fill_pct"] = fill_pct
            intent["volume"] = volume
            # Hard safety check — never enter the candidate set if cross-risk
            assert intent["post_price"] < intent["best_ask"], (
                f"SAFETY: post {intent['post_price']} >= ask {intent['best_ask']}"
            )
            # Outrights: flat-sized per level until kelly_solver is wired.
            intent["contracts"] = LEVEL_CONTRACTS
            candidates.append({
                "ticker": ticker,
                "title": title,
                "player": player,
                "market": mtype,
                **intent,
            })

    # ── Append H2H matchup candidates (different gates + size). ─────────
    if globals().get("_skip_matchups"):
        print("\n  [matchups] skipped via --no-matchups")
    else:
        print("\n  [matchups] fetching H2H markets...")
        matchup_cands = scan_matchups()
        candidates.extend(matchup_cands)
        print(f"  [matchups] {len(matchup_cands)} candidate(s) added "
              f"(edge>={MATCHUP_MIN_EDGE_PP}pp, contracts={MATCHUP_LEVEL_CONTRACTS}/level)")

    candidates.sort(key=lambda r: r["edge_pp"], reverse=True)

    if not candidates:
        print("  No mid-maker candidates pass the gate.")
        return candidates, prob_lookup

    print(f"\n  {len(candidates)} candidate(s):")
    hdr = (f"{'ticker':<28} {'player':<22} {'mkt':<7} {'side':<4} "
           f"{'type':<8} {'bid':>6} {'ask':>6} {'mid':>6} {'post':>6} "
           f"{'tick':>5} {'sim':>6} {'edge':>5} {'kelly':>6} {'qty':>4}  vs-ask")
    print("  " + hdr)
    print("  " + "-" * len(hdr))
    for c in candidates:
        tick_str = f"{c['tick']*100:.1f}c"
        print(f"  {c['ticker']:<28} {c['player'][:22]:<22} "
              f"{c['market']:<7} {c['side']:<4} {c['post_type']:<8} "
              f"{c['best_bid']*100:>5.1f}c {c['best_ask']*100:>5.1f}c "
              f"{c['mid']*100:>5.2f}c {c['post_price']*100:>5.1f}c "
              f"{tick_str:>5} "
              f"{c['sim_prob']*100:>5.1f}c {c['edge_pp']:>4.1f}% "
              f"{c['kelly_f']*100:>5.1f}% {c['contracts']:>4d}  "
              f"safe (-{(c['best_ask']-c['post_price'])*100:.1f}c)")
    return candidates, prob_lookup


# ── Live-trade preconditions (I/O wrappers around maker_guard's pure guards) ──
def _parse_dt_to_ts(s):
    """Best-effort parse of a tee-time / last_updated string to a unix ts, or
    None. NOTE: naive strings are read in the server's local time. The live guard
    only uses these for (a) a tee-time window and (b) DataGolf-feed freshness — a
    RELATIVE check — and treats negative age as live, so minor TZ slop is safe
    (worst case = a missed window, never trading during play, since a live feed
    going fresh makes DataGolf override to halt regardless)."""
    if not s or not isinstance(s, str):
        return None
    s2 = s.strip().replace("UTC", "").replace("Z", "").strip()
    for f in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
              "%m/%d/%Y %H:%M", "%Y-%m-%d %I:%M%p", "%I:%M%p"):
        try:
            return _dtmod.datetime.strptime(s2, f).timestamp()
        except ValueError:
            continue
    return None


def _current_play_round():
    """Round currently in play / next to play = (round just completed) + 1, 1..4."""
    try:
        import sheet_config
        rc = int(float(sheet_config.get_param("round", "0") or 0))
    except Exception:
        rc = 0
    return max(1, min(4, rc + 1))


def _check_fairs_fresh():
    payload = _fetch_sim_fairs()
    return maker_guard.check_fairs_fresh_payload(
        payload, time.time(), tourney=tourney, current_round=_current_play_round())


def _check_risk_limits():
    """Realized-loss circuit breaker + balance-vs-caps sanity. (ok, reason).
    FAIL-CLOSED: if the portfolio endpoints are unreadable we refuse to trade
    rather than guess — losses freeing cap room is exactly the state we must
    never trade blind through."""
    try:
        caps = maker_guard.caps_from_env()
        now = time.time()
        if maker_guard._envf("MAKER_MAX_DAILY_LOSS_USD", 200.0) > 0:
            setts = [s for s in list_settlements()
                     if _is_golf_ticker(s.get("ticker", ""))]
            ok, reason = maker_guard.check_daily_loss(setts, now)
            if not ok:
                return (ok, reason)
        else:
            reason = "daily-loss breaker disabled"
        golf_positions = [p for p in list_positions()
                          if _is_golf_ticker(p.get("ticker", ""))]
        exposure = maker_guard.build_exposure(
            golf_positions, list_resting_orders(golf_only=True))
        ok2, reason2 = maker_guard.check_balance_caps(
            get_balance_usd(), exposure["total"], caps)
        if not ok2:
            return (ok2, reason2)
        return (True, f"{reason}; {reason2}")
    except Exception as e:
        return (False, f"risk limits unreadable ({e}) — fail-closed")


def _check_not_live():
    """(ok_to_trade, reason). ok = NOT live. Schedule (tee-times) is the default;
    the DataGolf live feed overrides on a confident signal. Best-effort I/O;
    fail-closed (assume live) when both are blind."""
    now = time.time()
    rnd = _current_play_round()
    api_key = os.getenv("DATAGOLF_API_KEY", "")
    sched = dg = None
    try:  # schedule from this round's tee times
        from api_utils import fetch_field_updates
        col = f"r{rnd}_teetime"
        fu = fetch_field_updates(api_key, teetime_col=col, fill_missing_teetimes=False)
        tts = [t for t in (_parse_dt_to_ts(str(x)) for x in fu[col].dropna().tolist()) if t]
        if tts:
            sched = maker_guard.schedule_live(min(tts), max(tts), now)
    except Exception as e:
        print(f"  [live] schedule (tee-times) unavailable: {e}")
    try:  # DataGolf live feed freshness
        from api_utils import fetch_live_stats
        ls = fetch_live_stats(rnd, api_key)
        lu = None
        if ls is not None and len(ls) and "last_updated" in ls.columns:
            lu = _parse_dt_to_ts(str(ls["last_updated"].iloc[0]))
        dg = maker_guard.datagolf_live(lu, now)
    except Exception as e:
        print(f"  [live] datagolf feed unavailable: {e}")
    is_live, reason = maker_guard.resolve_live(sched, dg)
    return (not is_live, f"round {rnd}: {reason}")


_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


def _fetch_maker_config():
    """Read the maker toggle config from the dashboard (/api/maker-config, public
    GET). Fail-safe: the pre-Wednesday rule defaults ON if unreachable (it's a
    safety rule). Returns a dict of toggles."""
    default = {"block_yes_outrights_pre_wed": True}
    url = os.getenv("KALSHI_EXEC_URL", "https://kalshi-exec.pages.dev") + "/api/maker-config"
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": _BROWSER_UA})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = _json.loads(r.read().decode())
        return {**default, **(data.get("config") or {})}
    except Exception as e:
        print(f"  [config] maker-config unreachable ({e}) — defaulting rule ON")
        return default


def _check_live_preconditions():
    """Full --live gate: kill switch -> fairs fresh (round-aware) -> risk limits
    (daily realized-loss breaker + balance-vs-caps) -> not live-round. Returns
    (ok, reason); the first failure wins. Each is fail-closed."""
    ok, reason = maker_guard.should_trade()
    if not ok:
        return (ok, reason)
    ok, reason = _check_fairs_fresh()
    if not ok:
        return (ok, reason)
    ok, reason = _check_risk_limits()
    if not ok:
        return (ok, reason)
    return _check_not_live()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="Actually post orders (default is dry-run print only)")
    ap.add_argument("--cancel-all", action="store_true",
                    help="Cancel every resting GOLF order owned by our key, then exit. "
                         "Non-golf orders are NEVER touched.")
    ap.add_argument("--cancel-script", action="store_true",
                    help="Cancel only the bot's script-placed golf orders "
                         "(client_order_id prefix match). Manual orders are "
                         "left resting. Softer than --cancel-all.")
    ap.add_argument("--list-orders", action="store_true",
                    help="Pure-GET auth smoke test. Lists resting orders with golf vs "
                         "non-golf breakdown, no DELETE or POST. Then exits.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap to top N candidates ranked by Kelly fraction desc. "
                         "Applies to both dry-run print and --live POSTs.")
    ap.add_argument("--preview", action="store_true",
                    help="Run scan, write proposals to permanent_data/maker_proposals.parquet, "
                         "and exit. No POSTs. Used by the maker_dashboard for "
                         "human review + edit before sending.")
    ap.add_argument("--no-matchups", action="store_true",
                    help="Skip H2H matchup scan entirely. Matchup cash reservation "
                         "is $0, so outright Kelly sizing sees full available balance.")
    ap.add_argument("--rungs", action="store_true",
                    help="Use the legacy static edge-rung ladder for OUTRIGHTS instead "
                         "of the passive-accumulation quote engine (the default).")
    ap.add_argument("--allow-pre-matchup", action="store_true",
                    help="Allow matchup scan to fall back to pre-tournament final_scores.npy "
                         "when no live final_scores file exists. ONLY safe pre-R1 — using stale "
                         "pre-tournament matchup probs mid-event will systematically mispice.")
    args = ap.parse_args()
    # Note: at module level, __main__ assignments are module-scoped (no function
    # frame), so this correctly mutates the module global used by scan_matchups.
    _allow_pre_matchup_fallback = bool(args.allow_pre_matchup)
    _skip_matchups = bool(args.no_matchups)
    _use_engine = not bool(args.rungs)  # passive-accumulation engine is the default outright path

    t0 = time.time()
    if args.list_orders:
        list_orders_smoke()
        _client.close()
        sys.exit(0)
    if args.cancel_all:
        cancel_all()
        _client.close()
        sys.exit(0)
    if args.cancel_script:
        cancel_script()
        _client.close()
        sys.exit(0)

    # ── Live-trade preconditions: kill switch -> sim fairs fresh -> NOT a live
    # round (schedule default + DataGolf override). Any failure halts --live and
    # PULLS the bot's own resting quotes so live risk comes off; manual orders are
    # left alone. (Computed before scan so a halted --live exits fast.) ──
    trade_ok, trade_reason = _check_live_preconditions()
    if args.live and not trade_ok:
        print(f"\n[HALT] not trading: {trade_reason}")
        print("[HALT] pulling the bot's resting script quotes (manual orders untouched)…")
        n = maker_guard.pull_script_quotes(
            lambda: list_resting_orders(golf_only=True), _is_script_order, cancel_order)
        print(f"[HALT] cancelled {n} script quote(s). Posting nothing. Exiting.")
        try:
            import maker_alerts
            maker_alerts.process_run(status="HALT",
                                     reason=" ".join(str(trade_reason).split()),
                                     post_report=None, fills=None, enabled=True)
        except Exception as _ae:
            print(f"  [alerts] alerting failed ({_ae}) — never blocks the run")
        _client.close()
        sys.exit(0)

    try:
        cands, prob_lookup = scan()
    except FileNotFoundError as _se:
        # No current sim fairs (e.g. between events). The fairs guard already
        # flags this; don't crash the (shadow) run — just quote nothing.
        print(f"\n[maker] no sim fairs to scan ({_se}). Nothing to quote this run.")
        cands, prob_lookup = [], {}
    if args.limit is not None and cands:
        before = len(cands)
        cands = sorted(cands, key=lambda c: c["kelly_f"], reverse=True)[:args.limit]
        print(f"\n[limit] kept top {len(cands)} of {before} candidates "
              f"(sorted by Kelly fraction desc):")
        for c in cands:
            print(f"  {c['ticker']:<28} {c['player'][:22]:<22} "
                  f"{c['market']:<7} {c['side']:<4} {c['post_type']:<8} "
                  f"post={c['post_price']*100:>5.1f}c "
                  f"(tick={c['tick']*100:.1f}c)  "
                  f"edge={c['edge_pp']:>4.1f}%  kelly={c['kelly_f']*100:>5.1f}%  "
                  f"qty={c['contracts']}")
    if args.preview:
        import datetime as _dt
        out_path = os.path.join("permanent_data", "maker_proposals.parquet")
        os.makedirs("permanent_data", exist_ok=True)

        # ── Kelly portfolio-sizing pass for OUTRIGHTS only ──────────────
        # Matchups stay at flat MATCHUP_LEVEL_CONTRACTS. Outrights get
        # resized to per-player joint-Kelly using the live Kalshi balance
        # minus matchup cash reservation, accounting for existing
        # exposure (positions + ALL resting golf orders). Solver failure
        # falls back to flat LEVEL_CONTRACTS sizing for outrights.
        try:
            from kelly_solver import compute_optimal_outright_sizing, _cand_key
            outright_cands = [c for c in cands if c.get("market") != "h2h"]
            matchup_cands = [c for c in cands if c.get("market") == "h2h"]
            sizing = compute_optimal_outright_sizing(
                outright_candidates=outright_cands,
                matchup_candidates=matchup_cands,
                prob_lookup=prob_lookup,
                kelly_fraction=0.5,
                max_total_fraction_per_player=0.25,
                verbose=True,
            )
            kept = []
            for c in cands:
                if c.get("market") == "h2h":
                    kept.append(c)
                    continue
                qty = sizing.get(_cand_key(c), 0)
                if qty < 1:
                    continue  # solver allocated nothing → drop from proposals
                c = dict(c)
                c["contracts"] = qty
                kept.append(c)
            print(f"  [kelly] outright proposals: {len(outright_cands)} candidates "
                  f"-> {len(kept) - len(matchup_cands)} after Kelly + existing-exposure")
            cands = kept
        except Exception as _ke:
            print(f"  [warn] Kelly sizing failed ({_ke}) — falling back to flat sizing")

        # ── Preview-time dedup for MATCHUPS only ───────────────────────
        # Matchups bypass Kelly's existing-exposure netting and are
        # flat-sized at 500. If we already have a same-price matchup
        # order resting, drop the duplicate proposal so the dashboard
        # doesn't show stale rows. Outright proposals are kept as-is so
        # you see the full Kelly slate — same-price collisions will still
        # be caught at send-time by /api/send's dedup.
        try:
            _existing_keys = {_resting_key(o)
                              for o in list_resting_orders(golf_only=True)}
            _before = len(cands)
            cands = [
                c for c in cands
                if c.get("market") != "h2h"
                or (c["ticker"], c["side"],
                    int(round(c["post_price"] * 1000))) not in _existing_keys
            ]
            _dropped = _before - len(cands)
            if _dropped:
                print(f"  [dedup] dropped {_dropped} matchup proposal(s) "
                      f"already resting at the same price")
        except Exception as _de:
            print(f"  [warn] matchup dedup failed ({_de}) — proposals may "
                  f"include same-price duplicates that will skip at send")

        # Serialize candidates to a flat DataFrame. Add a stable per-row id so
        # the dashboard can track edits, plus the scan timestamp.
        scan_ts = _dt.datetime.now().isoformat(timespec="seconds")
        rows = []
        for c in cands:
            rows.append({
                "row_id": f"{c['ticker']}|{c['side']}|"
                          f"{int(round(c['post_price'] * 1000))}",
                "scan_ts": scan_ts,
                "ticker": c["ticker"],
                "title": c.get("title", ""),
                "player": c["player"],
                "market": c["market"],
                "side": c["side"],
                "post_type": c["post_type"],
                "best_bid": float(c["best_bid"]),
                "best_ask": float(c["best_ask"]),
                "mid": float(c["mid"]),
                "post_price": float(c["post_price"]),
                "tick": float(c["tick"]),
                "sim_prob": float(c["sim_prob"]),
                "edge_pp": float(c["edge_pp"]),
                "kelly_f": float(c["kelly_f"]),
                "fill_pct": float(c.get("fill_pct", 0.0)),
                "volume": float(c.get("volume", 0.0)),
                "kelly_contracts": int(c["contracts"]),
                "edit_contracts": int(c["contracts"]),  # default = recommended
                "include": True,
            })
        df = pd.DataFrame(rows)
        df.to_parquet(out_path, index=False)
        print(f"\n[preview] wrote {len(df)} proposals to {out_path}")
        print(f"[preview] Launch dashboard: python -m maker_dashboard.server")
        _client.close()
        sys.exit(0)
    # ── Exposure governor: trim/drop candidates so held + resting + new $ stay
    # within caps (env-overridable, see maker_guard). Per-(ticker,side) caps
    # count current held + resting, so they also bound inventory. Applies to
    # --live AND dry-run so the shadow shows the real plan. Fail-closed on --live. ──
    gov_report = None
    if cands:
        try:
            # GOLF ONLY: filter positions to golf tickers so unrelated markets
            # (e.g. Bitcoin) don't consume the maker's caps.
            golf_positions = [p for p in list_positions()
                              if _is_golf_ticker(p.get("ticker", ""))]
            golf_resting = list_resting_orders(golf_only=True)
            exposure = maker_guard.build_exposure(golf_positions, golf_resting)
            # The bot's own resting quotes, keyed at exact price: a candidate at
            # the same price replaces (keeps/re-arms) its twin in reconcile, so
            # the governor nets it out instead of double-charging it.
            own_resting = {}
            for o in golf_resting:
                if not _is_script_order(o):
                    continue
                k = _resting_key(o)
                own_resting[k] = own_resting.get(k, 0.0) + _resting_count(o) * (k[2] / 1000.0)
            before = len(cands)
            _pe = sorted(exposure["per_event"].items(), key=lambda x: -x[1])[:4]
            print(f"\n[governor] held+resting golf exposure (netted): total ${exposure['total']:.0f}"
                  + (" | " + ", ".join(f"{e}=${v:.0f}" for e, v in _pe) if _pe else ""))
            cands, gov = maker_guard.apply_exposure_caps(cands, exposure,
                                                         own_resting=own_resting)
            gov_report = gov
            caps = gov["caps"]
            print(f"\n[governor] caps: market=${caps['per_market_usd']:.0f} "
                  f"event=${caps['per_event_usd']:.0f} total=${caps['total_usd']:.0f} "
                  f"new/run=${caps['max_new_usd_run']:.0f} orders/run={caps['max_orders_run']}")
            print(f"[governor] {before} candidate(s) -> {gov['kept']} kept "
                  f"({gov['trimmed']} trimmed, {gov['dropped']} dropped); "
                  f"committing ${gov['new_usd']:.0f} new across {gov['orders']} order(s)")
            for d in gov["dropped_detail"][:8]:
                print(f"    drop {d['ticker']} {d['side']} @{float(d['post_price'])*100:.1f}c -> {d['why']}")
            for d in gov["trimmed_detail"][:8]:
                print(f"    trim {d['ticker']} {d['side']} {d['from']}->{d['to']}")
        except Exception as e:
            print(f"\n[governor] FAILED to compute exposure ({e}).")
            if args.live:
                print("[governor] fail-closed under --live: posting nothing.")
                _client.close()
                sys.exit(1)
            print("[governor] dry-run: continuing WITHOUT caps applied.")

    # ── Write the cockpit snapshot (permanent_data/maker_state.json) for the
    # dashboard Maker tab. push_maker_state.py POSTs it to /api/maker. Written in
    # every mode (shadow / dry / live) so the cockpit always reflects the last run. ──
    _committed = (gov_report["new_usd"] if gov_report
                  else sum(float(c.get("post_price", 0)) * int(c.get("contracts", 0)) for c in cands))
    _state = {
        "ts": _dtmod.datetime.now().isoformat(timespec="seconds"),
        "mode": "shadow" if _shadow_mode() else ("live" if args.live else "dry"),
        "status": "TRADE" if trade_ok else "HALT",
        "reason": " ".join(str(trade_reason).split()),
        "tourney": tourney,
        "quotes": [{
            "ticker": c.get("ticker"), "player": c.get("player"), "market": c.get("market"),
            "side": c.get("side"), "price": float(c.get("post_price", 0)),
            "size": int(c.get("contracts", 0)), "edge_pp": float(c.get("edge_pp", 0)),
            "fair": float(c.get("sim_prob", 0)), "best_bid": float(c.get("best_bid", 0)),
            "best_ask": float(c.get("best_ask", 0)), "action": c.get("engine_action", ""),
        } for c in sorted(cands, key=lambda x: x.get("edge_pp", 0), reverse=True)],
        "totals": {"quotes": len(cands), "committed": round(_committed, 2),
                   "caps": (gov_report["caps"] if gov_report else None)},
        "rules": globals().get("_rule_status") or {
            "block_yes_pre_wed": {"enabled": True,
                                  "active_now": bool(maker_guard.before_wed_cutoff()),
                                  "blocked": 0}},
        "fairs": {
            "generated_at": _SIM_FAIRS_META.get("generated_at"),
            "sim_run_at": _SIM_FAIRS_META.get("sim_run_at"),
            "round": _SIM_FAIRS_META.get("round"),
            "source": _SIM_FAIRS_META.get("source"),
            # age is measured from sim_run_at (when the sim ran) — the stamp the
            # freshness gate trusts; falls back to generated_at for legacy payloads.
            "age_hours": (round((time.time() - _ft) / 3600.0, 1)
                          if (_ft := (maker_guard.parse_fairs_ts(_SIM_FAIRS_META.get("sim_run_at"))
                                      or maker_guard.parse_fairs_ts(_SIM_FAIRS_META.get("generated_at"))))
                          else None),
        },
    }
    try:
        os.makedirs("permanent_data", exist_ok=True)
        with open(os.path.join("permanent_data", "maker_state.json"), "w", encoding="utf-8") as _f:
            _json.dump(_state, _f, indent=2)
        print(f"  [maker-state] wrote permanent_data/maker_state.json "
              f"({_state['status']}, {len(cands)} quote(s))")
    except Exception as _e:
        print(f"  [maker-state] write failed: {_e}")

    _post_report = None
    if args.live:
        if not cands:
            # Cancel-only reconcile: with an empty candidate set every resting
            # script quote is stale (fairs gone / between events / all gated out)
            # and must be pulled NOW, not left at stale prices until its TTL.
            print("\n[live] no candidates — cancel-only reconcile "
                  "(pulling any resting script quotes).")
        _post_report = reconcile_and_post(cands)
        print(f"\n[maker] done in {time.time()-t0:.1f}s")
    else:
        print(f"\n[maker] {len(cands)} intents in {time.time()-t0:.1f}s "
              f"— DRY RUN, nothing sent. Re-run with --live to post.")
        # ── structured shadow summary (pipe-delimited; parsed by shadow_digest.py) ──
        committed = (gov_report["new_usd"] if gov_report
                     else sum(float(c.get("post_price", 0)) * int(c.get("contracts", 0)) for c in cands))
        reason = " ".join(str(trade_reason).split())  # collapse whitespace/newlines
        print(f"[SHADOW] precond={'TRADE' if trade_ok else 'HALT'} | quotes={len(cands)} "
              f"| committed={committed:.0f} | reason={reason}")
        for c in sorted(cands, key=lambda x: x.get("edge_pp", 0), reverse=True)[:12]:
            print(f"[SHADOW-Q] {c.get('player', '?')} | {c.get('market', '?')} | "
                  f"{c.get('side', '?')} | {float(c.get('post_price', 0)) * 100:.1f} "
                  f"| {float(c.get('edge_pp', 0)):+.1f}")
        if not trade_ok:
            print(f"[guard] NOTE: --live precondition NOT met ({trade_reason}); "
                  f"--live would pull quotes and post nothing.")

    # ── Dead-man alerting (Telegram): HALT<->TRADE transitions, post-fail streaks,
    # new fills. Best-effort — can never break a maker run. Active under --live
    # (or MAKER_ALERTS=1 to force in dry/shadow). ──
    try:
        import maker_alerts
        _alerts_on = args.live or maker_alerts.alerts_forced()
        _new_fills = None  # None = not fetched (keeps the fill watermark intact)
        if _alerts_on:
            try:
                _new_fills = [f for f in list_fills()
                              if _is_golf_ticker(f.get("ticker", ""))]
            except Exception as _fe:
                print(f"  [alerts] fills fetch failed ({_fe})")
        maker_alerts.process_run(status=_state["status"], reason=_state["reason"],
                                 post_report=_post_report, fills=_new_fills,
                                 enabled=_alerts_on)
    except Exception as _ae:
        print(f"  [alerts] alerting failed ({_ae}) — never blocks the run")
    _client.close()
