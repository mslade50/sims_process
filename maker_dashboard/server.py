"""Local Flask server for the Kalshi maker preview-and-send dashboard.

Run with:
    set KALSHI_MAKER_ENABLED=1
    python -m maker_dashboard.server

Bound to 127.0.0.1:8051 by default — never exposed to the network. The
KALSHI_MAKER_ENABLED env-var gate is the explicit kill-switch so this
can never run if the Render Procfile (or anything else) tries to start it.

Routes:
    GET  /                  index.html
    GET  /static/<path>     css/js
    GET  /api/proposals     latest proposals (JSON) from permanent_data/maker_proposals.parquet
    POST /api/send          accept user-edited row set, route each through post_limit()
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory


HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
PROPOSALS_PATH = os.path.join(REPO_ROOT, "permanent_data", "maker_proposals.parquet")
PNL_HISTORY_PATH = os.path.join(REPO_ROOT, "permanent_data", "pnl_history.parquet")
# Durable raw-API stores. Kalshi's /portfolio/fills feed only retains ~2 months,
# so we persist every fill/settlement we've ever seen and only top up the recent
# pages on each load — otherwise historical cost basis vanishes as fills age out.
FILLS_STORE_PATH = os.path.join(REPO_ROOT, "permanent_data", "kalshi_fills.parquet")
SETTLEMENTS_STORE_PATH = os.path.join(REPO_ROOT, "permanent_data", "kalshi_settlements.parquet")

# In-memory PnL cache. The /markets batched call is cheap but we still don't
# want to hammer Kalshi on every UI filter toggle. Frontend reads from cache
# until it ages out (or the user clicks Refresh).
_PNL_CACHE = {"data": None, "fetched_at": None}
_PNL_CACHE_TTL_SEC = 60

# NoVig edges cache. Public GraphQL is fine to hit on demand, but the user
# can spam-click filters; 60s in-memory is plenty.
_NOVIG_CACHE = {"data": None, "fetched_at": None, "tournament_name": ""}
_NOVIG_CACHE_TTL_SEC = 60

# Ticker prefix → market_type. Mirrors OUTRIGHT_SERIES in kalshi_maker plus h2h.
_MARKET_TYPE_PREFIXES = (
    ("KXPGATOUR", "winner"),
    ("KXPGATOP5", "top_5"),
    ("KXPGATOP10", "top_10"),
    ("KXPGATOP20", "top_20"),
    ("KXPGAH2H", "h2h"),
)


def _classify_ticker(ticker):
    """Parse (market_type, event_code, player_code) from a Kalshi PGA ticker.
    e.g. KXPGATOP10-PGC26-RFOX -> ('top_10', 'PGC26', 'RFOX'). Returns
    ('', '', '') if the ticker doesn't match the known PGA shape."""
    for prefix, mt in _MARKET_TYPE_PREFIXES:
        if ticker.startswith(prefix + "-"):
            parts = ticker.split("-", 2)
            event_code = parts[1] if len(parts) >= 2 else ""
            player_code = parts[2] if len(parts) >= 3 else ""
            return mt, event_code, player_code
    return "", "", ""


def _load_store(path):
    """Load a persisted raw-API store (fills/settlements) as a list of dicts.
    Returns [] if the file is missing or unreadable."""
    if not os.path.exists(path):
        return []
    try:
        return pd.read_parquet(path).to_dict(orient="records")
    except Exception as e:
        print(f"[pnl] failed to read store {os.path.basename(path)}: {e}")
        return []


def _persist_store(records, path, id_field):
    """Atomically write the merged record set, deduped on id_field (last wins).
    tempfile + os.replace so a crash mid-write can't corrupt the store."""
    import tempfile
    if not records:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame(records).drop_duplicates(subset=[id_field], keep="last")
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".parquet")
        os.close(fd)
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[pnl] failed to persist store {os.path.basename(path)}: {e}")


def _fetch_incremental(authed_request, base_path, list_key, id_field, known_ids):
    """Paginate a Kalshi portfolio endpoint newest-first, returning only the
    records whose id_field is not already in known_ids.

    The feed is append-only and time-ordered, so once a page contributes
    nothing new we've reached the part of history we already hold and can stop.
    On a cold store (known_ids empty) this walks every page (full first sync);
    afterwards it only pulls the most-recent page(s). Raises RuntimeError on a
    non-200 so the caller can surface it like the old inline fetch did."""
    new_records, cursor = [], None
    while True:
        path = f"{base_path}?limit=200"
        if cursor:
            path += f"&cursor={cursor}"
        r = authed_request("GET", path)
        if r.status_code != 200:
            raise RuntimeError(f"{list_key} {r.status_code}: {r.text[:200]}")
        data = r.json()
        page = data.get(list_key, [])
        page_new = [rec for rec in page if rec.get(id_field) not in known_ids]
        new_records.extend(page_new)
        cursor = data.get("cursor")
        if not cursor:
            break
        # Reached known history — everything older is already stored.
        if known_ids and not page_new:
            break
    return new_records


def _sync_pnl_stores(authed_request):
    """Top up the durable fills/settlements stores and return the merged record
    lists + how many were newly pulled. Shared by the /api/pnl route and the
    headless `--sync` CLI. Cron the CLI regularly (e.g. alongside the nightly
    job) so fills are captured before they age out of Kalshi's ~2-month feed."""
    stored_fills = _load_store(FILLS_STORE_PATH)
    known_fill_ids = {f.get("fill_id") for f in stored_fills}
    new_fills = _fetch_incremental(
        authed_request, "/trade-api/v2/portfolio/fills",
        "fills", "fill_id", known_fill_ids)
    all_fills = stored_fills + new_fills
    if new_fills:
        _persist_store(all_fills, FILLS_STORE_PATH, "fill_id")

    stored_setts = _load_store(SETTLEMENTS_STORE_PATH)
    known_setts = {s.get("ticker") for s in stored_setts}
    new_setts = _fetch_incremental(
        authed_request, "/trade-api/v2/portfolio/settlements",
        "settlements", "ticker", known_setts)
    settlements = stored_setts + new_setts
    if new_setts:
        _persist_store(settlements, SETTLEMENTS_STORE_PATH, "ticker")
    return all_fills, settlements, len(new_fills), len(new_setts)


def _fetch_market_summary(tickers):
    """Batch-fetch {ticker: {title, yes_bid, yes_ask}} from /markets. One call
    handles 100 tickers; we chunk for safety. Far cheaper than per-ticker
    /orderbook calls (which are also throttled by Kalshi)."""
    sys.path.insert(0, REPO_ROOT)
    from kalshi_maker import _client, KALSHI_API
    todo = list(dict.fromkeys(t for t in tickers if t))
    CHUNK = 100
    out = {}
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i + CHUNK]
        try:
            r = _client.get(f"{KALSHI_API}/markets",
                            params={"tickers": ",".join(chunk), "limit": CHUNK})
            if r.status_code == 200:
                for m in r.json().get("markets", []):
                    t = m.get("ticker", "")
                    yb = m.get("yes_bid_dollars")
                    ya = m.get("yes_ask_dollars")
                    if yb is None:
                        yb = (m.get("yes_bid") or 0) / 100.0
                    if ya is None:
                        ya = (m.get("yes_ask") or 0) / 100.0
                    out[t] = {
                        "title": m.get("title", ""),
                        "yes_bid": float(yb or 0),
                        "yes_ask": float(ya or 0),
                        "status": m.get("status", ""),
                    }
        except Exception:
            pass
    return out

app = Flask(__name__, static_folder=os.path.join(HERE, "static"))

# Module-level title cache keyed by ticker. We batch-fetch on demand and
# never expire — within a session, ticker→title is effectively static
# (market titles don't change once created).
_TITLE_CACHE = {}


def _fetch_titles(tickers):
    """Populate _TITLE_CACHE for any tickers we haven't seen yet.
    Returns the merged {ticker: title} subset for the requested tickers.

    Kalshi's /markets endpoint accepts `tickers=t1,t2,...` (comma-separated)
    and returns titles in one round-trip. We chunk to 100 per call to stay
    well under any URL-length cap.
    """
    sys.path.insert(0, REPO_ROOT)
    from kalshi_maker import _client, KALSHI_API
    todo = [t for t in tickers if t and t not in _TITLE_CACHE]
    todo = list(dict.fromkeys(todo))  # dedup preserving order
    CHUNK = 100
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i + CHUNK]
        try:
            r = _client.get(f"{KALSHI_API}/markets",
                            params={"tickers": ",".join(chunk), "limit": CHUNK})
            r.raise_for_status()
            for m in r.json().get("markets", []):
                _TITLE_CACHE[m.get("ticker", "")] = m.get("title", "")
        except Exception:
            # Cache an empty string so we don't retry endlessly; the UI will
            # just show the raw ticker for these.
            for t in chunk:
                _TITLE_CACHE.setdefault(t, "")
    return {t: _TITLE_CACHE.get(t, "") for t in tickers}


def _enabled():
    return os.getenv("KALSHI_MAKER_ENABLED") == "1"


@app.route("/")
def index():
    if not _enabled():
        return (
            "<h1>Kalshi maker dashboard disabled</h1>"
            "<p>Set <code>KALSHI_MAKER_ENABLED=1</code> in the local env to enable.</p>",
            403,
        )
    return send_from_directory(HERE, "index.html")


@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory(os.path.join(HERE, "static"), path)


@app.route("/api/proposals")
def proposals():
    if not _enabled():
        return jsonify({"error": "disabled"}), 403
    if not os.path.exists(PROPOSALS_PATH):
        return jsonify({"rows": [], "scan_ts": None, "stale_minutes": None,
                        "note": "No proposals file. Run `python kalshi_maker.py --preview`."})
    df = pd.read_parquet(PROPOSALS_PATH)
    rows = df.to_dict(orient="records")
    scan_ts = rows[0].get("scan_ts") if rows else None
    stale_minutes = None
    if scan_ts:
        try:
            stale_minutes = int((dt.datetime.now() - dt.datetime.fromisoformat(scan_ts))
                                .total_seconds() // 60)
        except Exception:
            pass
    return jsonify({"rows": rows, "scan_ts": scan_ts, "stale_minutes": stale_minutes})


@app.route("/api/send", methods=["POST"])
def send():
    if not _enabled():
        return jsonify({"error": "disabled"}), 403
    body = request.get_json(silent=True) or {}
    rows = body.get("rows") or []
    if not rows:
        return jsonify({"error": "no rows submitted"}), 400

    # Optional global order expiry (unix seconds). When present, every order
    # in this batch is posted with time_in_force=good_till_canceled and the
    # same expiration_ts. Kalshi auto-cancels any unfilled remainder at that
    # wall-clock time. Absent / None = true GTC (the prior default).
    raw_exp = body.get("expiration_ts")
    if raw_exp in (None, "", 0, "0"):
        expiration_ts = None
    else:
        try:
            expiration_ts = int(raw_exp)
        except (TypeError, ValueError):
            return jsonify({"error": f"bad expiration_ts: {raw_exp!r}"}), 400
        # Sanity: must be in the future. A past timestamp would be rejected
        # by Kalshi per-order; fail the whole batch upfront with a clear msg.
        if expiration_ts <= int(dt.datetime.now().timestamp()):
            return jsonify({"error": f"expiration_ts {expiration_ts} is in the past"}), 400

    # Lazy import so the page can load even if kalshi_maker has issues at startup.
    sys.path.insert(0, REPO_ROOT)
    from kalshi_maker import post_limit, _is_golf_ticker, list_resting_orders, _resting_key

    # Snapshot existing (ticker, side, milli-dollars) so we can dedup against
    # orders the user might not realize are already on the book.
    try:
        existing_keys = {_resting_key(o) for o in list_resting_orders(golf_only=True)}
    except Exception as e:
        return jsonify({"error": f"list_resting_orders failed: {e}"}), 500

    details = []
    posted = failed = skipped = 0
    for row in rows:
        ticker = row.get("ticker", "")
        side = row.get("side", "")
        try:
            post_price = float(row.get("post_price"))
            contracts = int(row.get("edit_contracts", 0))
        except (TypeError, ValueError):
            details.append({**_row_summary(row), "status": "failed",
                            "message": "bad post_price or edit_contracts"})
            failed += 1
            continue
        if contracts <= 0:
            skipped += 1
            details.append({**_row_summary(row), "status": "skipped",
                            "message": "edit_contracts <= 0"})
            continue
        if not _is_golf_ticker(ticker):
            details.append({**_row_summary(row), "status": "failed",
                            "message": "non-golf ticker — refusing"})
            failed += 1
            continue
        key = (ticker, side, int(round(post_price * 1000)))
        if key in existing_keys:
            details.append({**_row_summary(row), "status": "skipped",
                            "message": "already resting at this price"})
            skipped += 1
            continue
        ok, info = post_limit(ticker, side, post_price, contracts,
                              expiration_ts=expiration_ts)
        if ok:
            posted += 1
            details.append({**_row_summary(row), "status": "posted",
                            "message": f"order_id={info}"})
        else:
            failed += 1
            details.append({**_row_summary(row), "status": "failed",
                            "message": str(info)[:200]})

    return jsonify({"posted": posted, "failed": failed, "skipped": skipped,
                    "details": details, "expiration_ts": expiration_ts})


def _row_summary(row):
    return {
        "row_id": row.get("row_id"),
        "ticker": row.get("ticker"),
        "side": row.get("side"),
        "post_price": row.get("post_price"),
        "contracts": row.get("edit_contracts"),
    }


@app.route("/api/open-orders")
def open_orders():
    if not _enabled():
        return jsonify({"error": "disabled"}), 403
    sys.path.insert(0, REPO_ROOT)
    from kalshi_maker import (
        list_resting_orders, _is_golf_ticker, _is_script_order,
        _resting_key, _resting_count, _extract_player, _norm,
    )
    raw = list_resting_orders(golf_only=False)
    titles = _fetch_titles([o.get("ticker", "") for o in raw])
    rows = []
    for o in raw:
        t = o.get("ticker", "")
        side = o.get("side", "")
        _, _, milli = _resting_key(o)
        price = milli / 1000.0
        rem = _resting_count(o)
        try:
            initial = int(round(float(o.get("initial_count_fp") or 0)))
        except (TypeError, ValueError):
            initial = 0
        try:
            fill = int(round(float(o.get("fill_count_fp") or 0)))
        except (TypeError, ValueError):
            fill = 0
        title = titles.get(t, "")
        player_raw = _extract_player(title)
        player = _norm(player_raw) if player_raw else ""
        rows.append({
            "order_id": o.get("order_id"),
            "ticker": t,
            "title": title,
            "player": player,
            "side": side,
            "price_dollars": price,
            "remaining_contracts": rem,
            "initial_count": initial,
            "fill_count": fill,
            "is_golf": _is_golf_ticker(t),
            "scope": "golf" if _is_golf_ticker(t) else "other",
            "placed_by": "script" if _is_script_order(o) else "manual",
            "created_time": o.get("created_time"),
            "client_order_id": o.get("client_order_id") or "",
        })
    # Sort by remaining contracts desc (per user spec).
    rows.sort(key=lambda r: r["remaining_contracts"], reverse=True)
    return jsonify({"rows": rows, "fetched_ts": dt.datetime.now().isoformat(timespec="seconds")})


@app.route("/api/positions")
def positions():
    if not _enabled():
        return jsonify({"error": "disabled"}), 403
    sys.path.insert(0, REPO_ROOT)
    from kalshi_maker import _authed_request, _is_golf_ticker

    out, cursor = [], None
    while True:
        path = "/trade-api/v2/portfolio/positions?limit=200"
        if cursor:
            path += f"&cursor={cursor}"
        r = _authed_request("GET", path)
        if r.status_code != 200:
            return jsonify({"error": f"kalshi {r.status_code}: {r.text[:200]}"}), 500
        data = r.json()
        out.extend(data.get("market_positions", []))
        cursor = data.get("cursor")
        if not cursor:
            break

    titles = _fetch_titles([p.get("ticker", "") for p in out])
    rows = []
    for p in out:
        ticker = p.get("ticker", "")
        try:
            pos_fp = float(p.get("position_fp") or 0)
        except (TypeError, ValueError):
            pos_fp = 0.0
        # Drop zero positions — Kalshi returns every market we've ever
        # touched, including fully-closed ones.
        if abs(pos_fp) < 1e-6:
            continue
        side = "yes" if pos_fp > 0 else "no"
        contracts = int(round(abs(pos_fp)))
        try:
            exposure = float(p.get("market_exposure_dollars") or 0)
        except (TypeError, ValueError):
            exposure = 0.0
        try:
            realized = float(p.get("realized_pnl_dollars") or 0)
        except (TypeError, ValueError):
            realized = 0.0
        try:
            fees = float(p.get("fees_paid_dollars") or 0)
        except (TypeError, ValueError):
            fees = 0.0
        avg_cost = exposure / contracts if contracts else 0.0
        max_gain = max(contracts - exposure, 0.0)  # if binary resolves our way
        rows.append({
            "ticker": ticker,
            "title": titles.get(ticker, ""),
            "side": side,
            "contracts": contracts,
            "avg_cost_dollars": avg_cost,
            "exposure_dollars": exposure,         # $ at risk if it goes against us
            "max_gain_dollars": max_gain,         # $ we'd win if it goes our way
            "realized_pnl_dollars": realized,
            "fees_paid_dollars": fees,
            "net_pnl_dollars": realized - fees,
            "resting_orders_count": int(p.get("resting_orders_count") or 0),
            "last_updated_ts": p.get("last_updated_ts"),
            "is_golf": _is_golf_ticker(ticker),
        })
    # Default sort: biggest exposure first
    rows.sort(key=lambda r: r["exposure_dollars"], reverse=True)
    return jsonify({"rows": rows,
                    "fetched_ts": dt.datetime.now().isoformat(timespec="seconds")})


@app.route("/api/fills")
def fills():
    if not _enabled():
        return jsonify({"error": "disabled"}), 403
    sys.path.insert(0, REPO_ROOT)
    from kalshi_maker import _authed_request, _is_golf_ticker

    # Paginate up to a reasonable cap. Default limit handles the common
    # "show me recent fills" use; raise via ?max= if needed.
    try:
        max_fills = int(request.args.get("max", 500))
    except ValueError:
        max_fills = 500

    out, cursor = [], None
    while len(out) < max_fills:
        path = "/trade-api/v2/portfolio/fills?limit=200"
        if cursor:
            path += f"&cursor={cursor}"
        r = _authed_request("GET", path)
        if r.status_code != 200:
            return jsonify({"error": f"kalshi {r.status_code}: {r.text[:200]}"}), 500
        data = r.json()
        out.extend(data.get("fills", []))
        cursor = data.get("cursor")
        if not cursor:
            break
    out = out[:max_fills]

    titles = _fetch_titles([f.get("ticker") or f.get("market_ticker") or "" for f in out])
    rows = []
    for f in out:
        ticker = f.get("ticker") or f.get("market_ticker") or ""
        side = f.get("side", "")
        try:
            count = int(round(float(f.get("count_fp") or 0)))
        except (TypeError, ValueError):
            count = 0
        # Fill price for our side = what we paid per contract
        try:
            if side == "yes":
                fill_price = float(f.get("yes_price_dollars") or 0)
            else:
                fill_price = float(f.get("no_price_dollars") or 0)
        except (TypeError, ValueError):
            fill_price = 0
        try:
            fee = float(f.get("fee_cost") or 0)
        except (TypeError, ValueError):
            fee = 0
        rows.append({
            "fill_id": f.get("fill_id"),
            "trade_id": f.get("trade_id"),
            "order_id": f.get("order_id"),
            "ticker": ticker,
            "title": titles.get(ticker, ""),
            "side": side,
            "action": f.get("action", ""),
            "count": count,
            "fill_price_dollars": fill_price,
            "yes_price_dollars": float(f.get("yes_price_dollars") or 0),
            "no_price_dollars": float(f.get("no_price_dollars") or 0),
            "is_taker": bool(f.get("is_taker")),
            "fee_cost": fee,
            "created_time": f.get("created_time"),
            "ts": f.get("ts"),
            "is_golf": _is_golf_ticker(ticker),
        })
    return jsonify({"rows": rows,
                    "fetched_ts": dt.datetime.now().isoformat(timespec="seconds"),
                    "count": len(rows)})


@app.route("/api/pnl")
def pnl():
    """All-time PnL across every Kalshi market the account has ever traded.

    Source-of-truth model (after two false starts):
      - /portfolio/positions: CURRENT state only. realized_pnl_dollars zeros
        out post-settlement, so it's not historical PnL.
      - /portfolio/settlements: ONLY captures the contracts held to
        settlement. Pre-settlement round-trip gains/losses are NOT in the
        cost basis — yes_total_cost_dollars only counts purchases that
        weren't closed out before expiry. Mid-event sells that recouped
        cost are invisible in this view.
      - /portfolio/fills: EVERY trade. Walking fills gives the true
        cashflow (sell_proceeds - buy_costs - fees). Combined with
        settlement revenue (for held-to-settlement payouts) and current
        position MTM (for open positions), this is correct accounting.

    Data flow:
      1. /portfolio/fills (paginated) → per-ticker cashflow + fees.
      2. /portfolio/settlements → settlement revenue per ticker.
      3. /portfolio/positions → current open position per ticker (for MTM).
      4. /markets (batched) → titles for all tickers + bid/ask for opens.
      5. permanent_data/pnl_history.parquet → title cache so we don't
         re-fetch titles for ancient tickers every minute.

    Definitions (slight abuse of terms, but matches dashboard UX):
      - realized_pnl = cashflow + settlement_revenue - fees
        (For fully-closed/settled tickers, this IS the final PnL.
        For still-open tickers, it's the cash crystallized so far —
        includes the cost basis of currently-held contracts as a negative,
        which gets offset by unrealized below.)
      - unrealized_pnl = position_count * current_mark (open positions only)
      - net_pnl = realized + unrealized
    A position with no sells has realized = -cost - fees and unrealized =
    position*mark; their sum is the correct (mark-cost)*position - fees.

    Cache: 60s in-memory. ?refresh=1 to force.
    """
    if not _enabled():
        return jsonify({"error": "disabled"}), 403

    force = request.args.get("refresh") == "1"
    now = dt.datetime.now()
    if (not force
            and _PNL_CACHE["data"] is not None
            and _PNL_CACHE["fetched_at"]
            and (now - _PNL_CACHE["fetched_at"]).total_seconds() < _PNL_CACHE_TTL_SEC):
        return jsonify({**_PNL_CACHE["data"], "from_cache": True})

    sys.path.insert(0, REPO_ROOT)
    from kalshi_maker import _authed_request, _is_golf_ticker
    from collections import defaultdict as _dd

    # ── 1+2. Fills & settlements — durable store + incremental top-up. We
    # persist every record ever seen (fills age out of Kalshi's feed after
    # ~2 months) and only pull pages newer than what we already hold. ──
    try:
        all_fills, settlements, n_new_fills, n_new_setts = _sync_pnl_stores(
            _authed_request)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    settlements_by_ticker = {s["ticker"]: s for s in settlements}

    # ── 3. Fetch current positions (for is_open + MTM). ─────────────────
    raw_positions, cursor = [], None
    while True:
        path = "/trade-api/v2/portfolio/positions?limit=200"
        if cursor:
            path += f"&cursor={cursor}"
        r = _authed_request("GET", path)
        if r.status_code != 200:
            return jsonify({"error": f"positions {r.status_code}: {r.text[:200]}"}), 500
        data = r.json()
        raw_positions.extend(data.get("market_positions", []))
        cursor = data.get("cursor")
        if not cursor:
            break
    pos_by_ticker = {}
    for p in raw_positions:
        try:
            pos_fp = float(p.get("position_fp") or 0)
        except (TypeError, ValueError):
            continue
        if abs(pos_fp) < 1e-6:
            continue
        pos_by_ticker[p.get("ticker", "")] = pos_fp

    # ── 4. Load historical ledger (title cache). ────────────────────────
    try:
        history = (pd.read_parquet(PNL_HISTORY_PATH) if os.path.exists(PNL_HISTORY_PATH)
                   else pd.DataFrame())
    except Exception:
        history = pd.DataFrame()
    cached_titles = ({} if history.empty
                     else {r["ticker"]: r.get("title", "")
                           for r in history.to_dict(orient="records")})

    # ── 5. Walk fills chronologically per ticker. ──────────────────────
    # Kalshi cashflow model: every fill (buy OR sell) adds inventory to
    # `fill.side` at the side-specific price and DEBITS the cash cost.
    # "Sell yes" is functionally "buy no" — the matching engine has us
    # acquire NO inventory at no_price as our share of a newly created
    # YES+NO pair. Kalshi reports it as action=sell on fill.side=no.
    # When YES and NO inventories coexist, Kalshi auto-converts pairs to
    # $1 cash apiece. `invested` is the gross $ paid across all fills
    # (the denominator for ROI).
    fills_by_ticker = _dd(list)
    for f in all_fills:
        tk = f.get("ticker") or f.get("market_ticker") or ""
        if tk:
            fills_by_ticker[tk].append(f)
    for tk in fills_by_ticker:
        fills_by_ticker[tk].sort(key=lambda f: f.get("ts") or 0)

    def _empty_bucket():
        return {"cashflow": 0.0, "fees": 0.0, "invested": 0.0,
                "yes_long": 0.0, "no_long": 0.0,
                "yes_buys": 0, "yes_sells": 0,
                "no_buys": 0, "no_sells": 0,
                "auto_converted": 0.0,
                "first_ts": None, "last_ts": None}

    by_t = {}
    for ticker, fs in fills_by_ticker.items():
        b = _empty_bucket()
        for f in fs:
            side = f.get("side", "")
            action = f.get("action", "")
            try:
                count = float(f.get("count_fp") or 0)
            except (TypeError, ValueError):
                count = 0.0
            if count <= 0:
                continue
            try:
                yes_p = float(f.get("yes_price_dollars") or 0)
                no_p = float(f.get("no_price_dollars") or 0)
            except (TypeError, ValueError):
                yes_p = no_p = 0.0
            price = yes_p if side == "yes" else no_p
            try:
                fee = float(f.get("fee_cost") or 0)
            except (TypeError, ValueError):
                fee = 0.0

            cnt_int = int(round(count))
            if side == "yes":
                b["yes_long"] += count
                if action == "buy":
                    b["yes_buys"] += cnt_int
                else:
                    b["yes_sells"] += cnt_int
            else:
                b["no_long"] += count
                if action == "buy":
                    b["no_buys"] += cnt_int
                else:
                    b["no_sells"] += cnt_int
            cost = price * count
            b["cashflow"] -= cost
            b["invested"] += cost
            b["fees"] += fee

            # Auto-conversion: any time both inventories are > 0, Kalshi
            # nets pairs into $1 cash each. Do it inline so subsequent
            # fills see the post-conv inventory (matters for /positions
            # cross-check on still-open tickers).
            if b["yes_long"] > 0 and b["no_long"] > 0:
                pairs = min(b["yes_long"], b["no_long"])
                b["yes_long"] -= pairs
                b["no_long"] -= pairs
                b["cashflow"] += pairs
                b["auto_converted"] += pairs

            ts = f.get("created_time") or f.get("ts") or ""
            if ts:
                if b["first_ts"] is None or str(ts) < str(b["first_ts"]):
                    b["first_ts"] = ts
                if b["last_ts"] is None or str(ts) > str(b["last_ts"]):
                    b["last_ts"] = ts
        by_t[ticker] = b

    # Settled-but-no-fills (super rare, e.g. legacy data) — still surface
    # the row with whatever the settlement record alone implies.
    for tk in settlements_by_ticker:
        if tk not in by_t:
            by_t[tk] = _empty_bucket()

    # ── 6. Batched /markets — title + bid/ask. ──────────────────────────
    all_tickers = list(by_t.keys())
    need_summary = [t for t in all_tickers
                    if t not in cached_titles or t in pos_by_ticker]
    summaries = _fetch_market_summary(need_summary)

    # ── 7. Build rows. ──────────────────────────────────────────────────
    rows = []
    new_for_ledger = []
    for ticker, b in by_t.items():
        cashflow = b["cashflow"]
        fees = b["fees"]
        sett = settlements_by_ticker.get(ticker)
        market_result = ""
        settled_time = None
        # Held-to-settlement counts/costs from the settlement record. Unlike the
        # /fills feed (which Kalshi only retains for ~2 months) these never age
        # out, so they are the authoritative cost basis for old settled tickers.
        held_yes = held_no = settle_cost = settle_fee = 0.0
        if sett is not None:
            market_result = sett.get("market_result", "")
            settled_time = sett.get("settled_time")
            held_yes = float(sett.get("yes_count_fp") or 0)
            held_no = float(sett.get("no_count_fp") or 0)
            settle_cost = (float(sett.get("yes_total_cost_dollars") or 0)
                           + float(sett.get("no_total_cost_dollars") or 0))
            settle_fee = float(sett.get("fee_cost") or 0)
        # Settlement payout = winning contracts held to settlement * $1. We do
        # NOT use the settlement `revenue` field: a recent Kalshi schema change
        # left it zero or partial on many winning tickers, which silently
        # understated wins. The count*result payout is always exact for binary
        # markets.
        if market_result == "yes":
            payout = held_yes
        elif market_result == "no":
            payout = held_no
        else:
            payout = 0.0

        pos_fp = pos_by_ticker.get(ticker)
        is_open = pos_fp is not None

        market_type, event_code, player_code = _classify_ticker(ticker)
        sm = summaries.get(ticker) or {}
        title = sm.get("title") or cached_titles.get(ticker, "")

        mark = None
        unreal = 0.0
        settlement_rev = 0.0
        if is_open:
            # ── OPEN: cost basis from fills (complete for in-window tickers),
            # valued at the current mid. ──
            side = "yes" if pos_fp > 0 else "no"
            contracts = int(round(abs(pos_fp)))
            realized = cashflow - fees
            cost_basis = abs(cashflow)
            invested = b["invested"]
            avg_cost = (-cashflow / contracts) if contracts > 0 else 0.0
            # MTM. Use post-auto-conv yes_long/no_long so each leg is valued at
            # its own mid (matters when raw fills span both sides before net).
            yes_bid = sm.get("yes_bid", 0.0)
            yes_ask = sm.get("yes_ask", 0.0)
            if yes_bid > 0 and yes_ask > 0 and yes_ask >= yes_bid:
                yes_mid = (yes_bid + yes_ask) / 2.0
                unreal = b["yes_long"] * yes_mid + b["no_long"] * (1.0 - yes_mid)
                mark = (unreal / contracts) if contracts else 0.0
            else:
                # Thin/empty book — fall back to avg cost (no MTM PnL).
                mark = (abs(cashflow) / contracts) if contracts else 0.0
                unreal = mark * contracts
        elif sett is not None:
            # ── SETTLED: price off the settlement record so realized PnL is
            # immune to fills aging out. A fills-only realized would credit the
            # payout against a vanished cost basis (phantom profit) for any
            # ticker older than the ~2-month fills window. When the fills ARE
            # complete (bucket inventory matches the settled counts) prefer the
            # fills cashflow, which also captures pre-settlement round-trips. ──
            settlement_rev = payout
            contracts = int(round(held_yes + held_no))
            side = "yes" if held_yes >= held_no else "no"
            fills_complete = (
                (b["yes_buys"] or b["no_buys"])
                and abs(b["yes_long"] - held_yes) < 1.0
                and abs(b["no_long"] - held_no) < 1.0
            )
            if fills_complete:
                realized = cashflow + payout - fees
                cost_basis = abs(cashflow)
                invested = b["invested"]
            else:
                realized = payout - settle_cost - settle_fee
                cost_basis = settle_cost
                invested = settle_cost
                fees = settle_fee
                cashflow = -settle_cost
            avg_cost = (cost_basis / contracts) if contracts > 0 else 0.0
        else:
            # ── CLOSED via fills only (no settlement record, not open): fully
            # round-tripped before expiry. Trust the fills cashflow. ──
            contracts = 0
            net_yes = b["yes_buys"] - b["yes_sells"]
            net_no = b["no_buys"] - b["no_sells"]
            side = ("yes" if abs(net_yes) >= abs(net_no)
                    else "no") if (net_yes or net_no) else ""
            realized = cashflow - fees
            cost_basis = abs(cashflow)
            invested = b["invested"]
            avg_cost = 0.0

        net = realized + unreal

        rows.append({
            "ticker": ticker,
            "title": title,
            "market_type": market_type,
            "event_code": event_code,
            "player_code": player_code,
            "is_golf": _is_golf_ticker(ticker),
            "is_open": is_open,
            "side": side,
            "contracts": contracts,
            "yes_buys": b["yes_buys"],
            "yes_sells": b["yes_sells"],
            "no_buys": b["no_buys"],
            "no_sells": b["no_sells"],
            "avg_cost_dollars": avg_cost,
            "exposure_dollars": cost_basis,
            "invested_dollars": invested,
            "auto_converted_pairs": b["auto_converted"],
            "cashflow_dollars": cashflow,
            "settlement_revenue_dollars": settlement_rev,
            "realized_pnl_dollars": realized,
            "unrealized_pnl_dollars": unreal,
            "fees_paid_dollars": fees,
            "mark_dollars": mark,
            "net_pnl_dollars": net,
            "market_result": market_result,
            "settled_time": settled_time,
            "first_fill_ts": b["first_ts"],
            "last_fill_ts": b["last_ts"],
            "return_pct": (unreal / cost_basis * 100.0) if (is_open and cost_basis > 0) else None,
            "last_updated_ts": settled_time or b["last_ts"],
        })

        if ticker not in cached_titles and title:
            new_for_ledger.append({
                "ticker": ticker,
                "title": title,
                "recorded_ts": now.isoformat(timespec="seconds"),
            })

    if new_for_ledger:
        try:
            os.makedirs(os.path.dirname(PNL_HISTORY_PATH), exist_ok=True)
            new_df = pd.DataFrame(new_for_ledger)
            combined = (new_df if history.empty
                        else pd.concat([history, new_df], ignore_index=True))
            combined = combined.drop_duplicates(subset=["ticker"], keep="last")
            combined.to_parquet(PNL_HISTORY_PATH, index=False)
        except Exception as e:
            print(f"[pnl] failed to persist ledger: {e}")

    totals = {
        "realized": sum(r["realized_pnl_dollars"] for r in rows),
        "unrealized": sum(r.get("unrealized_pnl_dollars", 0.0) for r in rows),
        "fees": sum(r["fees_paid_dollars"] for r in rows),
        "net": sum(r["net_pnl_dollars"] for r in rows),
        "invested": sum(r.get("invested_dollars", 0.0) for r in rows),
        "open_count": sum(1 for r in rows if r["is_open"]),
        "settled_count": sum(1 for r in rows if not r["is_open"]),
    }

    result = {
        "rows": rows,
        "totals": totals,
        "fetched_ts": now.isoformat(timespec="seconds"),
        "new_settled_count": len(new_for_ledger),
        # Store telemetry: how much came from the durable parquet vs. freshly
        # pulled this load. fills_new/settlements_new should be small after the
        # first sync — that's the "only scrape recent" win.
        "store": {
            "fills_total": len(all_fills), "fills_new": n_new_fills,
            "settlements_total": len(settlements), "settlements_new": n_new_setts,
        },
    }
    _PNL_CACHE["data"] = result
    _PNL_CACHE["fetched_at"] = now
    return jsonify({**result, "from_cache": False})


@app.route("/api/novig")
def novig_edges():
    """Live NoVig outright edges (read-only).

    Public GraphQL endpoint, no auth. Tournament is the current `tourney`
    from sim_inputs; sim probs are loaded from the latest persisted
    rank_probs + simulated_probs files. 60s cache; ?refresh=1 to force.
    """
    if not _enabled():
        return jsonify({"error": "disabled"}), 403

    force = request.args.get("refresh") == "1"
    now = dt.datetime.now()
    if (not force
            and _NOVIG_CACHE["data"] is not None
            and _NOVIG_CACHE["fetched_at"]
            and (now - _NOVIG_CACHE["fetched_at"]).total_seconds() < _NOVIG_CACHE_TTL_SEC):
        return jsonify({**_NOVIG_CACHE["data"], "from_cache": True})

    sys.path.insert(0, REPO_ROOT)
    try:
        from sim_inputs import tourney
    except Exception as e:
        return jsonify({"error": f"cannot read sim_inputs.tourney: {e}"}), 500

    try:
        from novig_pricer import price_novig_outrights
        df = price_novig_outrights(tourney, project_root=REPO_ROOT)
    except Exception as e:
        return jsonify({"error": f"novig_pricer failed: {e}"}), 500

    tournament_name = ""
    if hasattr(df, "attrs"):
        tournament_name = df.attrs.get("tournament", "") or ""

    rows = [] if df.empty else df.to_dict(orient="records")
    positive_edges = int((df["edge"] > 0).sum()) if not df.empty and "edge" in df.columns else 0

    result = {
        "rows": rows,
        "tournament_name": tournament_name,
        "tourney": tourney,
        "positive_edges": positive_edges,
        "fetched_ts": now.isoformat(timespec="seconds"),
    }
    _NOVIG_CACHE["data"] = result
    _NOVIG_CACHE["fetched_at"] = now
    _NOVIG_CACHE["tournament_name"] = tournament_name
    return jsonify({**result, "from_cache": False})


if __name__ == "__main__":
    if not _enabled():
        print("Refusing to start: KALSHI_MAKER_ENABLED is not set to '1'.")
        print("On Windows PowerShell: $env:KALSHI_MAKER_ENABLED='1'")
        sys.exit(1)
    # Headless store sync — run this on a schedule so fills are captured before
    # they age out of Kalshi's ~2-month feed, no browser/server needed.
    if "--sync" in sys.argv:
        sys.path.insert(0, REPO_ROOT)
        from kalshi_maker import _authed_request
        _, _, nf, ns = _sync_pnl_stores(_authed_request)
        print(f"[pnl sync] +{nf} fills, +{ns} settlements persisted to "
              f"{os.path.relpath(os.path.dirname(FILLS_STORE_PATH), REPO_ROOT)}/")
        sys.exit(0)
    print("[maker dashboard] http://127.0.0.1:8051/")
    # Bind to localhost only — never expose to network.
    app.run(host="127.0.0.1", port=8051, debug=False)
