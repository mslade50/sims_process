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

# In-memory PnL cache. The /markets batched call is cheap but we still don't
# want to hammer Kalshi on every UI filter toggle. Frontend reads from cache
# until it ages out (or the user clicks Refresh).
_PNL_CACHE = {"data": None, "fetched_at": None}
_PNL_CACHE_TTL_SEC = 60

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
        _resting_key, _resting_count,
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
        rows.append({
            "order_id": o.get("order_id"),
            "ticker": t,
            "title": titles.get(t, ""),
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

    # ── 1. Fetch all fills (paginated, all-time). ───────────────────────
    all_fills, cursor = [], None
    while True:
        path = "/trade-api/v2/portfolio/fills?limit=200"
        if cursor:
            path += f"&cursor={cursor}"
        r = _authed_request("GET", path)
        if r.status_code != 200:
            return jsonify({"error": f"fills {r.status_code}: {r.text[:200]}"}), 500
        data = r.json()
        all_fills.extend(data.get("fills", []))
        cursor = data.get("cursor")
        if not cursor:
            break

    # ── 2. Fetch settlements (paginated, all-time). ─────────────────────
    settlements, cursor = [], None
    while True:
        path = "/trade-api/v2/portfolio/settlements?limit=200"
        if cursor:
            path += f"&cursor={cursor}"
        r = _authed_request("GET", path)
        if r.status_code != 200:
            return jsonify({"error": f"settlements {r.status_code}: {r.text[:200]}"}), 500
        data = r.json()
        settlements.extend(data.get("settlements", []))
        cursor = data.get("cursor")
        if not cursor:
            break
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

    # ── 5. Walk fills → per-ticker cashflow + fees + side activity. ─────
    # cashflow = sum over fills of (+price * count for sells, -price * count
    # for buys). NEGATIVE when net long-bought, POSITIVE when net sold.
    # For YES/NO inference: when no live position remains, infer side from
    # which side had more activity. (Doesn't affect math, only display.)
    by_t = _dd(lambda: {"cashflow": 0.0, "fees": 0.0,
                        "yes_buys": 0, "yes_sells": 0,
                        "no_buys": 0, "no_sells": 0,
                        "first_ts": None, "last_ts": None})
    for f in all_fills:
        ticker = f.get("ticker") or f.get("market_ticker") or ""
        if not ticker:
            continue
        side = f.get("side", "")
        action = f.get("action", "")
        try:
            count = int(round(float(f.get("count_fp") or 0)))
        except (TypeError, ValueError):
            count = 0
        if count <= 0:
            continue
        try:
            raw_price = (f.get("yes_price_dollars") if side == "yes"
                         else f.get("no_price_dollars"))
            price = float(raw_price or 0)
        except (TypeError, ValueError):
            price = 0.0
        try:
            fee = float(f.get("fee_cost") or 0)
        except (TypeError, ValueError):
            fee = 0.0

        b = by_t[ticker]
        if action == "buy":
            b["cashflow"] -= price * count
            if side == "yes":
                b["yes_buys"] += count
            else:
                b["no_buys"] += count
        elif action == "sell":
            b["cashflow"] += price * count
            if side == "yes":
                b["yes_sells"] += count
            else:
                b["no_sells"] += count
        b["fees"] += fee

        ts = f.get("created_time") or f.get("ts") or ""
        if ts:
            if b["first_ts"] is None or ts < b["first_ts"]:
                b["first_ts"] = ts
            if b["last_ts"] is None or ts > b["last_ts"]:
                b["last_ts"] = ts

    # Settled-but-no-fills (super rare, e.g. legacy data) — still surface
    # the row with whatever the settlement record alone implies.
    for tk in settlements_by_ticker:
        if tk not in by_t:
            by_t[tk]  # creates default zeros

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
        settlement_rev = 0.0
        market_result = ""
        settled_time = None
        if sett is not None:
            settlement_rev = float(sett.get("revenue") or 0) / 100.0
            market_result = sett.get("market_result", "")
            settled_time = sett.get("settled_time")
            # fees on the settlement row are usually duplicates of per-fill
            # fees, so we don't double-count.

        pos_fp = pos_by_ticker.get(ticker)
        is_open = pos_fp is not None
        if is_open:
            side = "yes" if pos_fp > 0 else "no"
            contracts = int(round(abs(pos_fp)))
        else:
            contracts = 0
            net_yes = b["yes_buys"] - b["yes_sells"]
            net_no = b["no_buys"] - b["no_sells"]
            side = ("yes" if abs(net_yes) >= abs(net_no)
                    else "no") if (net_yes or net_no) else ""

        market_type, event_code, player_code = _classify_ticker(ticker)
        sm = summaries.get(ticker) or {}
        title = sm.get("title") or cached_titles.get(ticker, "")

        # MTM for open positions
        mark = None
        unreal = 0.0
        if is_open:
            yes_bid = sm.get("yes_bid", 0.0)
            yes_ask = sm.get("yes_ask", 0.0)
            if yes_bid > 0 and yes_ask > 0 and yes_ask >= yes_bid:
                yes_mid = (yes_bid + yes_ask) / 2.0
                mark = yes_mid if side == "yes" else (1.0 - yes_mid)
                unreal = mark * contracts
            else:
                # Thin/empty book — assume mark = avg cost (no MTM PnL).
                # avg cost here = absolute cost basis of open position;
                # without FIFO we approximate as |cashflow| / contracts.
                mark = (abs(cashflow) / contracts) if contracts else 0.0
                unreal = mark * contracts

        realized = cashflow + settlement_rev - fees
        net = realized + unreal
        # avg_cost is conceptually -cashflow / contracts when net long. We
        # only display this for open positions; for settled it's misleading.
        avg_cost = (-cashflow / contracts) if (is_open and contracts > 0) else 0.0
        cost_basis = abs(cashflow)

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
        "open_count": sum(1 for r in rows if r["is_open"]),
        "settled_count": sum(1 for r in rows if not r["is_open"]),
    }

    result = {
        "rows": rows,
        "totals": totals,
        "fetched_ts": now.isoformat(timespec="seconds"),
        "new_settled_count": len(new_for_ledger),
    }
    _PNL_CACHE["data"] = result
    _PNL_CACHE["fetched_at"] = now
    return jsonify({**result, "from_cache": False})


if __name__ == "__main__":
    if not _enabled():
        print("Refusing to start: KALSHI_MAKER_ENABLED is not set to '1'.")
        print("On Windows PowerShell: $env:KALSHI_MAKER_ENABLED='1'")
        sys.exit(1)
    print("[maker dashboard] http://127.0.0.1:8051/")
    # Bind to localhost only — never expose to network.
    app.run(host="127.0.0.1", port=8051, debug=False)
