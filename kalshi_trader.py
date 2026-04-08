"""Kalshi order generation, approval, and execution.

CLI modes:
    python kalshi_trader.py                  # Compute + approve + submit
    python kalshi_trader.py --dry-run        # Compute + display only
    python kalshi_trader.py --status         # Balance + positions + open orders
    python kalshi_trader.py --cancel-all     # Cancel all open orders
    python kalshi_trader.py --reconcile      # Sync fills to local ledger
"""

import argparse
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from kalshi_client import KalshiClient

# -- Config ------------------------------------------------------------

KALSHI_BANKROLL = float(os.environ.get("KALSHI_BANKROLL", "200"))
KELLY_FRACTION = 0.60
MIN_EDGE_OUTRIGHT = 0.5   # percentage points
MIN_EDGE_WINNER = 0.5
MAX_SPREAD = 0.12         # skip markets with bid/ask spread > 12 cents
MAX_CONTRACTS = 50        # per market
MAX_RISK_PCT = 0.40       # max 40% of bankroll in open risk

LEDGER_PATH = Path(__file__).parent / "permanent_data" / "kalshi_orders.parquet"

# Series tickers for golf outrights
OUTRIGHT_SERIES = {
    "KXPGATOP5": "top_5",
    "KXPGATOP10": "top_10",
    "KXPGATOP20": "top_20",
    "KXPGATOUR": "winner",
}

# Map market type → sim probability column (try nodh first, fallback to regular)
TYPE_TO_COL = {
    "top_5": ("top_5_nodh", "top_5"),
    "top_10": ("top_10_nodh", "top_10"),
    "top_20": ("top_20_nodh", "top_20"),
    "winner": ("simulated_win_prob", "simulated_win_prob"),
}

MIN_EDGE = {
    "top_5": MIN_EDGE_OUTRIGHT,
    "top_10": MIN_EDGE_OUTRIGHT,
    "top_20": MIN_EDGE_OUTRIGHT,
    "winner": MIN_EDGE_WINNER,
}


# -- Name Matching -----------------------------------------------------

try:
    from sim_inputs import name_replacements
except ImportError:
    name_replacements = {}


def norm(s: str) -> str:
    """Normalize player name to 'last, first' format."""
    x = s.strip().lower()
    if "," not in x:
        parts = x.rsplit(" ", 1)
        if len(parts) == 2:
            x = f"{parts[1]}, {parts[0]}"
    return name_replacements.get(x, x)


def extract_player_from_title(title: str) -> str:
    """Extract player name from Kalshi market title."""
    # Outright: "...: Will X finish top N?"
    m = re.match(r".*:\s*Will (.+?) (?:finish|make|miss|lead|win)", title)
    if m:
        return m.group(1).strip()
    # Winner: "Will X win the Tournament?"
    m = re.match(r"Will (.+?) win the ", title)
    if m:
        return m.group(1).strip()
    return ""


def detect_tournament_from_title(title: str) -> str:
    """Extract tournament name from market title."""
    m = re.search(r"(?:at|in|win) the (.+?)\?", title)
    if m:
        return m.group(1).strip()
    m = re.match(r"(.+?):\s*Will", title)
    if m:
        return m.group(1).strip()
    return ""


# -- Sim Loading -------------------------------------------------------

def find_sim_file(tourney_slug: str | None = None) -> Path | None:
    """Find the most recent sim probabilities CSV."""
    sim_dir = Path(__file__).parent
    candidates = []

    # Try live first, then pre-tournament
    for pattern in ["top_finish_probs_live_*.csv", "top_finish_probs_*.csv"]:
        for f in sim_dir.glob(pattern):
            if tourney_slug and tourney_slug not in f.stem:
                continue
            candidates.append(f)

    if not candidates:
        return None

    # Most recently modified
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_sim_probs(path: Path) -> pd.DataFrame:
    """Load sim probabilities, handling both nodh and regular columns."""
    df = pd.read_csv(path)
    # Ensure player_name column exists
    if "player_name" not in df.columns:
        # Try first column
        df.rename(columns={df.columns[0]: "player_name"}, inplace=True)
    return df


def get_prob_col(df: pd.DataFrame, market_type: str) -> str | None:
    """Get the right probability column for a market type."""
    primary, fallback = TYPE_TO_COL.get(market_type, (None, None))
    if primary and primary in df.columns:
        return primary
    if fallback and fallback in df.columns:
        return fallback
    return None


# -- Market Fetching ---------------------------------------------------

def fetch_live_markets(client: KalshiClient) -> list[dict]:
    """Fetch all open golf outright markets from Kalshi."""
    all_markets = []
    for series_ticker, mtype in OUTRIGHT_SERIES.items():
        try:
            mkts = client.get_markets(series_ticker)
            for m in mkts:
                m["_market_type"] = mtype
            all_markets.extend(mkts)
        except Exception as e:
            print(f"  Warning: failed to fetch {series_ticker}: {e}")
    return all_markets


def filter_current_tournament(markets: list[dict]) -> tuple[list[dict], str]:
    """Filter markets to current tournament (most common in titles)."""
    from collections import Counter
    tourn_counts = Counter()
    for m in markets:
        t = detect_tournament_from_title(m.get("title", ""))
        if t:
            tourn_counts[t] += 1

    if not tourn_counts:
        return markets, "Unknown"

    tournament = tourn_counts.most_common(1)[0][0]

    # Filter to only this tournament
    filtered = []
    for m in markets:
        t = detect_tournament_from_title(m.get("title", ""))
        if t == tournament or not t:
            filtered.append(m)

    return filtered, tournament


# -- Order Computation -------------------------------------------------

def compute_orders(
    sim_probs: pd.DataFrame,
    markets: list[dict],
    existing_positions: list[dict],
    bankroll: float,
    client: KalshiClient | None = None,
) -> list[dict]:
    """Compute proposed orders from sim probs vs live markets.

    If client is provided, fetches orderbook depth for edge-passing markets
    and caps contract count at available depth.
    """
    import time as _time

    EDGE_RETAIN = 0.005  # 0.5% edge retention for depth calc

    # Build position lookup: ticker → net contracts
    pos_lookup = {}
    for p in existing_positions:
        ticker = p.get("ticker", "")
        # Kalshi positions have yes/no quantities
        yes_qty = p.get("position", 0)
        pos_lookup[ticker] = yes_qty

    # Track total open risk
    total_risk = sum(
        abs(qty) * 0.50  # conservative estimate at 50c avg price
        for qty in pos_lookup.values()
    )

    orders = []
    for m in markets:
        ticker = m.get("ticker", "")
        title = m.get("title", "")
        mtype = m.get("_market_type", "")

        bid = float(m.get("yes_bid", 0) or 0) / 100.0  # cents → dollars
        ask = float(m.get("yes_ask", 0) or 0) / 100.0

        # Also try dollar fields (API format varies)
        if bid == 0 and ask == 0:
            bid = float(m.get("yes_bid_dollars") or 0)
            ask = float(m.get("yes_ask_dollars") or 0)

        if bid <= 0 or ask <= 0:
            continue

        spread = ask - bid
        if spread > MAX_SPREAD:
            continue

        mid = (bid + ask) / 2.0

        # Extract and normalize player name
        player_raw = extract_player_from_title(title)
        if not player_raw:
            continue
        player = norm(player_raw)

        # Look up sim probability
        prob_col = get_prob_col(sim_probs, mtype)
        if not prob_col:
            continue

        match = sim_probs[sim_probs["player_name"] == player]
        if match.empty:
            continue

        sim_prob = float(match.iloc[0][prob_col])
        if sim_prob <= 0:
            continue

        # Compute edge
        edge = (sim_prob - mid) * 100.0  # in percentage points
        min_edge = MIN_EDGE.get(mtype, MIN_EDGE_OUTRIGHT)

        # Determine side
        # price_cents = the yes_price we send to Kalshi API
        # cost_cents  = what we actually pay per contract (= price for YES, 100-price for NO)
        if edge >= min_edge:
            side = "yes"
            # Post at fair - 0.5pp to retain minimum edge on fill
            price_cents = math.floor((sim_prob - 0.005) * 100)
            # Skip if current bid is already at or above our price (no room)
            if bid >= price_cents / 100.0:
                continue
        elif (mid - sim_prob) * 100.0 >= min_edge:
            # NO side: edge = mid - sim_prob (we think YES is overpriced)
            side = "no"
            edge = (mid - sim_prob) * 100.0
            # Our NO fair value in yes_price terms: sim_prob + 0.5pp
            # (posting YES side higher = cheaper NO for us)
            price_cents = math.ceil((sim_prob + 0.005) * 100)
            # Skip if no room — NO ask = 1 - YES bid
            no_cost = (100 - price_cents) / 100.0
            no_ask = 1.0 - bid
            if no_ask <= no_cost:
                continue
        else:
            continue

        # Clamp price
        price_cents = max(1, min(99, price_cents))

        # Cost per contract: what we actually pay
        if side == "yes":
            cost_cents = price_cents
        else:
            cost_cents = 100 - price_cents
        cost_dollars = cost_cents / 100.0

        # Kelly sizing (based on cost, not yes_price)
        if side == "yes":
            b = (1.0 / cost_dollars) - 1.0
            f_star = (b * sim_prob - (1 - sim_prob)) / b if b > 0 else 0
        else:
            b = (1.0 / cost_dollars) - 1.0
            no_prob = 1.0 - sim_prob
            f_star = (b * no_prob - sim_prob) / b if b > 0 else 0

        f_star = max(f_star, 0)
        raw_contracts = math.floor(bankroll * KELLY_FRACTION * f_star / cost_dollars)
        contracts = min(raw_contracts, MAX_CONTRACTS)

        if contracts <= 0:
            continue

        # Check existing position
        existing = pos_lookup.get(ticker, 0)
        if side == "yes" and existing > 0:
            contracts = max(0, contracts - existing)
        elif side == "no" and existing < 0:
            contracts = max(0, contracts - abs(existing))

        if contracts <= 0:
            continue

        # Orderbook depth cap: fetch depth and limit contracts
        depth = None
        if client is not None:
            try:
                ob = client.get_orderbook(ticker)
                _time.sleep(0.05)
                yes_levels = ob.get("yes", [])
                no_levels = ob.get("no", [])

                if side == "yes":
                    max_yes_cents = int((sim_prob - EDGE_RETAIN) * 100)
                    depth = sum(qty for (no_price, qty) in no_levels
                                if (100 - no_price) <= max_yes_cents)
                else:
                    max_no_cents = int(((1 - sim_prob) - EDGE_RETAIN) * 100)
                    depth = sum(qty for (yes_price, qty) in yes_levels
                                if (100 - yes_price) <= max_no_cents)

                if depth is not None and depth > 0:
                    contracts = min(contracts, depth)
                elif depth == 0:
                    continue  # no fillable depth at our edge
            except Exception:
                pass  # proceed without depth cap

        if contracts <= 0:
            continue

        # Check risk budget (risk = cost per contract * count)
        order_risk = contracts * cost_dollars
        if total_risk + order_risk > bankroll * MAX_RISK_PCT:
            # Reduce to fit
            room = bankroll * MAX_RISK_PCT - total_risk
            if room <= 0:
                continue
            contracts = min(contracts, math.floor(room / cost_dollars))
            if contracts <= 0:
                continue
            order_risk = contracts * cost_dollars

        total_risk += order_risk

        orders.append({
            "ticker": ticker,
            "player": player,
            "player_raw": player_raw,
            "market_type": mtype,
            "side": side,
            "count": contracts,
            "price_cents": price_cents,      # yes_price for API
            "cost_cents": cost_cents,         # actual cost per contract
            "sim_prob": sim_prob,
            "mid": mid,
            "edge": edge,
            "risk": order_risk,
            "depth": depth,
        })

    # Sort by edge descending
    orders.sort(key=lambda o: o["edge"], reverse=True)
    return orders


# -- Display -----------------------------------------------------------

def display_status(client: KalshiClient):
    """Display balance, positions, and open orders."""
    bal = client.get_balance()
    positions = client.get_positions()
    orders = client.get_open_orders()

    print(f"\n  Balance: ${bal / 100:.2f}")
    print(f"  Open positions: {len(positions)}")
    print(f"  Open orders: {len(orders)}")

    if positions:
        print("\n  -- Positions --")
        print(f"  {'Ticker':<40} {'Side':<5} {'Qty':>5}")
        print(f"  {'-'*40} {'-'*5} {'-'*5}")
        for p in positions:
            qty = p.get("position", 0)
            side = "YES" if qty > 0 else "NO"
            print(f"  {p.get('ticker', ''):<40} {side:<5} {abs(qty):>5}")

    if orders:
        print("\n  -- Open Orders --")
        print(f"  {'Ticker':<40} {'Side':<5} {'Qty':>5} {'Price':>7}")
        print(f"  {'-'*40} {'-'*5} {'-'*5} {'-'*7}")
        for o in orders:
            side = o.get("side", "").upper()
            price = o.get("yes_price", 0)
            print(f"  {o.get('ticker', ''):<40} {side:<5} {o.get('remaining_count', 0):>5} {price:>6}c")

    print()


def display_orders(orders: list[dict], tournament: str, balance_cents: int, positions: list, open_orders: list):
    """Display proposed orders for approval."""
    print()
    print("=" * 90)
    print(f"  PROPOSED KALSHI ORDERS - {tournament}")
    print(f"  Balance: ${balance_cents / 100:.2f} | Positions: {len(positions)} | Open orders: {len(open_orders)}")
    print("=" * 90)
    print()

    hdr = f"  {'#':>3}  {'Player':<22} {'Market':<8} {'Side':<5} {'Qty':>4} {'Cost':>6} {'Fair':>7} {'Risk':>7} {'Edge':>6} {'SimProb':>8} {'Depth':>6}"
    print(hdr)
    print(f"  {'-'*3}  {'-'*22} {'-'*8} {'-'*5} {'-'*4} {'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*8} {'-'*6}")

    total_risk = 0.0
    for i, o in enumerate(orders, 1):
        # Fair from the perspective of the side we're trading
        if o['side'] == 'yes':
            fair_cents = o['sim_prob'] * 100
        else:
            fair_cents = (1 - o['sim_prob']) * 100
        depth_str = f"{o['depth']:>6}" if o.get('depth') is not None else "   —  "
        print(
            f"  {i:>3}  {o['player']:<22} {o['market_type']:<8} "
            f"{o['side'].upper():<5} {o['count']:>4} "
            f"{o['cost_cents']:>5}c {fair_cents:>6.1f}c "
            f"${o['risk']:>5.2f} {o['edge']:>+5.1f} {o['sim_prob']:>7.1%} {depth_str}"
        )
        total_risk += o["risk"]

    print()
    print(f"  TOTAL RISK: ${total_risk:.2f} ({total_risk / KALSHI_BANKROLL * 100:.1f}% of bankroll)")
    print()


def prompt_approval(orders: list[dict]) -> list[dict]:
    """Prompt user to approve orders. Returns approved subset."""
    response = input("  Submit all? [Y/n/1,3]: ").strip()

    if not response or response.lower() == "y":
        return orders
    elif response.lower() == "n":
        return []
    else:
        # Parse comma-separated indices
        try:
            indices = [int(x.strip()) for x in response.split(",")]
            return [orders[i - 1] for i in indices if 1 <= i <= len(orders)]
        except (ValueError, IndexError):
            print("  Invalid selection, aborting.")
            return []


# -- Order Submission --------------------------------------------------

def submit_orders(client: KalshiClient, orders: list[dict]) -> list[dict]:
    """Submit approved orders and return results."""
    results = []
    for o in orders:
        try:
            resp = client.place_order(
                ticker=o["ticker"],
                side=o["side"],
                price_cents=o["price_cents"],
                count=o["count"],
            )
            order_id = resp.get("order", {}).get("order_id", "unknown")
            status = resp.get("order", {}).get("status", "unknown")
            print(f"  [OK] {o['player']} {o['market_type']} - {o['side'].upper()} x{o['count']} @ {o['price_cents']}c → {status} ({order_id})")
            results.append({**o, "order_id": order_id, "status": status})
        except Exception as e:
            print(f"  [FAIL] {o['player']} {o['market_type']} - FAILED: {e}")
            results.append({**o, "order_id": None, "status": "error", "error": str(e)})
    return results


# -- Ledger ------------------------------------------------------------

def save_to_ledger(results: list[dict]):
    """Append submitted orders to the local parquet ledger."""
    if not results:
        return

    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for r in results:
        rows.append({
            "order_id": r.get("order_id"),
            "timestamp": now,
            "ticker": r["ticker"],
            "market_type": r["market_type"],
            "player_name": r["player"],
            "side": r["side"],
            "price_cents": r["price_cents"],
            "count": r["count"],
            "status": r.get("status", "unknown"),
            "sim_prob": r["sim_prob"],
            "edge": r["edge"],
            "fill_price": None,
            "fill_count": 0,
            "pnl": None,
        })

    new_df = pd.DataFrame(rows)

    if LEDGER_PATH.exists():
        existing = pd.read_parquet(LEDGER_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(LEDGER_PATH, index=False)
    print(f"\n  Ledger updated: {LEDGER_PATH} ({len(combined)} total rows)")


def reconcile_fills(client: KalshiClient):
    """Sync fills from Kalshi to local ledger."""
    if not LEDGER_PATH.exists():
        print("  No ledger found. Nothing to reconcile.")
        return

    ledger = pd.read_parquet(LEDGER_PATH)
    fills = client.get_fills()

    if not fills:
        print("  No fills found.")
        return

    # Build fill lookup: order_id → fill details
    fill_lookup = {}
    for f in fills:
        oid = f.get("order_id", "")
        if oid not in fill_lookup:
            fill_lookup[oid] = {"fill_count": 0, "fill_price_sum": 0}
        count = f.get("count", 0)
        price = f.get("yes_price", 0)
        fill_lookup[oid]["fill_count"] += count
        fill_lookup[oid]["fill_price_sum"] += price * count

    updated = 0
    for idx, row in ledger.iterrows():
        oid = row.get("order_id")
        if oid and oid in fill_lookup:
            fl = fill_lookup[oid]
            if fl["fill_count"] > 0:
                avg_price = fl["fill_price_sum"] / fl["fill_count"]
                ledger.at[idx, "fill_count"] = fl["fill_count"]
                ledger.at[idx, "fill_price"] = avg_price
                ledger.at[idx, "status"] = "filled" if fl["fill_count"] >= row["count"] else "partial"
                updated += 1

    ledger.to_parquet(LEDGER_PATH, index=False)
    print(f"  Reconciled {updated} orders with fills.")
    print(f"  Ledger: {LEDGER_PATH}")


# -- Main --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Kalshi golf order manager")
    parser.add_argument("--dry-run", action="store_true", help="Compute + display only")
    parser.add_argument("--status", action="store_true", help="Show balance, positions, orders")
    parser.add_argument("--cancel-all", action="store_true", help="Cancel all open orders")
    parser.add_argument("--reconcile", action="store_true", help="Sync fills to local ledger")
    parser.add_argument("--tourney", type=str, default=None, help="Tournament slug filter")
    args = parser.parse_args()

    client = KalshiClient()

    if args.status:
        display_status(client)
        return

    if args.cancel_all:
        n = client.cancel_all_orders()
        print(f"  Cancelled {n} orders.")
        return

    if args.reconcile:
        reconcile_fills(client)
        return

    # -- Full order flow ------------------------------------------

    # 1. Load sim probabilities
    sim_path = find_sim_file(args.tourney)
    if not sim_path:
        print("  No sim file found. Run simulation first.")
        sys.exit(1)
    print(f"  Sim file: {sim_path.name}")

    sim_probs = load_sim_probs(sim_path)
    print(f"  Players in sim: {len(sim_probs)}")

    # 2. Fetch live markets
    print("  Fetching live Kalshi markets...")
    all_markets = fetch_live_markets(client)
    markets, tournament = filter_current_tournament(all_markets)
    print(f"  Tournament: {tournament} ({len(markets)} markets)")

    # 3. Account state
    balance = client.get_balance()
    positions = client.get_positions()
    open_orders = client.get_open_orders()

    # 4. Compute orders (with orderbook depth enrichment)
    orders = compute_orders(sim_probs, markets, positions, KALSHI_BANKROLL, client=client)

    if not orders:
        print("\n  No orders meet edge/sizing criteria.")
        return

    # 5. Display
    display_orders(orders, tournament, balance, positions, open_orders)

    if args.dry_run:
        print("  [DRY RUN - no orders submitted]")
        return

    # 6. Approval
    approved = prompt_approval(orders)
    if not approved:
        print("  Aborted.")
        return

    # 7. Submit
    print(f"\n  Submitting {len(approved)} orders...")
    results = submit_orders(client, approved)

    # 8. Save to ledger
    save_to_ledger(results)


if __name__ == "__main__":
    main()
