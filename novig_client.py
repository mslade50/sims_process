"""NoVig (NBX) authenticated API client.

Thin wrapper around the NBX Sports Trading API. OAuth 2.0 client credentials.
Token auto-refreshes every 25 minutes (expires at 30).

Credentials from .env:
    NOVIG_CLIENT_ID      — OAuth client ID (request from NoVig)
    NOVIG_CLIENT_SECRET   — OAuth client secret
"""

import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

API_BASE = "https://api.novig.us"
TOKEN_PATH = "/nbx/v1/auth/emm-token"

HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
}


class NovigClient:
    """Authenticated NBX API client using OAuth 2.0 client credentials."""

    def __init__(self, base_url: str | None = None):
        self._base = (base_url or API_BASE).rstrip("/")
        self._client_id = os.environ["NOVIG_CLIENT_ID"]
        self._client_secret = os.environ["NOVIG_CLIENT_SECRET"]
        self._http = httpx.Client(timeout=15.0)
        self._token: str | None = None
        self._token_expiry: float = 0.0

    # ── Auth ──────────────────────────────────────────────────────────

    def _ensure_token(self):
        """Refresh OAuth token if expired or missing."""
        if self._token and time.time() < self._token_expiry:
            return
        resp = self._http.post(
            self._base + TOKEN_PATH,
            headers=HEADERS,
            json={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        # Refresh 5 min early (token lasts 30 min)
        expires_in = data.get("expires_in", 1800)
        self._token_expiry = time.time() + expires_in - 300

    def _auth_headers(self) -> dict:
        self._ensure_token()
        return {
            **HEADERS,
            "Authorization": f"Bearer {self._token}",
        }

    # ── HTTP helpers ──────────────────────────────────────────────────

    def _get(self, path: str, params: dict | None = None) -> dict | list:
        resp = self._http.get(
            self._base + path,
            headers=self._auth_headers(),
            params=params,
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def _post(self, path: str, json_data=None) -> dict | list:
        resp = self._http.post(
            self._base + path,
            headers=self._auth_headers(),
            json=json_data,
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def _delete(self, path: str, json_data=None) -> dict | list:
        resp = self._http.request(
            "DELETE",
            self._base + path,
            headers=self._auth_headers(),
            json=json_data,
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    # ── Account ───────────────────────────────────────────────────────

    def get_balance(self, currency: str = "CASH") -> float:
        """Get wallet balance. Returns dollars."""
        data = self._get("/nbx/v2/emm/account/balance", {"currency": currency})
        return float(data.get("balance", 0))

    def get_open_costs(self, currency: str = "CASH") -> dict:
        """Get open costs and available credit."""
        return self._get("/nbx/v2/emm/account/open-costs-and-credit", {"currency": currency})

    # ── Events & Markets ──────────────────────────────────────────────

    def get_leagues(self) -> list[str]:
        """Get all available league identifiers."""
        return self._get("/nbx/v2/emm/event-metadata/leagues")

    def get_events(
        self,
        league: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Fetch events with optional filters.

        Args:
            league: e.g. 'PGA', 'NFL', 'NBA'
            status: INITIAL, OPEN_PREGAME, CLOSED_PREGAME, OPEN_INGAME, FINAL, etc.
            limit: max 100
            offset: pagination offset
        """
        params = {"limit": limit, "offset": offset}
        if league:
            params["league"] = league
        if status:
            params["status"] = status
        return self._get("/nbx/v2/emm/events", params)

    def get_event(self, event_id: str) -> dict:
        """Get a single event by ID."""
        return self._get(f"/nbx/v2/emm/events/{event_id}")

    def get_markets_by_event(self, event_id: str, currency: str = "CASH") -> list[dict]:
        """Get all markets + orderbook for an event."""
        return self._get(
            f"/nbx/v2/emm/events/getMarketsByEvent/{event_id}",
            {"currency": currency},
        )

    def get_markets_by_events(self, event_ids: list[str], currency: str = "CASH") -> list[dict]:
        """Batch fetch markets + orderbooks for multiple events (max 50)."""
        return self._post(
            "/nbx/v2/emm/events/getMarketsByEvents",
            {"eventIds": event_ids, "currency": currency},
        )

    def get_open_markets(
        self,
        league: str | None = None,
        market_type: str | None = None,
    ) -> list[dict]:
        """Get all open markets with optional filters."""
        params = {}
        if league:
            params["league"] = league
        if market_type:
            params["marketType"] = market_type
        return self._get("/nbx/v2/emm/markets/open", params)

    def get_market(self, market_id: str) -> dict:
        """Get a single market by ID."""
        return self._get(f"/nbx/v2/emm/markets/{market_id}")

    # ── Orderbook ─────────────────────────────────────────────────────

    def get_orderbook(self, market_id: str, currency: str = "CASH") -> dict:
        """Get orderbook for a market.

        Returns dict with marketId, marketDescription, and outcomeLadders.
        Each ladder has outcomeId and bids: [{price, qty, ...}].

        Prices are decimal probabilities (0.000-1.000).
        Quantities are in cents.
        """
        return self._get(f"/nbx/v2/emm/book/{market_id}", {"currency": currency})

    # ── Orders ────────────────────────────────────────────────────────

    def place_order(
        self,
        outcome_id: str,
        price: float,
        qty_cents: int,
        currency: str = "CASH",
        tif: str = "GTC",
    ) -> dict:
        """Place a limit order.

        Args:
            outcome_id: Target outcome UUID
            price: Limit price as decimal probability (0.001-0.999, max 3 decimals)
            qty_cents: Quantity in cents (positive integer)
            currency: CASH or COIN
            tif: Time-in-force — GTC (good-til-cancel), IOC, FOK

        Returns:
            Order response with id, status, etc.
        """
        body = {
            "outcomeId": outcome_id,
            "price": round(price, 3),
            "qty": qty_cents,
            "currency": currency,
            "tif": tif,
        }
        return self._post("/nbx/v2/emm/orders/place", body)

    def place_orders(self, orders: list[dict]) -> list[dict]:
        """Place multiple orders (max 256).

        Each order dict: {outcomeId, price, qty, currency, tif, ttl?, flags?}
        """
        return self._post("/nbx/v2/emm/orders/batch", orders)

    def cancel_order(self, order_id: str) -> dict:
        """Cancel a single order."""
        return self._delete(f"/nbx/v2/emm/orders/{order_id}")

    def cancel_orders(self, order_ids: list[str]) -> list[dict]:
        """Cancel multiple orders (max 256)."""
        return self._delete("/nbx/v2/emm/orders/batch", order_ids)

    def cancel_all_orders(self) -> list[dict]:
        """Cancel all open orders."""
        return self._delete("/nbx/v2/emm/kill")

    def cancel_orders_by_event(self, event_id: str) -> list[dict]:
        """Cancel all orders for an event."""
        return self._delete(f"/nbx/v2/emm/orders/event/{event_id}")

    def cancel_orders_by_market(self, market_id: str) -> list[dict]:
        """Cancel all orders for a market."""
        return self._delete(f"/nbx/v2/emm/orders/market/{market_id}")

    def get_order(self, order_id: str) -> dict:
        """Get a single order by ID."""
        return self._get(f"/nbx/v2/emm/orders/{order_id}")

    def get_orders(
        self,
        currency: str = "CASH",
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Get all orders with optional filters."""
        params = {"currency": currency, "limit": limit, "offset": offset}
        if status:
            params["status"] = status
        return self._get("/nbx/v2/emm/orders/all", params)

    # ── Positions & Fills ─────────────────────────────────────────────

    def get_positions(self, currency: str = "CASH", outcome_ids: list[str] | None = None) -> list[dict]:
        """Get all open positions."""
        params = {"currency": currency}
        if outcome_ids:
            params["outcomeIds"] = outcome_ids
        return self._get("/nbx/v2/emm/positions/all", params)

    def get_fills(
        self,
        currency: str = "CASH",
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Get fill history."""
        return self._get("/nbx/v2/emm/fills/all", {
            "currency": currency,
            "limit": limit,
            "offset": offset,
        })

    # ── Transactions ──────────────────────────────────────────────────

    def get_transactions(
        self,
        currency: str = "CASH",
        limit: int = 100,
        offset: int = 0,
        tx_type: str | None = None,
    ) -> list[dict]:
        """Get transaction history."""
        params = {"currency": currency, "limit": limit, "offset": offset}
        if tx_type:
            params["type"] = tx_type
        return self._get("/nbx/v2/emm/transactions", params)


if __name__ == "__main__":
    c = NovigClient()
    bal = c.get_balance()
    print(f"Balance: ${bal:.2f}")

    leagues = c.get_leagues()
    print(f"Leagues: {leagues}")

    pga_events = c.get_events(league="PGA", status="OPEN_PREGAME")
    for e in pga_events:
        print(f"  {e.get('description', 'Unknown')} [{e.get('id', '')}] — {e.get('status', '')}")
        for mid in e.get("marketIds", [])[:3]:
            print(f"    market: {mid}")
