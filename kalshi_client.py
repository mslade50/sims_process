"""Kalshi authenticated API client.

Thin wrapper around the Kalshi trading API. RSA-PSS signing, no login endpoint.
Each request gets 3 headers:
    KALSHI-ACCESS-KEY       — API key ID
    KALSHI-ACCESS-SIGNATURE — base64(RSA-PSS-sign(timestamp + method + path))
    KALSHI-ACCESS-TIMESTAMP — Unix millis

Credentials from .env:
    KALSHI_API_KEY           — API key ID
    KALSHI_PRIVATE_KEY_PATH  — Path to RSA private key PEM
"""

import base64
import os
import time
from pathlib import Path

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from dotenv import load_dotenv

load_dotenv()

API_BASE = "https://api.elections.kalshi.com/trade-api/v2"

HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
}


class KalshiClient:
    """Authenticated Kalshi API client using RSA-PSS request signing."""

    def __init__(self):
        self.api_key = os.environ["KALSHI_API_KEY"]
        key_path = os.environ.get(
            "KALSHI_PRIVATE_KEY_PATH",
            str(Path.home() / ".kalshi" / "kalshi_private_key.pem"),
        )
        with open(key_path, "rb") as f:
            self._private_key = serialization.load_pem_private_key(f.read(), password=None)
        self._client = httpx.Client(timeout=15.0)

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        """RSA-PSS sign: timestamp_ms + method + path."""
        message = (timestamp_ms + method + path).encode("utf-8")
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _auth_headers(self, method: str, path: str) -> dict:
        """Build auth headers for a request."""
        ts = str(int(time.time() * 1000))
        sig = self._sign(ts, method, path)
        return {
            **HEADERS,
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": sig,
        }

    def _request(self, method: str, path: str, **kwargs) -> dict:
        """Make an authenticated request."""
        url = API_BASE + path
        # Sign with full path including /trade-api/v2 prefix
        sign_path = "/trade-api/v2" + path
        headers = self._auth_headers(method.upper(), sign_path)
        resp = self._client.request(method.upper(), url, headers=headers, **kwargs)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def _get(self, path: str, params: dict | None = None) -> dict:
        return self._request("GET", path, params=params)

    def _post(self, path: str, json_data: dict | None = None) -> dict:
        return self._request("POST", path, json=json_data)

    def _delete(self, path: str) -> dict:
        return self._request("DELETE", path)

    # ── Account ──────────────────────────────────────────────────────

    def get_balance(self) -> int:
        """Get account balance in cents."""
        data = self._get("/portfolio/balance")
        return data.get("balance", 0)

    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        data = self._get("/portfolio/positions")
        # API returns {market_positions: [...]}
        return data.get("market_positions", [])

    def get_open_orders(self) -> list[dict]:
        """Get all resting (open) orders."""
        all_orders = []
        cursor = None
        while True:
            params = {"status": "resting"}
            if cursor:
                params["cursor"] = cursor
            data = self._get("/portfolio/orders", params=params)
            orders = data.get("orders", [])
            all_orders.extend(orders)
            cursor = data.get("cursor")
            if not cursor or len(orders) < 100:
                break
        return all_orders

    # ── Orders ───────────────────────────────────────────────────────

    def place_order(
        self,
        ticker: str,
        side: str,
        price_cents: int,
        count: int,
        client_order_id: str | None = None,
    ) -> dict:
        """Place a limit order.

        Args:
            ticker: Market ticker (e.g. 'KXPGATOP5-26MAR28-XSCH')
            side: 'yes' or 'no'
            price_cents: Limit price in cents (1-99)
            count: Number of contracts
            client_order_id: Optional idempotency key

        Returns:
            Order response dict with order_id, status, etc.
        """
        body = {
            "ticker": ticker,
            "action": "buy",
            "side": side,
            "type": "limit",
            "yes_price": price_cents if side == "yes" else (100 - price_cents),
            "count": count,
        }
        if client_order_id:
            body["client_order_id"] = client_order_id
        return self._post("/portfolio/orders", json_data=body)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a single order. Returns True if successful."""
        try:
            self._delete(f"/portfolio/orders/{order_id}")
            return True
        except httpx.HTTPStatusError:
            return False

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count cancelled."""
        orders = self.get_open_orders()
        cancelled = 0
        for order in orders:
            if self.cancel_order(order["order_id"]):
                cancelled += 1
        return cancelled

    # ── Orderbook ─────────────────────────────────────────────────────

    def get_orderbook(self, ticker: str, depth: int = 0) -> dict:
        """Get orderbook for a market.

        Args:
            ticker: Market ticker
            depth: Number of levels (0 = all)

        Returns:
            dict with 'yes' and 'no' lists of [price_dollars, quantity] tuples,
            sorted best-to-worst.  Prices are floats in dollar units (0.0–1.0)
            to preserve sub-cent / deci-cent precision.
        """
        params = {"depth": depth} if depth > 0 else {}
        data = self._get(f"/markets/{ticker}/orderbook", params=params)
        ob = data.get("orderbook", data.get("orderbook_fp", data))

        def parse_levels(raw, scale):
            levels = []
            for item in (raw or []):
                if isinstance(item, list) and len(item) == 2:
                    price = float(item[0]) * scale  # normalized to dollars
                    qty = int(float(item[1]))
                    levels.append((price, qty))
            return levels

        # Contract: ALWAYS dollars. The legacy 'yes'/'no' shape carries integer
        # cents and must be scaled — without this the function's unit was
        # shape-dependent (2026-08 audit note).
        raw_yes = ob.get("yes_dollars")
        raw_no = ob.get("no_dollars")
        yes_levels = parse_levels(raw_yes, 1.0) if raw_yes is not None \
            else parse_levels(ob.get("yes", []), 0.01)
        no_levels = parse_levels(raw_no, 1.0) if raw_no is not None \
            else parse_levels(ob.get("no", []), 0.01)

        return {"yes": yes_levels, "no": no_levels}

    # ── Fills ────────────────────────────────────────────────────────

    def get_fills(self, ticker: str | None = None) -> list[dict]:
        """Get fill history, optionally filtered by ticker."""
        all_fills = []
        cursor = None
        while True:
            params = {}
            if ticker:
                params["ticker"] = ticker
            if cursor:
                params["cursor"] = cursor
            data = self._get("/portfolio/fills", params=params)
            fills = data.get("fills", [])
            all_fills.extend(fills)
            cursor = data.get("cursor")
            if not cursor or len(fills) < 100:
                break
        return all_fills

    # ── Market Data (public, but signed for consistency) ─────────

    def get_market(self, ticker: str) -> dict:
        """Get a single market by ticker."""
        data = self._get(f"/markets/{ticker}")
        return data.get("market", data)

    def get_markets(self, series_ticker: str) -> list[dict]:
        """Fetch all open markets for a series, handling pagination + 429 retry.

        Kalshi throttles burst paginated reads — page 2+ of large series like
        KXPGATOP20 / KXPGATOUR will 429 if hit back-to-back. Retry with
        retry-after backoff (5 attempts, capped at 5s each) before giving up.
        """
        all_mkts = []
        cursor = None
        while True:
            params = {"limit": 200, "status": "open", "series_ticker": series_ticker}
            if cursor:
                params["cursor"] = cursor
            data = None
            for attempt in range(5):
                try:
                    data = self._get("/markets", params=params)
                    break
                except httpx.HTTPStatusError as e:
                    if e.response.status_code != 429:
                        raise
                    wait = float(e.response.headers.get("retry-after", 1.0)) * (attempt + 1)
                    time.sleep(min(wait, 5.0))
            if data is None:
                data = self._get("/markets", params=params)  # final attempt, let it raise
            mkts = data.get("markets", [])
            all_mkts.extend(mkts)
            cursor = data.get("cursor")
            if not cursor or len(mkts) < 200:
                break
        return all_mkts


if __name__ == "__main__":
    c = KalshiClient()
    bal = c.get_balance()
    print(f"Balance: ${bal / 100:.2f}")
