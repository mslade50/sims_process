"""Push model-edge proposals from the local sim to the Kalshi-exec app.

The sim writes `permanent_data/maker_proposals.parquet` (see kalshi_maker.py).
This reads it, keeps the strongest edge per ticker, and POSTs the set to the
serverless app's `/api/proposals` so the order ticket can overlay model fair /
edge per market. Display only — this never places an order.

Auth: the app's proposals POST is gated by the `X-Proposals-Token` header, which
must match the Cloudflare secret `PROPOSALS_TOKEN`. Set it locally:

    setx PROPOSALS_TOKEN "<the token>"          # or pass --token
    python kalshi_exec/push_proposals.py --push  # actually send

Without --push it is a DRY RUN: it prints what it would send and POSTs nothing.

Usage:
    python kalshi_exec/push_proposals.py                 # dry run
    python kalshi_exec/push_proposals.py --push          # send to prod
    python kalshi_exec/push_proposals.py --url http://127.0.0.1:8788 --push
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

DEFAULT_URL = "https://kalshi-exec.pages.dev"
DEFAULT_PARQUET = Path(__file__).resolve().parent.parent / "permanent_data" / "maker_proposals.parquet"

# Columns we forward (best-effort; missing ones are dropped per row).
FIELDS = ["ticker", "side", "sim_prob", "edge_pp", "best_bid", "best_ask", "post_price", "kelly_f"]


def _num(v) -> float | None:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def build_rows(df: pd.DataFrame) -> list[dict]:
    """One proposal per ticker — keep the row with the largest |edge_pp|."""
    if "ticker" not in df.columns:
        raise SystemExit("proposals parquet has no 'ticker' column — nothing to push")
    if "edge_pp" in df.columns:
        df = df.reindex(df["edge_pp"].abs().sort_values(ascending=False).index)
    seen: set[str] = set()
    rows: list[dict] = []
    for rec in df.to_dict("records"):
        ticker = str(rec.get("ticker") or "").strip()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        row = {"ticker": ticker}
        for f in FIELDS[1:]:
            if f in rec:
                val = rec[f] if f == "side" else _num(rec[f])
                if val is not None:
                    row[f] = val
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--url", default=os.getenv("KALSHI_EXEC_URL", DEFAULT_URL))
    ap.add_argument("--token", default=os.getenv("PROPOSALS_TOKEN", ""))
    ap.add_argument("--push", action="store_true", help="actually POST (default: dry run)")
    args = ap.parse_args()

    if not args.parquet.exists():
        print(f"no proposals file at {args.parquet} — run the maker preview first")
        return 1
    df = pd.read_parquet(args.parquet)
    rows = build_rows(df)
    scan_ts = str(df["scan_ts"].iloc[0]) if "scan_ts" in df.columns and len(df) else ""
    payload = {"scan_ts": scan_ts, "rows": rows}

    print(f"{len(rows)} proposals (scan_ts={scan_ts or 'n/a'}) -> {args.url}/api/proposals")
    for r in rows[:8]:
        edge = r.get("edge_pp")
        print(f"  {r['ticker']:<28} {r.get('side','?'):<3} "
              f"fair={(_num(r.get('sim_prob')) or 0) * 100:5.1f}c edge={edge if edge is None else f'{edge:+.1f}pp'}")
    if len(rows) > 8:
        print(f"  … +{len(rows) - 8} more")

    if not args.push:
        print("\nDRY RUN — pass --push to send. Nothing posted.")
        return 0
    if not args.token:
        print("\nrefusing to push: no token (set PROPOSALS_TOKEN or pass --token)", file=sys.stderr)
        return 2

    req = urllib.request.Request(
        f"{args.url}/api/proposals",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-Proposals-Token": args.token},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            print("response:", resp.status, resp.read().decode()[:300])
    except urllib.error.HTTPError as e:
        print("push failed:", e.code, e.read().decode()[:300], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
