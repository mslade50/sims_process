"""Push model-edge proposals from the local sim to the Kalshi-exec app.

The sim writes `permanent_data/maker_proposals.parquet` (see kalshi_maker.py).
This reads it, keeps the strongest edge per ticker, and POSTs the set to the
serverless app's `/api/proposals` so the order ticket can overlay model fair /
edge per market. Display only — this never places an order.

Auth (two layers since 2026-08-19):
1. Cloudflare Access fronts the whole app — this script authenticates with the
   "kalshi-exec-push" service token via CF-Access-Client-Id/Secret headers,
   read from CF_ACCESS_CLIENT_ID / CF_ACCESS_CLIENT_SECRET (kalshi_exec/.env,
   gitignored, or process env). Without them Access 302s to the login page.
2. The app's own `X-Proposals-Token` header, which must match the Cloudflare
   secret `PROPOSALS_TOKEN`. Set it locally:

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


def _load_local_env() -> None:
    """Best-effort load of kalshi_exec/.env (no python-dotenv dependency)."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


_load_local_env()

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

    # Cloudflare Access service-token headers (the platform gate; app token below
    # is the second factor). Missing creds -> Access will 302 to the login page.
    cf_id = os.getenv("CF_ACCESS_CLIENT_ID", "")
    cf_secret = os.getenv("CF_ACCESS_CLIENT_SECRET", "")
    if not (cf_id and cf_secret):
        print("\nrefusing to push: CF_ACCESS_CLIENT_ID/SECRET not set "
              "(kalshi_exec/.env) — Access would redirect this request to the "
              "login page", file=sys.stderr)
        return 2

    # Browser UA: Cloudflare's edge bot-protection blocks the default Python
    # urllib client signature (CF error 1010).
    req = urllib.request.Request(
        f"{args.url}/api/proposals",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Proposals-Token": args.token,
            "CF-Access-Client-Id": cf_id,
            "CF-Access-Client-Secret": cf_secret,
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
        },
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
