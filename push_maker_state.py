"""Push the maker cockpit snapshot (permanent_data/maker_state.json) to the
kalshi-exec dashboard's /api/maker so the Maker tab shows what the bot wants to
do right now. Display only — never places an order.

    python push_maker_state.py                  # push the latest snapshot
    python push_maker_state.py --dry-run        # print, don't POST

Auth: the X-Maker-Token header must match the Cloudflare secret MAKER_STATE_TOKEN
(or PROPOSALS_TOKEN). Set MAKER_STATE_TOKEN locally (or pass --token). maker_shadow.py
calls this automatically when a token is present.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://kalshi-exec.pages.dev"
STATE_PATH = Path(__file__).resolve().parent / "permanent_data" / "maker_state.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", type=Path, default=STATE_PATH)
    ap.add_argument("--url", default=os.getenv("KALSHI_EXEC_URL", DEFAULT_URL))
    ap.add_argument("--token", default=os.getenv("MAKER_STATE_TOKEN") or os.getenv("PROPOSALS_TOKEN", ""))
    ap.add_argument("--dry-run", action="store_true", help="print, do not POST")
    args = ap.parse_args()

    if not args.state.exists():
        print(f"no state at {args.state} — run the maker (or maker_shadow.py) first")
        return 1
    payload = args.state.read_text(encoding="utf-8")
    try:
        s = json.loads(payload)
    except Exception as e:
        print("state file is not valid JSON:", e)
        return 1

    print(f"{s.get('status')} · {s.get('totals', {}).get('quotes', 0)} quote(s) · "
          f"mode={s.get('mode')} -> {args.url}/api/maker")
    if args.dry_run:
        print("dry-run — not posted.")
        return 0
    if not args.token:
        print("no token (set MAKER_STATE_TOKEN / PROPOSALS_TOKEN or pass --token)", file=sys.stderr)
        return 2

    req = urllib.request.Request(
        f"{args.url}/api/maker", data=payload.encode(),
        headers={"Content-Type": "application/json", "X-Maker-Token": args.token}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            print("response:", r.status, r.read().decode()[:200])
    except urllib.error.HTTPError as e:
        print("push failed:", e.code, e.read().decode()[:200], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
