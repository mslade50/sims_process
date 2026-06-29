"""One-line digest of a shadow log — skim what the maker would have done.

    python shadow_digest.py            # the latest shadow log
    python shadow_digest.py <file>     # a specific log
    python shadow_digest.py --tail 10  # one line per the last N logs

Parses the pipe-delimited [SHADOW] / [SHADOW-Q] lines the maker emits in dry-run.
"""
from __future__ import annotations

import pathlib
import re
import sys

LOGDIR = pathlib.Path(__file__).resolve().parent / "permanent_data" / "shadow_logs"


def _when(name):
    m = re.search(r"shadow_(\d{8})_(\d{6})", name)
    if not m:
        return "?"
    d, t = m.group(1), m.group(2)
    return f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}"


def _fields(line):
    out = {}
    for part in line.split("|"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def digest(path: pathlib.Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    when = _when(path.name)

    sline = next((ln for ln in text.splitlines() if ln.startswith("[SHADOW] ")), None)
    if not sline:
        tag = "ERROR" if "Traceback" in text else "no-summary"
        return f"{when} | {tag:<6} | (see {path.name})"

    f = _fields(sline[len("[SHADOW] "):])
    precond = f.get("precond", "?")
    reason = f.get("reason", "")
    n = f.get("quotes", "0")
    committed = f.get("committed", "0")
    status = "TRADE" if precond == "TRADE" else f"HALT ({reason})"

    quotes = []
    for ln in text.splitlines():
        if ln.startswith("[SHADOW-Q] "):
            p = [x.strip() for x in ln[len("[SHADOW-Q] "):].split("|")]
            if len(p) >= 5:
                player, market, side, _price, edge = p[0], p[1], p[2], p[3], p[4]
                quotes.append(f"{player} {market} {side} {edge}c")

    out = f"{when} | {status:<28} | {n:>3} quotes / ${committed} committed"
    if quotes:
        out += " | top: " + ", ".join(quotes[:4])
    return out


def main():
    args = sys.argv[1:]
    if args and args[0] == "--tail":
        n = int(args[1]) if len(args) > 1 else 10
        logs = sorted(LOGDIR.glob("shadow_*.log"))[-n:]
        if not logs:
            print("no shadow logs in", LOGDIR)
            return 1
        for p in logs:
            print(digest(p))
        return 0

    path = pathlib.Path(args[0]) if args else None
    if path is None:
        logs = sorted(LOGDIR.glob("shadow_*.log"))
        path = logs[-1] if logs else None
    if not path or not path.exists():
        print("no shadow log found (looked in", LOGDIR, ")")
        return 1
    print(digest(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
