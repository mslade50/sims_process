"""Shadow run for the maker — dry-run against LIVE data with orders HARD-disabled.

It NEVER posts or cancels: three independent safeties —
  1. no --live (plain dry-run never calls the post/cancel path), AND
  2. MAKER_SHADOW=1 (post_limit/cancel_order become no-ops at the API boundary), AND
  3. the kill switch / live-round / fairs guards still gate everything.

Use it to validate the passive-accumulation engine + the guards against real
Kalshi books, DataGolf live status, and the current sim fairs — logging exactly
what it WOULD quote. Output streams to the console and to
permanent_data/shadow_logs/shadow_<ts>.log.

    python maker_shadow.py

Schedule it (Windows Task Scheduler / cron) every ~15-30 min on tournament days.
Needs the same local env as the maker: .env (KALSHI/DATAGOLF keys), the .kalshi
PEM, credentials.json, and the current sim fair files
(rank_probs_*_<tourney>.parquet) — which is why this runs on the machine where the
sim ran, not a bare CI checkout.
"""
import datetime
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def main():
    env = dict(os.environ)
    env["MAKER_SHADOW"] = "1"  # hard no-orders at the API boundary (belt + suspenders)

    logdir = HERE / "permanent_data" / "shadow_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = logdir / f"shadow_{ts}.log"

    # PLAIN dry-run — never --live. (Intentionally not passing through argv so a
    # stray --live can't reach the maker.)
    cmd = [sys.executable, str(HERE / "kalshi_maker.py")]
    header = f"=== shadow run {ts}  (MAKER_SHADOW=1, dry-run, NO orders) ===\n"
    print(header, end="")

    with open(logfile, "w", encoding="utf-8") as f:
        f.write(header)
        proc = subprocess.Popen(
            cmd, cwd=str(HERE), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        proc.wait()
        footer = f"\n=== shadow exit code {proc.returncode} ===\n"
        print(footer, end="")
        f.write(footer)

    print(f"[shadow] log: {logfile}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
